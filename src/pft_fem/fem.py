"""
Finite Element Method solver for tumor growth simulation.

Implements a coupled reaction-diffusion and mechanical deformation model
for simulating tumor growth in brain tissue.

Mass Effect Model:
    The tumor growth model simulates the creation of new volume within the
    tumor matrix over time. As new tumor cells are created (conceptually new
    nodes), they displace surrounding brain tissue. The brain tissue is
    constrained within the posterior cranial fossa (fixed skull boundary),
    leading to compression of tissue structures.

    Key components:
    1. Reaction-diffusion: Models cell proliferation and migration, effectively
       creating new cells over time that increase tumor volume.
    2. Eigenstrain formulation: Volumetric expansion creates stress that pushes
       tissue outward from tumor regions.
    3. Radial mass effect: Additional force directed radially outward from
       tumor center, modeling pressure from new volume addition.
    4. Fixed boundary: Skull constraint prevents tissue from escaping, creating
       compression of brain tissue against the posterior fossa boundary.

Supports:
- Anisotropic material properties for white matter (fiber-aligned resistance)
- Compressible gray matter with uniform mechanical response
- Skull/boundary immovability constraints
- Tissue-specific diffusion and growth parameters
- Mass effect scaling for visible tissue displacement
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Callable, Dict, List, Any, TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, lsqr, splu

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from .mesh import TetMesh


# =============================================================================
# JIT-compiled assembly functions (50-100x faster than pure Python)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _assemble_mass_matrix_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled mass matrix assembly.

        Returns COO format arrays (rows, cols, data) for sparse matrix construction.
        """
        num_elements = len(elements)
        # Each element contributes 4x4 = 16 entries
        total_entries = num_elements * 16

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]

            for i in range(4):
                for j in range(4):
                    if i == j:
                        val = vol / 10.0
                    else:
                        val = vol / 20.0

                    rows[idx] = elem[i]
                    cols[idx] = elem[j]
                    data[idx] = val
                    idx += 1

        return rows, cols, data

    @njit(cache=True)
    def _assemble_diffusion_matrix_isotropic_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        shape_gradients: np.ndarray,
        diffusion_coeffs: np.ndarray,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled isotropic diffusion matrix assembly.

        For elements without fiber directions (gray matter, CSF).
        """
        num_elements = len(elements)
        total_entries = num_elements * 16

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]
            grads = shape_gradients[e]  # (4, 3)
            D = diffusion_coeffs[e]

            for i in range(4):
                for j in range(4):
                    # Isotropic: K_ij = D * V * (grad_i . grad_j)
                    val = D * vol * (
                        grads[i, 0] * grads[j, 0] +
                        grads[i, 1] * grads[j, 1] +
                        grads[i, 2] * grads[j, 2]
                    )

                    rows[idx] = elem[i]
                    cols[idx] = elem[j]
                    data[idx] = val
                    idx += 1

        return rows, cols, data

    @njit(cache=True)
    def _assemble_diffusion_matrix_anisotropic_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        shape_gradients: np.ndarray,
        diffusion_coeffs: np.ndarray,
        fiber_directions: np.ndarray,
        tissue_types: np.ndarray,
        white_matter_type: int,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled anisotropic diffusion matrix assembly (legacy version).

        Handles both isotropic and anisotropic elements based on tissue type.
        White matter uses anisotropic diffusion (2x along fibers, 0.5x perpendicular).

        Note: This is the legacy version with fixed anisotropy ratio.
        Use _assemble_diffusion_matrix_fa_anisotropic_jit for FA-dependent anisotropy.
        """
        num_elements = len(elements)
        total_entries = num_elements * 16

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]
            grads = shape_gradients[e]  # (4, 3)
            D_base = diffusion_coeffs[e]
            tissue = tissue_types[e]

            # Build diffusion tensor
            if tissue == white_matter_type:
                # Anisotropic diffusion for white matter
                f = fiber_directions[e]
                f_norm = np.sqrt(f[0]**2 + f[1]**2 + f[2]**2)
                if f_norm > 1e-10:
                    f = f / f_norm

                D_para = D_base * 2.0
                D_perp = D_base * 0.5

                # D_tensor = D_perp * I + (D_para - D_perp) * f ⊗ f
                D_tensor = np.zeros((3, 3))
                for a in range(3):
                    D_tensor[a, a] = D_perp
                    for b in range(3):
                        D_tensor[a, b] += (D_para - D_perp) * f[a] * f[b]
            else:
                # Isotropic diffusion
                D_tensor = np.zeros((3, 3))
                D_tensor[0, 0] = D_base
                D_tensor[1, 1] = D_base
                D_tensor[2, 2] = D_base

            for i in range(4):
                for j in range(4):
                    # K_ij = V * grad_i^T * D * grad_j
                    val = 0.0
                    for a in range(3):
                        for b in range(3):
                            val += grads[i, a] * D_tensor[a, b] * grads[j, b]
                    val *= vol

                    rows[idx] = elem[i]
                    cols[idx] = elem[j]
                    data[idx] = val
                    idx += 1

        return rows, cols, data

    @njit(cache=True)
    def _assemble_diffusion_matrix_fa_anisotropic_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        shape_gradients: np.ndarray,
        diffusion_coeffs: np.ndarray,
        fiber_directions: np.ndarray,
        fa_values: np.ndarray,
        tissue_types: np.ndarray,
        white_matter_type: int,
        fa_anisotropy_factor: float,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled FA-dependent anisotropic diffusion matrix assembly.

        Diffusion anisotropy is proportional to local FA values:
        D_parallel/D_perpendicular = 1 + k * FA, where k is fa_anisotropy_factor.

        This provides more realistic tumor spread patterns, as tumor cells
        preferentially migrate along white matter tracts with high FA.

        Literature reference: Tumor cells show enhanced migration along high-FA
        white matter tracts (Giese et al., 2003; Painter & Hillen, 2013).

        Args:
            elements: Element connectivity array
            volumes: Element volumes
            shape_gradients: Shape function gradients per element
            diffusion_coeffs: Base diffusion coefficient per element
            fiber_directions: Fiber orientation vectors per element
            fa_values: Fractional anisotropy per element (0-1)
            tissue_types: Tissue type per element
            white_matter_type: Integer value for white matter tissue type
            fa_anisotropy_factor: Scaling factor k for FA-dependent anisotropy (typically 4-9)
            num_nodes: Total number of nodes
        """
        num_elements = len(elements)
        total_entries = num_elements * 16

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]
            grads = shape_gradients[e]  # (4, 3)
            D_base = diffusion_coeffs[e]
            tissue = tissue_types[e]
            fa = fa_values[e]

            # Build diffusion tensor
            if tissue == white_matter_type and fa > 0.1:
                # FA-dependent anisotropic diffusion for white matter
                f = fiber_directions[e]
                f_norm = np.sqrt(f[0]**2 + f[1]**2 + f[2]**2)
                if f_norm > 1e-10:
                    f = f / f_norm

                # FA-dependent anisotropy ratio: ratio = 1 + k * FA
                # For FA=0: isotropic (ratio=1)
                # For FA=1 with k=6: ratio=7 (strong anisotropy)
                anisotropy_ratio = 1.0 + fa_anisotropy_factor * fa

                # D_parallel = D_base * sqrt(ratio) to maintain geometric mean
                # D_perpendicular = D_base / sqrt(ratio)
                # This preserves D_para * D_perp^2 ≈ D_base^3 (trace preservation)
                sqrt_ratio = np.sqrt(anisotropy_ratio)
                D_para = D_base * sqrt_ratio
                D_perp = D_base / sqrt_ratio

                # D_tensor = D_perp * I + (D_para - D_perp) * f ⊗ f
                D_tensor = np.zeros((3, 3))
                for a in range(3):
                    D_tensor[a, a] = D_perp
                    for b in range(3):
                        D_tensor[a, b] += (D_para - D_perp) * f[a] * f[b]
            else:
                # Isotropic diffusion (gray matter, CSF, or low-FA white matter)
                D_tensor = np.zeros((3, 3))
                D_tensor[0, 0] = D_base
                D_tensor[1, 1] = D_base
                D_tensor[2, 2] = D_base

            for i in range(4):
                for j in range(4):
                    # K_ij = V * grad_i^T * D * grad_j
                    val = 0.0
                    for a in range(3):
                        for b in range(3):
                            val += grads[i, a] * D_tensor[a, b] * grads[j, b]
                    val *= vol

                    rows[idx] = elem[i]
                    cols[idx] = elem[j]
                    data[idx] = val
                    idx += 1

        return rows, cols, data

    @njit(cache=True)
    def _compute_element_stiffness_isotropic(
        grads: np.ndarray,
        vol: float,
        lam: float,
        mu: float,
    ) -> np.ndarray:
        """Compute 12x12 element stiffness matrix for isotropic material."""
        Ke = np.zeros((12, 12))

        # Constitutive matrix (isotropic linear elasticity)
        C = np.array([
            [lam + 2*mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2*mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2*mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ])

        for i in range(4):
            dNdx_i, dNdy_i, dNdz_i = grads[i, 0], grads[i, 1], grads[i, 2]
            Bi = np.array([
                [dNdx_i, 0.0, 0.0],
                [0.0, dNdy_i, 0.0],
                [0.0, 0.0, dNdz_i],
                [dNdy_i, dNdx_i, 0.0],
                [dNdz_i, 0.0, dNdx_i],
                [0.0, dNdz_i, dNdy_i],
            ])

            for j in range(4):
                dNdx_j, dNdy_j, dNdz_j = grads[j, 0], grads[j, 1], grads[j, 2]
                Bj = np.array([
                    [dNdx_j, 0.0, 0.0],
                    [0.0, dNdy_j, 0.0],
                    [0.0, 0.0, dNdz_j],
                    [dNdy_j, dNdx_j, 0.0],
                    [dNdz_j, 0.0, dNdx_j],
                    [0.0, dNdz_j, dNdy_j],
                ])

                # K_ij = V * Bi^T @ C @ Bj
                for a in range(3):
                    for b in range(3):
                        val = 0.0
                        for k in range(6):
                            for l in range(6):
                                val += Bi[k, a] * C[k, l] * Bj[l, b]
                        Ke[i*3 + a, j*3 + b] = vol * val

        return Ke

    @njit(cache=True)
    def _assemble_stiffness_matrix_isotropic_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        shape_gradients: np.ndarray,
        lam_values: np.ndarray,
        mu_values: np.ndarray,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled stiffness matrix assembly for isotropic materials.
        """
        num_elements = len(elements)
        # Each element contributes 12x12 = 144 entries
        total_entries = num_elements * 144

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]
            grads = shape_gradients[e]
            lam = lam_values[e]
            mu = mu_values[e]

            # Compute element stiffness matrix
            Ke = _compute_element_stiffness_isotropic(grads, vol, lam, mu)

            # Assemble into global arrays
            for i in range(4):
                for j in range(4):
                    for di in range(3):
                        for dj in range(3):
                            global_i = elem[i] * 3 + di
                            global_j = elem[j] * 3 + dj
                            local_i = i * 3 + di
                            local_j = j * 3 + dj

                            rows[idx] = global_i
                            cols[idx] = global_j
                            data[idx] = Ke[local_i, local_j]
                            idx += 1

        return rows, cols, data

if TYPE_CHECKING:
    from .biophysical_constraints import BiophysicalConstraints, AnisotropicMaterialProperties


class TissueType(Enum):
    """Brain tissue types with different mechanical properties."""

    GRAY_MATTER = 0
    WHITE_MATTER = 1
    CSF = 2
    TUMOR = 3
    EDEMA = 4
    SKULL = 5  # Immovable boundary


@dataclass
class MaterialProperties:
    """
    Material properties for brain tissue FEM simulation.

    Properties are based on literature values for brain tissue mechanics.

    Gray matter: More compressible (lower Poisson ratio), isotropic
    White matter: Nearly incompressible, anisotropic along fiber direction
    """

    # Elastic properties
    young_modulus: float = 3000.0  # Pa (brain tissue ~1-10 kPa)
    poisson_ratio: float = 0.45  # Nearly incompressible

    # Tumor growth parameters
    # These defaults model a non-infiltrative, expansile mass (e.g., pilocytic astrocytoma)
    # that grows as a solid mass with minimal invasion into surrounding tissue
    proliferation_rate: float = 0.04  # 1/day - higher rate for solid mass growth
    diffusion_coefficient: float = 0.01  # mm^2/day - very low for minimal infiltration
    carrying_capacity: float = 1.0  # Normalized max cell density

    # Mechanical coupling (eigenstrain formulation)
    # This represents volumetric strain per unit cell density.
    # Value of 0.25 means 25% volumetric expansion at full tumor density.
    # Higher value produces more displacement for solid mass tumors.
    growth_stress_coefficient: float = 0.25  # Volumetric strain per unit density

    # Mass effect parameters for tumor-induced tissue displacement
    # These control how tumor growth creates new volume that displaces tissue
    # Higher values for expansile tumors that push tissue aside rather than infiltrate
    mass_effect_scaling: float = 30.0  # Amplification of displacement from mass addition
    radial_displacement_factor: float = 12.0  # Additional radial outward force

    # Volume-preserving mass effect (physics-based formulation)
    # When enabled, replaces empirical scaling with volumetric strain model
    use_volume_preserving_mass_effect: bool = True
    # Characteristic decay length for radial pressure (multiples of tumor radius)
    pressure_decay_length_factor: float = 2.0

    # Adaptive time stepping parameters
    # When enabled, time step is adjusted based on density change rate
    use_adaptive_stepping: bool = True
    # Minimum allowed time step (days)
    dt_min: float = 0.1
    # Maximum allowed time step (days)
    dt_max: float = 5.0
    # Target relative density change per step (lower = smaller steps)
    adaptive_target_change: float = 0.05
    # Density change thresholds for step adjustment
    adaptive_increase_threshold: float = 0.01  # Increase dt if change < this
    adaptive_decrease_threshold: float = 0.1   # Decrease dt if change > this

    # Ventricular compliance modeling
    # CSF-filled ventricles act as compliant regions that absorb displacement
    # before surrounding parenchyma is compressed
    use_ventricular_compliance: bool = True
    # Stiffness reduction factor for ventricular/CSF regions
    # Lower value = more compliant (easier to compress)
    # Range: 0.001-0.1 typical (100-1000x softer than parenchyma)
    ventricular_compliance_factor: float = 0.01
    # Bulk modulus for CSF (Pa) - very soft, nearly incompressible
    csf_bulk_modulus: float = 100.0

    # Anisotropy parameters for white matter
    anisotropy_ratio: float = 2.0  # Ratio of parallel/perpendicular stiffness
    fiber_direction: Optional[NDArray[np.float64]] = None  # Local fiber direction

    # FA-dependent diffusion anisotropy (for tumor spread modeling)
    # D_parallel/D_perpendicular = 1 + fa_anisotropy_factor * FA
    # Literature suggests k=4-9 for realistic tumor migration patterns
    fa_anisotropy_factor: float = 6.0

    # Tissue-specific multipliers
    tissue_stiffness_multipliers: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 1.0,
        TissueType.WHITE_MATTER: 1.2,  # Slightly stiffer
        TissueType.CSF: 0.01,  # Very soft (fluid)
        TissueType.TUMOR: 2.0,  # Tumors are often stiffer
        TissueType.EDEMA: 0.5,  # Softened tissue
        TissueType.SKULL: 1000.0,  # Very stiff (immovable)
    })

    # Tissue diffusion multipliers - configured for non-infiltrative tumor
    # Low values prevent infiltration; tumor cells stay within tumor boundary
    tissue_diffusion_multipliers: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 0.5,  # Reduced - limits infiltration
        TissueType.WHITE_MATTER: 0.8,  # Reduced - less fiber tract spread
        TissueType.CSF: 0.01,  # Strong barrier to invasion
        TissueType.TUMOR: 1.0,  # Normal diffusion within tumor mass
        TissueType.EDEMA: 0.3,  # Reduced - limits spread through edema
        TissueType.SKULL: 0.0,  # No diffusion through skull
    })

    # Tissue-specific Poisson ratios (compressibility)
    tissue_poisson_ratios: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 0.40,  # More compressible (uniform compression)
        TissueType.WHITE_MATTER: 0.45,  # Nearly incompressible
        TissueType.CSF: 0.499,  # Essentially incompressible fluid
        TissueType.TUMOR: 0.42,
        TissueType.EDEMA: 0.48,
        TissueType.SKULL: 0.30,  # Standard bone value
    })

    # Tissue-specific carrying capacity multipliers
    # Higher values allow more tumor growth in those regions (less resistance)
    tissue_carrying_capacity_multipliers: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 1.0,  # Baseline
        TissueType.WHITE_MATTER: 1.0,  # Same as gray matter
        TissueType.CSF: 1.0,  # Same as tissue - prevents cystic compartmentalization
        TissueType.TUMOR: 1.0,  # Normal within tumor
        TissueType.EDEMA: 1.0,  # Same as tissue
        TissueType.SKULL: 0.0,  # No growth in skull
    })

    @classmethod
    def for_tissue(cls, tissue_type: TissueType) -> "MaterialProperties":
        """Create material properties for a specific tissue type."""
        base = cls()
        stiffness_mult = base.tissue_stiffness_multipliers.get(tissue_type, 1.0)
        diffusion_mult = base.tissue_diffusion_multipliers.get(tissue_type, 1.0)
        poisson = base.tissue_poisson_ratios.get(tissue_type, 0.45)

        return cls(
            young_modulus=base.young_modulus * stiffness_mult,
            poisson_ratio=poisson,
            proliferation_rate=base.proliferation_rate,
            diffusion_coefficient=base.diffusion_coefficient * diffusion_mult,
            carrying_capacity=base.carrying_capacity,
            growth_stress_coefficient=base.growth_stress_coefficient,
        )

    @classmethod
    def gray_matter(cls) -> "MaterialProperties":
        """
        Create material properties for gray matter.

        Gray matter is modeled as uniformly compressible with isotropic properties.
        Lower Poisson ratio allows volume change under pressure.
        """
        return cls(
            young_modulus=2500.0,  # Pa - softer than white matter
            poisson_ratio=0.40,  # More compressible than white matter
            proliferation_rate=0.01,
            diffusion_coefficient=0.1,
            carrying_capacity=1.0,
            growth_stress_coefficient=0.15,  # 15% volumetric strain at full density
            mass_effect_scaling=15.0,  # Amplification for visible displacement (~3mm)
            radial_displacement_factor=5.0,  # Radial force from tumor center
            anisotropy_ratio=1.0,  # Isotropic
            fiber_direction=None,
        )

    @classmethod
    def white_matter(
        cls,
        fiber_direction: Optional[NDArray[np.float64]] = None,
    ) -> "MaterialProperties":
        """
        Create material properties for white matter.

        White matter resists stretching along the fiber direction (transversely isotropic).
        Higher stiffness parallel to fibers, nearly incompressible.
        """
        return cls(
            young_modulus=3500.0,  # Pa - stiffer than gray matter
            poisson_ratio=0.45,  # Nearly incompressible
            proliferation_rate=0.01,
            diffusion_coefficient=0.2,  # Faster along fibers
            carrying_capacity=1.0,
            growth_stress_coefficient=0.15,  # 15% volumetric strain at full density
            mass_effect_scaling=15.0,  # Amplification for visible displacement (~3mm)
            radial_displacement_factor=5.0,  # Radial force from tumor center
            anisotropy_ratio=2.0,  # 2x stiffer along fibers
            fiber_direction=fiber_direction,
        )

    def lame_parameters(self) -> Tuple[float, float]:
        """Compute Lamé parameters from Young's modulus and Poisson ratio."""
        E = self.young_modulus
        nu = self.poisson_ratio

        # First Lamé parameter (lambda)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        # Second Lamé parameter (mu, shear modulus)
        mu = E / (2 * (1 + nu))

        return lam, mu

    def get_constitutive_matrix(
        self,
        fiber_direction: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Get the 6x6 constitutive matrix (stress-strain relationship).

        For isotropic materials (gray matter), returns standard isotropic matrix.
        For anisotropic materials (white matter), returns transversely isotropic
        matrix with fiber direction as the axis of symmetry.

        Args:
            fiber_direction: Override fiber direction (uses self.fiber_direction if None)

        Returns:
            6x6 constitutive matrix in Voigt notation
        """
        fiber_dir = fiber_direction if fiber_direction is not None else self.fiber_direction

        if fiber_dir is None or self.anisotropy_ratio == 1.0:
            # Isotropic (gray matter)
            return self._isotropic_constitutive_matrix()
        else:
            # Transversely isotropic (white matter)
            return self._anisotropic_constitutive_matrix(fiber_dir)

    def _isotropic_constitutive_matrix(self) -> NDArray[np.float64]:
        """Build isotropic constitutive matrix."""
        lam, mu = self.lame_parameters()

        C = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ], dtype=np.float64)

        return C

    def _anisotropic_constitutive_matrix(
        self,
        fiber_direction: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Build transversely isotropic constitutive matrix for white matter.

        The fiber direction is the axis of symmetry (stiffer along fibers).
        This resists stretching in the fiber direction.
        """
        E_perp = self.young_modulus
        E_para = E_perp * self.anisotropy_ratio
        nu_perp = self.poisson_ratio
        nu_para = self.poisson_ratio * 0.8  # Lower along fibers

        # Shear moduli
        G_perp = E_perp / (2 * (1 + nu_perp))
        G_para = E_para / (2 * (1 + nu_para))

        # Build compliance matrix in local (fiber-aligned) coordinates
        nu21 = nu_para * E_perp / E_para

        S = np.zeros((6, 6))
        S[0, 0] = 1 / E_para  # Along fiber
        S[1, 1] = 1 / E_perp  # Perpendicular
        S[2, 2] = 1 / E_perp  # Perpendicular
        S[0, 1] = -nu_para / E_para
        S[1, 0] = -nu21 / E_perp
        S[0, 2] = -nu_para / E_para
        S[2, 0] = -nu21 / E_perp
        S[1, 2] = -nu_perp / E_perp
        S[2, 1] = -nu_perp / E_perp
        S[3, 3] = 1 / G_perp  # sigma_23
        S[4, 4] = 1 / G_para  # sigma_13
        S[5, 5] = 1 / G_para  # sigma_12

        # Invert to get stiffness
        try:
            C_local = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self._isotropic_constitutive_matrix()

        # Rotate to global coordinates
        C_global = self._rotate_constitutive_to_global(C_local, fiber_direction)

        return C_global

    def _rotate_constitutive_to_global(
        self,
        C_local: NDArray[np.float64],
        fiber_dir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Rotate constitutive matrix from fiber-aligned to global coordinates."""
        # Normalize fiber direction
        f = fiber_dir / np.linalg.norm(fiber_dir)

        # Build orthonormal basis with f as first axis
        if abs(f[0]) < 0.9:
            t = np.array([1.0, 0.0, 0.0])
        else:
            t = np.array([0.0, 1.0, 0.0])

        n1 = np.cross(f, t)
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.cross(f, n1)

        # Rotation matrix (columns are local axes in global coords)
        R = np.column_stack([f, n1, n2])

        # Build 6x6 transformation matrix for Voigt notation
        T = self._build_voigt_rotation(R)

        # Transform: C_global = T^T * C_local * T
        return T.T @ C_local @ T

    def _build_voigt_rotation(self, R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build 6x6 Voigt rotation matrix."""
        T = np.zeros((6, 6))

        # Index pairs for Voigt notation
        pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for I in range(6):
            i, j = pairs[I]
            for J in range(6):
                k, l = pairs[J]
                if I < 3 and J < 3:
                    T[I, J] = R[i, k] * R[j, l]
                elif I < 3:
                    T[I, J] = R[i, k] * R[j, l] + R[i, l] * R[j, k]
                elif J < 3:
                    T[I, J] = R[i, k] * R[j, l]
                else:
                    T[I, J] = R[i, k] * R[j, l] + R[i, l] * R[j, k]

        return T


@dataclass
class TumorState:
    """
    Current state of the tumor simulation.

    Attributes:
        cell_density: Tumor cell density at each node (0 to 1).
        displacement: Tissue displacement at each node (N, 3).
        stress: Stress tensor at each element (M, 6) in Voigt notation.
        time: Current simulation time in days.
        tumor_center: Current tumor center of mass in mm.
        initial_volume: Initial tumor volume at t=0 (for mass effect calculation).
        current_volume: Current tumor volume (for tracking growth).
        ventricular_volume: Current ventricular volume (for compliance tracking).
        initial_ventricular_volume: Initial ventricular volume at t=0.
    """

    cell_density: NDArray[np.float64]
    displacement: NDArray[np.float64]
    stress: NDArray[np.float64]
    time: float = 0.0
    tumor_center: Optional[NDArray[np.float64]] = None
    initial_volume: float = 0.0
    current_volume: float = 0.0
    ventricular_volume: float = 0.0
    initial_ventricular_volume: float = 0.0

    @classmethod
    def initial(
        cls,
        mesh: TetMesh,
        seed_center: NDArray[np.float64],
        seed_radius: float = 2.5,
        seed_density: float = 0.5,
    ) -> "TumorState":
        """
        Create initial tumor state with a seed tumor.

        Args:
            mesh: FEM mesh.
            seed_center: Center of initial tumor seed in mm.
            seed_radius: Radius of initial tumor in mm.
            seed_density: Initial cell density in seed region.

        Returns:
            Initial TumorState.
        """
        num_nodes = mesh.num_nodes
        num_elements = mesh.num_elements

        # Ensure seed_center is a numpy array
        seed_center = np.asarray(seed_center, dtype=np.float64)

        # Initialize cell density with Gaussian seed
        distances = np.linalg.norm(mesh.nodes - seed_center, axis=1)
        cell_density = seed_density * np.exp(-(distances / seed_radius) ** 2)

        # Initialize displacement to zero
        displacement = np.zeros((num_nodes, 3), dtype=np.float64)

        # Initialize stress to zero
        stress = np.zeros((num_elements, 6), dtype=np.float64)

        # Compute initial tumor volume (approximate from Gaussian seed)
        # Volume where density > 0.1 threshold
        initial_vol = (4.0 / 3.0) * np.pi * (seed_radius ** 3) * seed_density

        return cls(
            cell_density=cell_density,
            displacement=displacement,
            stress=stress,
            time=0.0,
            tumor_center=seed_center.copy(),
            initial_volume=initial_vol,
            current_volume=initial_vol,
            ventricular_volume=0.0,  # Will be computed by solver if needed
            initial_ventricular_volume=0.0,
        )


# Default coarse mesh voxel size for fast simulation (mm)
# Increased from 3.0 to 4.0 for faster simulation with acceptable accuracy
DEFAULT_COARSE_MESH_VOXEL_SIZE = 4.0


@dataclass
class SolverConfig:
    """
    Configuration for FEM solver performance and accuracy tradeoffs.

    Provides options for approximate solutions that trade accuracy for speed:
    - AMG preconditioning for faster CG convergence
    - Reduced tolerance for fewer iterations
    - Coarse mesh with high-resolution output interpolation
    - Maximum iteration limits

    Example usage:
        # Fast coarse mode (recommended default, ~50-100x speedup)
        config = SolverConfig.fast_coarse()
        solver = TumorGrowthSolver(mesh, solver_config=config)

        # Fast approximate mode (3-10x speedup, same resolution)
        config = SolverConfig.fast()
        solver = TumorGrowthSolver(mesh, solver_config=config)

        # High accuracy mode (slower but more precise)
        config = SolverConfig.accurate()
        solver = TumorGrowthSolver(mesh, solver_config=config)
    """

    # Mechanical solver (CG) settings
    mechanical_tol: float = 1e-6  # Relative tolerance for CG convergence
    mechanical_maxiter: int = 1000  # Maximum CG iterations

    # AMG preconditioning settings
    use_amg: bool = True  # Use algebraic multigrid preconditioning
    amg_cycle: str = "V"  # AMG cycle type: "V", "W", or "F"
    amg_strength: str = "symmetric"  # Strength of connection: "symmetric" or "classical"

    # Diffusion solver settings (uses direct solver by default)
    diffusion_use_iterative: bool = False  # Use iterative solver for diffusion
    diffusion_tol: float = 1e-8  # Tolerance for iterative diffusion solver

    # Mesh resolution settings for multi-resolution simulation
    mesh_voxel_size: float = 1.0  # Mesh voxel size in mm (larger = coarser = faster)
    output_at_full_resolution: bool = True  # Interpolate output to original resolution

    # Cached AMG preconditioner (built lazily)
    _amg_preconditioner: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def default(cls) -> "SolverConfig":
        """
        Default configuration using coarse mesh for speed.

        Uses 3mm mesh voxels with AMG preconditioning and interpolates
        output to full resolution. This is the recommended default for
        most applications.

        Typical speedup: ~50-100x vs fine mesh + standard solver
        Accuracy: Smooth deformation field, suitable for most applications
        """
        return cls(
            mechanical_tol=1e-6,
            mechanical_maxiter=1000,
            use_amg=True,
            amg_cycle="V",
            mesh_voxel_size=DEFAULT_COARSE_MESH_VOXEL_SIZE,
            output_at_full_resolution=True,
        )

    @classmethod
    def fast_coarse(cls) -> "SolverConfig":
        """
        Fastest configuration using coarse mesh + reduced tolerance.

        Combines coarse mesh (3mm voxels), AMG preconditioning, and
        reduced solver tolerance for maximum speed. Output is interpolated
        to full resolution.

        Typical speedup: ~100-200x vs fine mesh + standard solver
        Accuracy: ~1-5% relative error in displacement field
        """
        return cls(
            mechanical_tol=1e-3,
            mechanical_maxiter=100,
            use_amg=True,
            amg_cycle="V",
            mesh_voxel_size=DEFAULT_COARSE_MESH_VOXEL_SIZE,
            output_at_full_resolution=True,
        )

    @classmethod
    def fast(cls) -> "SolverConfig":
        """
        Fast configuration with same mesh resolution.

        Uses reduced tolerance and AMG preconditioning for 3-10x speedup
        at the same mesh resolution. Use this when you need the full
        mesh detail but want faster solving.

        Typical speedup: 3-10x
        Accuracy: ~0.1-1% relative error
        """
        return cls(
            mechanical_tol=1e-3,
            mechanical_maxiter=100,
            use_amg=True,
            amg_cycle="V",
            mesh_voxel_size=1.0,  # Same as atlas resolution
            output_at_full_resolution=True,
        )

    @classmethod
    def accurate(cls) -> "SolverConfig":
        """
        High accuracy configuration for final results.

        Uses fine mesh (1mm voxels), tight tolerance, and direct solver.
        Use this when maximum accuracy is required.

        Typical speedup: 1x (baseline)
        Accuracy: Maximum precision
        """
        return cls(
            mechanical_tol=1e-8,
            mechanical_maxiter=2000,
            use_amg=False,
            mesh_voxel_size=1.0,
            output_at_full_resolution=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mechanical_tol": self.mechanical_tol,
            "mechanical_maxiter": self.mechanical_maxiter,
            "use_amg": self.use_amg,
            "amg_cycle": self.amg_cycle,
            "amg_strength": self.amg_strength,
            "diffusion_use_iterative": self.diffusion_use_iterative,
            "diffusion_tol": self.diffusion_tol,
            "mesh_voxel_size": self.mesh_voxel_size,
            "output_at_full_resolution": self.output_at_full_resolution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolverConfig":
        """Create from dictionary."""
        return cls(
            mechanical_tol=data.get("mechanical_tol", 1e-6),
            mechanical_maxiter=data.get("mechanical_maxiter", 1000),
            use_amg=data.get("use_amg", True),
            amg_cycle=data.get("amg_cycle", "V"),
            amg_strength=data.get("amg_strength", "symmetric"),
            diffusion_use_iterative=data.get("diffusion_use_iterative", False),
            diffusion_tol=data.get("diffusion_tol", 1e-8),
            mesh_voxel_size=data.get("mesh_voxel_size", DEFAULT_COARSE_MESH_VOXEL_SIZE),
            output_at_full_resolution=data.get("output_at_full_resolution", True),
        )


class TumorGrowthSolver:
    """
    FEM solver for tumor growth in brain tissue.

    Implements a coupled model:
    1. Reaction-diffusion equation for tumor cell density
    2. Linear elasticity for tissue deformation
    3. Mass effect: tumor growth causes tissue displacement

    The model uses operator splitting:
    - Diffusion step (implicit)
    - Reaction step (explicit)
    - Mechanical equilibrium (static)

    Supports biophysical constraints:
    - Anisotropic white matter: Resists stretching along fiber direction
    - Compressible gray matter: Uniform volumetric compression
    - Skull boundary: Immovable (fixed displacement)
    - Tissue-specific diffusion: Faster along white matter tracts
    """

    def __init__(
        self,
        mesh: TetMesh,
        properties: Optional[MaterialProperties] = None,
        boundary_condition: str = "fixed",
        biophysical_constraints: Optional["BiophysicalConstraints"] = None,
        solver_config: Optional[SolverConfig] = None,
    ):
        """
        Initialize the tumor growth solver.

        Args:
            mesh: Tetrahedral mesh for FEM.
            properties: Material properties (uses defaults if None).
            boundary_condition: Boundary condition type:
                - "fixed": All boundary nodes are immovable
                - "skull": Only skull boundary nodes are fixed
                - "skull_csf_free": Skull fixed, CSF nodes free (recommended for
                  posterior fossa tumors - allows expansion into fourth ventricle)
            biophysical_constraints: Optional biophysical constraints for tissue-specific
                                    material properties, fiber orientation, and boundaries.
            solver_config: Solver configuration for performance/accuracy tradeoffs.
                          Use SolverConfig.fast() for approximate solutions,
                          SolverConfig.accurate() for high precision.
        """
        self.mesh = mesh
        self.properties = properties or MaterialProperties()
        self.boundary_condition = boundary_condition
        self.biophysical_constraints = biophysical_constraints
        self.solver_config = solver_config or SolverConfig.default()

        # Tissue and fiber data from biophysical constraints
        self._node_tissues: Optional[NDArray[np.int32]] = None
        self._node_fiber_directions: Optional[NDArray[np.float64]] = None
        self._node_fa_values: Optional[NDArray[np.float64]] = None
        self._element_fa_values: Optional[NDArray[np.float64]] = None
        self._element_properties: Optional[List[MaterialProperties]] = None

        # Cached AMG preconditioner (built lazily on first solve)
        self._amg_ml: Any = None

        # Cached diffusion system matrix factorization (for efficiency)
        self._cached_diffusion_dt: Optional[float] = None
        self._cached_diffusion_lu: Any = None

        # Initialize biophysical data if constraints provided
        if biophysical_constraints is not None:
            self._initialize_biophysical_data()

        # Precompute element matrices
        self._element_volumes = mesh.compute_element_volumes()
        self._shape_gradients = self._compute_shape_gradients()

        # Build system matrices
        self._mass_matrix = self._build_mass_matrix()
        self._stiffness_matrix = self._build_stiffness_matrix()
        self._diffusion_matrix = self._build_diffusion_matrix()

    def _initialize_biophysical_data(self) -> None:
        """Initialize tissue types, fiber directions, and FA values from biophysical constraints."""
        bc = self.biophysical_constraints

        # Load all constraint data
        bc.load_all_constraints()

        # Assign tissue types to nodes
        self._node_tissues = bc.assign_node_tissues(self.mesh.nodes)

        # Get fiber directions at all nodes
        self._node_fiber_directions = bc.get_fiber_directions_at_nodes(self.mesh.nodes)

        # Get FA values at all nodes for FA-dependent anisotropic diffusion
        self._node_fa_values = self._get_fa_values_at_nodes(bc)

        # Compute element-averaged FA values
        self._element_fa_values = np.zeros(len(self.mesh.elements), dtype=np.float64)
        for e, elem in enumerate(self.mesh.elements):
            self._element_fa_values[e] = np.mean(self._node_fa_values[elem])

        # Build element-specific material properties
        self._element_properties = []
        for e, elem in enumerate(self.mesh.elements):
            # Use dominant tissue type in element
            elem_tissues = self._node_tissues[elem]
            dominant_tissue = int(np.median(elem_tissues))

            # Get average fiber direction for element
            elem_fibers = self._node_fiber_directions[elem]
            avg_fiber = np.mean(elem_fibers, axis=0)
            norm = np.linalg.norm(avg_fiber)
            if norm > 1e-6:
                avg_fiber = avg_fiber / norm
            else:
                avg_fiber = np.array([1.0, 0.0, 0.0])

            # Create tissue-specific properties
            if dominant_tissue == 3:  # WHITE_MATTER (from BrainTissue enum)
                props = MaterialProperties.white_matter(fiber_direction=avg_fiber)
                # Copy FA anisotropy factor from global properties
                props.fa_anisotropy_factor = self.properties.fa_anisotropy_factor
            elif dominant_tissue == 2:  # GRAY_MATTER
                props = MaterialProperties.gray_matter()
            elif dominant_tissue == 1:  # CSF
                # Apply ventricular compliance if enabled
                if self.properties.use_ventricular_compliance:
                    # CSF regions are very compliant (soft)
                    # Stiffness is reduced by compliance factor
                    csf_stiffness = self.properties.csf_bulk_modulus
                    compliance_factor = self.properties.ventricular_compliance_factor
                    effective_stiffness = csf_stiffness * compliance_factor
                else:
                    effective_stiffness = 100.0  # Default soft CSF

                props = MaterialProperties(
                    young_modulus=effective_stiffness,
                    poisson_ratio=0.499,  # Nearly incompressible fluid
                    diffusion_coefficient=0.01,
                )
            else:
                props = MaterialProperties()

            self._element_properties.append(props)

    def _get_fa_values_at_nodes(
        self,
        bc: "BiophysicalConstraints",
    ) -> NDArray[np.float64]:
        """
        Get fractional anisotropy (FA) values at mesh nodes.

        FA values are used for FA-dependent anisotropic diffusion, where
        tumor cells preferentially migrate along high-FA white matter tracts.

        Args:
            bc: Biophysical constraints with fiber orientation data

        Returns:
            FA values at each node (0-1)
        """
        n_nodes = len(self.mesh.nodes)
        fa_values = np.zeros(n_nodes, dtype=np.float64)

        try:
            fibers = bc.load_fiber_orientation()
            segmentation = bc.load_tissue_segmentation()

            # Sample FA at each node position
            for i, pos in enumerate(self.mesh.nodes):
                try:
                    # Convert physical to voxel coordinates
                    if fibers.affine is not None:
                        inv_affine = np.linalg.inv(fibers.affine)
                        pos_h = np.append(pos, 1.0)
                        voxel = (inv_affine @ pos_h)[:3]
                    else:
                        voxel = pos / np.array(segmentation.voxel_size)

                    # Get voxel indices (clamp to valid range)
                    shape = fibers.fractional_anisotropy.shape
                    ix = int(np.clip(np.round(voxel[0]), 0, shape[0] - 1))
                    iy = int(np.clip(np.round(voxel[1]), 0, shape[1] - 1))
                    iz = int(np.clip(np.round(voxel[2]), 0, shape[2] - 1))

                    fa_values[i] = fibers.fractional_anisotropy[ix, iy, iz]
                except (IndexError, ValueError):
                    fa_values[i] = 0.0

        except Exception:
            # Fall back to default FA=0.5 for all nodes
            fa_values[:] = 0.5

        return fa_values

    def _get_element_tissue_type(self, element_idx: int) -> TissueType:
        """Get the dominant tissue type for an element."""
        if self._node_tissues is None:
            return TissueType.GRAY_MATTER

        elem = self.mesh.elements[element_idx]
        elem_tissues = self._node_tissues[elem]
        dominant = int(np.median(elem_tissues))

        # Map from BrainTissue enum values
        tissue_map = {
            0: TissueType.CSF,  # BACKGROUND treated as CSF
            1: TissueType.CSF,
            2: TissueType.GRAY_MATTER,
            3: TissueType.WHITE_MATTER,
            4: TissueType.SKULL,
            5: TissueType.SKULL,  # SCALP treated as skull boundary
        }
        return tissue_map.get(dominant, TissueType.GRAY_MATTER)

    def _get_skull_boundary_nodes(self) -> NDArray[np.int32]:
        """Get nodes on the skull boundary for immovable constraint.

        Uses anatomical skull boundary detection when biophysical constraints
        are available, but falls back to mesh boundary nodes if:
        - No biophysical constraints are provided
        - Anatomical detection returns no or too few boundary nodes

        A minimum of 3 boundary nodes is required to prevent rigid body motion
        in the FEM solve. Using mesh boundary ensures the matrix is never singular.
        """
        min_boundary_nodes = 3  # Minimum to prevent rigid body motion

        if self.biophysical_constraints is None:
            return self.mesh.boundary_nodes

        # Try anatomical skull boundary detection
        anatomical_boundary = self.biophysical_constraints.get_boundary_nodes(self.mesh.nodes)

        # Fall back to mesh boundary if anatomical detection fails
        if len(anatomical_boundary) < min_boundary_nodes:
            if len(self.mesh.boundary_nodes) >= min_boundary_nodes:
                return self.mesh.boundary_nodes
            # If mesh boundary is also insufficient, use all boundary nodes
            # This ensures the matrix is never singular
            return np.unique(np.concatenate([
                anatomical_boundary,
                self.mesh.boundary_nodes
            ])).astype(np.int32)

        return anatomical_boundary

    def _get_csf_nodes(self) -> NDArray[np.int32]:
        """
        Get nodes in CSF regions (including fourth ventricle).

        These nodes can be given free boundary conditions since CSF
        can be displaced with minimal resistance.

        Returns:
            Array of node indices in CSF regions.
        """
        if self._node_tissues is None:
            return np.array([], dtype=np.int32)

        # BrainTissue.CSF = 1
        csf_mask = self._node_tissues == 1
        return np.where(csf_mask)[0].astype(np.int32)

    def _get_anterior_brainstem_boundary_nodes(self) -> NDArray[np.int32]:
        """
        Get nodes on the anterior boundary of the brainstem (clivus).

        These nodes should always have fixed boundary conditions to
        represent the clivus (petrous bone) that constrains the anterior
        brainstem, preventing unrealistic anterior tumor expansion.

        Returns:
            Array of node indices on the anterior brainstem boundary.
        """
        if self.biophysical_constraints is None:
            return np.array([], dtype=np.int32)

        return self.biophysical_constraints._compute_anterior_brainstem_boundary(
            self.mesh.nodes
        )

    def _get_node_tissue_type(self, node_idx: int) -> TissueType:
        """Get the tissue type for a node."""
        if self._node_tissues is None:
            return TissueType.GRAY_MATTER

        brain_tissue = self._node_tissues[node_idx]

        # Map from BrainTissue enum values to TissueType
        tissue_map = {
            0: TissueType.CSF,  # BACKGROUND treated as CSF
            1: TissueType.CSF,
            2: TissueType.GRAY_MATTER,
            3: TissueType.WHITE_MATTER,
            4: TissueType.SKULL,
            5: TissueType.SKULL,  # SCALP treated as skull boundary
        }
        return tissue_map.get(brain_tissue, TissueType.GRAY_MATTER)

    def _compute_node_carrying_capacities(self) -> NDArray[np.float64]:
        """
        Compute tissue-specific carrying capacities for all nodes.

        CSF-filled spaces have higher carrying capacity (lower resistance),
        allowing tumors to grow more easily into these regions. This models
        the biophysical reality that tumors preferentially expand into the
        fourth ventricle and other CSF spaces due to minimal tissue resistance.

        Returns:
            Array of carrying capacity values for each node.
        """
        n_nodes = self.mesh.num_nodes
        base_K = self.properties.carrying_capacity
        capacities = np.full(n_nodes, base_K, dtype=np.float64)

        if self._node_tissues is not None:
            # Apply tissue-specific multipliers
            multipliers = self.properties.tissue_carrying_capacity_multipliers

            for i in range(n_nodes):
                tissue_type = self._get_node_tissue_type(i)
                mult = multipliers.get(tissue_type, 1.0)
                capacities[i] = base_K * mult

        return capacities

    def _compute_shape_gradients(self) -> List[NDArray[np.float64]]:
        """Compute shape function gradients for each element."""
        gradients = []

        for elem in self.mesh.elements:
            coords = self.mesh.nodes[elem]  # (4, 3)

            # Jacobian matrix for linear tetrahedron
            # J = [x1-x0, x2-x0, x3-x0]^T
            J = np.array([
                coords[1] - coords[0],
                coords[2] - coords[0],
                coords[3] - coords[0],
            ])

            # Inverse Jacobian
            try:
                J_inv = np.linalg.inv(J)
            except np.linalg.LinAlgError:
                # Degenerate element
                J_inv = np.zeros((3, 3))

            # Shape function gradients in physical coordinates
            # For linear tetrahedron: dN/dx = J^(-T) * dN/dxi
            # dN/dxi for reference tetrahedron:
            # N0 = 1 - xi - eta - zeta -> grad = [-1, -1, -1]
            # N1 = xi -> grad = [1, 0, 0]
            # N2 = eta -> grad = [0, 1, 0]
            # N3 = zeta -> grad = [0, 0, 1]
            ref_grads = np.array([
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.float64)

            # Transform to physical coordinates
            phys_grads = ref_grads @ J_inv
            gradients.append(phys_grads)

        return gradients

    def _build_mass_matrix(self) -> sparse.csr_matrix:
        """Build the mass matrix for the reaction-diffusion equation."""
        n = self.mesh.num_nodes

        if HAS_NUMBA:
            # Use JIT-compiled assembly (50-100x faster)
            rows, cols, data = _assemble_mass_matrix_jit(
                self.mesh.elements,
                self._element_volumes,
                n,
            )
        else:
            # Fallback to pure Python
            rows, cols, data = [], [], []

            for e, elem in enumerate(self.mesh.elements):
                vol = self._element_volumes[e]

                # Mass matrix for linear tetrahedron
                # M_ij = integral(Ni * Nj) = V/20 * (1 + delta_ij)
                for i in range(4):
                    for j in range(4):
                        if i == j:
                            val = vol / 10.0
                        else:
                            val = vol / 20.0

                        rows.append(elem[i])
                        cols.append(elem[j])
                        data.append(val)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _build_diffusion_matrix(self) -> sparse.csr_matrix:
        """
        Build the diffusion matrix with tissue-specific coefficients.

        White matter: FA-dependent anisotropic diffusion, faster along fiber direction
        Gray matter: Isotropic diffusion
        CSF: Reduced diffusion (barrier to invasion)

        When FA values are available, uses FA-dependent anisotropy where the
        diffusion ratio D_parallel/D_perpendicular = 1 + k * FA. This provides
        more realistic tumor spread patterns along high-FA white matter tracts.
        """
        n = self.mesh.num_nodes
        num_elements = len(self.mesh.elements)

        # Prepare arrays for JIT compilation
        shape_grads_array = np.array(self._shape_gradients)  # (num_elements, 4, 3)

        # Build diffusion coefficient array
        diffusion_coeffs = np.empty(num_elements, dtype=np.float64)
        if self._element_properties is not None:
            for e in range(num_elements):
                diffusion_coeffs[e] = self._element_properties[e].diffusion_coefficient
        else:
            diffusion_coeffs[:] = self.properties.diffusion_coefficient

        # Check if we have anisotropic elements (white matter with fiber directions)
        has_anisotropic = (
            self._element_properties is not None and
            any(p.fiber_direction is not None for p in self._element_properties)
        )

        # Check if we have FA values for FA-dependent anisotropy
        has_fa_values = self._element_fa_values is not None

        if HAS_NUMBA and not has_anisotropic:
            # Use fast isotropic JIT assembly
            rows, cols, data = _assemble_diffusion_matrix_isotropic_jit(
                self.mesh.elements,
                self._element_volumes,
                shape_grads_array,
                diffusion_coeffs,
                n,
            )
        elif HAS_NUMBA and has_anisotropic and has_fa_values:
            # Use FA-dependent anisotropic JIT assembly (most realistic)
            fiber_directions = np.zeros((num_elements, 3), dtype=np.float64)
            tissue_types = np.zeros(num_elements, dtype=np.int32)

            for e in range(num_elements):
                tissue_types[e] = self._get_element_tissue_type(e).value
                if self._element_properties[e].fiber_direction is not None:
                    fiber_directions[e] = self._element_properties[e].fiber_direction

            fa_anisotropy_factor = self.properties.fa_anisotropy_factor

            rows, cols, data = _assemble_diffusion_matrix_fa_anisotropic_jit(
                self.mesh.elements,
                self._element_volumes,
                shape_grads_array,
                diffusion_coeffs,
                fiber_directions,
                self._element_fa_values,
                tissue_types,
                TissueType.WHITE_MATTER.value,
                fa_anisotropy_factor,
                n,
            )
        elif HAS_NUMBA and has_anisotropic:
            # Use legacy fixed-ratio anisotropic JIT assembly (backward compatible)
            fiber_directions = np.zeros((num_elements, 3), dtype=np.float64)
            tissue_types = np.zeros(num_elements, dtype=np.int32)

            for e in range(num_elements):
                tissue_types[e] = self._get_element_tissue_type(e).value
                if self._element_properties[e].fiber_direction is not None:
                    fiber_directions[e] = self._element_properties[e].fiber_direction

            rows, cols, data = _assemble_diffusion_matrix_anisotropic_jit(
                self.mesh.elements,
                self._element_volumes,
                shape_grads_array,
                diffusion_coeffs,
                fiber_directions,
                tissue_types,
                TissueType.WHITE_MATTER.value,
                n,
            )
        else:
            # Fallback to pure Python (also supports FA-dependent anisotropy)
            rows, cols, data = [], [], []

            for e, elem in enumerate(self.mesh.elements):
                vol = self._element_volumes[e]
                grads = self._shape_gradients[e]  # (4, 3)

                # Get tissue-specific diffusion coefficient
                if self._element_properties is not None:
                    D = self._element_properties[e].diffusion_coefficient
                    fiber_dir = self._element_properties[e].fiber_direction
                else:
                    D = self.properties.diffusion_coefficient
                    fiber_dir = None

                # Get FA value for this element (for FA-dependent anisotropy)
                fa_value = 0.5  # default
                if self._element_fa_values is not None:
                    fa_value = self._element_fa_values[e]

                # Build diffusion tensor
                if fiber_dir is not None and self._get_element_tissue_type(e) == TissueType.WHITE_MATTER:
                    # FA-dependent anisotropic diffusion: faster along fibers
                    D_tensor = self._build_anisotropic_diffusion_tensor(D, fiber_dir, fa_value)
                else:
                    # Isotropic diffusion
                    D_tensor = D * np.eye(3)

                # Diffusion matrix: K_ij = integral(grad(Ni) . D . grad(Nj))
                for i in range(4):
                    for j in range(4):
                        # For anisotropic: grad_i . D . grad_j
                        val = vol * grads[i] @ D_tensor @ grads[j]

                        rows.append(elem[i])
                        cols.append(elem[j])
                        data.append(val)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _build_anisotropic_diffusion_tensor(
        self,
        D_base: float,
        fiber_direction: NDArray[np.float64],
        fa_value: float = 0.5,
    ) -> NDArray[np.float64]:
        """
        Build FA-dependent anisotropic diffusion tensor for white matter.

        Diffusion anisotropy is proportional to local FA values:
        D_parallel/D_perpendicular = 1 + k * FA

        This models the biophysical observation that tumor cells preferentially
        migrate along white matter tracts with high FA (coherent fiber bundles).

        Args:
            D_base: Baseline diffusion coefficient
            fiber_direction: Unit vector of local fiber orientation
            fa_value: Fractional anisotropy (0-1), default 0.5 for backward compatibility

        Returns:
            3x3 diffusion tensor matrix
        """
        # Normalize fiber direction
        f = fiber_direction / np.linalg.norm(fiber_direction)

        # Get FA anisotropy factor from properties
        fa_anisotropy_factor = getattr(self.properties, 'fa_anisotropy_factor', 6.0)

        # FA-dependent anisotropy ratio
        # For FA=0: isotropic (ratio=1)
        # For FA=1 with k=6: strong anisotropy (ratio=7)
        anisotropy_ratio = 1.0 + fa_anisotropy_factor * fa_value

        # Preserve geometric mean to maintain overall diffusion magnitude
        # D_para * D_perp^2 ≈ D_base^3
        sqrt_ratio = np.sqrt(anisotropy_ratio)
        D_parallel = D_base * sqrt_ratio
        D_perpendicular = D_base / sqrt_ratio

        # Diffusion tensor: D = D_perp * I + (D_para - D_perp) * f ⊗ f
        D_tensor = D_perpendicular * np.eye(3) + (D_parallel - D_perpendicular) * np.outer(f, f)

        return D_tensor

    def _build_stiffness_matrix(self) -> sparse.csr_matrix:
        """
        Build the elastic stiffness matrix (3D linear elasticity).

        Uses tissue-specific material properties when biophysical constraints
        are available:
        - White matter: Anisotropic, stiffer along fiber direction
        - Gray matter: Isotropic, more compressible
        - CSF: Very soft, nearly incompressible
        """
        n = self.mesh.num_nodes
        num_elements = len(self.mesh.elements)

        # Check if we have anisotropic elements (white matter with fiber directions)
        has_anisotropic = (
            self._element_properties is not None and
            any(p.fiber_direction is not None for p in self._element_properties)
        )

        # Prepare arrays for JIT compilation
        shape_grads_array = np.array(self._shape_gradients)  # (num_elements, 4, 3)

        # Build Lame parameter arrays
        lam_values = np.empty(num_elements, dtype=np.float64)
        mu_values = np.empty(num_elements, dtype=np.float64)

        if self._element_properties is not None:
            for e in range(num_elements):
                lam, mu = self._element_properties[e].lame_parameters()
                lam_values[e] = lam
                mu_values[e] = mu
        else:
            lam, mu = self.properties.lame_parameters()
            lam_values[:] = lam
            mu_values[:] = mu

        if HAS_NUMBA and not has_anisotropic:
            # Use JIT-compiled assembly for all-isotropic case (50-100x faster)
            rows, cols, data = _assemble_stiffness_matrix_isotropic_jit(
                self.mesh.elements,
                self._element_volumes,
                shape_grads_array,
                lam_values,
                mu_values,
                n,
            )
        else:
            # Fallback to Python (required for anisotropic elements)
            rows, cols, data = [], [], []

            for e, elem in enumerate(self.mesh.elements):
                vol = self._element_volumes[e]
                grads = self._shape_gradients[e]  # (4, 3)

                # Get tissue-specific material properties
                if self._element_properties is not None:
                    elem_props = self._element_properties[e]
                else:
                    elem_props = self.properties

                # Build element stiffness matrix (12x12)
                # Use anisotropic formulation for white matter
                if elem_props.fiber_direction is not None:
                    Ke = self._element_stiffness_anisotropic(
                        grads, vol, elem_props
                    )
                else:
                    lam, mu = elem_props.lame_parameters()
                    Ke = self._element_stiffness(grads, vol, lam, mu)

                # Assemble into global matrix
                for i in range(4):
                    for j in range(4):
                        for di in range(3):  # DOF dimension
                            for dj in range(3):
                                global_i = elem[i] * 3 + di
                                global_j = elem[j] * 3 + dj
                                local_i = i * 3 + di
                                local_j = j * 3 + dj

                                rows.append(global_i)
                                cols.append(global_j)
                                data.append(Ke[local_i, local_j])

        K = sparse.csr_matrix((data, (rows, cols)), shape=(3 * n, 3 * n))

        # Apply boundary conditions
        if self.boundary_condition == "fixed":
            K = self._apply_fixed_bc(K)
        elif self.boundary_condition == "skull":
            K = self._apply_skull_bc(K)
        elif self.boundary_condition == "skull_csf_free":
            K = self._apply_skull_csf_free_bc(K)

        return K

    def _element_stiffness_anisotropic(
        self,
        grads: NDArray[np.float64],
        vol: float,
        props: MaterialProperties,
    ) -> NDArray[np.float64]:
        """
        Compute element stiffness matrix with anisotropic constitutive law.

        Used for white matter elements to resist stretching along fiber direction.
        """
        Ke = np.zeros((12, 12))

        # Get anisotropic constitutive matrix
        C = props.get_constitutive_matrix()

        for i in range(4):
            for j in range(4):
                # B matrices for nodes i and j
                Bi = self._strain_displacement_matrix(grads[i])
                Bj = self._strain_displacement_matrix(grads[j])

                # K_ij = V * Bi^T * C * Bj
                Kij = vol * Bi.T @ C @ Bj

                # Insert into element matrix
                Ke[i*3:(i+1)*3, j*3:(j+1)*3] = Kij

        return Ke

    def _apply_skull_bc(
        self,
        K: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """
        Apply skull boundary conditions (immovable).

        Uses biophysical constraints to identify skull boundary nodes,
        or falls back to mesh boundary nodes.
        """
        K = K.tolil()

        skull_nodes = self._get_skull_boundary_nodes()

        for node_idx in skull_nodes:
            for dof in range(3):
                global_dof = node_idx * 3 + dof
                # Set row to zero except diagonal
                K[global_dof, :] = 0
                K[:, global_dof] = 0
                K[global_dof, global_dof] = 1.0

        return K.tocsr()

    def _apply_skull_csf_free_bc(
        self,
        K: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """
        Apply boundary conditions with free CSF (fourth ventricle) regions.

        This boundary condition mode:
        1. Fixes skull boundary nodes (immovable, like brain-skull interface)
        2. Fixes anterior brainstem boundary (clivus - always fixed as bone)
        3. Leaves other CSF nodes free (zero traction)

        This models the biophysical reality that:
        - CSF in the fourth ventricle can be displaced with minimal resistance
        - The anterior brainstem is bounded by the clivus (petrous bone), which
          acts as a hard constraint preventing anterior tumor expansion

        Returns:
            Modified stiffness matrix with boundary conditions applied.
        """
        K = K.tolil()

        # Get skull boundary nodes (fixed)
        skull_nodes = set(self._get_skull_boundary_nodes())

        # Get CSF nodes (free - do not constrain these)
        csf_nodes = set(self._get_csf_nodes())

        # Get anterior brainstem boundary (clivus - always fixed, even if CSF)
        anterior_brainstem = set(self._get_anterior_brainstem_boundary_nodes())

        # CSF nodes that should remain free (exclude anterior brainstem boundary)
        free_csf_nodes = csf_nodes - anterior_brainstem

        # Constrained nodes: skull + anterior brainstem, minus free CSF
        constrained_nodes = (skull_nodes | anterior_brainstem) - free_csf_nodes

        for node_idx in constrained_nodes:
            for dof in range(3):
                global_dof = node_idx * 3 + dof
                # Set row to zero except diagonal
                K[global_dof, :] = 0
                K[:, global_dof] = 0
                K[global_dof, global_dof] = 1.0

        return K.tocsr()

    def _element_stiffness(
        self,
        grads: NDArray[np.float64],
        vol: float,
        lam: float,
        mu: float,
    ) -> NDArray[np.float64]:
        """Compute element stiffness matrix for 3D linear elasticity."""
        Ke = np.zeros((12, 12))

        # Constitutive matrix (isotropic linear elasticity)
        C = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ])

        for i in range(4):
            for j in range(4):
                # B matrix (strain-displacement) for nodes i and j
                Bi = self._strain_displacement_matrix(grads[i])
                Bj = self._strain_displacement_matrix(grads[j])

                # K_ij = integral(Bi^T * C * Bj) = V * Bi^T * C * Bj
                Kij = vol * Bi.T @ C @ Bj

                # Insert into element matrix
                Ke[i*3:(i+1)*3, j*3:(j+1)*3] = Kij

        return Ke

    def _strain_displacement_matrix(
        self,
        grad: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build strain-displacement matrix B for a node."""
        # B relates strain to displacement: epsilon = B * u
        # For 3D: epsilon = [exx, eyy, ezz, 2*exy, 2*exz, 2*eyz]
        dNdx, dNdy, dNdz = grad

        B = np.array([
            [dNdx, 0, 0],
            [0, dNdy, 0],
            [0, 0, dNdz],
            [dNdy, dNdx, 0],
            [dNdz, 0, dNdx],
            [0, dNdz, dNdy],
        ])

        return B

    def _apply_fixed_bc(
        self,
        K: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Apply fixed boundary conditions to stiffness matrix."""
        K = K.tolil()

        for node_idx in self.mesh.boundary_nodes:
            for dof in range(3):
                global_dof = node_idx * 3 + dof
                # Set row to zero except diagonal
                K[global_dof, :] = 0
                K[:, global_dof] = 0
                K[global_dof, global_dof] = 1.0

        return K.tocsr()

    def step(
        self,
        state: TumorState,
        dt: float,
    ) -> TumorState:
        """
        Perform one time step of the simulation.

        Models tumor growth as the creation of new volume within the tumor matrix,
        which displaces and compresses surrounding brain tissue constrained within
        the posterior cranial fossa.

        Args:
            state: Current tumor state.
            dt: Time step in days.

        Returns:
            Updated TumorState with new density, displacement, stress, and volume.
        """
        # Step 1: Reaction-diffusion for tumor cell density
        # This models cell proliferation and migration, effectively creating new cells
        new_density = self._reaction_diffusion_step(state.cell_density, dt)

        # Step 2: Compute new tumor center and volume
        # The volume increase represents new nodes being created in the tumor
        new_center = self._compute_tumor_centroid(new_density)
        new_volume = self._compute_tumor_volume_internal(new_density)

        # Step 3: Compute growth-induced force with mass effect
        # The force models new tumor volume displacing surrounding tissue
        # Pass volume information for volume-preserving mass effect formulation
        force = self._compute_growth_force(
            new_density,
            tumor_center=new_center,
            initial_volume=state.initial_volume,
            current_volume=new_volume,
        )

        # Step 4: Solve mechanical equilibrium
        # Tissue displacement from tumor mass effect
        new_displacement = self._solve_mechanical_equilibrium(force)

        # Step 5: Compute stress (shows compression in surrounding tissue)
        new_stress = self._compute_stress(new_displacement)

        # Step 6: Compute ventricular volume (for compliance tracking)
        new_ventricular_volume = self._compute_ventricular_volume(new_displacement)

        return TumorState(
            cell_density=new_density,
            displacement=new_displacement,
            stress=new_stress,
            time=state.time + dt,
            tumor_center=new_center,
            initial_volume=state.initial_volume,
            current_volume=new_volume,
            ventricular_volume=new_ventricular_volume,
            initial_ventricular_volume=state.initial_ventricular_volume,
        )

    def _compute_tumor_volume_internal(
        self,
        density: NDArray[np.float64],
        threshold: float = 0.1,
    ) -> float:
        """
        Compute tumor volume from density field.

        Args:
            density: Tumor cell density at each node.
            threshold: Density threshold for tumor boundary.

        Returns:
            Tumor volume in mm^3.
        """
        volume = 0.0
        for e, elem in enumerate(self.mesh.elements):
            elem_density = np.mean(density[elem])
            if elem_density >= threshold:
                volume += self._element_volumes[e] * elem_density
        return volume

    def _compute_ventricular_volume(
        self,
        displacement: NDArray[np.float64],
    ) -> float:
        """
        Compute current ventricular volume accounting for displacement.

        This tracks how much the ventricles have compressed due to tumor
        mass effect. The volume is computed by summing CSF element volumes
        and adjusting for local volumetric strain from displacement.

        Args:
            displacement: Current displacement field at each node.

        Returns:
            Current ventricular volume in mm^3.
        """
        if self._node_tissues is None:
            return 0.0

        volume = 0.0
        for e, elem in enumerate(self.mesh.elements):
            # Check if this element is CSF (ventricular)
            elem_tissues = self._node_tissues[elem]
            if np.median(elem_tissues) != 1:  # BrainTissue.CSF = 1
                continue

            # Get element volume
            base_volume = self._element_volumes[e]

            # Compute volumetric strain from displacement divergence
            grads = self._shape_gradients[e]
            div_u = 0.0
            for i in range(4):
                u_node = displacement[elem[i]]
                # div(u) = du_x/dx + du_y/dy + du_z/dz
                div_u += np.dot(grads[i], u_node)

            # Deformed volume = base_volume * (1 + div(u))
            deformed_volume = base_volume * (1.0 + div_u)
            volume += max(0.0, deformed_volume)  # Ensure non-negative

        return volume

    def compute_initial_ventricular_volume(self) -> float:
        """
        Compute the initial ventricular volume (before any displacement).

        This should be called when creating the initial TumorState to
        establish the baseline ventricular volume for compliance tracking.

        Returns:
            Initial ventricular volume in mm^3.
        """
        if self._node_tissues is None:
            return 0.0

        volume = 0.0
        for e, elem in enumerate(self.mesh.elements):
            elem_tissues = self._node_tissues[elem]
            if np.median(elem_tissues) == 1:  # BrainTissue.CSF = 1
                volume += self._element_volumes[e]

        return volume

    def _reaction_diffusion_step(
        self,
        density: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Solve reaction-diffusion equation for one time step.

        Uses implicit Euler for diffusion, explicit for reaction.
        dc/dt = D * laplacian(c) + rho * c * (1 - c/K)

        Tissue-specific carrying capacity allows tumors to grow more easily
        into CSF-filled spaces (like the fourth ventricle), modeling the
        biophysical reality that these fluid-filled regions offer minimal
        resistance to tumor expansion.

        Performance optimization: The system matrix (M + dt*D) is factorized
        once using LU decomposition and cached. Subsequent solves with the
        same dt reuse the factorization, providing ~5-10x speedup.
        """
        rho = self.properties.proliferation_rate

        # Get tissue-specific carrying capacities
        # CSF regions have higher capacity (lower resistance to growth)
        K = self._compute_node_carrying_capacities()

        # Reaction term (logistic growth) with per-node carrying capacity
        # Avoid division by zero for skull nodes (K=0)
        K_safe = np.maximum(K, 1e-10)
        reaction = rho * density * (1 - density / K_safe)

        # For skull nodes (K=0), force reaction to be negative (no growth)
        skull_mask = K < 1e-10
        reaction[skull_mask] = -rho * density[skull_mask]

        # Right-hand side: M * (c_n + dt * reaction)
        rhs = self._mass_matrix @ (density + dt * reaction)

        # Use cached LU factorization if available and dt hasn't changed
        if self._cached_diffusion_dt == dt and self._cached_diffusion_lu is not None:
            # Fast solve using cached factorization
            try:
                new_density = self._cached_diffusion_lu.solve(rhs)
            except Exception:
                # Fallback if cached solve fails
                self._cached_diffusion_lu = None
                return self._reaction_diffusion_step_direct(density, dt, rhs, K_safe)
        else:
            # Build and cache new factorization
            new_density = self._reaction_diffusion_step_direct(density, dt, rhs, K_safe)

        # Ensure non-negative and bounded by local carrying capacity
        new_density = np.clip(new_density, 0, K_safe)

        return new_density

    def _reaction_diffusion_step_direct(
        self,
        density: NDArray[np.float64],
        dt: float,
        rhs: NDArray[np.float64],
        K_safe: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve reaction-diffusion with direct solver and cache factorization.

        This builds the system matrix, factorizes it using LU decomposition,
        caches the factorization for future solves with the same dt, and
        returns the solution.

        Args:
            density: Current tumor cell density at each node.
            dt: Time step in days.
            rhs: Right-hand side vector.
            K_safe: Carrying capacity array with minimum bounds.

        Returns:
            New density values after one time step.
        """
        # System matrix: M + dt * D (implicit diffusion)
        A = self._mass_matrix + dt * self._diffusion_matrix

        # Add small regularization to handle singular/near-singular matrices
        # This is necessary because pure Neumann BCs (zero flux) leave the matrix
        # singular with a null space of constant vectors
        n = self.mesh.num_nodes
        reg_strength = 1e-10 * A.diagonal().mean()
        A_reg = A + sparse.diags([reg_strength], [0], shape=(n, n), format='csr')

        # Try to cache LU factorization for faster subsequent solves
        try:
            # Convert to CSC format for splu (more efficient)
            A_csc = A_reg.tocsc()
            lu = splu(A_csc)

            # Cache the factorization
            self._cached_diffusion_dt = dt
            self._cached_diffusion_lu = lu

            # Solve using the factorization
            new_density = lu.solve(rhs)
        except Exception:
            # LU factorization failed, fall back to direct solve
            self._cached_diffusion_lu = None
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Matrix is exactly singular')
                try:
                    new_density = spsolve(A_reg, rhs)
                except Exception:
                    # Fallback to least-squares solver for robustness
                    result = lsqr(A_reg, rhs, atol=1e-10, btol=1e-10)
                    new_density = result[0]

        return new_density

    def _compute_growth_force(
        self,
        density: NDArray[np.float64],
        tumor_center: Optional[NDArray[np.float64]] = None,
        initial_volume: float = 0.0,
        current_volume: float = 0.0,
    ) -> NDArray[np.float64]:
        """
        Compute force vector due to tumor growth (mass effect).

        This method models tumor growth as the creation of new material volume
        that displaces surrounding brain tissue. Two formulations are available:

        1. Legacy formulation (use_volume_preserving_mass_effect=False):
           - Eigenstrain formulation with empirical scaling factors
           - Radial force inversely proportional to distance

        2. Volume-preserving formulation (use_volume_preserving_mass_effect=True):
           - Physics-based volumetric strain: ε_v = dV / (3 * V_tumor)
           - Exponential pressure decay: P(r) = P_0 * exp(-r/λ)
           - No arbitrary scaling factors - derived from volume change

        The volume-preserving formulation is recommended for non-infiltrative
        mass effect tumors (medulloblastoma, pilocytic astrocytoma) as it
        produces more physically realistic tissue displacement.

        Args:
            density: Current tumor cell density at each node.
            tumor_center: Center of tumor mass for radial force calculation.
                         If None, computed from density-weighted centroid.
            initial_volume: Initial tumor volume at t=0 (for volume-preserving mode).
            current_volume: Current tumor volume (for volume-preserving mode).

        Returns:
            Force vector (3*N,) for mechanical equilibrium.
        """
        # Choose formulation based on material property setting
        if self.properties.use_volume_preserving_mass_effect:
            return self._compute_growth_force_volume_preserving(
                density, tumor_center, initial_volume, current_volume
            )
        else:
            return self._compute_growth_force_legacy(
                density, tumor_center
            )

    def _compute_growth_force_volume_preserving(
        self,
        density: NDArray[np.float64],
        tumor_center: Optional[NDArray[np.float64]] = None,
        initial_volume: float = 0.0,
        current_volume: float = 0.0,
    ) -> NDArray[np.float64]:
        """
        Compute growth force using physics-based volume-preserving formulation.

        This formulation derives the force from actual volume change:
        - Volumetric strain: ε_v = (V_current - V_initial) / V_initial
        - Radial stress with exponential decay: σ_r = E * ε_v * exp(-r/λ)
        - Decay length λ = factor * tumor_radius (typically factor=2)

        Reference: Clatz et al., "Realistic Simulation of the 3D Growth of
        Brain Tumors in MR Images..." (2005)

        Args:
            density: Current tumor cell density at each node.
            tumor_center: Center of tumor mass (computed if None).
            initial_volume: Initial tumor volume at t=0.
            current_volume: Current tumor volume.

        Returns:
            Force vector (3*N,) for mechanical equilibrium.
        """
        n = self.mesh.num_nodes
        force = np.zeros(3 * n)

        # Compute tumor center if not provided
        if tumor_center is None:
            tumor_center = self._compute_tumor_centroid(density)

        # Compute current tumor volume if not provided
        if current_volume <= 0:
            current_volume = self._compute_tumor_volume_internal(density)

        # Handle edge case of no initial volume
        if initial_volume <= 0:
            initial_volume = current_volume * 0.1  # Assume 10x growth

        # Get base constitutive matrix
        lam, mu = self.properties.lame_parameters()
        E = self.properties.young_modulus
        C_base = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ])

        # Isotropic growth strain direction
        growth_strain_dir = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        # Volume-preserving formulation
        # Volume strain = (V_new - V_old) / V_old
        volume_strain = (current_volume - initial_volume) / max(initial_volume, 1e-6)

        # Estimate tumor radius from volume (assuming spherical)
        tumor_radius = (3.0 * current_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        tumor_radius = max(tumor_radius, 1.0)  # Minimum 1mm radius

        # Characteristic decay length for radial pressure
        decay_length_factor = self.properties.pressure_decay_length_factor
        decay_length = decay_length_factor * tumor_radius

        for e, elem in enumerate(self.mesh.elements):
            vol = self._element_volumes[e]
            grads = self._shape_gradients[e]

            # Average density in element
            elem_density = np.mean(density[elem])

            # Skip elements with negligible tumor density
            if elem_density < 1e-6:
                continue

            # Get element-specific constitutive matrix if available
            if self._element_properties is not None:
                C = self._element_properties[e].get_constitutive_matrix()
                E_elem = self._element_properties[e].young_modulus
            else:
                C = C_base
                E_elem = E

            # Element centroid for distance calculation
            elem_centroid = np.mean(self.mesh.nodes[elem], axis=0)
            distance = np.linalg.norm(elem_centroid - tumor_center)

            # Volume-preserving eigenstrain formulation
            # Eigenstrain = volumetric_strain / 3 (for isotropic expansion)
            # Modified by density and distance-dependent decay
            eigenstrain_magnitude = (volume_strain / 3.0) * elem_density

            # Apply exponential decay with distance from tumor center
            # This models the pressure wave propagating outward
            decay_factor = np.exp(-distance / decay_length)
            eigenstrain_magnitude *= decay_factor

            growth_strain = eigenstrain_magnitude * growth_strain_dir

            # Growth stress: σ_growth = C * ε_growth
            growth_stress = C @ growth_strain

            # Compute equivalent nodal forces
            for i in range(4):
                node_idx = elem[i]
                B_i = self._strain_displacement_matrix(grads[i])

                # Eigenstrain-based force
                f_i = vol * B_i.T @ growth_stress

                # Add radial component for volume expansion
                # This ensures outward displacement from tumor center
                node_pos = self.mesh.nodes[node_idx]
                radial_vec = node_pos - tumor_center
                radial_dist = np.linalg.norm(radial_vec)

                if radial_dist > 1e-6:
                    radial_dir = radial_vec / radial_dist

                    # Radial stress from volume expansion with exponential decay
                    # σ_radial = E * ε_v * density * exp(-r/λ)
                    radial_stress = (
                        E_elem * volume_strain * elem_density *
                        np.exp(-radial_dist / decay_length)
                    )

                    # Convert stress to nodal force (stress * area ~ stress * vol^(2/3))
                    radial_force_magnitude = radial_stress * (vol ** (2.0 / 3.0)) / 4.0

                    f_i += radial_force_magnitude * radial_dir

                # Assemble into global force vector
                for d in range(3):
                    global_dof = node_idx * 3 + d
                    force[global_dof] += f_i[d]

        return force

    def _compute_growth_force_legacy(
        self,
        density: NDArray[np.float64],
        tumor_center: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Compute growth force using legacy empirical formulation.

        This is the original formulation with empirical scaling factors.
        Kept for backward compatibility.

        Args:
            density: Current tumor cell density at each node.
            tumor_center: Center of tumor mass for radial force calculation.

        Returns:
            Force vector (3*N,) for mechanical equilibrium.
        """
        n = self.mesh.num_nodes
        alpha = self.properties.growth_stress_coefficient
        mass_scaling = self.properties.mass_effect_scaling
        radial_factor = self.properties.radial_displacement_factor
        force = np.zeros(3 * n)

        # Compute tumor center of mass if not provided
        if tumor_center is None:
            tumor_center = self._compute_tumor_centroid(density)

        # Get base constitutive matrix (will be modified per-element if needed)
        lam, mu = self.properties.lame_parameters()
        C_base = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ])

        # Isotropic growth strain direction: [1, 1, 1, 0, 0, 0]
        # This represents uniform volumetric expansion
        growth_strain_dir = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        # Compute total tumor volume for mass effect scaling
        total_tumor_volume = 0.0
        for e, elem in enumerate(self.mesh.elements):
            elem_density = np.mean(density[elem])
            if elem_density > 0.1:
                total_tumor_volume += self._element_volumes[e] * elem_density

        # Scale mass effect by tumor volume (larger tumors create more displacement)
        volume_factor = 1.0 + np.log1p(total_tumor_volume / 100.0)

        for e, elem in enumerate(self.mesh.elements):
            vol = self._element_volumes[e]
            grads = self._shape_gradients[e]

            # Average density in element
            elem_density = np.mean(density[elem])

            # Skip elements with negligible tumor density
            if elem_density < 1e-6:
                continue

            # Get element-specific constitutive matrix if available
            if self._element_properties is not None:
                C = self._element_properties[e].get_constitutive_matrix()
            else:
                C = C_base

            # Enhanced growth strain with mass effect scaling
            # The mass_scaling amplifies displacement to model new volume creation
            effective_alpha = alpha * mass_scaling * volume_factor
            growth_strain = effective_alpha * elem_density * growth_strain_dir

            # Growth stress: σ_growth = C * ε_growth
            growth_stress = C @ growth_strain

            # Compute equivalent nodal forces: f_i = V * B_i^T * σ_growth
            # Each node contributes via its strain-displacement matrix
            for i in range(4):
                node_idx = elem[i]
                B_i = self._strain_displacement_matrix(grads[i])

                # Standard eigenstrain force
                f_i = vol * B_i.T @ growth_stress

                # Add radial mass-effect force
                # This models new material pushing tissue outward from tumor center
                node_pos = self.mesh.nodes[node_idx]
                radial_vec = node_pos - tumor_center
                radial_dist = np.linalg.norm(radial_vec)

                if radial_dist > 1e-6:
                    # Normalize to get direction
                    radial_dir = radial_vec / radial_dist

                    # Radial force decreases with distance (pressure decay)
                    # Force is proportional to density and inversely to distance
                    radial_magnitude = (
                        radial_factor * elem_density * vol * mu
                        / (radial_dist + 1.0)  # +1 prevents singularity at center
                    )

                    # Add radial component to force
                    f_i += radial_magnitude * radial_dir

                # Assemble into global force vector
                for d in range(3):
                    global_dof = node_idx * 3 + d
                    force[global_dof] += f_i[d]

        return force

    def _compute_tumor_centroid(
        self,
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the density-weighted centroid of the tumor.

        Args:
            density: Tumor cell density at each node.

        Returns:
            3D coordinates of tumor center of mass.
        """
        # Weighted average of node positions
        total_mass = 0.0
        centroid = np.zeros(3)

        for e, elem in enumerate(self.mesh.elements):
            elem_density = np.mean(density[elem])
            if elem_density > 0.01:
                vol = self._element_volumes[e]
                mass = vol * elem_density
                elem_centroid = np.mean(self.mesh.nodes[elem], axis=0)
                centroid += mass * elem_centroid
                total_mass += mass

        if total_mass > 1e-6:
            centroid /= total_mass
        else:
            # Default to mesh center if no tumor
            centroid = np.mean(self.mesh.nodes, axis=0)

        return centroid

    def _solve_mechanical_equilibrium(
        self,
        force: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve static mechanical equilibrium: K * u = f.

        Uses skull boundary nodes (immovable) when biophysical constraints
        are provided, otherwise uses mesh boundary nodes.

        Performance optimization:
        - Uses AMG preconditioning when enabled (3-10x faster convergence)
        - Configurable tolerance for speed/accuracy tradeoff
        """
        n = self.mesh.num_nodes
        config = self.solver_config

        # Get boundary nodes based on constraint type
        if self.boundary_condition == "skull":
            boundary_nodes = self._get_skull_boundary_nodes()
        else:
            boundary_nodes = self.mesh.boundary_nodes

        # Apply boundary conditions to force vector
        for node_idx in boundary_nodes:
            for dof in range(3):
                force[node_idx * 3 + dof] = 0.0

        # Build AMG preconditioner if needed (lazy initialization)
        preconditioner = None
        if config.use_amg and HAS_PYAMG:
            if self._amg_ml is None:
                # Build AMG hierarchy (one-time cost, reused for all solves)
                self._amg_ml = pyamg.smoothed_aggregation_solver(
                    self._stiffness_matrix,
                    strength=config.amg_strength,
                    max_coarse=500,
                )
            preconditioner = self._amg_ml.aspreconditioner(cycle=config.amg_cycle)

        # Solve using conjugate gradient with optional AMG preconditioning
        # Suppress singular matrix warnings - these can occur with ill-conditioned
        # meshes but the solve typically still produces reasonable results
        # Note: scipy >= 1.12 renamed 'tol' to 'rtol'
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Matrix is exactly singular')
            try:
                u_flat, info = cg(
                    self._stiffness_matrix,
                    force,
                    rtol=config.mechanical_tol,
                    maxiter=config.mechanical_maxiter,
                    M=preconditioner,
                )
            except TypeError:
                # Fallback for older scipy versions
                u_flat, info = cg(
                    self._stiffness_matrix,
                    force,
                    tol=config.mechanical_tol,
                    maxiter=config.mechanical_maxiter,
                    M=preconditioner,
                )

            if info != 0:
                # Fall back to direct solver if CG did not converge
                try:
                    u_flat = spsolve(self._stiffness_matrix, force)
                except Exception:
                    # Last resort: use least-squares solver
                    result = lsqr(self._stiffness_matrix, force, atol=1e-8, btol=1e-8)
                    u_flat = result[0]

        # Reshape to (n, 3)
        displacement = u_flat.reshape((n, 3))

        return displacement

    def _compute_stress(
        self,
        displacement: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute stress tensor at each element.

        Uses tissue-specific constitutive matrices when biophysical
        constraints are available (anisotropic for white matter).
        """
        stress = np.zeros((self.mesh.num_elements, 6))

        for e, elem in enumerate(self.mesh.elements):
            grads = self._shape_gradients[e]

            # Compute strain from displacement
            strain = np.zeros(6)
            for i in range(4):
                B = self._strain_displacement_matrix(grads[i])
                strain += B @ displacement[elem[i]]

            # Get tissue-specific constitutive matrix
            if self._element_properties is not None:
                C = self._element_properties[e].get_constitutive_matrix()
            else:
                lam, mu = self.properties.lame_parameters()
                C = np.array([
                    [lam + 2*mu, lam, lam, 0, 0, 0],
                    [lam, lam + 2*mu, lam, 0, 0, 0],
                    [lam, lam, lam + 2*mu, 0, 0, 0],
                    [0, 0, 0, mu, 0, 0],
                    [0, 0, 0, 0, mu, 0],
                    [0, 0, 0, 0, 0, mu],
                ])

            # Compute stress: sigma = C * epsilon
            stress[e] = C @ strain

        return stress

    def simulate(
        self,
        initial_state: TumorState,
        duration: float,
        dt: float = 1.0,
        callback: Optional[Callable[[TumorState, int], None]] = None,
        use_adaptive_stepping: Optional[bool] = None,
    ) -> List[TumorState]:
        """
        Run simulation for a specified duration.

        Supports adaptive time stepping when enabled, which uses larger time
        steps when tumor growth is slow and smaller steps during rapid expansion.
        This improves efficiency without sacrificing accuracy.

        Args:
            initial_state: Initial tumor state.
            duration: Simulation duration in days.
            dt: Initial (or fixed) time step in days.
            callback: Optional callback function called after each step.
                     Signature: callback(state, step_idx) -> None
            use_adaptive_stepping: Override property setting for adaptive stepping.
                                  If None, uses self.properties.use_adaptive_stepping.

        Returns:
            List of TumorState objects at each time step.
        """
        # Determine if adaptive stepping is enabled
        adaptive = use_adaptive_stepping
        if adaptive is None:
            adaptive = self.properties.use_adaptive_stepping

        if adaptive:
            return self._simulate_adaptive(initial_state, duration, dt, callback)
        else:
            return self._simulate_fixed(initial_state, duration, dt, callback)

    def _simulate_fixed(
        self,
        initial_state: TumorState,
        duration: float,
        dt: float,
        callback: Optional[Callable[[TumorState, int], None]] = None,
    ) -> List[TumorState]:
        """
        Run simulation with fixed time steps.

        Args:
            initial_state: Initial tumor state.
            duration: Simulation duration in days.
            dt: Time step in days.
            callback: Optional callback function.

        Returns:
            List of TumorState objects at each time step.
        """
        states = [initial_state]
        current_state = initial_state
        num_steps = int(duration / dt)

        for step_idx in range(num_steps):
            current_state = self.step(current_state, dt)
            states.append(current_state)

            if callback is not None:
                callback(current_state, step_idx)

        return states

    def _simulate_adaptive(
        self,
        initial_state: TumorState,
        duration: float,
        dt: float,
        callback: Optional[Callable[[TumorState, int], None]] = None,
    ) -> List[TumorState]:
        """
        Run simulation with adaptive time stepping.

        Adjusts time step based on density change rate:
        - If change rate < threshold_low: increase dt (up to dt_max)
        - If change rate > threshold_high: decrease dt (down to dt_min)

        This improves efficiency during slow growth phases while maintaining
        accuracy during rapid expansion.

        Args:
            initial_state: Initial tumor state.
            duration: Simulation duration in days.
            dt: Initial time step in days.
            callback: Optional callback function.

        Returns:
            List of TumorState objects at each time step.
        """
        states = [initial_state]
        current_state = initial_state
        current_time = 0.0
        step_idx = 0

        # Get adaptive parameters from properties
        dt_min = self.properties.dt_min
        dt_max = min(self.properties.dt_max, duration / 2)  # Don't exceed half duration
        increase_threshold = self.properties.adaptive_increase_threshold
        decrease_threshold = self.properties.adaptive_decrease_threshold

        current_dt = dt

        while current_time < duration:
            # Adjust dt to not exceed remaining time
            actual_dt = min(current_dt, duration - current_time)

            # Store old density for change rate calculation
            old_density = current_state.cell_density.copy()

            # Take time step
            current_state = self.step(current_state, actual_dt)
            states.append(current_state)
            current_time += actual_dt

            # Compute density change rate
            density_change_rate = self._compute_density_change_rate(
                old_density, current_state.cell_density
            )

            # Adapt time step for next iteration
            current_dt = self._compute_adaptive_dt(
                density_change_rate, current_dt, dt_min, dt_max,
                increase_threshold, decrease_threshold
            )

            if callback is not None:
                callback(current_state, step_idx)

            step_idx += 1

        return states

    def _compute_density_change_rate(
        self,
        old_density: NDArray[np.float64],
        new_density: NDArray[np.float64],
    ) -> float:
        """
        Compute the relative density change rate.

        Args:
            old_density: Previous density field.
            new_density: Current density field.

        Returns:
            Relative change rate: ||c_new - c_old|| / ||c_old||
        """
        diff = new_density - old_density
        diff_norm = np.linalg.norm(diff)
        old_norm = np.linalg.norm(old_density)

        if old_norm < 1e-10:
            return 1.0  # Large change from near-zero

        return diff_norm / old_norm

    def _compute_adaptive_dt(
        self,
        density_change_rate: float,
        current_dt: float,
        dt_min: float,
        dt_max: float,
        increase_threshold: float,
        decrease_threshold: float,
    ) -> float:
        """
        Compute adapted time step based on density change rate.

        Args:
            density_change_rate: Current relative change rate.
            current_dt: Current time step.
            dt_min: Minimum allowed time step.
            dt_max: Maximum allowed time step.
            increase_threshold: Increase dt if change < this.
            decrease_threshold: Decrease dt if change > this.

        Returns:
            New time step value.
        """
        if density_change_rate < increase_threshold:
            # Slow growth - increase time step
            new_dt = current_dt * 1.5
        elif density_change_rate > decrease_threshold:
            # Rapid growth - decrease time step
            new_dt = current_dt * 0.5
        else:
            # Moderate growth - keep current step
            new_dt = current_dt

        # Clamp to allowed range
        return max(dt_min, min(new_dt, dt_max))

    def compute_tumor_volume(
        self,
        state: TumorState,
        threshold: float = 0.1,
    ) -> float:
        """
        Compute tumor volume above a density threshold.

        Args:
            state: Current tumor state.
            threshold: Density threshold for tumor boundary.

        Returns:
            Tumor volume in mm^3.
        """
        volume = 0.0

        for e, elem in enumerate(self.mesh.elements):
            elem_density = np.mean(state.cell_density[elem])
            if elem_density >= threshold:
                volume += self._element_volumes[e]

        return volume

    def compute_max_displacement(self, state: TumorState) -> float:
        """Compute maximum displacement magnitude."""
        magnitudes = np.linalg.norm(state.displacement, axis=1)
        return float(np.max(magnitudes))

    def compute_von_mises_stress(
        self,
        state: TumorState,
    ) -> NDArray[np.float64]:
        """Compute von Mises stress at each element."""
        stress = state.stress
        # Voigt notation: [sxx, syy, szz, sxy, sxz, syz]
        sxx, syy, szz = stress[:, 0], stress[:, 1], stress[:, 2]
        sxy, sxz, syz = stress[:, 3], stress[:, 4], stress[:, 5]

        # von Mises: sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2 + 6*(sxy^2+sxz^2+syz^2)))
        vm = np.sqrt(0.5 * (
            (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 +
            6 * (sxy**2 + sxz**2 + syz**2)
        ))

        return vm

    # =========================================================================
    # Serialization Methods for Precomputed Solvers
    # =========================================================================

    def save(self, directory: str) -> None:
        """
        Save precomputed solver state to a directory.

        Saves all matrices, mesh, and precomputed data for fast loading later.
        This avoids the expensive matrix assembly step when using default parameters.

        Args:
            directory: Directory path to save solver data.
        """
        import json
        import pickle
        from pathlib import Path
        from .mesh import save_mesh

        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save mesh
        save_mesh(self.mesh, str(save_dir / "mesh.vtu"))

        # Save boundary nodes separately (not stored in meshio format)
        np.save(save_dir / "boundary_nodes.npy", self.mesh.boundary_nodes)

        # Save system matrices (scipy sparse format)
        matrices_dir = save_dir / "matrices"
        matrices_dir.mkdir(exist_ok=True)
        sparse.save_npz(matrices_dir / "mass_matrix.npz", self._mass_matrix)
        sparse.save_npz(matrices_dir / "stiffness_matrix.npz", self._stiffness_matrix)
        sparse.save_npz(matrices_dir / "diffusion_matrix.npz", self._diffusion_matrix)

        # Save precomputed element data
        precomputed_dir = save_dir / "precomputed"
        precomputed_dir.mkdir(exist_ok=True)
        np.save(precomputed_dir / "element_volumes.npy", self._element_volumes)
        with open(precomputed_dir / "shape_gradients.pkl", "wb") as f:
            pickle.dump(self._shape_gradients, f)

        # Save biophysical data if present
        if self._node_tissues is not None:
            np.save(precomputed_dir / "node_tissues.npy", self._node_tissues)
        if self._node_fiber_directions is not None:
            np.save(precomputed_dir / "node_fiber_directions.npy", self._node_fiber_directions)

        # Save element properties as simplified format
        if self._element_properties is not None:
            elem_props_data = []
            for props in self._element_properties:
                elem_props_data.append({
                    "young_modulus": props.young_modulus,
                    "poisson_ratio": props.poisson_ratio,
                    "proliferation_rate": props.proliferation_rate,
                    "diffusion_coefficient": props.diffusion_coefficient,
                    "carrying_capacity": props.carrying_capacity,
                    "growth_stress_coefficient": props.growth_stress_coefficient,
                    "mass_effect_scaling": props.mass_effect_scaling,
                    "radial_displacement_factor": props.radial_displacement_factor,
                    "anisotropy_ratio": props.anisotropy_ratio,
                    "fiber_direction": props.fiber_direction.tolist() if props.fiber_direction is not None else None,
                })
            with open(precomputed_dir / "element_properties.json", "w") as f:
                json.dump(elem_props_data, f)

        # Save metadata
        metadata = {
            "version": "1.2",
            "boundary_condition": self.boundary_condition,
            "num_nodes": len(self.mesh.nodes),
            "num_elements": len(self.mesh.elements),
            "has_biophysical_constraints": self._node_tissues is not None,
            "properties": {
                "young_modulus": self.properties.young_modulus,
                "poisson_ratio": self.properties.poisson_ratio,
                "proliferation_rate": self.properties.proliferation_rate,
                "diffusion_coefficient": self.properties.diffusion_coefficient,
                "carrying_capacity": self.properties.carrying_capacity,
                "growth_stress_coefficient": self.properties.growth_stress_coefficient,
                "mass_effect_scaling": self.properties.mass_effect_scaling,
                "radial_displacement_factor": self.properties.radial_displacement_factor,
                "anisotropy_ratio": self.properties.anisotropy_ratio,
            },
            "solver_config": self.solver_config.to_dict(),
        }
        with open(save_dir / "solver_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "TumorGrowthSolver":
        """
        Load precomputed solver from a directory.

        This bypasses expensive matrix assembly by loading pre-built matrices.

        Args:
            directory: Directory containing saved solver data.

        Returns:
            TumorGrowthSolver with precomputed matrices loaded.
        """
        import json
        import pickle
        from pathlib import Path
        from .mesh import load_mesh

        load_dir = Path(directory)

        if not load_dir.exists():
            raise FileNotFoundError(f"Solver directory not found: {directory}")

        # Load metadata
        with open(load_dir / "solver_metadata.json") as f:
            metadata = json.load(f)

        # Load mesh
        mesh = load_mesh(str(load_dir / "mesh.vtu"))

        # Load boundary nodes
        boundary_nodes_path = load_dir / "boundary_nodes.npy"
        if boundary_nodes_path.exists():
            mesh.boundary_nodes = np.load(boundary_nodes_path)

        # Create properties from metadata
        props_data = metadata.get("properties", {})
        properties = MaterialProperties(
            young_modulus=props_data.get("young_modulus", 3000.0),
            poisson_ratio=props_data.get("poisson_ratio", 0.45),
            proliferation_rate=props_data.get("proliferation_rate", 0.01),
            diffusion_coefficient=props_data.get("diffusion_coefficient", 0.1),
            carrying_capacity=props_data.get("carrying_capacity", 1.0),
            growth_stress_coefficient=props_data.get("growth_stress_coefficient", 0.15),
            mass_effect_scaling=props_data.get("mass_effect_scaling", 3.0),
            radial_displacement_factor=props_data.get("radial_displacement_factor", 1.5),
            anisotropy_ratio=props_data.get("anisotropy_ratio", 2.0),
        )

        # Load solver config if present (v1.1+), otherwise use default
        solver_config_data = metadata.get("solver_config")
        if solver_config_data:
            solver_config = SolverConfig.from_dict(solver_config_data)
        else:
            solver_config = SolverConfig.default()

        # Create solver instance without building matrices
        solver = object.__new__(cls)
        solver.mesh = mesh
        solver.properties = properties
        solver.boundary_condition = metadata.get("boundary_condition", "fixed")
        solver.biophysical_constraints = None  # Not serialized
        solver.solver_config = solver_config
        solver._amg_ml = None  # Will be built lazily on first solve

        # Load system matrices
        matrices_dir = load_dir / "matrices"
        solver._mass_matrix = sparse.load_npz(matrices_dir / "mass_matrix.npz")
        solver._stiffness_matrix = sparse.load_npz(matrices_dir / "stiffness_matrix.npz")
        solver._diffusion_matrix = sparse.load_npz(matrices_dir / "diffusion_matrix.npz")

        # Load precomputed element data
        precomputed_dir = load_dir / "precomputed"
        solver._element_volumes = np.load(precomputed_dir / "element_volumes.npy")
        with open(precomputed_dir / "shape_gradients.pkl", "rb") as f:
            solver._shape_gradients = pickle.load(f)

        # Load biophysical data if present
        node_tissues_path = precomputed_dir / "node_tissues.npy"
        solver._node_tissues = np.load(node_tissues_path) if node_tissues_path.exists() else None

        fiber_dir_path = precomputed_dir / "node_fiber_directions.npy"
        solver._node_fiber_directions = np.load(fiber_dir_path) if fiber_dir_path.exists() else None

        # Load element properties if present
        elem_props_path = precomputed_dir / "element_properties.json"
        if elem_props_path.exists():
            with open(elem_props_path) as f:
                elem_props_data = json.load(f)
            solver._element_properties = []
            for ep in elem_props_data:
                fiber_dir = np.array(ep["fiber_direction"]) if ep["fiber_direction"] else None
                solver._element_properties.append(MaterialProperties(
                    young_modulus=ep["young_modulus"],
                    poisson_ratio=ep["poisson_ratio"],
                    proliferation_rate=ep["proliferation_rate"],
                    diffusion_coefficient=ep["diffusion_coefficient"],
                    carrying_capacity=ep["carrying_capacity"],
                    growth_stress_coefficient=ep["growth_stress_coefficient"],
                    mass_effect_scaling=ep.get("mass_effect_scaling", 3.0),
                    radial_displacement_factor=ep.get("radial_displacement_factor", 1.5),
                    anisotropy_ratio=ep["anisotropy_ratio"],
                    fiber_direction=fiber_dir,
                ))
        else:
            solver._element_properties = None

        return solver

    @classmethod
    def load_default(
        cls,
        solver_config: Optional[SolverConfig] = None,
    ) -> "TumorGrowthSolver":
        """
        Load the precomputed default solver for posterior fossa simulations.

        This provides fast initialization (~100ms vs ~10s) by loading
        precomputed matrices built with default parameters:
        - MNI152 space with posterior fossa restriction
        - Bundled SUIT atlas regions
        - Bundled MNI152 tissue segmentation
        - Bundled HCP1065 DTI fiber orientations
        - Default tumor origin at vermis [1, -61, -34] MNI

        Args:
            solver_config: Optional solver configuration to override defaults.
                          Use SolverConfig.fast() for approximate (faster) solutions,
                          SolverConfig.accurate() for high precision.

        Returns:
            TumorGrowthSolver ready for simulation.

        Raises:
            FileNotFoundError: If precomputed solver data not found.

        Example:
            # Fast approximate mode (3-10x speedup)
            solver = TumorGrowthSolver.load_default(SolverConfig.fast())

            # Default mode with AMG preconditioning
            solver = TumorGrowthSolver.load_default()
        """
        from pathlib import Path

        default_solver_dir = Path(__file__).parent.parent.parent / "data" / "solvers" / "default_posterior_fossa"

        if not default_solver_dir.exists():
            raise FileNotFoundError(
                f"Precomputed default solver not found at {default_solver_dir}. "
                "Run 'python -m pft_fem.create_default_solver' to generate it."
            )

        solver = cls.load(str(default_solver_dir))

        # Override solver config if provided
        if solver_config is not None:
            solver.solver_config = solver_config
            solver._amg_ml = None  # Reset AMG preconditioner

        return solver
