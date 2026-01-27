#!/usr/bin/env python3
"""
Create precomputed default FEM solver for fast initialization.

This script generates a precomputed solver with default parameters:
- MNI152 space with posterior fossa restriction
- Bundled SUIT atlas regions
- Bundled MNI152 tissue segmentation
- Bundled HCP1065 DTI fiber orientations
- Fixed boundary conditions

Two mesh generation modes are available:
1. **DTI-guided (default)**: Creates mesh following white matter fiber tracts
   - Nodes placed along principal diffusion directions from DTI
   - Gray matter nodes attached to white matter skeleton
   - Preserves biophysical connectivity for accurate tumor diffusion
   - Allows aggressive coarsening while maintaining accuracy

2. **Voxel-based (legacy)**: Simple voxel-to-tetrahedra conversion
   - Uniform grid structure
   - Faster to generate but less biophysically accurate

The precomputed solver can be loaded with TumorGrowthSolver.load_default()
for ~100x faster initialization compared to building from scratch.

Usage:
    python -m pft_fem.create_default_solver                    # DTI-guided (default)
    python -m pft_fem.create_default_solver --method voxel     # Voxel-based
    python -m pft_fem.create_default_solver --wm-spacing 8.0   # Coarser WM nodes
"""

import sys
import time
from pathlib import Path


def create_default_solver(
    output_dir: Path = None,
    mesh_voxel_size: float = 4.0,
    method: str = "dti",
    wm_node_spacing: float = 6.0,
    gm_node_spacing: float = 8.0,
) -> None:
    """
    Create and save the default precomputed solver.

    Args:
        output_dir: Directory to save solver. Defaults to data/solvers/default_posterior_fossa.
        mesh_voxel_size: Mesh voxel size in mm (for voxel method). Default 4.0mm.
        method: Mesh generation method - "dti" (default) or "voxel".
        wm_node_spacing: White matter node spacing in mm (for DTI method). Default 6.0mm.
        gm_node_spacing: Gray matter node spacing in mm (for DTI method). Default 8.0mm.
    """
    print("Creating precomputed default FEM solver...")
    print(f"  Method: {method}")
    if method == "dti":
        print(f"  WM node spacing: {wm_node_spacing}mm")
        print(f"  GM node spacing: {gm_node_spacing}mm")
    else:
        print(f"  Mesh voxel size: {mesh_voxel_size}mm")
    print("=" * 60)

    # Import here to show progress
    print("Loading modules...")
    start = time.time()

    import numpy as np
    from scipy import ndimage

    from .biophysical_constraints import BiophysicalConstraints
    from .mesh import MeshGenerator
    from .fem import TumorGrowthSolver, SolverConfig

    print(f"  Modules loaded in {time.time() - start:.2f}s")

    # Determine output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "solvers" / "default_posterior_fossa"
    output_dir = Path(output_dir)

    # Step 1: Load biophysical constraints
    print("\nStep 1: Loading biophysical constraints...")
    start = time.time()

    bc = BiophysicalConstraints(
        posterior_fossa_only=True,
        use_suit_space=False,  # Use MNI152 space
        use_dti_constraints=True,
    )
    bc.load_all_constraints()

    print(f"  Tissue segmentation loaded: {bc._segmentation.labels.shape}")
    print(f"  Fiber orientations loaded: {bc._fibers.vectors.shape}")
    print(f"  Loaded in {time.time() - start:.2f}s")

    # Step 2: Create mesh using selected method
    if method == "dti":
        mesh = _create_dti_guided_mesh(bc, wm_node_spacing, gm_node_spacing)
        effective_voxel_size = wm_node_spacing  # Use WM spacing as effective resolution
    else:
        mesh = _create_voxel_mesh(bc, mesh_voxel_size)
        effective_voxel_size = mesh_voxel_size

    # Step 3: Build FEM solver with biophysical constraints and default config
    print("\nStep 3: Building FEM solver with tissue-specific properties...")
    start = time.time()

    # Use default solver config with appropriate mesh settings
    solver_config = SolverConfig.default()
    solver_config = SolverConfig(
        mechanical_tol=solver_config.mechanical_tol,
        mechanical_maxiter=solver_config.mechanical_maxiter,
        use_amg=solver_config.use_amg,
        amg_cycle=solver_config.amg_cycle,
        amg_strength=solver_config.amg_strength,
        mesh_voxel_size=effective_voxel_size,
        output_at_full_resolution=True,
    )

    solver = TumorGrowthSolver(
        mesh=mesh,
        boundary_condition="skull_csf_free",  # Fixed skull, free CSF (fourth ventricle)
        biophysical_constraints=bc,
        solver_config=solver_config,
    )

    print(f"  Mass matrix: {solver._mass_matrix.shape}, nnz={solver._mass_matrix.nnz}")
    print(f"  Stiffness matrix: {solver._stiffness_matrix.shape}, nnz={solver._stiffness_matrix.nnz}")
    print(f"  Diffusion matrix: {solver._diffusion_matrix.shape}, nnz={solver._diffusion_matrix.nnz}")
    print(f"  Solver config: mesh_voxel_size={effective_voxel_size}mm, use_amg={solver_config.use_amg}")
    print(f"  Solver built in {time.time() - start:.2f}s")

    # Step 4: Save precomputed solver
    print(f"\nStep 4: Saving precomputed solver to {output_dir}...")
    start = time.time()

    solver.save(str(output_dir))

    # Calculate saved data size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Saved in {time.time() - start:.2f}s")

    # Create README
    method_desc = "DTI-guided (fiber-aligned)" if method == "dti" else "Voxel-based"
    readme_content = f"""# Precomputed Default FEM Solver

This directory contains a precomputed FEM solver for posterior fossa tumor simulations.

## Mesh Generation Method

**{method_desc}**

{"This mesh was generated using DTI-guided mesh generation, which:" if method == "dti" else "This mesh was generated using voxel-based conversion, which:"}
{"- Places white matter nodes along principal diffusion directions from DTI" if method == "dti" else "- Subdivides each voxel into tetrahedra"}
{"- Connects gray matter nodes to the white matter skeleton" if method == "dti" else "- Creates a uniform grid structure"}
{"- Preserves biophysical connectivity for accurate tumor diffusion" if method == "dti" else "- Simple and fast to generate"}
{"- Allows coarsening while maintaining fiber tract topology" if method == "dti" else "- Less biophysically accurate at coarse resolutions"}

## Parameters

- **Coordinate space**: MNI152
- **Region**: Posterior fossa (cerebellum + brainstem)
- **Tissue segmentation**: MNI152 FAST segmentation (GM/WM/CSF)
- **Fiber orientations**: HCP1065 DTI atlas
- **Boundary condition**: Skull fixed, CSF free (fourth ventricle can be displaced)
{f"- **WM node spacing**: {wm_node_spacing} mm" if method == "dti" else f"- **Voxel size**: {mesh_voxel_size} mm"}
{f"- **GM node spacing**: {gm_node_spacing} mm" if method == "dti" else ""}

## Usage

```python
from pft_fem import TumorGrowthSolver, TumorState

# Load precomputed solver (~100ms vs ~10s from scratch)
solver = TumorGrowthSolver.load_default()

# Create initial tumor state
state = TumorState.initial(
    mesh=solver.mesh,
    seed_center=[2.0, -49.0, -35.0],  # Vermis/fourth ventricle in MNI coordinates
)

# Run simulation
for _ in range(100):
    state = solver.step(state, dt=0.1)
```

## Files

- `mesh.vtu` - Tetrahedral mesh (meshio format)
- `boundary_nodes.npy` - Fixed boundary node indices
- `solver_metadata.json` - Solver configuration and parameters
- `matrices/` - Precomputed sparse system matrices
  - `mass_matrix.npz` - Mass matrix for time integration
  - `stiffness_matrix.npz` - Mechanical stiffness matrix
  - `diffusion_matrix.npz` - Tumor diffusion matrix
- `precomputed/` - Precomputed element and node data
  - `element_volumes.npy` - Volume of each tetrahedron
  - `shape_gradients.pkl` - Shape function gradients
  - `node_tissues.npy` - Tissue type at each node
  - `node_fiber_directions.npy` - Fiber direction at each node
  - `element_properties.json` - Material properties per element

## Regenerating

To regenerate this precomputed solver:

```bash
# DTI-guided mesh (default, recommended)
python -m pft_fem.create_default_solver

# Voxel-based mesh (legacy)
python -m pft_fem.create_default_solver --method voxel
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 60)
    print("Precomputed default solver created successfully!")
    print(f"  Method: {method_desc}")
    print(f"  Nodes: {mesh.num_nodes}, Elements: {mesh.num_elements}")
    print(f"Load with: TumorGrowthSolver.load_default()")


def _create_dti_guided_mesh(bc, wm_node_spacing: float, gm_node_spacing: float):
    """Create mesh using DTI-guided method."""
    import time
    from .dti_mesh import DTIGuidedMeshGenerator, DTIMeshConfig

    print("\nStep 2: Creating DTI-guided mesh from fiber orientations...")
    start = time.time()

    # Configure DTI mesh generation
    config = DTIMeshConfig(
        wm_node_spacing=wm_node_spacing,
        gm_node_spacing=gm_node_spacing,
        fa_threshold=0.2,
        min_tract_length=10.0,
        seed_density=0.01,
    )

    # Create DTI-guided mesh generator
    generator = DTIGuidedMeshGenerator(
        fiber_orientation=bc._fibers,
        tissue_segmentation=bc._segmentation,
        config=config,
        posterior_fossa_only=True,
    )

    # Generate mesh (this prints its own progress)
    mesh = generator.generate_mesh()

    print(f"  DTI-guided mesh created in {time.time() - start:.2f}s")
    print(f"  Nodes: {mesh.num_nodes}")
    print(f"  Elements: {mesh.num_elements}")
    print(f"  Boundary nodes: {len(mesh.boundary_nodes)}")

    return mesh


def _create_voxel_mesh(bc, mesh_voxel_size: float):
    """Create mesh using voxel-based method (legacy)."""
    import time
    import numpy as np
    from scipy import ndimage
    from .mesh import MeshGenerator

    print("\nStep 2: Creating voxel-based mesh from posterior fossa region...")
    start = time.time()

    # Get posterior fossa bounding box mask
    pf_bounds_mask = bc.compute_posterior_fossa_mask()
    print(f"  Posterior fossa bounds shape: {pf_bounds_mask.shape}")
    print(f"  Posterior fossa bounds voxels: {pf_bounds_mask.sum()}")

    # Get actual brain tissue from segmentation (GM + WM, excluding CSF/background)
    tissue_labels = bc._segmentation.labels
    brain_tissue_mask = (tissue_labels == 2) | (tissue_labels == 3)  # GM=2, WM=3
    print(f"  Brain tissue voxels (GM+WM): {brain_tissue_mask.sum()}")

    # Combine: only brain tissue within posterior fossa bounds
    mask = pf_bounds_mask & brain_tissue_mask
    print(f"  Combined mask voxels (posterior fossa tissue): {mask.sum()}")

    # Get the affine from the segmentation
    original_affine = bc._segmentation.affine
    if original_affine is None:
        original_affine = np.eye(4)
        original_affine[0, 3] = -90
        original_affine[1, 3] = -126
        original_affine[2, 3] = -72

    # Downsample mask if using coarse mesh
    original_voxel_size = 1.0
    coarse_factor = mesh_voxel_size / original_voxel_size

    if coarse_factor > 1.5:
        print(f"  Downsampling mask by factor {coarse_factor:.1f} for coarse mesh...")
        zoom_factors = tuple(1.0 / coarse_factor for _ in range(3))
        mask_coarse = ndimage.zoom(
            mask.astype(np.float32),
            zoom_factors,
            order=0
        ) > 0.5
        coarse_voxel_size = (mesh_voxel_size, mesh_voxel_size, mesh_voxel_size)
        coarse_affine = original_affine.copy()
        print(f"  Coarse mask shape: {mask_coarse.shape}")
        print(f"  Coarse mask voxels: {mask_coarse.sum()}")
    else:
        mask_coarse = mask
        coarse_voxel_size = (original_voxel_size, original_voxel_size, original_voxel_size)
        coarse_affine = original_affine

    # Create mesh from mask
    mesh_gen = MeshGenerator()
    mesh = mesh_gen.from_mask(mask_coarse, voxel_size=coarse_voxel_size, affine=coarse_affine)

    print(f"  Voxel-based mesh created in {time.time() - start:.2f}s")
    print(f"  Nodes: {mesh.num_nodes}")
    print(f"  Elements: {mesh.num_elements}")
    print(f"  Boundary nodes: {len(mesh.boundary_nodes)}")

    return mesh


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create precomputed default FEM solver for fast initialization."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/solvers/default_posterior_fossa)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dti", "voxel"],
        default="dti",
        help="Mesh generation method: 'dti' (default) or 'voxel' (legacy)",
    )
    parser.add_argument(
        "--mesh-voxel-size",
        type=float,
        default=4.0,
        help="Mesh voxel size in mm for voxel method (default: 4.0)",
    )
    parser.add_argument(
        "--wm-spacing",
        type=float,
        default=6.0,
        help="White matter node spacing in mm for DTI method (default: 6.0)",
    )
    parser.add_argument(
        "--gm-spacing",
        type=float,
        default=8.0,
        help="Gray matter node spacing in mm for DTI method (default: 8.0)",
    )

    args = parser.parse_args()

    try:
        create_default_solver(
            output_dir=args.output_dir,
            mesh_voxel_size=args.mesh_voxel_size,
            method=args.method,
            wm_node_spacing=args.wm_spacing,
            gm_node_spacing=args.gm_spacing,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
