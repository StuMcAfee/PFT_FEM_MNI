"""
PFT_FEM: Posterior Fossa Tumor Finite Element Modeling

A pipeline for simulating MRI images by modeling tumor growth in the
posterior cranial fossa using finite element methods.

Default configuration uses MNI152 space (ICBM 2009c) with:
- Non-skull-stripped T1 template for skull boundary constraints
- HCP1065 DTI-based fiber orientation for anisotropic tissue properties
- Posterior fossa restriction (cerebellum + brainstem)
- Tumor origin at MNI coordinates (2, -49, -35) - vermis/fourth ventricle

Key features:
- White matter: Anisotropic properties based on DTI fiber orientation
- Gray matter: Isotropic, uniformly compressible
- Skull boundary: Fixed displacement from non-skull-stripped T1
- Tissue-specific diffusion: Enhanced along white matter tracts

Quick start:
    from pft_fem import MNIAtlasLoader, MRISimulator, TumorParameters

    # Load MNI atlas (default: non-skull-stripped T1, DTI enabled)
    loader = MNIAtlasLoader()
    atlas = loader.load()

    # Run simulation with default MNI tumor origin
    simulator = MRISimulator(atlas)
    result = simulator.run_full_pipeline(duration_days=365)
"""

__version__ = "0.1.0"

from .atlas import SUITAtlasLoader, MNIAtlasLoader, DefaultAtlasLoader, AtlasProcessor
from .mesh import MeshGenerator, TetMesh
from .dti_mesh import (
    DTIGuidedMeshGenerator,
    DTIMeshConfig,
    WhiteMatterGraph,
    GrayMatterNodes,
    create_dti_guided_mesh,
)
from .fem import TumorGrowthSolver, MaterialProperties, TissueType, TumorState, SolverConfig
from .simulation import MRISimulator, TumorParameters, MRISequence, SimulationResult
from .io import NIfTIWriter, load_nifti, save_nifti
from .transforms import SpatialTransform, ANTsTransformExporter, compute_transform_from_simulation
from .biophysical_constraints import (
    BiophysicalConstraints,
    BrainTissue,
    TissueSegmentation,
    FiberOrientation,
    AnisotropicMaterialProperties,
    SUITPyIntegration,
    MNIAtlasLoader as MNIBiophysicalLoader,  # Alias for backwards compatibility
    SpaceTransformer,
    DEFAULT_TUMOR_ORIGIN_MNI,
    POSTERIOR_FOSSA_BOUNDS_MNI,
    # Regional material properties (literature-based)
    REGIONAL_STIFFNESS,
    REGIONAL_POISSON_RATIO,
    ANATOMICAL_REGIONS_MNI,
    RegionalMaterialProperties,
    get_anatomical_region,
    get_regional_properties,
    get_regional_stiffness,
    get_regional_poisson_ratio,
)

__all__ = [
    # Atlas loading (MNI-first)
    "MNIAtlasLoader",
    "DefaultAtlasLoader",
    "SUITAtlasLoader",  # Legacy SUIT support
    "AtlasProcessor",
    # Mesh generation
    "MeshGenerator",
    "TetMesh",
    # DTI-guided mesh generation
    "DTIGuidedMeshGenerator",
    "DTIMeshConfig",
    "WhiteMatterGraph",
    "GrayMatterNodes",
    "create_dti_guided_mesh",
    # FEM solver
    "TumorGrowthSolver",
    "MaterialProperties",
    "TissueType",
    "TumorState",
    "SolverConfig",
    # MRI simulation
    "MRISimulator",
    "TumorParameters",
    "MRISequence",
    "SimulationResult",
    # I/O
    "NIfTIWriter",
    "load_nifti",
    "save_nifti",
    # Spatial transforms
    "SpatialTransform",
    "ANTsTransformExporter",
    "compute_transform_from_simulation",
    # Biophysical constraints
    "BiophysicalConstraints",
    "BrainTissue",
    "TissueSegmentation",
    "FiberOrientation",
    "AnisotropicMaterialProperties",
    "SUITPyIntegration",
    "SpaceTransformer",
    # Default parameters (MNI space)
    "DEFAULT_TUMOR_ORIGIN_MNI",
    "POSTERIOR_FOSSA_BOUNDS_MNI",
    # Regional material properties (literature-based)
    "REGIONAL_STIFFNESS",
    "REGIONAL_POISSON_RATIO",
    "ANATOMICAL_REGIONS_MNI",
    "RegionalMaterialProperties",
    "get_anatomical_region",
    "get_regional_properties",
    "get_regional_stiffness",
    "get_regional_poisson_ratio",
]
