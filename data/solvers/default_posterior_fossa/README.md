# Precomputed Default FEM Solver

This directory contains a precomputed FEM solver for posterior fossa tumor simulations.

## Mesh Generation Method

**DTI-guided (fiber-aligned)**

This mesh was generated using DTI-guided mesh generation, which:
- Places white matter nodes along principal diffusion directions from DTI
- Connects gray matter nodes to the white matter skeleton
- Preserves biophysical connectivity for accurate tumor diffusion
- Allows coarsening while maintaining fiber tract topology

## Parameters

- **Coordinate space**: MNI152
- **Region**: Posterior fossa (cerebellum + brainstem)
- **Tissue segmentation**: MNI152 FAST segmentation (GM/WM/CSF)
- **Fiber orientations**: HCP1065 DTI atlas
- **Boundary condition**: Skull fixed, CSF free (fourth ventricle can be displaced)
- **WM node spacing**: 6.0 mm
- **GM node spacing**: 8.0 mm

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
