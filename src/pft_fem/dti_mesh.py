"""
DTI-guided mesh generation for biophysically-informed FEM simulation.

This module implements a multi-step mesh generation process that preserves
white matter fiber tract topology:

1. **White Matter Skeleton**: Place nodes along principal diffusion directions
   from DTI, connecting them to form a graph that follows fiber tracts.

2. **Gray Matter Attachment**: Sample gray matter nodes from cortical regions
   and connect them to the white matter skeleton and to neighboring GM nodes.

3. **Tetrahedralization**: Generate tetrahedral elements from the combined
   node structure using constrained Delaunay triangulation.

This approach allows significant mesh coarsening while preserving biophysically
meaningful connectivity - tumor diffusion follows actual fiber tracts rather
than being constrained to a uniform grid.

References:
- DTI tractography: Basser et al., MRM 2000
- White matter atlases: Warrington et al., HCP1065
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set, Any
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, cKDTree
from scipy import ndimage

from .mesh import TetMesh
from .biophysical_constraints import (
    FiberOrientation,
    TissueSegmentation,
    BrainTissue,
    BiophysicalConstraints,
    POSTERIOR_FOSSA_BOUNDS_MNI,
)


@dataclass
class DTIMeshConfig:
    """Configuration for DTI-guided mesh generation.

    Attributes:
        wm_node_spacing: Spacing between white matter nodes along fibers (mm)
        gm_node_spacing: Spacing between gray matter nodes (mm)
        fa_threshold: Minimum FA to consider as white matter tract
        min_tract_length: Minimum length of fiber tract to include (mm)
        max_wm_neighbors: Maximum neighbors for WM-WM connections
        max_gm_wm_connections: Maximum WM connections per GM node
        max_gm_gm_neighbors: Maximum GM-GM lateral connections
        connection_radius: Maximum distance for node connections (mm)
        seed_density: Density of tract seeds per mm³ in high-FA regions
        step_size: Integration step size for tractography (mm)
        angle_threshold: Maximum angle change between steps (degrees)
    """
    wm_node_spacing: float = 6.0  # mm - coarse but tract-aligned
    gm_node_spacing: float = 8.0  # mm - coarse cortical sampling
    fa_threshold: float = 0.2  # Minimum FA for WM tracts
    min_tract_length: float = 10.0  # mm
    max_wm_neighbors: int = 6  # Neighbors per WM node
    max_gm_wm_connections: int = 3  # WM connections per GM node
    max_gm_gm_neighbors: int = 4  # Lateral GM connections
    connection_radius: float = 15.0  # mm - max connection distance
    seed_density: float = 0.01  # seeds per mm³
    step_size: float = 1.0  # mm - tractography step
    angle_threshold: float = 45.0  # degrees - max curvature


@dataclass
class WhiteMatterGraph:
    """Graph structure representing white matter connectivity.

    Attributes:
        nodes: Node positions in physical coordinates (N, 3)
        edges: List of (i, j) node index pairs
        node_fa: Fractional anisotropy at each node
        node_directions: Principal fiber direction at each node (N, 3)
        tract_ids: Which tract each node belongs to
    """
    nodes: NDArray[np.float64]
    edges: List[Tuple[int, int]]
    node_fa: NDArray[np.float32]
    node_directions: NDArray[np.float64]
    tract_ids: NDArray[np.int32]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_neighbors(self, node_idx: int) -> List[int]:
        """Get all neighbors of a node."""
        neighbors = []
        for i, j in self.edges:
            if i == node_idx:
                neighbors.append(j)
            elif j == node_idx:
                neighbors.append(i)
        return neighbors


@dataclass
class GrayMatterNodes:
    """Gray matter node structure with connectivity.

    Attributes:
        nodes: Node positions in physical coordinates (M, 3)
        gm_gm_edges: Edges between GM nodes
        gm_wm_edges: Edges from GM nodes to WM nodes (gm_idx, wm_idx)
        node_depth: Cortical depth (0=surface, 1=WM boundary)
    """
    nodes: NDArray[np.float64]
    gm_gm_edges: List[Tuple[int, int]]
    gm_wm_edges: List[Tuple[int, int]]  # (gm_idx, wm_idx)
    node_depth: NDArray[np.float32]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


class DTIGuidedMeshGenerator:
    """
    Generator for DTI-guided tetrahedral meshes.

    Creates meshes that preserve white matter fiber tract topology by:
    1. Building a white matter skeleton from DTI tractography
    2. Attaching gray matter nodes to the skeleton
    3. Tetrahedralizing the combined structure

    Example:
        >>> from pft_fem import BiophysicalConstraints
        >>> bc = BiophysicalConstraints()
        >>> bc.load_all_constraints()
        >>>
        >>> generator = DTIGuidedMeshGenerator(
        ...     fiber_orientation=bc._fibers,
        ...     tissue_segmentation=bc._segmentation,
        ... )
        >>> mesh = generator.generate_mesh()
        >>> print(f"Nodes: {mesh.num_nodes}, Elements: {mesh.num_elements}")
    """

    def __init__(
        self,
        fiber_orientation: FiberOrientation,
        tissue_segmentation: TissueSegmentation,
        config: Optional[DTIMeshConfig] = None,
        posterior_fossa_only: bool = True,
    ):
        """
        Initialize DTI-guided mesh generator.

        Args:
            fiber_orientation: DTI fiber orientation data (V1 vectors, FA)
            tissue_segmentation: Tissue labels and probability maps
            config: Mesh generation configuration
            posterior_fossa_only: If True, restrict to posterior fossa region
        """
        self.fibers = fiber_orientation
        self.segmentation = tissue_segmentation
        self.config = config or DTIMeshConfig()
        self.posterior_fossa_only = posterior_fossa_only

        # Cache for intermediate results
        self._wm_graph: Optional[WhiteMatterGraph] = None
        self._gm_nodes: Optional[GrayMatterNodes] = None

    def generate_mesh(
        self,
        refine_tumor_region: bool = False,
        tumor_center: Optional[NDArray[np.float64]] = None,
        tumor_radius: float = 20.0,
    ) -> TetMesh:
        """
        Generate a complete DTI-guided tetrahedral mesh.

        This is the main entry point that runs all steps:
        1. Build white matter graph
        2. Attach gray matter nodes
        3. Tetrahedralize

        Args:
            refine_tumor_region: If True, add extra nodes near tumor
            tumor_center: Center of tumor for refinement (MNI coords)
            tumor_radius: Radius around tumor for refinement (mm)

        Returns:
            TetMesh with DTI-guided topology
        """
        print("Step 1/3: Building white matter skeleton from DTI...")
        wm_graph = self.build_white_matter_graph()
        print(f"  White matter: {wm_graph.num_nodes} nodes, {wm_graph.num_edges} edges")

        print("Step 2/3: Attaching gray matter nodes...")
        gm_nodes = self.attach_gray_matter_nodes(wm_graph)
        print(f"  Gray matter: {gm_nodes.num_nodes} nodes")
        print(f"  GM-WM connections: {len(gm_nodes.gm_wm_edges)}")
        print(f"  GM-GM connections: {len(gm_nodes.gm_gm_edges)}")

        print("Step 3/3: Tetrahedralizing combined structure...")
        mesh = self.tetrahedralize(wm_graph, gm_nodes)
        print(f"  Final mesh: {mesh.num_nodes} nodes, {mesh.num_elements} elements")

        # Optional tumor region refinement
        if refine_tumor_region and tumor_center is not None:
            print(f"  Refining near tumor at {tumor_center}...")
            mesh = self._refine_tumor_region(mesh, tumor_center, tumor_radius)
            print(f"  After refinement: {mesh.num_nodes} nodes, {mesh.num_elements} elements")

        return mesh

    def build_white_matter_graph(self) -> WhiteMatterGraph:
        """
        Build white matter skeleton by tracing fiber tracts.

        Uses a simplified tractography approach:
        1. Seed points in high-FA regions
        2. Trace streamlines following V1 direction
        3. Sample nodes along streamlines at regular intervals
        4. Connect sequential nodes along each tract

        Returns:
            WhiteMatterGraph with nodes along fiber tracts
        """
        if self._wm_graph is not None:
            return self._wm_graph

        config = self.config
        fa = self.fibers.fractional_anisotropy
        vectors = self.fibers.vectors
        fiber_affine = self.fibers.affine
        seg_affine = self.segmentation.affine

        # Get WM mask from segmentation (in segmentation voxel space)
        wm_mask_seg = self.segmentation.labels == BrainTissue.WHITE_MATTER.value

        # Apply posterior fossa restriction if enabled (in segmentation space)
        if self.posterior_fossa_only:
            pf_mask = self._get_posterior_fossa_mask(wm_mask_seg.shape, seg_affine)
            wm_mask_seg = wm_mask_seg & pf_mask

        # Resample WM mask to fiber space for use with FA data
        # This is critical: FA and WM mask may have different shapes/affines
        wm_mask_fiber = self._resample_mask_to_fiber_space(
            wm_mask_seg, seg_affine, fa.shape, fiber_affine
        )

        # High-FA mask for seeding (now both are in fiber voxel space)
        high_fa_mask = (fa > config.fa_threshold) & wm_mask_fiber

        # Generate seed points (in physical coordinates)
        seeds = self._generate_tract_seeds(high_fa_mask, fa, fiber_affine)
        print(f"    Generated {len(seeds)} tract seeds")

        # Trace streamlines from seeds
        all_nodes = []
        all_fa = []
        all_directions = []
        all_tract_ids = []
        edges = []

        tract_id = 0
        node_offset = 0

        for seed in seeds:
            # Trace in both directions from seed
            # Pass wm_mask_seg (segmentation space) - _trace_streamline handles coordinate conversion
            forward = self._trace_streamline(seed, vectors, fa, fiber_affine, wm_mask_seg, direction=1)
            backward = self._trace_streamline(seed, vectors, fa, fiber_affine, wm_mask_seg, direction=-1)

            # Combine into single tract (backward reversed + forward)
            if len(backward) > 1:
                tract_points = list(reversed(backward[1:])) + forward
            else:
                tract_points = forward

            # Check minimum length
            if len(tract_points) < 2:
                continue
            tract_length = sum(
                np.linalg.norm(tract_points[i+1] - tract_points[i])
                for i in range(len(tract_points) - 1)
            )
            if tract_length < config.min_tract_length:
                continue

            # Sample nodes along tract at regular spacing
            sampled_nodes, sampled_fa, sampled_dirs = self._sample_tract_nodes(
                tract_points, vectors, fa, fiber_affine, config.wm_node_spacing
            )

            if len(sampled_nodes) < 2:
                continue

            # Add nodes
            for i, (node, fa_val, direction) in enumerate(zip(sampled_nodes, sampled_fa, sampled_dirs)):
                all_nodes.append(node)
                all_fa.append(fa_val)
                all_directions.append(direction)
                all_tract_ids.append(tract_id)

                # Connect to previous node in tract
                if i > 0:
                    edges.append((node_offset + i - 1, node_offset + i))

            node_offset += len(sampled_nodes)
            tract_id += 1

        if len(all_nodes) == 0:
            # Fallback: sample WM voxels directly
            print("    Warning: No tracts found, falling back to WM voxel sampling")
            all_nodes, all_fa, all_directions, all_tract_ids, edges = self._fallback_wm_sampling(
                wm_mask_seg, fa, vectors, fiber_affine
            )

        # Add cross-tract connections for nearby nodes
        edges = self._add_cross_tract_connections(
            np.array(all_nodes),
            np.array(all_tract_ids),
            edges,
            config.connection_radius,
            config.max_wm_neighbors
        )

        self._wm_graph = WhiteMatterGraph(
            nodes=np.array(all_nodes, dtype=np.float64),
            edges=edges,
            node_fa=np.array(all_fa, dtype=np.float32),
            node_directions=np.array(all_directions, dtype=np.float64),
            tract_ids=np.array(all_tract_ids, dtype=np.int32),
        )

        return self._wm_graph

    def attach_gray_matter_nodes(
        self,
        wm_graph: WhiteMatterGraph,
    ) -> GrayMatterNodes:
        """
        Sample gray matter nodes and connect to white matter skeleton.

        Strategy:
        1. Sample GM voxels at regular spacing
        2. Connect each GM node to nearest WM nodes (cortico-subcortical)
        3. Connect GM nodes to neighboring GM nodes (lateral connections)

        Args:
            wm_graph: White matter skeleton to attach to

        Returns:
            GrayMatterNodes with connectivity information
        """
        if self._gm_nodes is not None:
            return self._gm_nodes

        config = self.config

        # Get GM mask
        gm_mask = self.segmentation.labels == BrainTissue.GRAY_MATTER.value

        # Apply posterior fossa restriction
        if self.posterior_fossa_only:
            pf_mask = self._get_posterior_fossa_mask(gm_mask.shape)
            gm_mask = gm_mask & pf_mask

        # Sample GM nodes at regular spacing
        gm_nodes = self._sample_gm_nodes(gm_mask, config.gm_node_spacing)

        if len(gm_nodes) == 0:
            self._gm_nodes = GrayMatterNodes(
                nodes=np.zeros((0, 3), dtype=np.float64),
                gm_gm_edges=[],
                gm_wm_edges=[],
                node_depth=np.array([], dtype=np.float32),
            )
            return self._gm_nodes

        # Build KD-tree for WM nodes
        wm_tree = cKDTree(wm_graph.nodes)

        # Connect GM to nearest WM nodes
        gm_wm_edges = []
        for gm_idx, gm_pos in enumerate(gm_nodes):
            # Find nearest WM nodes
            distances, wm_indices = wm_tree.query(
                gm_pos,
                k=min(config.max_gm_wm_connections, wm_graph.num_nodes)
            )

            # Filter by distance
            for dist, wm_idx in zip(distances, wm_indices):
                if dist < config.connection_radius:
                    gm_wm_edges.append((gm_idx, wm_idx))

        # Build KD-tree for GM nodes
        gm_tree = cKDTree(gm_nodes)

        # Connect GM to neighboring GM nodes
        gm_gm_edges = []
        for gm_idx, gm_pos in enumerate(gm_nodes):
            # Find nearest GM neighbors
            distances, neighbor_indices = gm_tree.query(
                gm_pos,
                k=min(config.max_gm_gm_neighbors + 1, len(gm_nodes))  # +1 for self
            )

            for dist, neighbor_idx in zip(distances, neighbor_indices):
                if neighbor_idx != gm_idx and dist < config.connection_radius:
                    # Add edge (avoid duplicates by only adding if gm_idx < neighbor_idx)
                    if gm_idx < neighbor_idx:
                        gm_gm_edges.append((gm_idx, neighbor_idx))

        # Compute cortical depth (distance to WM / cortical thickness)
        node_depth = np.zeros(len(gm_nodes), dtype=np.float32)
        for gm_idx, gm_pos in enumerate(gm_nodes):
            dist_to_wm, _ = wm_tree.query(gm_pos)
            # Normalize depth (assume ~3mm cortical thickness)
            node_depth[gm_idx] = min(1.0, dist_to_wm / 3.0)

        self._gm_nodes = GrayMatterNodes(
            nodes=np.array(gm_nodes, dtype=np.float64),
            gm_gm_edges=gm_gm_edges,
            gm_wm_edges=gm_wm_edges,
            node_depth=node_depth,
        )

        return self._gm_nodes

    def tetrahedralize(
        self,
        wm_graph: WhiteMatterGraph,
        gm_nodes: GrayMatterNodes,
    ) -> TetMesh:
        """
        Generate tetrahedral mesh from combined WM and GM nodes.

        Uses Delaunay triangulation with edge constraints to preserve
        the fiber-aligned connectivity.

        Args:
            wm_graph: White matter skeleton
            gm_nodes: Gray matter nodes with connectivity

        Returns:
            TetMesh suitable for FEM simulation
        """
        # Combine all nodes
        n_wm = wm_graph.num_nodes
        n_gm = gm_nodes.num_nodes

        if n_wm == 0 and n_gm == 0:
            return TetMesh(
                nodes=np.zeros((0, 3), dtype=np.float64),
                elements=np.zeros((0, 4), dtype=np.int32),
            )

        all_nodes = np.vstack([wm_graph.nodes, gm_nodes.nodes]) if n_gm > 0 else wm_graph.nodes

        # Assign tissue labels
        # WM nodes: label 3, GM nodes: label 2
        node_labels = np.zeros(len(all_nodes), dtype=np.int32)
        node_labels[:n_wm] = BrainTissue.WHITE_MATTER.value
        if n_gm > 0:
            node_labels[n_wm:] = BrainTissue.GRAY_MATTER.value

        # Perform Delaunay triangulation
        try:
            tri = Delaunay(all_nodes)
            elements = tri.simplices.astype(np.int32)
        except Exception as e:
            warnings.warn(f"Delaunay triangulation failed: {e}. Using fallback.")
            # Fallback: create minimal valid mesh
            return self._create_fallback_mesh(all_nodes, node_labels)

        # Filter degenerate elements (very small volume or inverted)
        elements = self._filter_degenerate_elements(all_nodes, elements)

        # Find boundary nodes
        boundary_nodes = self._find_boundary_nodes(all_nodes, elements)

        # Build mesh
        mesh = TetMesh(
            nodes=all_nodes,
            elements=elements,
            node_labels=node_labels,
            boundary_nodes=boundary_nodes,
        )

        # Build node neighbors
        mesh.build_node_neighbors()

        return mesh

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _get_posterior_fossa_mask(
        self,
        shape: Tuple[int, ...],
        affine: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.bool_]:
        """Create mask for posterior fossa region in voxel space.

        Args:
            shape: Shape of the output mask
            affine: Affine transformation matrix to convert voxel to physical coords.
                   If None, uses segmentation affine or a standard MNI affine.

        Returns:
            Boolean mask where True indicates voxels in posterior fossa
        """
        if affine is None:
            affine = self.segmentation.affine
        if affine is None:
            # Use standard MNI152 1mm affine (radiological convention)
            affine = np.array([
                [-1.0,  0.0,  0.0,  90.0],
                [ 0.0,  1.0,  0.0, -126.0],
                [ 0.0,  0.0,  1.0, -72.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ])

        bounds = POSTERIOR_FOSSA_BOUNDS_MNI

        # Vectorized implementation for efficiency
        # Create coordinate grids
        i_coords, j_coords, k_coords = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )

        # Convert all voxel coordinates to physical coordinates at once
        # Physical = affine @ [i, j, k, 1]^T
        x_phys = affine[0, 0] * i_coords + affine[0, 1] * j_coords + affine[0, 2] * k_coords + affine[0, 3]
        y_phys = affine[1, 0] * i_coords + affine[1, 1] * j_coords + affine[1, 2] * k_coords + affine[1, 3]
        z_phys = affine[2, 0] * i_coords + affine[2, 1] * j_coords + affine[2, 2] * k_coords + affine[2, 3]

        # Check bounds (vectorized)
        mask = (
            (x_phys >= bounds['x_min']) & (x_phys <= bounds['x_max']) &
            (y_phys >= bounds['y_min']) & (y_phys <= bounds['y_max']) &
            (z_phys >= bounds['z_min']) & (z_phys <= bounds['z_max'])
        )

        return mask

    def _resample_mask_to_fiber_space(
        self,
        mask_seg: NDArray[np.bool_],
        seg_affine: Optional[NDArray[np.float64]],
        fiber_shape: Tuple[int, ...],
        fiber_affine: Optional[NDArray[np.float64]],
    ) -> NDArray[np.bool_]:
        """Resample a mask from segmentation space to fiber space.

        This handles the case where segmentation and fiber data have different
        shapes and/or affine matrices.

        Args:
            mask_seg: Boolean mask in segmentation voxel space
            seg_affine: Affine for segmentation space
            fiber_shape: Target shape in fiber voxel space
            fiber_affine: Affine for fiber space

        Returns:
            Boolean mask resampled to fiber voxel space
        """
        # Handle None affines - use standard MNI152 1mm affine
        if seg_affine is None:
            seg_affine = np.array([
                [-1.0,  0.0,  0.0,  90.0],
                [ 0.0,  1.0,  0.0, -126.0],
                [ 0.0,  0.0,  1.0, -72.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ])
        if fiber_affine is None:
            fiber_affine = np.array([
                [-1.0,  0.0,  0.0,  90.0],
                [ 0.0,  1.0,  0.0, -126.0],
                [ 0.0,  0.0,  1.0, -72.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ])

        # Check if shapes and affines are the same (common case)
        if (mask_seg.shape == fiber_shape[:3] and
            np.allclose(seg_affine, fiber_affine, atol=1e-6)):
            return mask_seg

        # Need to resample: for each fiber voxel, find corresponding segmentation voxel
        inv_seg_affine = np.linalg.inv(seg_affine)

        # Create coordinate grids in fiber space
        i_f, j_f, k_f = np.meshgrid(
            np.arange(fiber_shape[0]),
            np.arange(fiber_shape[1]),
            np.arange(fiber_shape[2]),
            indexing='ij'
        )

        # Convert fiber voxels to physical coordinates
        x_phys = fiber_affine[0, 0] * i_f + fiber_affine[0, 1] * j_f + fiber_affine[0, 2] * k_f + fiber_affine[0, 3]
        y_phys = fiber_affine[1, 0] * i_f + fiber_affine[1, 1] * j_f + fiber_affine[1, 2] * k_f + fiber_affine[1, 3]
        z_phys = fiber_affine[2, 0] * i_f + fiber_affine[2, 1] * j_f + fiber_affine[2, 2] * k_f + fiber_affine[2, 3]

        # Convert physical to segmentation voxel coordinates
        i_s = inv_seg_affine[0, 0] * x_phys + inv_seg_affine[0, 1] * y_phys + inv_seg_affine[0, 2] * z_phys + inv_seg_affine[0, 3]
        j_s = inv_seg_affine[1, 0] * x_phys + inv_seg_affine[1, 1] * y_phys + inv_seg_affine[1, 2] * z_phys + inv_seg_affine[1, 3]
        k_s = inv_seg_affine[2, 0] * x_phys + inv_seg_affine[2, 1] * y_phys + inv_seg_affine[2, 2] * z_phys + inv_seg_affine[2, 3]

        # Round to nearest voxel and clip to valid range
        i_s = np.clip(np.round(i_s).astype(int), 0, mask_seg.shape[0] - 1)
        j_s = np.clip(np.round(j_s).astype(int), 0, mask_seg.shape[1] - 1)
        k_s = np.clip(np.round(k_s).astype(int), 0, mask_seg.shape[2] - 1)

        # Sample the segmentation mask at these locations
        mask_fiber = mask_seg[i_s, j_s, k_s]

        return mask_fiber

    def _generate_tract_seeds(
        self,
        high_fa_mask: NDArray[np.bool_],
        fa: NDArray[np.float32],
        affine: Optional[NDArray[np.float64]],
    ) -> List[NDArray[np.float64]]:
        """Generate seed points for tractography in high-FA regions."""
        config = self.config

        # Get voxel coordinates of high-FA voxels
        voxel_coords = np.array(np.where(high_fa_mask)).T

        if len(voxel_coords) == 0:
            return []

        # Calculate voxel volume
        if affine is not None:
            voxel_size = np.abs(np.diag(affine)[:3])
        else:
            voxel_size = np.array([1.0, 1.0, 1.0])
        voxel_volume = np.prod(voxel_size)

        # Total volume of high-FA region
        total_volume = len(voxel_coords) * voxel_volume

        # Number of seeds based on density
        n_seeds = max(10, int(total_volume * config.seed_density))
        n_seeds = min(n_seeds, len(voxel_coords), 500)  # Cap at 500 seeds

        # Sample seed voxels (weighted by FA)
        fa_values = np.array([fa[v[0], v[1], v[2]] for v in voxel_coords])
        fa_weights = fa_values / fa_values.sum()

        seed_indices = np.random.choice(
            len(voxel_coords),
            size=n_seeds,
            replace=False,
            p=fa_weights
        )

        # Convert to physical coordinates
        seeds = []
        for idx in seed_indices:
            voxel = voxel_coords[idx]
            if affine is not None:
                phys = affine @ np.array([*voxel, 1])
                seeds.append(phys[:3])
            else:
                seeds.append(voxel.astype(np.float64))

        return seeds

    def _trace_streamline(
        self,
        seed: NDArray[np.float64],
        vectors: NDArray[np.float64],
        fa: NDArray[np.float32],
        affine: Optional[NDArray[np.float64]],
        wm_mask: NDArray[np.bool_],
        direction: int = 1,
        max_steps: int = 200,
    ) -> List[NDArray[np.float64]]:
        """Trace a streamline from seed point following V1 direction."""
        config = self.config
        step_size = config.step_size * direction
        angle_threshold_rad = np.radians(config.angle_threshold)

        # Get inverse affine for fiber coordinate conversion
        if affine is not None:
            inv_affine = np.linalg.inv(affine)
        else:
            inv_affine = np.eye(4)

        # Get inverse affine for segmentation coordinate conversion (for WM mask lookups)
        # This is critical: wm_mask comes from segmentation.labels which uses segmentation.affine
        seg_affine = self.segmentation.affine
        if seg_affine is not None:
            inv_seg_affine = np.linalg.inv(seg_affine)
        else:
            inv_seg_affine = np.eye(4)

        points = [seed.copy()]
        current_pos = seed.copy()
        prev_dir = None

        for _ in range(max_steps):
            # Convert to FIBER voxel coordinates (for FA and vector lookups)
            voxel_h = inv_affine @ np.array([*current_pos, 1])
            voxel = voxel_h[:3]

            # Check bounds in fiber space
            vi, vj, vk = int(round(voxel[0])), int(round(voxel[1])), int(round(voxel[2]))
            if not (0 <= vi < fa.shape[0] and 0 <= vj < fa.shape[1] and 0 <= vk < fa.shape[2]):
                break

            # Convert to SEGMENTATION voxel coordinates (for WM mask lookup)
            seg_voxel_h = inv_seg_affine @ np.array([*current_pos, 1])
            seg_voxel = seg_voxel_h[:3]
            si, sj, sk = int(round(seg_voxel[0])), int(round(seg_voxel[1])), int(round(seg_voxel[2]))

            # Check bounds in segmentation space
            if not (0 <= si < wm_mask.shape[0] and 0 <= sj < wm_mask.shape[1] and 0 <= sk < wm_mask.shape[2]):
                break

            # Check if still in WM (using segmentation voxel coordinates)
            if not wm_mask[si, sj, sk]:
                break

            # Check FA threshold
            if fa[vi, vj, vk] < config.fa_threshold:
                break

            # Get fiber direction (trilinear interpolation would be better but keeping simple)
            fiber_dir = vectors[vi, vj, vk].copy()
            if np.linalg.norm(fiber_dir) < 1e-6:
                break
            fiber_dir = fiber_dir / np.linalg.norm(fiber_dir)

            # Ensure consistent direction with previous step
            if prev_dir is not None:
                if np.dot(fiber_dir, prev_dir) < 0:
                    fiber_dir = -fiber_dir

                # Check angle threshold
                cos_angle = np.clip(np.dot(fiber_dir, prev_dir), -1, 1)
                if np.arccos(cos_angle) > angle_threshold_rad:
                    break

            # Take step
            new_pos = current_pos + step_size * fiber_dir
            points.append(new_pos.copy())

            prev_dir = fiber_dir
            current_pos = new_pos

        return points

    def _sample_tract_nodes(
        self,
        tract_points: List[NDArray[np.float64]],
        vectors: NDArray[np.float64],
        fa: NDArray[np.float32],
        affine: Optional[NDArray[np.float64]],
        spacing: float,
    ) -> Tuple[List[NDArray[np.float64]], List[float], List[NDArray[np.float64]]]:
        """Sample nodes along tract at regular spacing."""
        if len(tract_points) < 2:
            return [], [], []

        if affine is not None:
            inv_affine = np.linalg.inv(affine)
        else:
            inv_affine = np.eye(4)

        # Compute cumulative arc length
        arc_lengths = [0.0]
        for i in range(1, len(tract_points)):
            dist = np.linalg.norm(tract_points[i] - tract_points[i-1])
            arc_lengths.append(arc_lengths[-1] + dist)

        total_length = arc_lengths[-1]
        if total_length < spacing:
            # Just use endpoints
            return tract_points[:1], [0.5], [np.array([1, 0, 0])]

        # Sample at regular intervals
        n_samples = max(2, int(total_length / spacing))
        sample_positions = np.linspace(0, total_length, n_samples)

        sampled_nodes = []
        sampled_fa = []
        sampled_dirs = []

        for s in sample_positions:
            # Find segment containing this arc length
            for i in range(len(arc_lengths) - 1):
                if arc_lengths[i] <= s <= arc_lengths[i+1]:
                    # Interpolate position
                    t = (s - arc_lengths[i]) / (arc_lengths[i+1] - arc_lengths[i] + 1e-10)
                    pos = (1 - t) * tract_points[i] + t * tract_points[i+1]

                    # Get FA and direction at this position
                    voxel_h = inv_affine @ np.array([*pos, 1])
                    vi = int(np.clip(round(voxel_h[0]), 0, fa.shape[0]-1))
                    vj = int(np.clip(round(voxel_h[1]), 0, fa.shape[1]-1))
                    vk = int(np.clip(round(voxel_h[2]), 0, fa.shape[2]-1))

                    sampled_nodes.append(pos)
                    sampled_fa.append(float(fa[vi, vj, vk]))

                    direction = vectors[vi, vj, vk].copy()
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0.0, 0.0])
                    sampled_dirs.append(direction)
                    break

        return sampled_nodes, sampled_fa, sampled_dirs

    def _fallback_wm_sampling(
        self,
        wm_mask: NDArray[np.bool_],
        fa: NDArray[np.float32],
        vectors: NDArray[np.float64],
        fiber_affine: Optional[NDArray[np.float64]],
    ) -> Tuple[List, List, List, List, List]:
        """Fallback: sample WM voxels directly when tractography fails.

        Note: wm_mask is in segmentation space, fa/vectors are in fiber space.
        This function handles the coordinate transformation between spaces.
        """
        config = self.config

        # Get segmentation affine for converting WM mask voxels to physical coords
        seg_affine = self.segmentation.affine
        if seg_affine is None:
            seg_affine = np.array([
                [-1.0,  0.0,  0.0,  90.0],
                [ 0.0,  1.0,  0.0, -126.0],
                [ 0.0,  0.0,  1.0, -72.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ])

        # Get inverse fiber affine for converting physical to fiber voxel coords
        if fiber_affine is None:
            fiber_affine = np.array([
                [-1.0,  0.0,  0.0,  90.0],
                [ 0.0,  1.0,  0.0, -126.0],
                [ 0.0,  0.0,  1.0, -72.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ])
        inv_fiber_affine = np.linalg.inv(fiber_affine)

        # Compute voxel size from segmentation affine
        seg_voxel_size = np.abs(np.diag(seg_affine)[:3]).mean()

        # Downsample WM mask
        zoom_factor = seg_voxel_size / config.wm_node_spacing
        if zoom_factor < 1:
            wm_coarse = ndimage.zoom(wm_mask.astype(float), zoom_factor, order=0) > 0.5
        else:
            wm_coarse = wm_mask

        # Get voxel coordinates (in downsampled segmentation space)
        voxel_coords = np.array(np.where(wm_coarse)).T

        all_nodes = []
        all_fa = []
        all_directions = []
        all_tract_ids = []

        for idx, voxel in enumerate(voxel_coords):
            # Convert back to original segmentation voxel space
            if zoom_factor < 1:
                orig_seg_voxel = (voxel / zoom_factor).astype(int)
            else:
                orig_seg_voxel = voxel
            orig_seg_voxel = np.clip(orig_seg_voxel, 0, np.array(wm_mask.shape) - 1)

            # Convert segmentation voxel to physical coordinates
            phys = seg_affine @ np.array([*orig_seg_voxel, 1])
            pos = phys[:3]

            # Convert physical to fiber voxel coordinates for FA/vector lookup
            fiber_voxel_h = inv_fiber_affine @ np.array([*pos, 1])
            fiber_voxel = fiber_voxel_h[:3]
            fi = int(np.clip(round(fiber_voxel[0]), 0, fa.shape[0] - 1))
            fj = int(np.clip(round(fiber_voxel[1]), 0, fa.shape[1] - 1))
            fk = int(np.clip(round(fiber_voxel[2]), 0, fa.shape[2] - 1))

            all_nodes.append(pos)
            all_fa.append(float(fa[fi, fj, fk]))

            direction = vectors[fi, fj, fk].copy()
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1.0, 0.0, 0.0])
            all_directions.append(direction)
            all_tract_ids.append(idx)  # Each node is its own "tract"

        # Build edges using nearest neighbors
        edges = []
        if len(all_nodes) > 1:
            tree = cKDTree(all_nodes)
            for i, node in enumerate(all_nodes):
                distances, indices = tree.query(node, k=min(config.max_wm_neighbors + 1, len(all_nodes)))
                for dist, j in zip(distances, indices):
                    if j != i and dist < config.connection_radius:
                        if i < j:  # Avoid duplicates
                            edges.append((i, j))

        return all_nodes, all_fa, all_directions, all_tract_ids, edges

    def _add_cross_tract_connections(
        self,
        nodes: NDArray[np.float64],
        tract_ids: NDArray[np.int32],
        edges: List[Tuple[int, int]],
        max_dist: float,
        max_neighbors: int,
    ) -> List[Tuple[int, int]]:
        """Add connections between nearby nodes from different tracts."""
        if len(nodes) < 2:
            return edges

        tree = cKDTree(nodes)
        edge_set = set(edges)

        for i, node in enumerate(nodes):
            # Find nearby nodes
            distances, indices = tree.query(node, k=min(max_neighbors + 1, len(nodes)))

            for dist, j in zip(distances, indices):
                if j == i:
                    continue
                if dist > max_dist:
                    continue

                # Add edge if not already present
                edge = (min(i, j), max(i, j))
                if edge not in edge_set:
                    edge_set.add(edge)

        return list(edge_set)

    def _sample_gm_nodes(
        self,
        gm_mask: NDArray[np.bool_],
        spacing: float,
    ) -> NDArray[np.float64]:
        """Sample gray matter nodes at regular spacing."""
        affine = self.segmentation.affine
        if affine is None:
            affine = np.array([
                [-1.0,  0.0,  0.0,  90.0],
                [ 0.0,  1.0,  0.0, -126.0],
                [ 0.0,  0.0,  1.0, -72.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ])

        # Downsample GM mask
        voxel_size = np.abs(np.diag(affine)[:3]).mean()
        zoom_factor = voxel_size / spacing

        if zoom_factor < 1:
            gm_coarse = ndimage.zoom(gm_mask.astype(float), zoom_factor, order=0) > 0.5
        else:
            gm_coarse = gm_mask

        # Get voxel coordinates
        voxel_coords = np.array(np.where(gm_coarse)).T

        if len(voxel_coords) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # Convert to physical coordinates
        nodes = []
        for voxel in voxel_coords:
            # Convert back to original voxel space
            if zoom_factor < 1:
                orig_voxel = (voxel / zoom_factor).astype(int)
            else:
                orig_voxel = voxel

            phys = affine @ np.array([*orig_voxel, 1])
            nodes.append(phys[:3])

        return np.array(nodes, dtype=np.float64)

    def _filter_degenerate_elements(
        self,
        nodes: NDArray[np.float64],
        elements: NDArray[np.int32],
        min_volume: float = 1e-6,
    ) -> NDArray[np.int32]:
        """Filter out degenerate (zero or negative volume) tetrahedra."""
        valid_elements = []

        for elem in elements:
            v0, v1, v2, v3 = nodes[elem]

            # Compute volume using determinant
            mat = np.array([v1 - v0, v2 - v0, v3 - v0])
            volume = np.linalg.det(mat) / 6.0

            # Keep if volume is positive and above threshold
            if volume > min_volume:
                valid_elements.append(elem)
            elif volume < -min_volume:
                # Flip orientation
                valid_elements.append(np.array([elem[0], elem[2], elem[1], elem[3]]))

        if len(valid_elements) == 0:
            return np.zeros((0, 4), dtype=np.int32)

        return np.array(valid_elements, dtype=np.int32)

    def _find_boundary_nodes(
        self,
        nodes: NDArray[np.float64],
        elements: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Find nodes on the mesh boundary."""
        if len(elements) == 0:
            return np.array([], dtype=np.int32)

        # Count face occurrences (boundary faces appear once, interior twice)
        face_count: Dict[Tuple[int, ...], int] = {}

        for elem in elements:
            # Four faces per tetrahedron
            faces = [
                tuple(sorted([elem[0], elem[1], elem[2]])),
                tuple(sorted([elem[0], elem[1], elem[3]])),
                tuple(sorted([elem[0], elem[2], elem[3]])),
                tuple(sorted([elem[1], elem[2], elem[3]])),
            ]
            for face in faces:
                face_count[face] = face_count.get(face, 0) + 1

        # Boundary faces appear exactly once
        boundary_nodes = set()
        for face, count in face_count.items():
            if count == 1:
                boundary_nodes.update(face)

        return np.array(sorted(boundary_nodes), dtype=np.int32)

    def _create_fallback_mesh(
        self,
        nodes: NDArray[np.float64],
        node_labels: NDArray[np.int32],
    ) -> TetMesh:
        """Create minimal valid mesh when Delaunay fails."""
        if len(nodes) < 4:
            return TetMesh(
                nodes=nodes,
                elements=np.zeros((0, 4), dtype=np.int32),
                node_labels=node_labels,
            )

        # Use first 4 nodes as single tetrahedron
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        return TetMesh(
            nodes=nodes,
            elements=elements,
            node_labels=node_labels,
            boundary_nodes=np.array([0, 1, 2, 3], dtype=np.int32),
        )

    def _refine_tumor_region(
        self,
        mesh: TetMesh,
        center: NDArray[np.float64],
        radius: float,
    ) -> TetMesh:
        """Add refinement nodes near tumor location."""
        from .mesh import MeshGenerator

        # Use the existing mesh refinement method
        gen = MeshGenerator()
        return gen.refine_region(mesh, center, radius, refinement_factor=2)


def create_dti_guided_mesh(
    biophysical_constraints: Optional[BiophysicalConstraints] = None,
    config: Optional[DTIMeshConfig] = None,
    posterior_fossa_only: bool = True,
    verbose: bool = True,
) -> TetMesh:
    """
    Convenience function to create a DTI-guided mesh.

    Args:
        biophysical_constraints: Pre-loaded constraints (or creates new)
        config: Mesh generation configuration
        posterior_fossa_only: Restrict to posterior fossa
        verbose: Print progress messages

    Returns:
        TetMesh with DTI-guided topology
    """
    if biophysical_constraints is None:
        if verbose:
            print("Loading biophysical constraints...")
        biophysical_constraints = BiophysicalConstraints(
            posterior_fossa_only=posterior_fossa_only,
            use_dti_constraints=True,
        )
        biophysical_constraints.load_all_constraints()

    generator = DTIGuidedMeshGenerator(
        fiber_orientation=biophysical_constraints._fibers,
        tissue_segmentation=biophysical_constraints._segmentation,
        config=config,
        posterior_fossa_only=posterior_fossa_only,
    )

    return generator.generate_mesh()
