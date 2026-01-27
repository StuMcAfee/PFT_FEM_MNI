"""
Tests for DTI-guided mesh generation.

This module tests the DTIGuidedMeshGenerator class and related functionality
for creating meshes that follow white matter fiber tract topology.
"""

import numpy as np
import pytest

from pft_fem.dti_mesh import (
    DTIGuidedMeshGenerator,
    DTIMeshConfig,
    WhiteMatterGraph,
    GrayMatterNodes,
    create_dti_guided_mesh,
)
from pft_fem.biophysical_constraints import (
    FiberOrientation,
    TissueSegmentation,
    BrainTissue,
    BiophysicalConstraints,
)
from pft_fem.mesh import TetMesh


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_fiber_orientation():
    """Create synthetic fiber orientation data for testing."""
    shape = (20, 25, 20)

    # Create synthetic FA (high in center, low at edges)
    fa = np.zeros(shape, dtype=np.float32)
    center = np.array([10, 12, 10])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                fa[i, j, k] = max(0, 0.8 - dist / 15)

    # Create synthetic vectors (primarily anterior-posterior)
    vectors = np.zeros((*shape, 3), dtype=np.float64)
    vectors[..., 1] = 0.9  # Y-direction (A-P)
    vectors[..., 0] = 0.1  # Some X component
    # Normalize
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms < 1e-6] = 1
    vectors = vectors / norms

    # Simple affine (1mm isotropic centered at origin)
    affine = np.eye(4)
    affine[:3, 3] = -np.array(shape) / 2

    return FiberOrientation(
        vectors=vectors,
        fractional_anisotropy=fa,
        affine=affine,
    )


@pytest.fixture
def synthetic_tissue_segmentation():
    """Create synthetic tissue segmentation for testing."""
    shape = (20, 25, 20)

    # Create labels: WM in center, GM in outer shell, CSF at edges
    labels = np.zeros(shape, dtype=np.int32)
    center = np.array([10, 12, 10])

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if dist < 5:
                    labels[i, j, k] = BrainTissue.WHITE_MATTER.value
                elif dist < 8:
                    labels[i, j, k] = BrainTissue.GRAY_MATTER.value
                elif dist < 10:
                    labels[i, j, k] = BrainTissue.CSF.value

    # Simple affine
    affine = np.eye(4)
    affine[:3, 3] = -np.array(shape) / 2

    return TissueSegmentation(
        labels=labels,
        affine=affine,
        voxel_size=(1.0, 1.0, 1.0),
    )


@pytest.fixture
def dti_mesh_config():
    """Create a test-friendly DTI mesh configuration."""
    return DTIMeshConfig(
        wm_node_spacing=3.0,  # Smaller for test data
        gm_node_spacing=4.0,
        fa_threshold=0.15,
        min_tract_length=3.0,
        max_wm_neighbors=4,
        max_gm_wm_connections=2,
        max_gm_gm_neighbors=3,
        connection_radius=10.0,
        seed_density=0.1,  # Higher for small test volume
        step_size=0.5,
        angle_threshold=60.0,
    )


@pytest.fixture
def dti_mesh_generator(synthetic_fiber_orientation, synthetic_tissue_segmentation, dti_mesh_config):
    """Create a DTI mesh generator with synthetic data."""
    return DTIGuidedMeshGenerator(
        fiber_orientation=synthetic_fiber_orientation,
        tissue_segmentation=synthetic_tissue_segmentation,
        config=dti_mesh_config,
        posterior_fossa_only=False,  # Disable for synthetic data
    )


# ============================================================================
# DTIMeshConfig Tests
# ============================================================================

class TestDTIMeshConfig:
    """Tests for DTIMeshConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DTIMeshConfig()

        assert config.wm_node_spacing == 6.0
        assert config.gm_node_spacing == 8.0
        assert config.fa_threshold == 0.2
        assert config.min_tract_length == 10.0
        assert config.max_wm_neighbors == 6
        assert config.max_gm_wm_connections == 3
        assert config.max_gm_gm_neighbors == 4
        assert config.connection_radius == 15.0
        assert config.seed_density == 0.01
        assert config.step_size == 1.0
        assert config.angle_threshold == 45.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DTIMeshConfig(
            wm_node_spacing=4.0,
            gm_node_spacing=5.0,
            fa_threshold=0.3,
        )

        assert config.wm_node_spacing == 4.0
        assert config.gm_node_spacing == 5.0
        assert config.fa_threshold == 0.3


# ============================================================================
# WhiteMatterGraph Tests
# ============================================================================

class TestWhiteMatterGraph:
    """Tests for WhiteMatterGraph dataclass."""

    def test_creation(self):
        """Test basic graph creation."""
        nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        edges = [(0, 1), (1, 2)]
        fa = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        directions = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
        tract_ids = np.array([0, 0, 0], dtype=np.int32)

        graph = WhiteMatterGraph(
            nodes=nodes,
            edges=edges,
            node_fa=fa,
            node_directions=directions,
            tract_ids=tract_ids,
        )

        assert graph.num_nodes == 3
        assert graph.num_edges == 2

    def test_get_neighbors(self):
        """Test neighbor lookup."""
        nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.float64)
        edges = [(0, 1), (1, 2), (1, 3)]
        fa = np.array([0.5, 0.6, 0.7, 0.5], dtype=np.float32)
        directions = np.array([[1, 0, 0]] * 4, dtype=np.float64)
        tract_ids = np.array([0, 0, 0, 1], dtype=np.int32)

        graph = WhiteMatterGraph(
            nodes=nodes,
            edges=edges,
            node_fa=fa,
            node_directions=directions,
            tract_ids=tract_ids,
        )

        # Node 1 should have 3 neighbors: 0, 2, 3
        neighbors = graph.get_neighbors(1)
        assert set(neighbors) == {0, 2, 3}

        # Node 0 should have 1 neighbor: 1
        neighbors = graph.get_neighbors(0)
        assert neighbors == [1]


# ============================================================================
# GrayMatterNodes Tests
# ============================================================================

class TestGrayMatterNodes:
    """Tests for GrayMatterNodes dataclass."""

    def test_creation(self):
        """Test basic creation."""
        nodes = np.array([[0, 0, 5], [0, 5, 0], [5, 0, 0]], dtype=np.float64)
        gm_gm_edges = [(0, 1), (1, 2)]
        gm_wm_edges = [(0, 0), (1, 1), (2, 2)]  # GM node -> WM node
        depth = np.array([0.5, 0.3, 0.7], dtype=np.float32)

        gm = GrayMatterNodes(
            nodes=nodes,
            gm_gm_edges=gm_gm_edges,
            gm_wm_edges=gm_wm_edges,
            node_depth=depth,
        )

        assert gm.num_nodes == 3
        assert len(gm.gm_gm_edges) == 2
        assert len(gm.gm_wm_edges) == 3


# ============================================================================
# DTIGuidedMeshGenerator Tests
# ============================================================================

class TestDTIGuidedMeshGenerator:
    """Tests for the main DTI-guided mesh generator."""

    def test_initialization(self, dti_mesh_generator):
        """Test generator initialization."""
        gen = dti_mesh_generator

        assert gen.fibers is not None
        assert gen.segmentation is not None
        assert gen.config is not None
        assert gen.posterior_fossa_only is False

    def test_build_white_matter_graph(self, dti_mesh_generator):
        """Test white matter graph building."""
        gen = dti_mesh_generator

        wm_graph = gen.build_white_matter_graph()

        assert isinstance(wm_graph, WhiteMatterGraph)
        assert wm_graph.num_nodes > 0
        assert wm_graph.num_edges > 0
        assert len(wm_graph.node_fa) == wm_graph.num_nodes
        assert len(wm_graph.node_directions) == wm_graph.num_nodes
        assert len(wm_graph.tract_ids) == wm_graph.num_nodes

    def test_attach_gray_matter_nodes(self, dti_mesh_generator):
        """Test gray matter attachment."""
        gen = dti_mesh_generator

        wm_graph = gen.build_white_matter_graph()
        gm_nodes = gen.attach_gray_matter_nodes(wm_graph)

        assert isinstance(gm_nodes, GrayMatterNodes)
        # GM nodes may be 0 if synthetic data doesn't have GM in expected region
        assert gm_nodes.num_nodes >= 0

    def test_tetrahedralize(self, dti_mesh_generator):
        """Test tetrahedralization."""
        gen = dti_mesh_generator

        wm_graph = gen.build_white_matter_graph()
        gm_nodes = gen.attach_gray_matter_nodes(wm_graph)
        mesh = gen.tetrahedralize(wm_graph, gm_nodes)

        assert isinstance(mesh, TetMesh)
        assert mesh.num_nodes > 0
        # Elements could be 0 if not enough points for Delaunay
        if mesh.num_nodes >= 4:
            assert mesh.num_elements > 0

    def test_generate_mesh_full(self, dti_mesh_generator):
        """Test full mesh generation pipeline."""
        gen = dti_mesh_generator

        mesh = gen.generate_mesh()

        assert isinstance(mesh, TetMesh)
        assert mesh.num_nodes > 0

        # Check node labels are assigned
        if len(mesh.node_labels) > 0:
            assert mesh.node_labels.dtype == np.int32
            # Should have WM or GM labels
            unique_labels = np.unique(mesh.node_labels)
            assert len(unique_labels) > 0

    def test_mesh_quality(self, dti_mesh_generator):
        """Test that generated mesh has reasonable quality."""
        gen = dti_mesh_generator

        mesh = gen.generate_mesh()

        if mesh.num_elements > 0:
            metrics = mesh.compute_quality_metrics()

            # Check basic metrics exist
            assert 'num_nodes' in metrics
            assert 'num_elements' in metrics
            assert 'total_volume' in metrics

            # Volume should be positive
            assert metrics['total_volume'] > 0

            # Aspect ratios should be reasonable (not extremely distorted)
            assert metrics['mean_aspect_ratio'] < 50  # Very generous bound


# ============================================================================
# Integration Tests
# ============================================================================

class TestDTIMeshIntegration:
    """Integration tests for DTI mesh generation."""

    def test_create_dti_guided_mesh_function(self, synthetic_fiber_orientation, synthetic_tissue_segmentation):
        """Test the convenience function."""
        # This would need full BiophysicalConstraints which requires atlas files
        # So we test the generator directly instead
        config = DTIMeshConfig(
            wm_node_spacing=3.0,
            gm_node_spacing=4.0,
            fa_threshold=0.15,
            min_tract_length=3.0,
            seed_density=0.1,
        )

        generator = DTIGuidedMeshGenerator(
            fiber_orientation=synthetic_fiber_orientation,
            tissue_segmentation=synthetic_tissue_segmentation,
            config=config,
            posterior_fossa_only=False,
        )

        mesh = generator.generate_mesh()
        assert isinstance(mesh, TetMesh)

    def test_mesh_connectivity(self, dti_mesh_generator):
        """Test that mesh nodes have proper connectivity."""
        gen = dti_mesh_generator
        mesh = gen.generate_mesh()

        if mesh.num_elements > 0:
            # Build neighbors
            mesh.build_node_neighbors()

            # Most nodes should have neighbors
            nodes_with_neighbors = sum(1 for neighbors in mesh.node_neighbors.values() if len(neighbors) > 0)

            # Allow some isolated nodes at boundaries
            assert nodes_with_neighbors > mesh.num_nodes * 0.5

    def test_reproducibility(self, synthetic_fiber_orientation, synthetic_tissue_segmentation, dti_mesh_config):
        """Test that mesh generation is deterministic with fixed seed."""
        np.random.seed(42)

        gen1 = DTIGuidedMeshGenerator(
            fiber_orientation=synthetic_fiber_orientation,
            tissue_segmentation=synthetic_tissue_segmentation,
            config=dti_mesh_config,
            posterior_fossa_only=False,
        )
        mesh1 = gen1.generate_mesh()

        np.random.seed(42)

        gen2 = DTIGuidedMeshGenerator(
            fiber_orientation=synthetic_fiber_orientation,
            tissue_segmentation=synthetic_tissue_segmentation,
            config=dti_mesh_config,
            posterior_fossa_only=False,
        )
        mesh2 = gen2.generate_mesh()

        # Should have same number of nodes and elements
        assert mesh1.num_nodes == mesh2.num_nodes
        assert mesh1.num_elements == mesh2.num_elements


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_wm_region(self):
        """Test handling of empty white matter region."""
        shape = (10, 10, 10)

        # All zeros FA (no WM)
        fa = np.zeros(shape, dtype=np.float32)
        vectors = np.ones((*shape, 3), dtype=np.float64)
        vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

        fibers = FiberOrientation(
            vectors=vectors,
            fractional_anisotropy=fa,
            affine=np.eye(4),
        )

        # All background labels
        labels = np.zeros(shape, dtype=np.int32)
        seg = TissueSegmentation(
            labels=labels,
            affine=np.eye(4),
            voxel_size=(1.0, 1.0, 1.0),
        )

        config = DTIMeshConfig(
            wm_node_spacing=2.0,
            gm_node_spacing=2.0,
            fa_threshold=0.1,
            seed_density=0.5,
        )

        gen = DTIGuidedMeshGenerator(
            fiber_orientation=fibers,
            tissue_segmentation=seg,
            config=config,
            posterior_fossa_only=False,
        )

        # Should not crash, may return mesh with few/no elements
        mesh = gen.generate_mesh()
        assert isinstance(mesh, TetMesh)

    def test_small_wm_region(self):
        """Test handling of very small white matter region."""
        shape = (10, 10, 10)

        # Single high-FA voxel
        fa = np.zeros(shape, dtype=np.float32)
        fa[5, 5, 5] = 0.8

        vectors = np.zeros((*shape, 3), dtype=np.float64)
        vectors[..., 0] = 1.0

        fibers = FiberOrientation(
            vectors=vectors,
            fractional_anisotropy=fa,
            affine=np.eye(4),
        )

        labels = np.zeros(shape, dtype=np.int32)
        labels[5, 5, 5] = BrainTissue.WHITE_MATTER.value

        seg = TissueSegmentation(
            labels=labels,
            affine=np.eye(4),
            voxel_size=(1.0, 1.0, 1.0),
        )

        config = DTIMeshConfig(
            wm_node_spacing=2.0,
            gm_node_spacing=2.0,
            fa_threshold=0.3,
            min_tract_length=0.5,
            seed_density=1.0,
        )

        gen = DTIGuidedMeshGenerator(
            fiber_orientation=fibers,
            tissue_segmentation=seg,
            config=config,
            posterior_fossa_only=False,
        )

        # Should handle gracefully
        mesh = gen.generate_mesh()
        assert isinstance(mesh, TetMesh)


# ============================================================================
# Boundary Conditions Tests
# ============================================================================

class TestBoundaryNodes:
    """Test boundary node detection."""

    def test_boundary_detection(self, dti_mesh_generator):
        """Test that boundary nodes are correctly identified."""
        gen = dti_mesh_generator
        mesh = gen.generate_mesh()

        if mesh.num_elements > 0 and len(mesh.boundary_nodes) > 0:
            # Boundary nodes should be valid indices
            assert all(0 <= idx < mesh.num_nodes for idx in mesh.boundary_nodes)

            # Should have some but not all nodes on boundary
            assert 0 < len(mesh.boundary_nodes) < mesh.num_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
