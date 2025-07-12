#!/usr/bin/env python3
"""
Unit tests for healthy reference integration functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from unittest.mock import patch, MagicMock

# Import the functions to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "script"))
from integrate_healthy_reference import (
    Config, 
    preprocess_data, 
    detect_abnormal_clusters,
    _extract_leukemia_type
)


class TestHealthyIntegration(unittest.TestCase):
    """Test cases for healthy reference integration functionality."""
    
    def setUp(self):
        """Set up test data and environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # Create mock AnnData objects for testing
        self.healthy_adata = self._create_mock_adata(100, 50, "healthy")
        self.leukemia_adata = self._create_mock_adata(150, 50, "leukemia")
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_adata(self, n_cells: int, n_genes: int, health_status: str) -> ad.AnnData:
        """Create a mock AnnData object for testing."""
        # Create random count data
        X = np.random.poisson(5, (n_cells, n_genes))
        
        # Create gene names
        var_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Create cell names
        obs_names = [f"Cell_{i}_{health_status}" for i in range(n_cells)]
        
        # Create observation metadata
        obs = pd.DataFrame({
            'health_status': health_status,
            'sample_type': health_status,
            'sample_id': f"test_{health_status}"
        }, index=obs_names)
        
        # Create variable metadata
        var = pd.DataFrame({
            'gene_name': var_names
        }, index=var_names)
        
        return ad.AnnData(X=X, obs=obs, var=var)
    
    def test_extract_leukemia_type(self):
        """Test leukemia type extraction from filename."""
        # Test ETV6-RUNX1
        filename = "data-raw-ETV6-RUNX1_1-matrix.h5ad"
        result = _extract_leukemia_type(filename)
        self.assertEqual(result, "ETV6-RUNX1")
        
        # Test PRE-T
        filename = "data-raw-PRE-T_2-matrix.h5ad"
        result = _extract_leukemia_type(filename)
        self.assertEqual(result, "PRE-T")
        
        # Test HHD
        filename = "data-raw-HHD_1-matrix.h5ad"
        result = _extract_leukemia_type(filename)
        self.assertEqual(result, "HHD")
        
        # Test unknown
        filename = "data-raw-UNKNOWN-matrix.h5ad"
        result = _extract_leukemia_type(filename)
        self.assertEqual(result, "unknown")
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Test basic preprocessing
        processed_adata = preprocess_data(self.healthy_adata.copy(), "test")
        
        # Check that preprocessing was applied
        self.assertIn('n_genes_by_counts', processed_adata.obs.columns)
        self.assertIn('total_counts', processed_adata.obs.columns)
        self.assertIn('pct_counts_mt', processed_adata.obs.columns)
        
        # Check that highly variable genes were selected
        self.assertLessEqual(processed_adata.n_vars, self.healthy_adata.n_vars)
        
        # Check that normalization was applied
        self.assertTrue(np.allclose(processed_adata.X.sum(axis=1), Config.TARGET_SUM, atol=1e-6))
    
    def test_detect_abnormal_clusters(self):
        """Test abnormal cluster detection."""
        # Create a mock integrated dataset with clusters
        integrated_adata = ad.concat([self.healthy_adata, self.leukemia_adata], join='outer')
        integrated_adata.obs_names_make_unique()
        
        # Add mock clustering results
        integrated_adata.obs['leiden'] = ['0'] * 50 + ['1'] * 50 + ['2'] * 50 + ['3'] * 50 + ['4'] * 50
        integrated_adata.obs['leiden'] = integrated_adata.obs['leiden'].astype('category')
        
        # Add health status
        integrated_adata.obs['health_status'] = ['healthy'] * 100 + ['leukemia'] * 150
        integrated_adata.obs['health_status'] = integrated_adata.obs['health_status'].astype('category')
        
        # Test abnormal cluster detection
        abnormal_clusters = detect_abnormal_clusters(integrated_adata)
        
        # Check that results are returned
        self.assertIsInstance(abnormal_clusters, pd.DataFrame)
        self.assertGreater(len(abnormal_clusters), 0)
        
        # Check required columns
        required_columns = [
            'cluster', 'total_cells', 'healthy_cells', 'leukemia_cells',
            'healthy_ratio', 'leukemia_ratio', 'is_abnormal', 'abnormality_score'
        ]
        for col in required_columns:
            self.assertIn(col, abnormal_clusters.columns)
    
    def test_config_parameters(self):
        """Test configuration parameters."""
        # Test that all required config parameters are defined
        self.assertIsInstance(Config.MIN_GENES, int)
        self.assertIsInstance(Config.MIN_CELLS, int)
        self.assertIsInstance(Config.N_HVG, int)
        self.assertIsInstance(Config.N_LATENT, int)
        self.assertIsInstance(Config.RESOLUTION, float)
        
        # Test that paths are Path objects
        self.assertIsInstance(Config.CACHE_DIR, Path)
        self.assertIsInstance(Config.OUTPUT_DIR, Path)
        
        # Test that prefixes are strings
        self.assertIsInstance(Config.HEALTHY_PREFIX, str)
        self.assertIsInstance(Config.LEUKEMIA_PREFIXES, list)
    
    @patch('integrate_healthy_reference.celltypist')
    def test_annotate_cell_types_no_celltypist(self, mock_celltypist):
        """Test cell type annotation when CellTypist is not available."""
        mock_celltypist = None
        
        # Create test data
        test_adata = self.healthy_adata.copy()
        
        # Import the function with mocked celltypist
        with patch.dict('sys.modules', {'celltypist': None}):
            from integrate_healthy_reference import annotate_cell_types
            result = annotate_cell_types(test_adata)
            
            # Should return the original adata unchanged
            self.assertEqual(result.n_obs, test_adata.n_obs)
            self.assertEqual(result.n_vars, test_adata.n_vars)
    
    def test_abnormal_cluster_threshold(self):
        """Test abnormal cluster detection threshold logic."""
        # Create test data with known ratios
        test_data = pd.DataFrame({
            'cluster': ['0', '1', '2', '3'],
            'total_cells': [100, 100, 100, 100],
            'healthy_cells': [90, 50, 10, 5],
            'leukemia_cells': [10, 50, 90, 95],
            'healthy_ratio': [0.9, 0.5, 0.1, 0.05],
            'leukemia_ratio': [0.1, 0.5, 0.9, 0.95],
            'is_abnormal': [False, False, True, True],
            'abnormality_score': [0.1, 0.5, 0.9, 0.95]
        })
        
        # Test that clusters with >80% leukemia cells are marked as abnormal
        abnormal_clusters = test_data[test_data['is_abnormal']]
        self.assertEqual(len(abnormal_clusters), 2)
        self.assertTrue(all(abnormal_clusters['leukemia_ratio'] > 0.8))
        
        # Test that clusters with <=80% leukemia cells are not marked as abnormal
        normal_clusters = test_data[~test_data['is_abnormal']]
        self.assertEqual(len(normal_clusters), 2)
        self.assertTrue(all(normal_clusters['leukemia_ratio'] <= 0.8))


if __name__ == '__main__':
    unittest.main() 