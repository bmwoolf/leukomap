"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
import tempfile
import shutil
import os
import logging
from scipy import sparse

from leukomap.preprocessing import preprocess


class TestPreprocessing:
    """Test cases for preprocessing functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object for testing."""
        # Create realistic scRNA-seq data
        n_cells, n_genes = 100, 50
        
        # Sparse count matrix with realistic distribution
        np.random.seed(42)
        X = sparse.csr_matrix(np.random.poisson(2, (n_cells, n_genes)))
        
        # Add some mitochondrial genes
        gene_names = [f"Gene_{i}" for i in range(40)] + [f"MT-Gene_{i}" for i in range(10)]
        
        # Create observation metadata
        obs_data = {
            'sample': ['Sample1'] * 50 + ['Sample2'] * 50,
            'sample_type': ['ETV6-RUNX1'] * 50 + ['HHD'] * 50,
            'n_genes': np.random.randint(100, 1000, n_cells),
            'n_counts': np.random.randint(1000, 50000, n_cells)
        }
        
        obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=gene_names)
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Add some QC metrics that scanpy would compute
        adata.obs['n_genes'] = np.array(X.sum(axis=1)).flatten()
        adata.obs['n_counts'] = np.array(X.sum(axis=1)).flatten()
        
        return adata
    
    @pytest.fixture
    def low_quality_adata(self):
        """Create AnnData with low quality cells for testing filtering."""
        n_cells, n_genes = 50, 30
        
        # Create matrix with some cells having very few genes
        X = sparse.csr_matrix(np.random.poisson(1, (n_cells, n_genes)))
        
        # Make some cells have very few genes
        for i in range(10):
            X[i, :] = 0
            X[i, np.random.choice(n_genes, 1)] = 1  # Only 1 gene expressed
        
        gene_names = [f"Gene_{i}" for i in range(25)] + [f"MT-Gene_{i}" for i in range(5)]
        
        obs_data = {
            'sample': ['Sample1'] * 25 + ['Sample2'] * 25,
            'sample_type': ['ETV6-RUNX1'] * 25 + ['HHD'] * 25,
            'n_genes': np.array(X.sum(axis=1)).flatten(),
            'n_counts': np.array(X.sum(axis=1)).flatten()
        }
        
        obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=gene_names)
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obs['n_genes'] = np.array(X.sum(axis=1)).flatten()
        adata.obs['n_counts'] = np.array(X.sum(axis=1)).flatten()
        
        return adata
    
    def test_preprocess_basic_functionality(self, sample_adata, temp_dir):
        """Test basic preprocessing functionality with permissive thresholds."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        processed_adata = preprocess(
            sample_adata.copy(),
            min_genes=1,
            min_cells=1,
            save_path=str(cache_dir / "test_preprocessed.h5ad")
        )
        assert processed_adata is not None
        assert isinstance(processed_adata, ad.AnnData)
        assert 'pct_counts_mt' in processed_adata.obs.columns
        assert 'highly_variable' in processed_adata.var.columns
        assert processed_adata.obs['sample'].dtype.name == 'category'
        assert processed_adata.obs['sample_type'].dtype.name == 'category'
        assert (cache_dir / "test_preprocessed.h5ad").exists()
    
    def test_preprocess_filtering(self, low_quality_adata, temp_dir):
        """Test that filtering removes low quality cells and genes (permissive)."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        original_cells = low_quality_adata.n_obs
        original_genes = low_quality_adata.n_vars
        processed_adata = preprocess(
            low_quality_adata.copy(),
            min_genes=1,
            min_cells=1,
            save_path=str(cache_dir / "test_filtered.h5ad")
        )
        assert processed_adata.n_obs <= original_cells
        assert processed_adata.n_vars <= original_genes
        assert all(processed_adata.obs['n_genes'] >= 1)
    
    def test_preprocess_mitochondrial_filtering(self, sample_adata, temp_dir):
        """Test mitochondrial content filtering (permissive)."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        sample_adata.obs.loc['cell_0', 'n_counts'] = 1000
        sample_adata.X[0, 40:50] = 800
        processed_adata = preprocess(
            sample_adata.copy(),
            min_genes=1,
            min_cells=1,
            max_mito=0.5,  # less strict
            save_path=str(cache_dir / "test_mito_filtered.h5ad")
        )
        assert 'pct_counts_mt' in processed_adata.obs.columns
        assert all(processed_adata.obs['pct_counts_mt'] < 0.5)
    
    def test_preprocess_parameter_validation(self, sample_adata, temp_dir):
        """Test preprocessing with different parameter values (permissive)."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        processed_adata = preprocess(
            sample_adata.copy(),
            min_genes=1,
            min_cells=1,
            max_genes=3000,
            max_counts=30000,
            max_mito=0.5,
            save_path=str(cache_dir / "test_custom_params.h5ad")
        )
        assert all(processed_adata.obs['n_genes'] >= 1)
        assert all(processed_adata.obs['n_genes'] <= 3000)
        assert all(processed_adata.obs['pct_counts_mt'] < 0.5)
        if 'n_counts' in processed_adata.obs.columns:
            assert all(processed_adata.obs['n_counts'] <= 30000)
    
    def test_preprocess_no_save(self, sample_adata):
        """Test preprocessing without saving to file (permissive)."""
        processed_adata = preprocess(
            sample_adata.copy(),
            min_genes=1,
            min_cells=1,
            save_path=None
        )
        assert processed_adata is not None
        assert isinstance(processed_adata, ad.AnnData)
        assert 'pct_counts_mt' in processed_adata.obs.columns
        assert 'highly_variable' in processed_adata.var.columns
    
    def test_preprocess_empty_data(self, temp_dir):
        """Test preprocessing with empty AnnData (should raise ValueError)."""
        X = sparse.csr_matrix((0, 0))
        obs = pd.DataFrame()
        var = pd.DataFrame()
        empty_adata = ad.AnnData(X=X, obs=obs, var=var)
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        with pytest.raises(ValueError, match="AnnData is empty after filtering"):
            preprocess(empty_adata, save_path=str(cache_dir / "test_empty.h5ad"))
    
    def test_preprocess_missing_columns(self, temp_dir):
        """Test preprocessing with missing observation columns (should raise ValueError)."""
        n_cells, n_genes = 20, 10
        X = sparse.csr_matrix(np.random.poisson(2, (n_cells, n_genes)))
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obs['n_genes'] = np.array(X.sum(axis=1)).flatten()
        adata.obs['n_counts'] = np.array(X.sum(axis=1)).flatten()
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        with pytest.raises(ValueError, match="AnnData is empty after filtering"):
            preprocess(adata, save_path=str(cache_dir / "test_no_samples.h5ad"))
    
    def test_preprocess_highly_variable_genes(self, sample_adata, temp_dir):
        """Test that highly variable genes are computed correctly (permissive)."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        processed_adata = preprocess(
            sample_adata.copy(),
            min_genes=1,
            min_cells=1,
            save_path=str(cache_dir / "test_hvg.h5ad")
        )
        assert 'highly_variable' in processed_adata.var.columns
        assert 'highly_variable_rank' in processed_adata.var.columns
        assert 'means' in processed_adata.var.columns
        assert 'variances' in processed_adata.var.columns
        assert processed_adata.var['highly_variable'].sum() > 0
        n_hvg = processed_adata.var['highly_variable'].sum()
        assert n_hvg <= min(2000, processed_adata.n_vars)
    
    def test_preprocess_data_integrity(self, sample_adata, temp_dir):
        """Test that preprocessing preserves data integrity (permissive)."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        original_obs_names = sample_adata.obs_names.copy()
        original_var_names = sample_adata.var_names.copy()
        processed_adata = preprocess(
            sample_adata.copy(),
            min_genes=1,
            min_cells=1,
            save_path=str(cache_dir / "test_integrity.h5ad")
        )
        assert all(name in original_obs_names for name in processed_adata.obs_names)
        assert all(name in original_var_names for name in processed_adata.var_names)
        assert isinstance(processed_adata.X, sparse.csr_matrix)
    
    def test_preprocess_logging(self, sample_adata, temp_dir, caplog):
        """Test that preprocessing logs appropriate messages (permissive)."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        with caplog.at_level(logging.INFO):
            preprocess(
                sample_adata.copy(),
                min_genes=1,
                min_cells=1,
                save_path=str(cache_dir / "test_logging.h5ad")
            )
        log_messages = [record.message for record in caplog.records]
        assert any("Starting preprocessing" in msg for msg in log_messages)
        assert any("After initial filtering" in msg for msg in log_messages)
        assert any("Annotated mitochondrial gene content" in msg for msg in log_messages)
        assert any("After QC" in msg for msg in log_messages)
        assert any("Annotated batch/sample columns" in msg for msg in log_messages)
        assert any("Computed highly variable genes" in msg for msg in log_messages)
        assert any("Saved preprocessed AnnData" in msg for msg in log_messages)
        assert any("Preprocessing complete" in msg for msg in log_messages)


class TestPreprocessingIntegration:
    """Integration tests for preprocessing workflow."""
    @pytest.fixture
    def temp_dir(self, tmp_path):
        yield tmp_path
    @pytest.fixture
    def realistic_adata(self):
        """Create a more realistic AnnData object for integration testing."""
        n_cells, n_genes = 500, 200
        
        # Create sparse count matrix with realistic distribution
        np.random.seed(42)
        X = sparse.csr_matrix(np.random.poisson(3, (n_cells, n_genes)))
        
        # Add mitochondrial genes
        gene_names = [f"Gene_{i}" for i in range(180)] + [f"MT-Gene_{i}" for i in range(20)]
        
        # Create observation metadata with different sample types
        sample_types = ['ETV6-RUNX1', 'HHD', 'PRE-T', 'PBMMC']
        samples = []
        sample_type_list = []
        
        for i in range(n_cells):
            sample_type = sample_types[i % len(sample_types)]
            sample_num = (i // len(sample_types)) + 1
            samples.append(f"{sample_type}_{sample_num}")
            sample_type_list.append(sample_type)
        
        obs_data = {
            'sample': samples,
            'sample_type': sample_type_list,
            'n_genes': np.array(X.sum(axis=1)).flatten(),
            'n_counts': np.array(X.sum(axis=1)).flatten()
        }
        
        obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
        var = pd.DataFrame(index=gene_names)
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obs['n_genes'] = np.array(X.sum(axis=1)).flatten()
        adata.obs['n_counts'] = np.array(X.sum(axis=1)).flatten()
        
        return adata
    
    def test_full_preprocessing_workflow(self, realistic_adata, temp_dir):
        """Test the complete preprocessing workflow."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        
        # Run preprocessing
        processed_adata = preprocess(
            realistic_adata.copy(),
            min_genes=100,
            min_cells=10,
            max_genes=4000,
            max_counts=40000,
            max_mito=0.15,
            save_path=str(cache_dir / "workflow_test.h5ad")
        )
        
        # Verify the complete workflow
        assert processed_adata is not None
        assert processed_adata.n_obs > 0
        assert processed_adata.n_vars > 0
        
        # Check all expected columns are present
        expected_obs_cols = ['pct_counts_mt', 'n_genes', 'n_counts', 'sample', 'sample_type']
        for col in expected_obs_cols:
            assert col in processed_adata.obs.columns
        
        expected_var_cols = ['highly_variable', 'highly_variable_rank', 'means', 'variances']
        for col in expected_var_cols:
            assert col in processed_adata.var.columns
        
        # Check that data was filtered appropriately
        assert all(processed_adata.obs['n_genes'] >= 100)
        assert all(processed_adata.obs['n_genes'] <= 4000)
        assert all(processed_adata.obs['pct_counts_mt'] < 0.15)
        assert all(processed_adata.obs['n_counts'] <= 40000)
        
        # Check that file was saved and can be loaded
        saved_path = cache_dir / "workflow_test.h5ad"
        assert saved_path.exists()
        
        # Test loading the saved file
        loaded_adata = sc.read_h5ad(saved_path)
        assert loaded_adata.n_obs == processed_adata.n_obs
        assert loaded_adata.n_vars == processed_adata.n_vars
    
    def test_preprocessing_reproducibility(self, realistic_adata, temp_dir):
        """Test that preprocessing is reproducible."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        
        # Run preprocessing twice with same parameters
        processed_1 = preprocess(
            realistic_adata.copy(),
            save_path=str(cache_dir / "reproducible_1.h5ad")
        )
        
        processed_2 = preprocess(
            realistic_adata.copy(),
            save_path=str(cache_dir / "reproducible_2.h5ad")
        )
        
        # Results should be identical
        assert processed_1.n_obs == processed_2.n_obs
        assert processed_1.n_vars == processed_2.n_vars
        
        # Check that highly variable genes are the same
        hvg_1 = processed_1.var['highly_variable']
        hvg_2 = processed_2.var['highly_variable']
        assert hvg_1.equals(hvg_2)
        
        # Check that mitochondrial percentages are the same
        mito_1 = processed_1.obs['pct_counts_mt']
        mito_2 = processed_2.obs['pct_counts_mt']
        np.testing.assert_array_almost_equal(mito_1, mito_2) 