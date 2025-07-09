"""
Unit tests for data loading module.
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
from scipy import sparse

from leukomap.data_loading import (
    load_data,
    _is_10x_format,
    _is_processed_format,
    _extract_sample_type,
    _validate_and_clean_data
)


class TestDataLoading:
    """Test cases for data loading functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_10x_data(self, temp_dir):
        """Create mock 10X Genomics data structure."""
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        
        # Create mock sample directories
        samples = ["ETV6-RUNX1_1", "HHD_1", "PBMMC_1"]
        
        for sample in samples:
            sample_dir = raw_dir / sample
            sample_dir.mkdir()
            
            # Create mock matrix.mtx file
            matrix_file = sample_dir / "matrix.mtx"
            with open(matrix_file, 'w') as f:
                f.write("%%MatrixMarket matrix coordinate integer general\n")
                f.write("3 5 6\n")  # 3 cells, 5 genes, 6 non-zero entries
                f.write("1 1 1\n")
                f.write("1 2 2\n")
                f.write("2 1 1\n")
                f.write("2 3 3\n")
                f.write("3 2 1\n")
                f.write("3 4 2\n")
            
            # Create mock features.tsv file
            features_file = sample_dir / "features.tsv"
            with open(features_file, 'w') as f:
                f.write("Gene1\tGene1\tGene Expression\n")
                f.write("Gene2\tGene2\tGene Expression\n")
                f.write("Gene3\tGene3\tGene Expression\n")
                f.write("Gene4\tGene4\tGene Expression\n")
                f.write("Gene5\tGene5\tGene Expression\n")
            
            # Create mock barcodes.tsv file
            barcodes_file = sample_dir / "barcodes.tsv"
            with open(barcodes_file, 'w') as f:
                f.write(f"{sample}_AAACCTGAGAAACCAT-1\n")
                f.write(f"{sample}_AAACCTGAGAAACCGC-1\n")
                f.write(f"{sample}_AAACCTGAGAAACCTA-1\n")
        
        return temp_dir
    
    @pytest.fixture
    def mock_annotations(self, temp_dir):
        """Create mock cell annotations."""
        annotations_dir = temp_dir / "annotations"
        annotations_dir.mkdir()
        
        annotations_file = annotations_dir / "GSE132509_cell_annotations.tsv"
        
        # Create mock annotations
        annotations_data = {
            'cell_id': [
                'ETV6-RUNX1_1_AAACCTGAGAAACCAT-1',
                'ETV6-RUNX1_1_AAACCTGAGAAACCGC-1',
                'HHD_1_AAACCTGAGAAACCAT-1',
                'PBMMC_1_AAACCTGAGAAACCAT-1'
            ],
            'cell_type': ['B_cell', 'B_cell', 'T_cell', 'B_cell'],
            'sample_type': ['ETV6-RUNX1', 'ETV6-RUNX1', 'HHD', 'PBMMC']
        }
        
        annotations_df = pd.DataFrame(annotations_data)
        annotations_df.set_index('cell_id', inplace=True)
        annotations_df.to_csv(annotations_file, sep='\t')
        
        return temp_dir
    
    def test_is_10x_format_true(self, mock_10x_data):
        """Test that 10X format detection works correctly."""
        assert _is_10x_format(mock_10x_data) is True
    
    def test_is_10x_format_false(self, temp_dir):
        """Test that non-10X format is detected correctly."""
        assert _is_10x_format(temp_dir) is False
    
    def test_is_processed_format_true(self, temp_dir):
        """Test that processed format detection works correctly."""
        # Create a mock h5ad file
        mock_file = temp_dir / "data.h5ad"
        mock_file.touch()
        assert _is_processed_format(temp_dir) is True
    
    def test_is_processed_format_false(self, temp_dir):
        """Test that non-processed format is detected correctly."""
        assert _is_processed_format(temp_dir) is False
    
    def test_extract_sample_type(self):
        """Test sample type extraction from sample names."""
        assert _extract_sample_type("ETV6-RUNX1_1") == "ETV6-RUNX1"
        assert _extract_sample_type("HHD_2") == "HHD"
        assert _extract_sample_type("PRE-T_1") == "PRE-T"
        assert _extract_sample_type("PBMMC_3") == "PBMMC"
        assert _extract_sample_type("Unknown_sample") == "Unknown"
    
    def test_validate_and_clean_data(self):
        """Test data validation and cleaning."""
        # Create mock AnnData
        X = sparse.csr_matrix(np.random.poisson(1, (10, 5)))
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(5)])
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Test validation
        cleaned_adata = _validate_and_clean_data(adata)
        
        assert cleaned_adata.n_obs == 10
        assert cleaned_adata.n_vars == 5
        assert 'data_source' in cleaned_adata.uns
        assert cleaned_adata.uns['n_cells'] == 10
        assert cleaned_adata.uns['n_genes'] == 5
    
    def test_validate_and_clean_data_empty(self):
        """Test validation with empty data."""
        # Create empty AnnData
        X = sparse.csr_matrix((0, 0))
        obs = pd.DataFrame()
        var = pd.DataFrame()
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No cells found in data"):
            _validate_and_clean_data(adata)
    
    def test_load_data_file_not_found(self, temp_dir):
        """Test loading data from non-existent directory."""
        non_existent_dir = temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            load_data(non_existent_dir)
    
    def test_load_data_unsupported_format(self, temp_dir):
        """Test loading data with unsupported format."""
        # Create directory with no recognizable format
        temp_dir.mkdir(exist_ok=True)
        
        with pytest.raises(ValueError, match="Unsupported data format"):
            load_data(temp_dir)
    
    def test_load_10x_data_success(self, mock_10x_data):
        """Test successful loading of 10X data."""
        # Note: This test requires actual 10X data files
        # For now, we'll test the structure detection
        assert _is_10x_format(mock_10x_data) is True
    
    def test_load_data_with_annotations(self, mock_10x_data, mock_annotations):
        """Test loading data with cell annotations."""
        # This test would require actual 10X data files
        # For now, we'll test that the annotation file exists
        annotations_file = mock_annotations / "annotations" / "GSE132509_cell_annotations.tsv"
        assert annotations_file.exists()


class TestDataLoadingIntegration:
    """Integration tests for data loading."""
    
    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object for testing."""
        # Create realistic scRNA-seq data
        n_cells, n_genes = 100, 50
        
        # Sparse count matrix with realistic distribution
        X = sparse.csr_matrix(np.random.negative_binomial(2, 0.1, (n_cells, n_genes)))
        
        # Cell metadata
        obs = pd.DataFrame({
            'sample': np.random.choice(['ETV6-RUNX1_1', 'HHD_1', 'PBMMC_1'], n_cells),
            'n_genes_by_counts': np.random.poisson(2000, n_cells),
            'total_counts': np.random.poisson(10000, n_cells),
            'pct_counts_mt': np.random.beta(2, 20, n_cells) * 10
        }, index=[f"cell_{i}" for i in range(n_cells)])
        
        # Gene metadata
        var = pd.DataFrame({
            'gene_ids': [f"ENSG{i:06d}" for i in range(n_genes)],
            'feature_types': ['Gene Expression'] * n_genes
        }, index=[f"gene_{i}" for i in range(n_genes)])
        
        return ad.AnnData(X=X, obs=obs, var=var)
    
    def test_ann_data_structure(self, sample_adata):
        """Test that AnnData has correct structure."""
        assert sample_adata.n_obs == 100
        assert sample_adata.n_vars == 50
        assert 'sample' in sample_adata.obs.columns
        assert 'gene_ids' in sample_adata.var.columns
        assert sparse.issparse(sample_adata.X)
    
    def test_data_validation_workflow(self, sample_adata):
        """Test the complete data validation workflow."""
        # Add some problematic data
        sample_adata.obs.loc['cell_0', 'n_genes_by_counts'] = 0  # Cell with no genes
        
        # Validate and clean
        cleaned_adata = _validate_and_clean_data(sample_adata)
        
        # Should remove the problematic cell
        assert cleaned_adata.n_obs == 99
        assert 'data_source' in cleaned_adata.uns
        assert cleaned_adata.uns['data_source'] == 'Caron et al. (2020) - GSE132509'


if __name__ == "__main__":
    pytest.main([__file__]) 


def test_data_loader_produces_valid_anndata():
    from leukomap.data_loading import load_data
    import anndata as ad
    import os
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    adata = load_data(data_dir)
    assert isinstance(adata, ad.AnnData)
    assert adata.n_obs > 0
    assert adata.n_vars > 0
    assert adata.X.shape == (adata.n_obs, adata.n_vars)
    assert not adata.obs.empty
    assert not adata.var.empty 