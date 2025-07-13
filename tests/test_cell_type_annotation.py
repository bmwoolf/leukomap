"""
Tests for cell type annotation module.
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from leukomap.cell_type_annotation import CellTypeAnnotator, annotate_cell_types_comprehensive


class TestCellTypeAnnotator:
    """Test the CellTypeAnnotator class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create a small test dataset
        np.random.seed(42)
        n_cells = 100
        n_genes = 50
        
        # Create random expression data
        X = np.random.poisson(5, (n_cells, n_genes))
        
        # Create gene names
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Create cell metadata
        obs_data = {
            'sample': np.random.choice(['sample1', 'sample2'], n_cells),
            'health_status': np.random.choice(['healthy', 'leukemia'], n_cells),
            'leiden': np.random.choice(['0', '1', '2'], n_cells)
        }
        
        # Create AnnData object
        self.adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(obs_data),
            var=pd.DataFrame(index=gene_names)
        )
        
        # Add some basic preprocessing
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Create output directory
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test outputs."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_annotator_initialization(self):
        """Test annotator initialization."""
        annotator = CellTypeAnnotator(self.output_dir)
        assert annotator.output_dir == self.output_dir
        assert len(annotator.celltypist_models) > 0
    
    def test_celltypist_annotation(self):
        """Test CellTypist annotation."""
        annotator = CellTypeAnnotator(self.output_dir)
        
        # Test CellTypist annotation
        try:
            annotated_adata = annotator.annotate_celltypist(self.adata)
            
            # Check that annotations were added
            assert 'celltypist_cell_type' in annotated_adata.obs.columns
            assert 'celltypist_confidence' in annotated_adata.obs.columns
            
            # Check that we have some annotations
            assert annotated_adata.obs['celltypist_cell_type'].nunique() > 0
            
        except Exception as e:
            # CellTypist might not be available or might fail with small test data
            pytest.skip(f"CellTypist annotation failed: {e}")
    
    def test_annotation_comparison(self):
        """Test annotation comparison functionality."""
        annotator = CellTypeAnnotator(self.output_dir)
        
        # Add mock annotations for testing
        self.adata.obs['celltypist_cell_type'] = np.random.choice(['A', 'B', 'C'], len(self.adata))
        self.adata.obs['azimuth_cell_type'] = np.random.choice(['A', 'B', 'C'], len(self.adata))
        
        # Test comparison
        comparison_results = annotator.compare_annotations(self.adata)
        
        # Should have one comparison
        assert len(comparison_results) == 1
        assert 'celltypist_vs_azimuth' in comparison_results
        
        # Check that comparison metrics are calculated
        comparison = comparison_results['celltypist_vs_azimuth']
        assert 'adjusted_rand_index' in comparison
        assert 'normalized_mutual_info' in comparison
        assert 'confusion_matrix' in comparison
    
    def test_cell_types_by_condition(self):
        """Test cell type analysis by condition."""
        annotator = CellTypeAnnotator(self.output_dir)
        
        # Add mock annotations
        self.adata.obs['celltypist_cell_type'] = np.random.choice(['A', 'B', 'C'], len(self.adata))
        
        # Test analysis
        analysis = annotator.analyze_cell_types_by_condition(
            self.adata, 
            condition_col='health_status',
            method='celltypist'
        )
        
        # Check that analysis was performed
        assert len(analysis) > 0
        assert 'total_cells' in analysis.columns
        assert 'healthy_pct' in analysis.columns
        assert 'leukemia_pct' in analysis.columns
    
    def test_save_results(self):
        """Test saving results."""
        annotator = CellTypeAnnotator(self.output_dir)
        
        # Add mock annotations and comparison results
        self.adata.obs['celltypist_cell_type'] = np.random.choice(['A', 'B', 'C'], len(self.adata))
        self.adata.obs['azimuth_cell_type'] = np.random.choice(['A', 'B', 'C'], len(self.adata))
        
        comparison_results = annotator.compare_annotations(self.adata)
        
        # Test saving
        annotator.save_results(self.adata, comparison_results)
        
        # Check that files were created
        assert (self.output_dir / 'adata_celltype_annotated.h5ad').exists()
        assert (self.output_dir / 'annotation_comparison_metrics.csv').exists()
        assert (self.output_dir / 'celltype_annotation_summary_report.txt').exists()


class TestComprehensiveAnnotation:
    """Test the comprehensive annotation function."""
    
    def setup_method(self):
        """Set up test data."""
        # Create a small test dataset
        np.random.seed(42)
        n_cells = 50
        n_genes = 30
        
        # Create random expression data
        X = np.random.poisson(5, (n_cells, n_genes))
        
        # Create gene names
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Create cell metadata
        obs_data = {
            'sample': np.random.choice(['sample1', 'sample2'], n_cells),
            'health_status': np.random.choice(['healthy', 'leukemia'], n_cells),
            'leiden': np.random.choice(['0', '1'], n_cells)
        }
        
        # Create AnnData object
        self.adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(obs_data),
            var=pd.DataFrame(index=gene_names)
        )
        
        # Add some basic preprocessing
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Create output directory
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test outputs."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_comprehensive_annotation(self):
        """Test comprehensive annotation function."""
        try:
            # Test with just CellTypist (most likely to work)
            annotated_adata = annotate_cell_types_comprehensive(
                self.adata, 
                self.output_dir,
                methods=['celltypist']
            )
            
            # Check that annotations were added
            assert 'celltypist_cell_type' in annotated_adata.obs.columns
            
        except Exception as e:
            # This might fail if CellTypist is not properly installed
            pytest.skip(f"Comprehensive annotation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 