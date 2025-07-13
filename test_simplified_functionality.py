#!/usr/bin/env python3
"""
Test script to verify simplified LeukoMap functionality.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from leukomap.core import AnalysisConfig, LeukoMapAnalysis
from leukomap.data import DataLoader, DataManager
from leukomap.preprocessing import PreprocessingPipeline, PreprocessingManager
from leukomap.cell_type_annotation import CellTypeAnnotator


def create_mock_data():
    """Create mock single-cell data for testing."""
    # Create mock expression matrix
    n_cells, n_genes = 100, 50
    X = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))
    
    # Create cell metadata
    obs = pd.DataFrame({
        'sample': ['sample_1'] * 50 + ['sample_2'] * 50,
        'sample_type': ['PBMMC'] * 50 + ['ETV6-RUNX1'] * 50,
        'total_counts': X.sum(axis=1),
        'n_genes_by_counts': (X > 0).sum(axis=1)
    }, index=[f'cell_{i}' for i in range(n_cells)])
    
    # Create gene metadata
    var = pd.DataFrame({
        'gene_ids': [f'gene_{i}' for i in range(n_genes)],
        'feature_types': ['Gene Expression'] * n_genes
    }, index=[f'gene_{i}' for i in range(n_genes)])
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    return adata


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    # Create mock data
    adata = create_mock_data()
    
    # Test DataLoader
    config = AnalysisConfig()
    loader = DataLoader(config)
    
    # Test validation
    assert loader.validate_input(adata) == True
    print("✓ Data loading validation passed")
    
    # Test DataManager
    manager = DataManager(config)
    validation_results = manager.validator.validate_adata(adata)
    
    assert validation_results['basic_stats']['n_cells'] == 100
    assert validation_results['basic_stats']['n_genes'] == 50
    print("✓ Data management validation passed")
    
    return adata


def test_preprocessing():
    """Test preprocessing functionality."""
    print("Testing preprocessing...")
    
    # Create mock data
    adata = create_mock_data()
    
    # Test PreprocessingPipeline
    config = AnalysisConfig(min_genes=1, min_cells=1)
    preprocessor = PreprocessingPipeline(config)
    
    # Process data
    processed_adata = preprocessor.process(adata)
    
    # Check that preprocessing worked
    assert processed_adata.n_obs <= adata.n_obs  # Some cells may be filtered
    assert processed_adata.n_vars <= adata.n_vars  # Some genes may be filtered
    assert 'highly_variable' in processed_adata.var.columns
    print("✓ Preprocessing pipeline passed")
    
    # Test PreprocessingManager
    manager = PreprocessingManager(config)
    manager_processed = manager.preprocess_data(adata, save_results=False)
    
    assert manager_processed.n_obs == processed_adata.n_obs
    print("✓ Preprocessing manager passed")
    
    return processed_adata


def test_cell_type_annotation():
    """Test cell type annotation functionality."""
    print("Testing cell type annotation...")
    
    # Create mock data
    adata = create_mock_data()
    
    # Test CellTypeAnnotator
    config = AnalysisConfig()
    annotator = CellTypeAnnotator(config)
    
    # Process data (should add mock annotations)
    annotated_adata = annotator.process(adata)
    
    # Check that annotations were added
    assert 'celltypist_cell_type' in annotated_adata.obs.columns
    assert 'celltypist_confidence' in annotated_adata.obs.columns
    assert 'health_status' in annotated_adata.obs.columns
    print("✓ Cell type annotation passed")
    
    return annotated_adata


def test_full_pipeline():
    """Test the complete simplified pipeline."""
    print("Testing full pipeline...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock data
        adata = create_mock_data()
        
        # Save mock data
        data_dir = temp_path / "mock_data"
        data_dir.mkdir()
        adata.write(data_dir / "data.h5ad")
        
        # Test full analysis
        config = AnalysisConfig(
            data_path=data_dir,
            output_dir=temp_path / "results",
            min_genes=1,
            min_cells=1
        )
        
        analysis = LeukoMapAnalysis(config)
        results = analysis.run_full_analysis()
        
        # Check that we got results
        assert len(results) > 0
        assert 'DataLoader' in results
        assert 'PreprocessingPipeline' in results
        print("✓ Full pipeline passed")
        
        # Test convenience functions
        from leukomap import load_and_preprocess, annotate_cells
        
        # Test load_and_preprocess
        processed = load_and_preprocess(str(data_dir), str(temp_path / "results2"))
        assert processed.n_obs > 0
        print("✓ Convenience function load_and_preprocess passed")
        
        # Test annotate_cells
        annotated = annotate_cells(processed, str(temp_path / "results3"))
        assert 'celltypist_cell_type' in annotated.obs.columns
        print("✓ Convenience function annotate_cells passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Simplified LeukoMap Functionality")
    print("=" * 60)
    
    try:
        # Test individual components
        test_data_loading()
        test_preprocessing()
        test_cell_type_annotation()
        
        # Test full pipeline
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("✓ Simplified codebase maintains functionality")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 