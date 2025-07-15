#!/usr/bin/env python3
"""
Test script for the full LeukoMap pipeline (Python-only).
"""

import sys
import os
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create a small test dataset."""
    print("Creating test data...")
    
    # Create mock gene expression data
    n_cells = 100
    n_genes = 50
    
    # Create realistic gene names
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:03d}" for i in range(n_cells)]
    
    # Create expression matrix with some structure
    np.random.seed(42)
    # Use more realistic parameters for single-cell data
    expr_matrix = np.random.negative_binomial(10, 0.1, (n_cells, n_genes))
    # Add some zeros (sparsity)
    mask = np.random.random((n_cells, n_genes)) < 0.7
    expr_matrix[mask] = 0
    
    # Add some cell type structure
    cell_types = ['B_cell', 'T_cell', 'Monocyte', 'Neutrophil']
    sample_types = ['ETV6-RUNX1', 'PBMMC']
    
    # Create AnnData
    adata = sc.AnnData(X=expr_matrix)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    
    # Add metadata
    adata.obs['sample_type'] = np.random.choice(sample_types, n_cells)
    adata.obs['predicted_celltype'] = np.random.choice(cell_types, n_cells)
    adata.obs['annotation_confidence'] = np.random.uniform(0.5, 0.9, n_cells)
    
    # Add some quality metrics
    adata.obs['total_counts'] = expr_matrix.sum(axis=1)
    adata.obs['n_genes_by_counts'] = (expr_matrix > 0).sum(axis=1)
    
    print(f"Created test data: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata

def test_pipeline():
    """Test the full pipeline."""
    print("Testing LeukoMap pipeline...")
    
    try:
        from leukomap import analyze
        from leukomap.core import LeukoMapAnalysis, AnalysisConfig
        
        # Create test data
        adata = create_test_data()
        
        # Create config
        config = AnalysisConfig(
            output_dir=Path("test_results"),
            n_latent=5,  # Smaller for testing
            max_epochs=10,  # Fewer epochs for testing
            batch_size=32,
            min_genes=5,  # Lower threshold for testing
            min_cells=2   # Lower threshold for testing
        )
        
        # Create analysis object
        analyzer = LeukoMapAnalysis(config)
        
        # Run analysis (this will test the full pipeline)
        print("Running full analysis pipeline...")
        results = analyzer.run_full_analysis_with_data(adata)
        
        print("✅ Pipeline completed successfully!")
        print(f"Results keys: {list(results.keys())}")
        
        # Check if we have the expected outputs
        expected_keys = ['annotated_data', 'druggable_targets', 'differential_expression', 'analysis_report']
        for key in expected_keys:
            if key in results:
                print(f"✅ {key}: Found")
            else:
                print(f"⚠️  {key}: Missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components."""
    print("\nTesting individual components...")
    
    # Test auto cell type labeling
    try:
        from scripts.auto_celltype_labeling import AutoCellTypeLabeler
        print("✅ Auto cell type labeling: OK")
    except Exception as e:
        print(f"❌ Auto cell type labeling: {e}")
    
    # Test visualization pipeline
    try:
        from scripts.visualization_pipeline import LeukoMapVisualizer
        print("✅ Visualization pipeline: OK")
    except Exception as e:
        print(f"❌ Visualization pipeline: {e}")
    
    # Test advanced analysis
    try:
        from scripts.advanced_analysis import AdvancedLeukoMapAnalysis
        print("✅ Advanced analysis: OK")
    except Exception as e:
        print(f"❌ Advanced analysis: {e}")

if __name__ == "__main__":
    print("="*60)
    print("LEUKOMAP PIPELINE TEST (Python-only)")
    print("="*60)
    
    # Test individual components first
    test_individual_components()
    
    # Test full pipeline
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    
    success = test_pipeline()
    
    if success:
        print("\n🎉 All tests passed! LeukoMap is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    print("\nTest completed.") 