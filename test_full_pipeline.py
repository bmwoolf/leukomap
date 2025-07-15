#!/usr/bin/env python3
"""
End-to-end test of the LeukoMap pipeline using real data.

This script tests the complete pipeline from data loading through analysis
using the real 10x Genomics data in the data/raw directory.
"""

import os
import sys
import logging
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from leukomap import analyze
from leukomap.core import AnalysisConfig
from leukomap.data_loading import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_pipeline():
    """Test the complete LeukoMap pipeline with real data."""
    
    # Check if real data exists
    data_path = Path("data")
    if not data_path.exists():
        logger.error("Data directory not found. Please ensure the data/raw directory contains 10x Genomics data.")
        return False
    
    # Create a temporary directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Configure the pipeline for testing with real data
        config = AnalysisConfig(
            data_path=data_path,
            output_dir=temp_path,
            max_epochs=50,  # Reduced for testing
            batch_size=128,
            learning_rate=1e-3,
            n_latent=10,
            n_hidden=128,
            n_layers=2,
            dropout_rate=0.1,
            n_neighbors=15,
            resolution=0.5,
            min_genes=200,
            min_cells=3,
            max_genes=2500,
            annotation_methods=['celltypist'],
            confidence_threshold=0.7
        )
        
        logger.info("Starting full pipeline test with real data...")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Output directory: {temp_path}")
        
        try:
            # Load the real 10x data using DataLoader
            loader = DataLoader(config)
            adata = loader.process()
            
            # Run the complete analysis pipeline
            results = analyze(adata, output_dir=temp_path)
            
            # Verify results
            if results is None:
                logger.error("Analysis returned None")
                return False
            
            # Check that key outputs were generated
            expected_files = [
                "preprocessing_report.txt",
                "cell_type_annotation_summary.md", 
                "differential_expression_summary.md",
                "analysis_report.txt"
            ]
            
            for file_name in expected_files:
                file_path = temp_path / "summaries" / file_name
                if not file_path.exists():
                    logger.warning(f"Expected output file not found: {file_path}")
                else:
                    logger.info(f"✓ Generated: {file_name}")
            
            # Check for key data files
            expected_data_files = [
                "adata_processed.h5ad",
                "adata_integrated.h5ad",
                "adata_annotated.h5ad"
            ]
            
            for file_name in expected_data_files:
                file_path = temp_path / "data" / file_name
                if not file_path.exists():
                    logger.warning(f"Expected data file not found: {file_path}")
                else:
                    logger.info(f"✓ Generated: {file_name}")
            
            # Check for visualizations
            expected_viz_files = [
                "umap_clusters.png",
                "umap_celltype.png", 
                "volcano_plots.png"
            ]
            
            for file_name in expected_viz_files:
                file_path = temp_path / "figures" / file_name
                if not file_path.exists():
                    logger.warning(f"Expected visualization not found: {file_path}")
                else:
                    logger.info(f"✓ Generated: {file_name}")
            
            logger.info("✓ Full pipeline test completed successfully!")
            logger.info(f"Results saved to: {temp_path}")
            
            # Print summary statistics
            if hasattr(results, 'adata') and results.adata is not None:
                logger.info(f"Final dataset: {results.adata.n_obs} cells, {results.adata.n_vars} genes")
                if 'leiden' in results.adata.obs.columns:
                    n_clusters = results.adata.obs['leiden'].nunique()
                    logger.info(f"Identified {n_clusters} clusters")
                if 'cell_type' in results.adata.obs.columns:
                    n_cell_types = results.adata.obs['cell_type'].nunique()
                    logger.info(f"Annotated {n_cell_types} cell types")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1) 