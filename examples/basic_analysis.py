#!/usr/bin/env python3
"""
LeukoMap End-to-End Analysis Example.

This example demonstrates the complete LeukoMap pipeline using the main analyze() function.
"""

import sys
import logging
from pathlib import Path
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse

# Import LeukoMap
import leukomap
from leukomap import analyze, load_data, preprocess, train_scvi, embed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")

def main():
    """Run the complete LeukoMap analysis example."""
    print("=" * 60)
    print("LeukoMap End-to-End Analysis Example")
    print("=" * 60)
    
    # Step 1: Check for existing data
    data_path = Path("cache/adata_preprocessed.h5ad")
    
    if data_path.exists():
        logger.info(f"✓ Found existing data: {data_path}")
        adata = sc.read_h5ad(data_path)
        logger.info(f"✓ Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        logger.warning("No existing data found. Creating mock data for demonstration...")
        adata = create_mock_data_for_demo()
        logger.info(f"✓ Created mock data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Step 2: Run complete analysis using the main analyze function
    logger.info("Running complete LeukoMap analysis...")
    
    try:
        results = analyze(
            scRNA_seq_data=adata,
            output_dir="results/end_to_end_example",
            max_epochs=50,  # Reduced for example
            n_latent=10,
            resolution=0.5
        )
        
        logger.info("✓ Analysis completed successfully!")
        
        # Display results summary
        print("\n" + "=" * 40)
        print("ANALYSIS RESULTS SUMMARY")
        print("=" * 40)
        
        if results['annotated_clusters'] is not None:
            adata_result = results['annotated_clusters']
            print(f"✓ Annotated clusters: {adata_result.n_obs} cells")
            if 'celltypist_cell_type' in adata_result.obs.columns:
                cell_types = adata_result.obs['celltypist_cell_type'].unique()
                print(f"✓ Cell types identified: {len(cell_types)}")
                print(f"  - {', '.join(cell_types[:5])}{'...' if len(cell_types) > 5 else ''}")
        
        if results['druggable_targets']:
            print(f"✓ Drug targets identified: {len(results['druggable_targets'])} clusters")
        
        if results['differential_expression']:
            print(f"✓ Differential expression: {len(results['differential_expression'])} clusters analyzed")
        
        if results['analysis_report']:
            print(f"✓ Analysis report: {results['analysis_report']}")
        
        print("\n" + "=" * 40)
        print("EXAMPLE COMPLETED SUCCESSFULLY")
        print("=" * 40)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

def create_mock_data_for_demo() -> ad.AnnData:
    """Create mock single-cell data for demonstration when real data is unavailable."""
    logger.info("Creating mock data for demonstration...")
    
    # Create realistic mock data
    n_cells, n_genes = 1000, 2000
    
    # Generate sparse count matrix with realistic distribution
    np.random.seed(42)
    X = scipy.sparse.csr_matrix(np.random.negative_binomial(2, 0.1, (n_cells, n_genes)))
    
    # Create cell metadata
    sample_types = ['ETV6-RUNX1', 'HHD', 'PRE-T', 'PBMMC']
    samples = np.random.choice(sample_types, n_cells, p=[0.3, 0.2, 0.2, 0.3])
    
    obs = pd.DataFrame({
        'sample': [f"{sample}_{i}" for i, sample in enumerate(samples)],
        'sample_type': samples,
        'n_genes_by_counts': np.random.poisson(2000, n_cells),
        'total_counts': np.random.poisson(10000, n_cells),
        'pct_counts_mt': np.random.beta(2, 20, n_cells) * 10
    }, index=[f"cell_{i}" for i in range(n_cells)])
    
    # Create gene metadata
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    var = pd.DataFrame({
        'gene_ids': [f"ENSG{i:06d}" for i in range(n_genes)],
        'feature_types': ['Gene Expression'] * n_genes
    }, index=gene_names)
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Add some basic QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    logger.info(f"Created mock data: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata

def demonstrate_individual_functions():
    """Demonstrate individual LeukoMap functions."""
    print("\n" + "=" * 40)
    print("INDIVIDUAL FUNCTION DEMONSTRATION")
    print("=" * 40)
    
    # Create mock data
    adata = create_mock_data_for_demo()
    
    # Demonstrate load_data function
    print("1. Data loading (mock data created)")
    
    # Demonstrate preprocess function
    print("2. Preprocessing...")
    adata_processed = preprocess(adata)
    print(f"   ✓ Preprocessed: {adata_processed.n_obs} cells, {adata_processed.n_vars} genes")
    
    # Demonstrate train_scvi function
    print("3. scVI training...")
    try:
        scvi_model = train_scvi(adata_processed)
        print("   ✓ scVI model trained")
        
        # Demonstrate embed function
        print("4. Generating embeddings...")
        adata_embedded = embed(scvi_model, adata_processed)
        print("   ✓ Embeddings generated")
        
    except Exception as e:
        print(f"   ⚠ scVI training failed (likely missing dependencies): {e}")
    
    print("\n" + "=" * 40)
    print("INDIVIDUAL FUNCTION DEMO COMPLETED")
    print("=" * 40)

if __name__ == "__main__":
    # Run the main example
    main()
    
    # Demonstrate individual functions
    demonstrate_individual_functions()
    
    print("\nThis example demonstrates:")
    print("✓ Complete end-to-end analysis using analyze() function")
    print("✓ Individual function usage (load_data, preprocess, train_scvi, embed)")
    print("✓ Error handling and fallback mechanisms")
    print("✓ Real data loading with mock data fallback")
    print("✓ Result summarization and reporting") 