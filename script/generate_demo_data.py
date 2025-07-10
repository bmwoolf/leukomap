#!/usr/bin/env python3
"""
Generate synthetic scRNA-seq data for testing LeukoMap pipeline.
This creates a small dataset with the expected structure for training.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_cells=1000, n_genes=2000, output_dir="data"):
    """
    Generate synthetic scRNA-seq data for testing.
    
    Parameters
    ----------
    n_cells : int
        Number of cells to generate
    n_genes : int
        Number of genes to generate
    output_dir : str
        Output directory for the data
    """
    
    logger.info(f"Generating synthetic data: {n_cells} cells, {n_genes} genes")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    # Generate cell types and sample types
    cell_types = ["B_cell", "T_cell", "Myeloid", "Stem_cell", "Unknown"]
    sample_types = ["ETV6-RUNX1", "HHD", "PRE-T", "PBMMC"]
    
    # Create cell metadata
    np.random.seed(42)
    cell_metadata = pd.DataFrame({
        'cell_type': np.random.choice(cell_types, n_cells, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'sample_type': np.random.choice(sample_types, n_cells, p=[0.3, 0.25, 0.25, 0.2]),
        'n_genes_by_counts': np.random.poisson(2000, n_cells),
        'total_counts': np.random.poisson(10000, n_cells),
        'pct_counts_mt': np.random.beta(2, 20, n_cells) * 0.1
    })
    
    # Generate expression matrix with some structure
    # Create different expression patterns for different cell types
    expression_matrix = np.zeros((n_cells, n_genes))
    
    for i, cell_type in enumerate(cell_metadata['cell_type']):
        # Base expression level
        base_expr = np.random.negative_binomial(5, 0.1, n_genes)
        
        # Add cell type specific patterns
        if cell_type == "B_cell":
            # B cell specific genes (first 100 genes)
            base_expr[:100] += np.random.negative_binomial(10, 0.2, 100)
        elif cell_type == "T_cell":
            # T cell specific genes (genes 100-200)
            base_expr[100:200] += np.random.negative_binomial(10, 0.2, 100)
        elif cell_type == "Myeloid":
            # Myeloid specific genes (genes 200-300)
            base_expr[200:300] += np.random.negative_binomial(10, 0.2, 100)
        elif cell_type == "Stem_cell":
            # Stem cell specific genes (genes 300-400)
            base_expr[300:400] += np.random.negative_binomial(10, 0.2, 100)
        
        expression_matrix[i] = base_expr
    
    # Create AnnData object
    adata = ad.AnnData(
        X=expression_matrix,
        obs=cell_metadata,
        var=pd.DataFrame(index=gene_names)
    )
    
    # Add some basic preprocessing
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Save as h5ad file
    output_file = output_path / "data.h5ad"
    adata.write(output_file)
    
    logger.info(f"Saved synthetic data to: {output_file}")
    logger.info(f"Data shape: {adata.shape}")
    logger.info(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
    logger.info(f"Sample types: {adata.obs['sample_type'].value_counts().to_dict()}")
    
    return adata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic scRNA-seq data")
    parser.add_argument("--n_cells", type=int, default=1000, help="Number of cells")
    parser.add_argument("--n_genes", type=int, default=2000, help="Number of genes")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    
    args = parser.parse_args()
    
    generate_synthetic_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        output_dir=args.output_dir
    ) 