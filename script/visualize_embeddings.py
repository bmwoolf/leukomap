#!/usr/bin/env python3
"""
Visualize scVI latent embeddings using UMAP.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_data(cache_dir="cache"):
    """Load the trained AnnData with latent embeddings."""
    adata_path = Path(cache_dir) / "adata_with_latent.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"Trained data not found: {adata_path}")
    
    logger.info(f"Loading trained data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    logger.info(f"Loaded data: {adata.shape}")
    logger.info(f"Available embeddings: {list(adata.obsm.keys())}")
    
    return adata

def compute_umap(adata, n_neighbors=15, min_dist=0.5, random_state=42):
    """Compute UMAP on scVI latent embeddings."""
    logger.info("Computing UMAP on scVI latent embeddings...")
    
    # Use scVI latent space for UMAP
    sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    
    logger.info("UMAP computation complete")
    return adata

def create_visualizations(adata, output_dir="cache"):
    """Create various UMAP visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up plotting style
    sc.settings.set_figure_params(dpi=100, frameon=False)
    sc.settings.verbosity = 1
    
    # 1. Sample type visualization
    logger.info("Creating sample type UMAP...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc.pl.umap(adata, color='sample_type', ax=ax, show=False, 
               title='UMAP: Sample Type', legend_loc='on data')
    plt.tight_layout()
    plt.savefig(output_path / "umap_sample_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cell type visualization (if available)
    if 'celltype' in adata.obs.columns:
        logger.info("Creating cell type UMAP...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sc.pl.umap(adata, color='celltype', ax=ax, show=False, 
                   title='UMAP: Cell Type', legend_loc='right margin')
        plt.tight_layout()
        plt.savefig(output_path / "umap_celltype.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Sample type with better colors
    logger.info("Creating enhanced sample type UMAP...")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sc.pl.umap(adata, color='sample_type', ax=ax, show=False, 
               title='UMAP: Pediatric Leukemia Samples', 
               palette=colors[:len(adata.obs['sample_type'].cat.categories)])
    plt.tight_layout()
    plt.savefig(output_path / "umap_sample_type_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Quality metrics
    logger.info("Creating quality metrics UMAP...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total counts
    sc.pl.umap(adata, color='total_counts', ax=axes[0,0], show=False, 
               title='Total Counts', use_raw=False)
    
    # Number of genes
    sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[0,1], show=False, 
               title='Number of Genes', use_raw=False)
    
    # Mitochondrial percentage
    if 'pct_counts_mt' in adata.obs.columns:
        sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1,0], show=False, 
                   title='Mitochondrial %', use_raw=False)
    
    # Sample
    sc.pl.umap(adata, color='sample', ax=axes[1,1], show=False, 
               title='Sample', legend_loc='right margin')
    
    plt.tight_layout()
    plt.savefig(output_path / "umap_quality_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to: {output_path}")

def create_summary_stats(adata, output_dir="cache"):
    """Create summary statistics of the dataset."""
    output_path = Path(output_dir)
    
    # Sample type distribution
    sample_stats = adata.obs['sample_type'].value_counts()
    
    # Cell type distribution (if available)
    celltype_stats = None
    if 'celltype' in adata.obs.columns:
        celltype_stats = adata.obs['celltype'].value_counts()
    
    # Quality metrics summary
    quality_stats = {
        'total_counts': adata.obs['total_counts'].describe(),
        'n_genes_by_counts': adata.obs['n_genes_by_counts'].describe(),
    }
    
    if 'pct_counts_mt' in adata.obs.columns:
        quality_stats['pct_counts_mt'] = adata.obs['pct_counts_mt'].describe()
    
    # Save summary
    with open(output_path / "dataset_summary.txt", 'w') as f:
        f.write("LeukoMap Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total cells: {adata.n_obs:,}\n")
        f.write(f"Total genes: {adata.n_vars:,}\n")
        f.write(f"Latent dimensions: {adata.obsm['X_scvi'].shape[1]}\n\n")
        
        f.write("Sample Type Distribution:\n")
        f.write("-" * 30 + "\n")
        for sample, count in sample_stats.items():
            f.write(f"{sample}: {count:,} cells ({count/adata.n_obs*100:.1f}%)\n")
        
        if celltype_stats is not None:
            f.write(f"\nCell Type Distribution:\n")
            f.write("-" * 30 + "\n")
            for celltype, count in celltype_stats.items():
                f.write(f"{celltype}: {count:,} cells ({count/adata.n_obs*100:.1f}%)\n")
        
        f.write(f"\nQuality Metrics Summary:\n")
        f.write("-" * 30 + "\n")
        for metric, stats in quality_stats.items():
            f.write(f"\n{metric}:\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Std: {stats['std']:.2f}\n")
            f.write(f"  Min: {stats['min']:.2f}\n")
            f.write(f"  Max: {stats['max']:.2f}\n")
    
    logger.info(f"Summary statistics saved to: {output_path / 'dataset_summary.txt'}")

def main():
    """Main function to run UMAP visualization."""
    try:
        # Load data
        adata = load_trained_data()
        
        # Compute UMAP
        adata = compute_umap(adata)
        
        # Create visualizations
        create_visualizations(adata)
        
        # Create summary statistics
        create_summary_stats(adata)
        
        # Save AnnData with UMAP
        adata.write("cache/adata_with_umap.h5ad")
        logger.info("Saved AnnData with UMAP to: cache/adata_with_umap.h5ad")
        
        logger.info("UMAP visualization complete!")
        
    except Exception as e:
        logger.error(f"Error in UMAP visualization: {e}")
        raise

if __name__ == "__main__":
    main() 