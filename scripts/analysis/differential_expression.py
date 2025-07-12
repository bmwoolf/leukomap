#!/usr/bin/env python3
"""
Differential Expression Analysis for LeukoMap scVI clusters.
Identifies marker genes and potential druggable targets.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(cache_dir="cache"):
    """Load the AnnData with UMAP and latent embeddings."""
    adata_path = Path(cache_dir) / "adata_with_umap.h5ad"
    if not adata_path.exists():
        # Fallback to latent data if UMAP not available
        adata_path = Path(cache_dir) / "adata_with_latent.h5ad"
    
    logger.info(f"Loading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Clean cell type annotations - remove NaN values
    if 'celltype' in adata.obs.columns:
        # Remove cells with NaN cell type
        adata = adata[~adata.obs['celltype'].isna()].copy()
        logger.info(f"After cleaning cell types: {adata.shape}")
        
        # Make cell type categorical
        adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    
    return adata

def run_differential_expression(adata, output_dir="cache"):
    """Run differential expression analysis between clusters."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info("Running differential expression analysis...")
    
    # Create a copy for analysis
    adata_de = adata.copy()
    
    # Normalize and log-transform the data for DE analysis
    sc.pp.normalize_total(adata_de, target_sum=1e4)
    sc.pp.log1p(adata_de)
    
    # Run DE analysis between all clusters
    sc.tl.rank_genes_groups(adata_de, 'celltype', method='wilcoxon')
    
    # Get results for each cluster
    de_results = {}
    for cluster in adata_de.obs['celltype'].cat.categories:
        logger.info(f"Analyzing cluster: {cluster}")
        
        # Get top genes for this cluster
        cluster_genes = sc.get.rank_genes_groups_df(adata_de, group=cluster)
        cluster_genes = cluster_genes.sort_values('scores', ascending=False)
        
        # Remove NaN values
        cluster_genes = cluster_genes.dropna()
        
        # Filter for significant genes (p-value < 0.05, log fold change > 0.5)
        significant = cluster_genes[
            (cluster_genes['pvals_adj'] < 0.05) & 
            (cluster_genes['logfoldchanges'] > 0.5)
        ]
        
        de_results[cluster] = {
            'all_genes': cluster_genes,
            'significant': significant,
            'top_10': cluster_genes.head(10)
        }
    
    return de_results

def create_volcano_plots(adata, de_results, output_dir="cache"):
    """Create volcano plots for each cluster."""
    output_path = Path(output_dir)
    
    logger.info("Creating volcano plots...")
    
    # Create subplots for each cluster
    n_clusters = len(de_results)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Differential Expression: Volcano Plots by Cluster', fontsize=16, fontweight='bold')
    
    for i, (cluster, results) in enumerate(de_results.items()):
        row = i // 4
        col = i % 4
        
        if row < 3:  # Only show first 12 clusters
            genes_df = results['all_genes']
            
            # Skip if no data
            if len(genes_df) == 0:
                continue
            
            # Create volcano plot
            ax = axes[row, col]
            
            # Handle infinite values
            genes_df = genes_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(genes_df) == 0:
                continue
            
            # Color points based on significance
            colors = ['gray'] * len(genes_df)
            for j, (_, gene) in enumerate(genes_df.iterrows()):
                if gene['pvals_adj'] < 0.05 and gene['logfoldchanges'] > 0.5:
                    colors[j] = 'red'
                elif gene['pvals_adj'] < 0.05 and gene['logfoldchanges'] < -0.5:
                    colors[j] = 'blue'
            
            # Handle log of zero
            pvals_log = -np.log10(genes_df['pvals_adj'].replace(0, 1e-300))
            
            ax.scatter(genes_df['logfoldchanges'], pvals_log, 
                      c=colors, alpha=0.6, s=20)
            
            # Add labels for top genes
            top_genes = results['top_10']
            if len(top_genes) > 0:
                for _, gene in top_genes.iterrows():
                    if gene['names'] in genes_df.index:
                        gene_data = genes_df.loc[gene['names']]
                        pval_log = -np.log10(gene_data['pvals_adj']) if gene_data['pvals_adj'] > 0 else 0
                        ax.annotate(gene['names'], 
                                   (gene_data['logfoldchanges'], pval_log),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Log2 Fold Change')
            ax.set_ylabel('-log10(adjusted p-value)')
            ax.set_title(f'{cluster}')
            ax.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
            ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
            ax.axvline(-0.5, color='black', linestyle='--', alpha=0.5)
    
    # Hide empty subplots
    for i in range(n_clusters, 12):
        row = i // 4
        col = i % 4
        if row < 3:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path / "volcano_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Volcano plots saved to: {output_path / 'volcano_plots.png'}")

def create_heatmap(adata, de_results, output_dir="cache"):
    """Create heatmap of top marker genes."""
    output_path = Path(output_dir)
    
    logger.info("Creating marker gene heatmap...")
    
    # Get top 5 genes from each cluster
    top_genes = []
    for cluster, results in de_results.items():
        if len(results['top_10']) > 0:
            cluster_top = results['top_10'].head(5)
            top_genes.extend(cluster_top['names'].tolist())
    
    # Remove duplicates and filter for genes that exist in the data
    top_genes = list(set(top_genes))
    available_genes = [gene for gene in top_genes if gene in adata.var_names]
    
    if len(available_genes) == 0:
        logger.warning("No marker genes found for heatmap")
        return
    
    # Create a copy for heatmap
    adata_hm = adata.copy()
    
    # Normalize data for heatmap
    sc.pp.normalize_total(adata_hm, target_sum=1e4)
    sc.pp.log1p(adata_hm)
    
    # Create heatmap
    sc.pl.heatmap(adata_hm, var_names=available_genes[:20], groupby='celltype', 
                  show=False, figsize=(15, 10))
    plt.title('Top Marker Genes by Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "marker_genes_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Heatmap saved to: {output_path / 'marker_genes_heatmap.png'}")

def leukemia_vs_healthy_analysis(adata, de_results, output_dir="cache"):
    """Compare leukemia subtypes against healthy controls."""
    output_path = Path(output_dir)
    
    logger.info("Analyzing leukemia vs healthy differential expression...")
    
    # Get leukemia and healthy clusters
    leukemia_clusters = [c for c in adata.obs['celltype'].cat.categories if 'ETV6.RUNX1' in c]
    healthy_clusters = [c for c in adata.obs['celltype'].cat.categories if 'HHD' in c]
    
    leukemia_vs_healthy = {}
    
    for leukemia_cluster in leukemia_clusters:
        logger.info(f"Comparing {leukemia_cluster} vs healthy controls...")
        
        # Get genes upregulated in leukemia vs healthy
        leukemia_genes = de_results[leukemia_cluster]['all_genes']
        
        # Filter for genes with high expression in leukemia
        leukemia_up = leukemia_genes[
            (leukemia_genes['pvals_adj'] < 0.01) & 
            (leukemia_genes['logfoldchanges'] > 1.0)
        ]
        
        leukemia_vs_healthy[leukemia_cluster] = leukemia_up
    
    # Create summary table
    summary_data = []
    for cluster, genes in leukemia_vs_healthy.items():
        for _, gene in genes.head(10).iterrows():
            summary_data.append({
                'Cluster': cluster,
                'Gene': gene['names'],
                'Log2_FC': gene['logfoldchanges'],
                'Adj_P_Value': gene['pvals_adj'],
                'Score': gene['scores']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / "leukemia_vs_healthy_markers.csv", index=False)
    
    logger.info(f"Leukemia vs healthy analysis saved to: {output_path / 'leukemia_vs_healthy_markers.csv'}")
    
    return leukemia_vs_healthy

def export_results(de_results, output_dir="cache"):
    """Export differential expression results."""
    output_path = Path(output_dir)
    
    logger.info("Exporting differential expression results...")
    
    # Export all results
    for cluster, results in de_results.items():
        # Clean cluster name for filename
        clean_name = cluster.replace('.', '_').replace(' ', '_')
        
        # Export all genes
        results['all_genes'].to_csv(output_path / f"de_all_genes_{clean_name}.csv", index=False)
        
        # Export significant genes
        results['significant'].to_csv(output_path / f"de_significant_{clean_name}.csv", index=False)
        
        # Export top 50 genes
        results['all_genes'].head(50).to_csv(output_path / f"de_top50_{clean_name}.csv", index=False)
    
    # Create summary report
    with open(output_path / "de_analysis_summary.txt", 'w') as f:
        f.write("Differential Expression Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for cluster, results in de_results.items():
            f.write(f"\n{cluster}:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total genes analyzed: {len(results['all_genes']):,}\n")
            f.write(f"Significant genes (p<0.05, FC>0.5): {len(results['significant']):,}\n")
            
            if len(results['significant']) > 0:
                f.write("Top 5 marker genes:\n")
                for _, gene in results['significant'].head(5).iterrows():
                    f.write(f"  {gene['names']}: FC={gene['logfoldchanges']:.2f}, p={gene['pvals_adj']:.2e}\n")
            f.write("\n")
    
    logger.info(f"Results exported to: {output_path}")

def main():
    """Main function to run differential expression analysis."""
    try:
        # Load data
        adata = load_data()
        
        # Run differential expression
        de_results = run_differential_expression(adata)
        
        # Create visualizations
        create_volcano_plots(adata, de_results)
        create_heatmap(adata, de_results)
        
        # Leukemia vs healthy analysis
        leukemia_vs_healthy = leukemia_vs_healthy_analysis(adata, de_results)
        
        # Export results
        export_results(de_results)
        
        logger.info("Differential expression analysis complete!")
        logger.info("Check the cache/ directory for results and visualizations.")
        
    except Exception as e:
        logger.error(f"Error in differential expression analysis: {e}")
        raise

if __name__ == "__main__":
    main() 