#!/usr/bin/env python3
"""
Resume Healthy Reference Integration from Cached Model.

This script resumes the integration process from a cached scVI model
to avoid retraining and complete the analysis efficiently.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, sparse
from sklearn.metrics import silhouette_score
import torch

# Import scVI modules
try:
    import scvi
    from scvi.model import SCVI, SCANVI
    from scvi.data import AnnDataManager
except ImportError:
    raise ImportError("scVI is not installed. Please install with: pip install scvi-tools")

# Import CellTypist for cell type annotation
try:
    import celltypist
    from celltypist import models
    from celltypist.annotate import annotate
except ImportError:
    print("Warning: CellTypist not installed. Cell type annotation will be skipped.")
    celltypist = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
class Config:
    """Configuration parameters for healthy reference integration."""
    
    # Data paths
    CACHE_DIR = Path("cache")
    OUTPUT_DIR = Path("outputs")
    MODEL_DIR = Path("cache/scvi_model")
    
    # Sample prefixes
    HEALTHY_PREFIX = "data-raw-PBMMC"
    LEUKEMIA_PREFIXES = [
        "data-raw-ETV6-RUNX1",
        "data-raw-PRE-T", 
        "data-raw-HHD"
    ]
    
    # Preprocessing parameters
    MIN_GENES = 200
    MIN_CELLS = 3
    N_HVG = 2000
    TARGET_SUM = 1e4
    
    # scVI parameters
    N_LATENT = 20
    N_LAYERS = 2
    N_HIDDEN = 128
    DROPOUT_RATE = 0.1
    MAX_EPOCHS = 400
    LEARNING_RATE = 0.001
    
    # Clustering parameters
    RESOLUTION = 0.5
    MIN_CLUSTER_SIZE = 50
    
    # Differential expression parameters
    MIN_FC = 0.5
    MAX_PVALUE = 0.05
    
    # CellTypist parameters
    CELLTYPIST_MODELS = ["Immune_All_High.pkl", "Immune_All_Low.pkl"]
    
    # Output file names
    INTEGRATED_DATA_FILE = "adata_integrated_healthy.h5ad"
    ABNORMAL_CLUSTERS_FILE = "abnormal_clusters_analysis.csv"
    HEALTHY_REFERENCE_FILE = "healthy_reference_summary.csv"
    INTEGRATION_REPORT_FILE = "healthy_integration_report.txt"


def setup_environment():
    """Set up the analysis environment."""
    logger.info("Setting up analysis environment...")
    
    # Create output directories
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Set scanpy settings
    sc.settings.verbosity = 3
    
    # Set random seeds
    np.random.seed(42)
    sc.settings.seed = 42
    if torch.cuda.is_available():
        torch.manual_seed(42)
    
    logger.info("Environment setup complete")


def load_cached_integrated_data() -> Tuple[ad.AnnData, SCVI]:
    """
    Load cached integrated data and scVI model.
    
    Returns:
        Tuple of (AnnData, SCVI model)
    """
    logger.info("Loading cached integrated data...")
    
    # Check if integrated data exists
    integrated_file = Config.CACHE_DIR / Config.INTEGRATED_DATA_FILE
    if not integrated_file.exists():
        raise FileNotFoundError(f"Integrated data not found: {integrated_file}")
    
    # Load integrated data
    adata = sc.read_h5ad(integrated_file)
    logger.info(f"Loaded integrated data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Load scVI model
    if not Config.MODEL_DIR.exists():
        raise FileNotFoundError(f"scVI model directory not found: {Config.MODEL_DIR}")
    
    # Setup scVI data
    scvi.data.setup_anndata(adata, batch_key="sample_id")
    
    # Load model
    model = SCVI.load(Config.MODEL_DIR, adata=adata)
    logger.info("Loaded cached scVI model")
    
    return adata, model


def perform_clustering_and_analysis(adata: ad.AnnData, model: SCVI) -> ad.AnnData:
    """
    Perform clustering and downstream analysis on integrated data.
    
    Args:
        adata: Integrated AnnData object
        model: Trained scVI model
    
    Returns:
        AnnData with clustering results
    """
    logger.info("Performing clustering and analysis...")
    
    # Get latent representation
    adata.obsm["X_scVI"] = model.get_latent_representation()
    
    # Compute neighbors
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
    
    # Perform clustering
    logger.info("Running Leiden clustering...")
    sc.tl.leiden(adata, resolution=Config.RESOLUTION)
    
    # Compute UMAP
    logger.info("Computing UMAP...")
    sc.tl.umap(adata, min_dist=0.1, spread=1.0)
    
    # Compute quality metrics
    logger.info("Computing quality metrics...")
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    return adata


def annotate_cell_types(adata: ad.AnnData) -> ad.AnnData:
    """
    Annotate cell types using CellTypist.
    
    Args:
        adata: AnnData object
    
    Returns:
        AnnData with cell type annotations
    """
    if celltypist is None:
        logger.warning("CellTypist not available, skipping cell type annotation")
        return adata
    
    logger.info("Annotating cell types with CellTypist...")
    
    try:
        # Use the high-confidence model
        predictions = annotate(adata, model='Immune_All_High.pkl', majority_voting=True)
        adata.obs['cell_type'] = predictions.predicted_labels.majority_voting
        adata.obs['cell_type_confidence'] = predictions.predicted_labels.majority_voting_score
        
        logger.info(f"Cell types annotated: {adata.obs['cell_type'].nunique()} unique types")
        
    except Exception as e:
        logger.warning(f"Cell type annotation failed: {e}")
        adata.obs['cell_type'] = 'Unknown'
        adata.obs['cell_type_confidence'] = 0.0
    
    return adata


def detect_abnormal_clusters(adata: ad.AnnData) -> pd.DataFrame:
    """
    Detect abnormal clusters by comparing to healthy reference.
    
    Args:
        adata: AnnData object with clustering results
    
    Returns:
        DataFrame with abnormal cluster analysis
    """
    logger.info("Detecting abnormal clusters...")
    
    # Calculate cluster composition
    cluster_comp = pd.crosstab(adata.obs['leiden'], adata.obs['health_status'])
    
    # Calculate percentage of leukemia cells in each cluster
    cluster_comp_pct = cluster_comp.div(cluster_comp.sum(axis=1), axis=0) * 100
    
    # Identify clusters with high leukemia percentage
    leukemia_threshold = 50  # Clusters with >50% leukemia cells
    abnormal_clusters = cluster_comp_pct[cluster_comp_pct['leukemia'] > leukemia_threshold]
    
    # Calculate additional metrics
    abnormal_analysis = []
    for cluster in abnormal_clusters.index:
        cluster_cells = adata[adata.obs['leiden'] == cluster]
        
        # Calculate metrics
        total_cells = len(cluster_cells)
        leukemia_cells = len(cluster_cells[cluster_cells.obs['health_status'] == 'leukemia'])
        leukemia_pct = (leukemia_cells / total_cells) * 100
        
        # Calculate average expression of top genes
        if 'cell_type' in cluster_cells.obs.columns:
            cell_type = cluster_cells.obs['cell_type'].mode().iloc[0] if len(cluster_cells.obs['cell_type'].mode()) > 0 else 'Unknown'
        else:
            cell_type = 'Unknown'
        
        abnormal_analysis.append({
            'cluster': cluster,
            'total_cells': total_cells,
            'leukemia_cells': leukemia_cells,
            'leukemia_percentage': leukemia_pct,
            'cell_type': cell_type,
            'abnormality_score': leukemia_pct / 100.0
        })
    
    abnormal_df = pd.DataFrame(abnormal_analysis)
    abnormal_df = abnormal_df.sort_values('abnormality_score', ascending=False)
    
    logger.info(f"Found {len(abnormal_df)} abnormal clusters")
    
    return abnormal_df


def perform_differential_expression(adata: ad.AnnData, abnormal_clusters: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform differential expression analysis for abnormal clusters.
    
    Args:
        adata: AnnData object
        abnormal_clusters: DataFrame with abnormal cluster info
    
    Returns:
        Dictionary of differential expression results
    """
    logger.info("Performing differential expression analysis...")
    
    de_results = {}
    
    for _, cluster_info in abnormal_clusters.iterrows():
        cluster = cluster_info['cluster']
        logger.info(f"Analyzing cluster {cluster}...")
        
        # Subset to cluster cells
        cluster_cells = adata[adata.obs['leiden'] == cluster]
        
        # Compare leukemia vs healthy within cluster
        if len(cluster_cells[cluster_cells.obs['health_status'] == 'healthy']) > 10:
            sc.tl.rank_genes_groups(cluster_cells, 'health_status', method='wilcoxon')
            
            # Extract results
            de_df = sc.get.rank_genes_groups_df(cluster_cells, group='leukemia')
            de_df = de_df[
                (de_df['logfoldchanges'].abs() > Config.MIN_FC) & 
                (de_df['pvals_adj'] < Config.MAX_PVALUE)
            ].sort_values('scores', ascending=False)
            
            de_results[f'cluster_{cluster}'] = de_df
            logger.info(f"Cluster {cluster}: {len(de_df)} significant genes")
        else:
            logger.warning(f"Cluster {cluster}: insufficient healthy cells for comparison")
    
    return de_results


def generate_visualizations(adata: ad.AnnData, abnormal_clusters: pd.DataFrame):
    """
    Generate comprehensive visualizations.
    
    Args:
        adata: AnnData object
        abnormal_clusters: DataFrame with abnormal cluster info
    """
    logger.info("Generating visualizations...")
    
    # Set up plotting
    sc.settings.set_figure_params(dpi=100, facecolor='white')
    
    # 1. UMAP colored by cluster
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc.pl.umap(adata, color='leiden', ax=ax, show=False, title='Clusters')
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'umap_clusters_healthy_integration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP colored by health status
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc.pl.umap(adata, color='health_status', ax=ax, show=False, title='Health Status')
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'umap_health_status_healthy_integration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. UMAP colored by sample type
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc.pl.umap(adata, color='sample_type', ax=ax, show=False, title='Sample Type')
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'umap_sample_type_healthy_integration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cell type annotation (if available)
    if 'cell_type' in adata.obs.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sc.pl.umap(adata, color='cell_type', ax=ax, show=False, title='Cell Type Annotation')
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'umap_celltype_healthy_integration.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Abnormal cluster analysis
    if len(abnormal_clusters) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Leukemia percentage per cluster
        abnormal_clusters.plot(x='cluster', y='leukemia_percentage', kind='bar', ax=ax1)
        ax1.set_title('Leukemia Percentage by Cluster')
        ax1.set_ylabel('Leukemia Cells (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Cell counts
        abnormal_clusters.plot(x='cluster', y='total_cells', kind='bar', ax=ax2)
        ax2.set_title('Total Cells per Cluster')
        ax2.set_ylabel('Cell Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'abnormal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Visualizations saved to outputs/")


def generate_report(adata: ad.AnnData, abnormal_clusters: pd.DataFrame, de_results: Dict[str, pd.DataFrame]):
    """
    Generate comprehensive analysis report.
    
    Args:
        adata: AnnData object
        abnormal_clusters: DataFrame with abnormal cluster info
        de_results: Dictionary of differential expression results
    """
    logger.info("Generating analysis report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HEALTHY REFERENCE INTEGRATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Dataset summary
    report_lines.append("DATASET SUMMARY:")
    report_lines.append(f"Total cells: {adata.n_obs:,}")
    report_lines.append(f"Total genes: {adata.n_vars:,}")
    report_lines.append(f"Number of clusters: {adata.obs['leiden'].nunique()}")
    report_lines.append("")
    
    # Health status distribution
    health_dist = adata.obs['health_status'].value_counts()
    report_lines.append("HEALTH STATUS DISTRIBUTION:")
    for status, count in health_dist.items():
        pct = (count / len(adata)) * 100
        report_lines.append(f"  {status}: {count:,} cells ({pct:.1f}%)")
    report_lines.append("")
    
    # Sample type distribution
    sample_dist = adata.obs['sample_type'].value_counts()
    report_lines.append("SAMPLE TYPE DISTRIBUTION:")
    for sample_type, count in sample_dist.items():
        pct = (count / len(adata)) * 100
        report_lines.append(f"  {sample_type}: {count:,} cells ({pct:.1f}%)")
    report_lines.append("")
    
    # Abnormal clusters
    report_lines.append("ABNORMAL CLUSTER ANALYSIS:")
    if len(abnormal_clusters) > 0:
        report_lines.append(f"Found {len(abnormal_clusters)} abnormal clusters:")
        for _, cluster_info in abnormal_clusters.iterrows():
            report_lines.append(f"  Cluster {cluster_info['cluster']}:")
            report_lines.append(f"    Total cells: {cluster_info['total_cells']:,}")
            report_lines.append(f"    Leukemia cells: {cluster_info['leukemia_cells']:,}")
            report_lines.append(f"    Leukemia percentage: {cluster_info['leukemia_percentage']:.1f}%")
            report_lines.append(f"    Cell type: {cluster_info['cell_type']}")
            report_lines.append(f"    Abnormality score: {cluster_info['abnormality_score']:.3f}")
            report_lines.append("")
    else:
        report_lines.append("No abnormal clusters detected.")
    report_lines.append("")
    
    # Differential expression summary
    report_lines.append("DIFFERENTIAL EXPRESSION SUMMARY:")
    for cluster_name, de_df in de_results.items():
        cluster = cluster_name.replace('cluster_', '')
        report_lines.append(f"  Cluster {cluster}: {len(de_df)} significant genes")
        if len(de_df) > 0:
            top_genes = de_df.head(5)['names'].tolist()
            report_lines.append(f"    Top genes: {', '.join(top_genes)}")
    report_lines.append("")
    
    # Quality metrics
    report_lines.append("QUALITY METRICS:")
    if 'n_genes_by_counts' in adata.obs.columns:
        report_lines.append(f"  Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.1f}")
        report_lines.append(f"  Median genes per cell: {adata.obs['n_genes_by_counts'].median():.1f}")
    if 'total_counts' in adata.obs.columns:
        report_lines.append(f"  Mean UMIs per cell: {adata.obs['total_counts'].mean():.1f}")
        report_lines.append(f"  Median UMIs per cell: {adata.obs['total_counts'].median():.1f}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("Analysis completed successfully!")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = Config.OUTPUT_DIR / Config.INTEGRATION_REPORT_FILE
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {report_path}")


def save_results(adata: ad.AnnData, abnormal_clusters: pd.DataFrame, de_results: Dict[str, pd.DataFrame]):
    """
    Save all analysis results.
    
    Args:
        adata: AnnData object
        abnormal_clusters: DataFrame with abnormal cluster info
        de_results: Dictionary of differential expression results
    """
    logger.info("Saving results...")
    
    # Save integrated data
    adata.write(Config.OUTPUT_DIR / Config.INTEGRATED_DATA_FILE)
    logger.info(f"Integrated data saved to {Config.OUTPUT_DIR / Config.INTEGRATED_DATA_FILE}")
    
    # Save abnormal clusters analysis
    abnormal_clusters.to_csv(Config.OUTPUT_DIR / Config.ABNORMAL_CLUSTERS_FILE, index=False)
    logger.info(f"Abnormal clusters analysis saved to {Config.OUTPUT_DIR / Config.ABNORMAL_CLUSTERS_FILE}")
    
    # Save differential expression results
    for cluster_name, de_df in de_results.items():
        cluster = cluster_name.replace('cluster_', '')
        de_file = Config.OUTPUT_DIR / f"de_cluster_{cluster}_vs_healthy.csv"
        de_df.to_csv(de_file, index=False)
        logger.info(f"Differential expression for cluster {cluster} saved to {de_file}")
    
    # Save healthy reference summary
    healthy_cells = adata[adata.obs['health_status'] == 'healthy']
    healthy_summary = {
        'total_healthy_cells': len(healthy_cells),
        'healthy_clusters': healthy_cells.obs['leiden'].nunique(),
        'healthy_cell_types': healthy_cells.obs['cell_type'].nunique() if 'cell_type' in healthy_cells.obs.columns else 0,
        'mean_genes_per_healthy_cell': healthy_cells.obs['n_genes_by_counts'].mean() if 'n_genes_by_counts' in healthy_cells.obs.columns else 0
    }
    healthy_df = pd.DataFrame([healthy_summary])
    healthy_df.to_csv(Config.OUTPUT_DIR / Config.HEALTHY_REFERENCE_FILE, index=False)
    logger.info(f"Healthy reference summary saved to {Config.OUTPUT_DIR / Config.HEALTHY_REFERENCE_FILE}")


def main():
    """Main analysis pipeline."""
    try:
        logger.info("Starting healthy reference integration analysis (resume mode)...")
        
        # Setup environment
        setup_environment()
        
        # Load cached data and model
        adata, model = load_cached_integrated_data()
        
        # Perform clustering and analysis
        adata = perform_clustering_and_analysis(adata, model)
        
        # Annotate cell types
        adata = annotate_cell_types(adata)
        
        # Detect abnormal clusters
        abnormal_clusters = detect_abnormal_clusters(adata)
        
        # Perform differential expression
        de_results = perform_differential_expression(adata, abnormal_clusters)
        
        # Generate visualizations
        generate_visualizations(adata, abnormal_clusters)
        
        # Generate report
        generate_report(adata, abnormal_clusters, de_results)
        
        # Save results
        save_results(adata, abnormal_clusters, de_results)
        
        logger.info("Healthy reference integration analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 