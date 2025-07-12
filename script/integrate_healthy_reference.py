#!/usr/bin/env python3
"""
Enhanced Healthy Reference Integration for LeukoMap.

This script implements comprehensive integration of healthy bone marrow reference data
to detect abnormal clusters in leukemia samples. Based on the approach from:
https://github.com/CBC-UCONN/Single-Cell-Transcriptomics

Features:
- Load and preprocess healthy PBMMC reference data
- Integrate with leukemia samples using scVI/scANVI
- Perform cell type annotation using CellTypist
- Detect abnormal clusters by comparing to healthy reference
- Generate comprehensive visualizations and reports
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


def load_healthy_reference_data() -> ad.AnnData:
    """
    Load and preprocess healthy PBMMC reference data.
    
    Returns:
        AnnData object with healthy reference data
    """
    logger.info("Loading healthy reference data...")
    
    healthy_files = sorted(Config.CACHE_DIR.glob(f"{Config.HEALTHY_PREFIX}*"))
    
    if not healthy_files:
        raise FileNotFoundError(f"No healthy reference files found with prefix: {Config.HEALTHY_PREFIX}")
    
    logger.info(f"Found {len(healthy_files)} healthy reference files")
    
    healthy_adatas = []
    for file_path in healthy_files:
        logger.info(f"Loading: {file_path.name}")
        adata = sc.read_h5ad(file_path)
        adata.obs['sample_id'] = file_path.stem
        adata.obs['sample_type'] = 'healthy'
        adata.obs['health_status'] = 'healthy'
        healthy_adatas.append(adata)
    
    # Concatenate healthy samples
    healthy_combined = ad.concat(healthy_adatas, join='outer', index_unique=None)
    healthy_combined.obs_names_make_unique()
    
    logger.info(f"Loaded {healthy_combined.n_obs} healthy cells with {healthy_combined.n_vars} genes")
    
    return healthy_combined


def load_leukemia_data() -> ad.AnnData:
    """
    Load and preprocess leukemia data.
    
    Returns:
        AnnData object with leukemia data
    """
    logger.info("Loading leukemia data...")
    
    leukemia_files = []
    for prefix in Config.LEUKEMIA_PREFIXES:
        files = sorted(Config.CACHE_DIR.glob(f"{prefix}*"))
        leukemia_files.extend(files)
    
    if not leukemia_files:
        raise FileNotFoundError(f"No leukemia files found with prefixes: {Config.LEUKEMIA_PREFIXES}")
    
    logger.info(f"Found {len(leukemia_files)} leukemia files")
    
    leukemia_adatas = []
    for file_path in leukemia_files:
        logger.info(f"Loading: {file_path.name}")
        adata = sc.read_h5ad(file_path)
        adata.obs['sample_id'] = file_path.stem
        adata.obs['sample_type'] = _extract_leukemia_type(file_path.name)
        adata.obs['health_status'] = 'leukemia'
        leukemia_adatas.append(adata)
    
    # Concatenate leukemia samples
    leukemia_combined = ad.concat(leukemia_adatas, join='outer', index_unique=None)
    leukemia_combined.obs_names_make_unique()
    
    logger.info(f"Loaded {leukemia_combined.n_obs} leukemia cells with {leukemia_combined.n_vars} genes")
    
    return leukemia_combined


def _extract_leukemia_type(filename: str) -> str:
    """Extract leukemia type from filename."""
    if "ETV6-RUNX1" in filename:
        return "ETV6-RUNX1"
    elif "PRE-T" in filename:
        return "PRE-T"
    elif "HHD" in filename:
        return "HHD"
    else:
        return "unknown"


def preprocess_data(adata: ad.AnnData, name: str) -> ad.AnnData:
    """
    Preprocess AnnData object for integration.
    
    Args:
        adata: AnnData object to preprocess
        name: Name for logging purposes
    
    Returns:
        Preprocessed AnnData object
    """
    logger.info(f"Preprocessing {name} data...")
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=Config.MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=Config.MIN_CELLS)
    
    # Calculate quality metrics
    adata.obs['n_genes_by_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    
    # Calculate mitochondrial percentage
    mito_genes = adata.var_names.str.startswith('MT-')
    if mito_genes.sum() > 0:
        adata.obs['pct_counts_mt'] = np.array(adata[:, mito_genes].X.sum(axis=1)).flatten() / adata.obs['total_counts'] * 100
    else:
        adata.obs['pct_counts_mt'] = 0
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=Config.TARGET_SUM)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=Config.N_HVG, subset=True, flavor="seurat_v3")
    
    logger.info(f"Preprocessed {name}: {adata.n_obs} cells, {adata.n_vars} genes")
    
    return adata


def integrate_datasets(healthy_adata: ad.AnnData, leukemia_adata: ad.AnnData) -> ad.AnnData:
    """
    Integrate healthy and leukemia datasets using scVI.
    
    Args:
        healthy_adata: Preprocessed healthy reference data
        leukemia_adata: Preprocessed leukemia data
    
    Returns:
        Integrated AnnData object
    """
    logger.info("Integrating healthy and leukemia datasets...")
    
    # Concatenate datasets
    adata_combined = ad.concat([healthy_adata, leukemia_adata], join='outer', index_unique=None)
    adata_combined.obs_names_make_unique()
    
    # Ensure categorical variables
    adata_combined.obs['sample_type'] = adata_combined.obs['sample_type'].astype('category')
    adata_combined.obs['health_status'] = adata_combined.obs['health_status'].astype('category')
    
    # Set up scVI
    SCVI.setup_anndata(adata_combined, batch_key='sample_type')
    
    # Train scVI model
    model = SCVI(
        adata_combined,
        n_latent=Config.N_LATENT,
        n_layers=Config.N_LAYERS,
        n_hidden=Config.N_HIDDEN,
        dropout_rate=Config.DROPOUT_RATE
    )
    
    logger.info("Training scVI model...")
    model.train(
        max_epochs=Config.MAX_EPOCHS,
        train_size=0.95,
        batch_size=64
    )
    
    # Get latent representation
    adata_combined.obsm['X_scvi'] = model.get_latent_representation()
    
    # Compute neighbors and UMAP
    sc.pp.neighbors(adata_combined, use_rep='X_scvi')
    sc.tl.umap(adata_combined)
    
    # Perform clustering
    sc.tl.leiden(adata_combined, resolution=Config.RESOLUTION)
    
    logger.info(f"Integration complete: {adata_combined.n_obs} cells, {len(adata_combined.obs['leiden'].cat.categories)} clusters")
    
    return adata_combined, model


def annotate_cell_types(adata: ad.AnnData) -> ad.AnnData:
    """
    Annotate cell types using CellTypist.
    
    Args:
        adata: AnnData object to annotate
    
    Returns:
        AnnData object with cell type annotations
    """
    if celltypist is None:
        logger.warning("CellTypist not available, skipping cell type annotation")
        return adata
    
    logger.info("Annotating cell types using CellTypist...")
    
    try:
        # Download models if needed
        for model_name in Config.CELLTYPIST_MODELS:
            models.download_models(model=[model_name], force_update=False)
        
        # Prepare data for CellTypist
        adata_ct = adata.copy()
        adata_ct.X = adata_ct.layers.get('counts', adata_ct.X)
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
        
        # Convert to dense if sparse
        if sparse.issparse(adata_ct.X):
            adata_ct.X = adata_ct.X.toarray()
        
        # Annotate with high-resolution model
        model_high = models.Model.load(model="Immune_All_High.pkl")
        predictions_high = celltypist.annotate(adata_ct, model=model_high, majority_voting=True)
        predictions_high_adata = predictions_high.to_adata()
        
        # Add predictions to original adata
        adata.obs['celltype_high'] = predictions_high_adata.obs['predicted_labels']
        adata.obs['celltype_confidence'] = predictions_high_adata.obs['over_clustering']
        
        logger.info("Cell type annotation complete")
        
    except Exception as e:
        logger.error(f"Cell type annotation failed: {e}")
    
    return adata


def detect_abnormal_clusters(adata: ad.AnnData) -> pd.DataFrame:
    """
    Detect abnormal clusters by comparing to healthy reference.
    
    Args:
        adata: Integrated AnnData object
    
    Returns:
        DataFrame with abnormal cluster analysis
    """
    logger.info("Detecting abnormal clusters...")
    
    abnormal_analysis = []
    
    # Get cluster information
    clusters = adata.obs['leiden'].cat.categories
    health_status = adata.obs['health_status'].cat.categories
    
    for cluster in clusters:
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_cells = adata[cluster_mask]
        
        # Calculate cluster composition
        total_cells = len(cluster_cells)
        healthy_cells = (cluster_cells.obs['health_status'] == 'healthy').sum()
        leukemia_cells = (cluster_cells.obs['health_status'] == 'leukemia').sum()
        
        healthy_ratio = healthy_cells / total_cells if total_cells > 0 else 0
        leukemia_ratio = leukemia_cells / total_cells if total_cells > 0 else 0
        
        # Determine if cluster is abnormal
        is_abnormal = leukemia_ratio > 0.8  # More than 80% leukemia cells
        
        # Calculate cluster characteristics
        cluster_analysis = {
            'cluster': cluster,
            'total_cells': total_cells,
            'healthy_cells': healthy_cells,
            'leukemia_cells': leukemia_cells,
            'healthy_ratio': healthy_ratio,
            'leukemia_ratio': leukemia_ratio,
            'is_abnormal': is_abnormal,
            'abnormality_score': leukemia_ratio,
            'sample_types': ', '.join(cluster_cells.obs['sample_type'].unique()),
            'cell_types': ', '.join(cluster_cells.obs.get('celltype_high', ['unknown']).unique()) if 'celltype_high' in cluster_cells.obs.columns else 'unknown'
        }
        
        abnormal_analysis.append(cluster_analysis)
    
    abnormal_df = pd.DataFrame(abnormal_analysis)
    abnormal_df = abnormal_df.sort_values('abnormality_score', ascending=False)
    
    logger.info(f"Detected {abnormal_df['is_abnormal'].sum()} abnormal clusters out of {len(abnormal_df)} total clusters")
    
    return abnormal_df


def perform_differential_expression(adata: ad.AnnData, abnormal_clusters: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform differential expression analysis for abnormal clusters.
    
    Args:
        adata: Integrated AnnData object
        abnormal_clusters: DataFrame with abnormal cluster information
    
    Returns:
        Dictionary of differential expression results
    """
    logger.info("Performing differential expression analysis...")
    
    de_results = {}
    
    # Get healthy reference cells
    healthy_mask = adata.obs['health_status'] == 'healthy'
    healthy_cells = adata[healthy_mask]
    
    # Analyze each abnormal cluster
    abnormal_cluster_list = abnormal_clusters[abnormal_clusters['is_abnormal']]['cluster'].tolist()
    
    for cluster in abnormal_cluster_list:
        logger.info(f"Analyzing cluster {cluster}...")
        
        # Get cluster cells
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_cells = adata[cluster_mask]
        
        # Create comparison groups
        comparison_adata = ad.concat([cluster_cells, healthy_cells], join='outer', index_unique=None)
        comparison_adata.obs['comparison_group'] = 'healthy'
        comparison_adata.obs.loc[cluster_mask, 'comparison_group'] = f'cluster_{cluster}'
        
        # Perform differential expression
        try:
            sc.tl.rank_genes_groups(
                comparison_adata, 
                groupby='comparison_group',
                groups=[f'cluster_{cluster}'],
                reference='healthy',
                method='wilcoxon'
            )
            
            # Extract results
            de_df = sc.get.rank_genes_groups_df(comparison_adata, group=f'cluster_{cluster}')
            de_df = de_df.sort_values('scores', ascending=False)
            
            # Filter significant genes
            significant_mask = (de_df['pvals_adj'] < Config.MAX_PVALUE) & (de_df['logfoldchanges'] > Config.MIN_FC)
            de_df['significant'] = significant_mask
            
            de_results[f'cluster_{cluster}'] = de_df
            
            logger.info(f"Cluster {cluster}: {significant_mask.sum()} significant genes")
            
        except Exception as e:
            logger.error(f"Differential expression failed for cluster {cluster}: {e}")
            de_results[f'cluster_{cluster}'] = pd.DataFrame()
    
    return de_results


def generate_visualizations(adata: ad.AnnData, abnormal_clusters: pd.DataFrame):
    """
    Generate comprehensive visualizations for healthy reference integration.
    
    Args:
        adata: Integrated AnnData object
        abnormal_clusters: DataFrame with abnormal cluster information
    """
    logger.info("Generating visualizations...")
    
    # Set up plotting
    sc.settings.figdir = Config.OUTPUT_DIR
    
    # 1. UMAP colored by health status
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Health status
    sc.pl.umap(adata, color='health_status', ax=axes[0,0], show=False, title='Health Status')
    
    # Sample type
    sc.pl.umap(adata, color='sample_type', ax=axes[0,1], show=False, title='Sample Type')
    
    # Clusters
    sc.pl.umap(adata, color='leiden', ax=axes[1,0], show=False, title='Clusters')
    
    # Cell types (if available)
    if 'celltype_high' in adata.obs.columns:
        sc.pl.umap(adata, color='celltype_high', ax=axes[1,1], show=False, title='Cell Types')
    else:
        sc.pl.umap(adata, color='total_counts', ax=axes[1,1], show=False, title='Total Counts')
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'healthy_integration_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Abnormal cluster visualization
    if len(abnormal_clusters) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Abnormality score distribution
        axes[0].hist(abnormal_clusters['abnormality_score'], bins=20, alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0.8, color='red', linestyle='--', label='Abnormal threshold')
        axes[0].set_xlabel('Abnormality Score (Leukemia Ratio)')
        axes[0].set_ylabel('Number of Clusters')
        axes[0].set_title('Cluster Abnormality Distribution')
        axes[0].legend()
        
        # Cluster composition
        abnormal_clusters.plot(x='cluster', y=['healthy_ratio', 'leukemia_ratio'], 
                              kind='bar', stacked=True, ax=axes[1])
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Cell Ratio')
        axes[1].set_title('Cluster Composition')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'abnormal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Quality metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total counts by health status
    sc.pl.violin(adata, keys='total_counts', groupby='health_status', ax=axes[0,0], show=False)
    axes[0,0].set_title('Total Counts by Health Status')
    
    # Genes by health status
    sc.pl.violin(adata, keys='n_genes_by_counts', groupby='health_status', ax=axes[0,1], show=False)
    axes[0,1].set_title('Genes by Health Status')
    
    # Mitochondrial percentage
    if 'pct_counts_mt' in adata.obs.columns:
        sc.pl.violin(adata, keys='pct_counts_mt', groupby='health_status', ax=axes[1,0], show=False)
        axes[1,0].set_title('Mitochondrial % by Health Status')
    
    # Cell type distribution
    if 'celltype_high' in adata.obs.columns:
        celltype_counts = adata.obs.groupby(['health_status', 'celltype_high']).size().unstack(fill_value=0)
        celltype_counts.plot(kind='bar', stacked=True, ax=axes[1,1])
        axes[1,1].set_title('Cell Type Distribution')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'quality_metrics_healthy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved to outputs/ directory")


def generate_report(adata: ad.AnnData, abnormal_clusters: pd.DataFrame, de_results: Dict[str, pd.DataFrame]):
    """
    Generate comprehensive analysis report.
    
    Args:
        adata: Integrated AnnData object
        abnormal_clusters: DataFrame with abnormal cluster information
        de_results: Dictionary of differential expression results
    """
    logger.info("Generating analysis report...")
    
    report_path = Config.OUTPUT_DIR / Config.INTEGRATION_REPORT_FILE
    
    with open(report_path, 'w') as f:
        f.write("Healthy Reference Integration Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset summary
        f.write("Dataset Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total cells: {adata.n_obs:,}\n")
        f.write(f"Total genes: {adata.n_vars:,}\n")
        f.write(f"Latent dimensions: {Config.N_LATENT}\n\n")
        
        # Health status distribution
        f.write("Health Status Distribution:\n")
        f.write("-" * 30 + "\n")
        health_counts = adata.obs['health_status'].value_counts()
        for status, count in health_counts.items():
            percentage = (count / len(adata)) * 100
            f.write(f"{status}: {count:,} cells ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Sample type distribution
        f.write("Sample Type Distribution:\n")
        f.write("-" * 30 + "\n")
        sample_counts = adata.obs['sample_type'].value_counts()
        for sample, count in sample_counts.items():
            percentage = (count / len(adata)) * 100
            f.write(f"{sample}: {count:,} cells ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Clustering summary
        f.write("Clustering Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total clusters: {len(adata.obs['leiden'].cat.categories)}\n")
        f.write(f"Clustering resolution: {Config.RESOLUTION}\n\n")
        
        # Abnormal clusters summary
        f.write("Abnormal Clusters Analysis:\n")
        f.write("-" * 30 + "\n")
        abnormal_count = abnormal_clusters['is_abnormal'].sum()
        total_clusters = len(abnormal_clusters)
        f.write(f"Abnormal clusters: {abnormal_count}/{total_clusters} ({abnormal_count/total_clusters*100:.1f}%)\n\n")
        
        if abnormal_count > 0:
            f.write("Top abnormal clusters:\n")
            top_abnormal = abnormal_clusters[abnormal_clusters['is_abnormal']].head(5)
            for _, row in top_abnormal.iterrows():
                f.write(f"  Cluster {row['cluster']}: {row['leukemia_cells']} leukemia cells, "
                       f"{row['abnormality_score']:.3f} abnormality score\n")
            f.write("\n")
        
        # Cell type annotation summary
        if 'celltype_high' in adata.obs.columns:
            f.write("Cell Type Annotation Summary:\n")
            f.write("-" * 35 + "\n")
            celltype_counts = adata.obs['celltype_high'].value_counts()
            f.write(f"Total cell types identified: {len(celltype_counts)}\n")
            f.write("Top cell types:\n")
            for celltype, count in celltype_counts.head(10).items():
                percentage = (count / len(adata)) * 100
                f.write(f"  {celltype}: {count:,} cells ({percentage:.1f}%)\n")
            f.write("\n")
        
        # Differential expression summary
        f.write("Differential Expression Summary:\n")
        f.write("-" * 35 + "\n")
        for cluster_name, de_df in de_results.items():
            if len(de_df) > 0:
                significant_count = de_df['significant'].sum()
                f.write(f"{cluster_name}: {significant_count} significant genes\n")
        f.write("\n")
        
        # Quality metrics
        f.write("Quality Metrics:\n")
        f.write("-" * 15 + "\n")
        for status in ['healthy', 'leukemia']:
            status_mask = adata.obs['health_status'] == status
            status_data = adata[status_mask]
            
            f.write(f"{status.capitalize()} cells:\n")
            f.write(f"  Mean total counts: {status_data.obs['total_counts'].mean():.0f}\n")
            f.write(f"  Mean genes per cell: {status_data.obs['n_genes_by_counts'].mean():.0f}\n")
            if 'pct_counts_mt' in status_data.obs.columns:
                f.write(f"  Mean mitochondrial %: {status_data.obs['pct_counts_mt'].mean():.2f}%\n")
            f.write("\n")
    
    logger.info(f"Report saved to: {report_path}")


def save_results(adata: ad.AnnData, abnormal_clusters: pd.DataFrame, de_results: Dict[str, pd.DataFrame]):
    """
    Save analysis results to files.
    
    Args:
        adata: Integrated AnnData object
        abnormal_clusters: DataFrame with abnormal cluster information
        de_results: Dictionary of differential expression results
    """
    logger.info("Saving analysis results...")
    
    # Save integrated data
    adata.write(Config.OUTPUT_DIR / Config.INTEGRATED_DATA_FILE)
    
    # Save abnormal clusters analysis
    abnormal_clusters.to_csv(Config.OUTPUT_DIR / Config.ABNORMAL_CLUSTERS_FILE, index=False)
    
    # Save differential expression results
    for cluster_name, de_df in de_results.items():
        if len(de_df) > 0:
            de_df.to_csv(Config.OUTPUT_DIR / f"de_{cluster_name}_vs_healthy.csv", index=False)
    
    # Save healthy reference summary
    healthy_mask = adata.obs['health_status'] == 'healthy'
    healthy_data = adata[healthy_mask]
    
    healthy_summary = {
        'total_healthy_cells': len(healthy_data),
        'healthy_samples': ', '.join(healthy_data.obs['sample_id'].unique()),
        'healthy_clusters': ', '.join(healthy_data.obs['leiden'].unique()),
        'mean_total_counts': healthy_data.obs['total_counts'].mean(),
        'mean_genes_per_cell': healthy_data.obs['n_genes_by_counts'].mean(),
        'cell_types': ', '.join(healthy_data.obs.get('celltype_high', ['unknown']).unique()) if 'celltype_high' in healthy_data.obs.columns else 'unknown'
    }
    
    pd.DataFrame([healthy_summary]).to_csv(Config.OUTPUT_DIR / Config.HEALTHY_REFERENCE_FILE, index=False)
    
    logger.info("Results saved to outputs/ directory")


def main():
    """Main analysis pipeline."""
    logger.info("Starting healthy reference integration analysis...")
    
    try:
        # Setup environment
        setup_environment()
        
        # Load data
        healthy_adata = load_healthy_reference_data()
        leukemia_adata = load_leukemia_data()
        
        # Preprocess data
        healthy_adata = preprocess_data(healthy_adata, "healthy")
        leukemia_adata = preprocess_data(leukemia_adata, "leukemia")
        
        # Integrate datasets
        integrated_adata, scvi_model = integrate_datasets(healthy_adata, leukemia_adata)
        
        # Annotate cell types
        integrated_adata = annotate_cell_types(integrated_adata)
        
        # Detect abnormal clusters
        abnormal_clusters = detect_abnormal_clusters(integrated_adata)
        
        # Perform differential expression
        de_results = perform_differential_expression(integrated_adata, abnormal_clusters)
        
        # Generate visualizations
        generate_visualizations(integrated_adata, abnormal_clusters)
        
        # Generate report
        generate_report(integrated_adata, abnormal_clusters, de_results)
        
        # Save results
        save_results(integrated_adata, abnormal_clusters, de_results)
        
        logger.info("Healthy reference integration analysis complete!")
        logger.info(f"Results saved to: {Config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 