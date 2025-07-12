#!/usr/bin/env python3
"""
Add CellTypist Cell Type Annotations to Integrated Data.

This script adds automated cell type annotations to our existing
healthy reference integration results.
"""

import os
import sys
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

# Import CellTypist
import celltypist
from celltypist import models
from celltypist.annotate import annotate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
class Config:
    """Configuration parameters for cell type annotation."""
    
    # Data paths
    OUTPUT_DIR = Path("outputs")
    INTEGRATED_DATA_FILE = "adata_integrated_healthy_quick.h5ad"
    
    # CellTypist parameters
    CELLTYPIST_MODEL = "Immune_All_High.pkl"
    
    # Output file names
    ANNOTATED_DATA_FILE = "adata_integrated_healthy_annotated.h5ad"
    CELLTYPE_SUMMARY_FILE = "celltype_annotation_summary.csv"
    CELLTYPE_VISUALIZATION_FILE = "umap_celltype_annotated.png"


def setup_environment():
    """Set up the analysis environment."""
    logger.info("Setting up analysis environment...")
    
    # Set scanpy settings
    sc.settings.verbosity = 3
    
    # Set random seeds
    np.random.seed(42)
    sc.settings.seed = 42
    
    logger.info("Environment setup complete")


def load_integrated_data() -> ad.AnnData:
    """
    Load the integrated data from our previous analysis.
    
    Returns:
        AnnData object with integrated data
    """
    logger.info("Loading integrated data...")
    
    data_file = Config.OUTPUT_DIR / Config.INTEGRATED_DATA_FILE
    if not data_file.exists():
        raise FileNotFoundError(f"Integrated data not found: {data_file}")
    
    adata = sc.read_h5ad(data_file)
    logger.info(f"Loaded integrated data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    return adata


def annotate_cell_types(adata: ad.AnnData) -> ad.AnnData:
    """
    Annotate cell types using CellTypist.
    
    Args:
        adata: AnnData object
    
    Returns:
        AnnData with cell type annotations
    """
    logger.info("Annotating cell types with CellTypist...")
    
    try:
        # Use the high-confidence immune model
        logger.info(f"Using CellTypist model: {Config.CELLTYPIST_MODEL}")
        predictions = annotate(adata, model=Config.CELLTYPIST_MODEL, majority_voting=True)
        
        # Add annotations to adata
        adata.obs['cell_type'] = predictions.predicted_labels.majority_voting
        adata.obs['cell_type_confidence'] = predictions.predicted_labels.majority_voting_score
        
        # Get detailed predictions
        adata.obs['cell_type_detailed'] = predictions.predicted_labels.predicted_labels
        
        logger.info(f"Cell types annotated: {adata.obs['cell_type'].nunique()} unique types")
        
        # Print cell type distribution
        cell_type_dist = adata.obs['cell_type'].value_counts()
        logger.info("Cell type distribution:")
        for cell_type, count in cell_type_dist.head(10).items():
            pct = (count / len(adata)) * 100
            logger.info(f"  {cell_type}: {count:,} cells ({pct:.1f}%)")
        
    except Exception as e:
        logger.error(f"Cell type annotation failed: {e}")
        raise
    
    return adata


def analyze_cell_types_by_health_status(adata: ad.AnnData) -> pd.DataFrame:
    """
    Analyze cell type distribution by health status.
    
    Args:
        adata: AnnData object with cell type annotations
    
    Returns:
        DataFrame with cell type analysis
    """
    logger.info("Analyzing cell types by health status...")
    
    # Create cross-tabulation
    cell_type_health = pd.crosstab(adata.obs['cell_type'], adata.obs['health_status'])
    
    # Calculate percentages
    cell_type_health_pct = cell_type_health.div(cell_type_health.sum(axis=1), axis=0) * 100
    
    # Add total counts
    cell_type_health['total_cells'] = cell_type_health.sum(axis=1)
    cell_type_health['healthy_pct'] = cell_type_health_pct['healthy']
    cell_type_health['leukemia_pct'] = cell_type_health_pct['leukemia']
    
    # Sort by total cells
    cell_type_health = cell_type_health.sort_values('total_cells', ascending=False)
    
    logger.info(f"Analyzed {len(cell_type_health)} cell types")
    
    return cell_type_health


def analyze_abnormal_clusters_with_cell_types(adata: ad.AnnData) -> pd.DataFrame:
    """
    Analyze abnormal clusters with cell type information.
    
    Args:
        adata: AnnData object with cell type annotations
    
    Returns:
        DataFrame with abnormal cluster analysis including cell types
    """
    logger.info("Analyzing abnormal clusters with cell type information...")
    
    # Get abnormal clusters (clusters with >50% leukemia cells)
    cluster_comp = pd.crosstab(adata.obs['leiden'], adata.obs['health_status'])
    cluster_comp_pct = cluster_comp.div(cluster_comp.sum(axis=1), axis=0) * 100
    abnormal_clusters = cluster_comp_pct[cluster_comp_pct['leukemia'] > 50]
    
    # Analyze each abnormal cluster
    abnormal_analysis = []
    for cluster in abnormal_clusters.index:
        cluster_cells = adata[adata.obs['leiden'] == cluster]
        
        # Basic metrics
        total_cells = len(cluster_cells)
        leukemia_cells = len(cluster_cells[cluster_cells.obs['health_status'] == 'leukemia'])
        leukemia_pct = (leukemia_cells / total_cells) * 100
        
        # Cell type analysis
        if 'cell_type' in cluster_cells.obs.columns:
            cell_type_dist = cluster_cells.obs['cell_type'].value_counts()
            dominant_cell_type = cell_type_dist.index[0] if len(cell_type_dist) > 0 else 'Unknown'
            cell_type_pct = (cell_type_dist.iloc[0] / total_cells) * 100 if len(cell_type_dist) > 0 else 0
        else:
            dominant_cell_type = 'Unknown'
            cell_type_pct = 0
        
        abnormal_analysis.append({
            'cluster': cluster,
            'total_cells': total_cells,
            'leukemia_cells': leukemia_cells,
            'leukemia_percentage': leukemia_pct,
            'dominant_cell_type': dominant_cell_type,
            'cell_type_percentage': cell_type_pct,
            'abnormality_score': leukemia_pct / 100.0
        })
    
    abnormal_df = pd.DataFrame(abnormal_analysis)
    abnormal_df = abnormal_df.sort_values('abnormality_score', ascending=False)
    
    logger.info(f"Analyzed {len(abnormal_df)} abnormal clusters with cell type information")
    
    return abnormal_df


def generate_celltype_visualizations(adata: ad.AnnData):
    """
    Generate visualizations with cell type annotations.
    
    Args:
        adata: AnnData object with cell type annotations
    """
    logger.info("Generating cell type visualizations...")
    
    # Set up plotting
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.facecolor'] = 'white'
    
    # UMAP colored by cell type
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sc.pl.umap(adata, color='cell_type', ax=ax, show=False, title='Cell Type Annotation')
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / Config.CELLTYPE_VISUALIZATION_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cell type distribution by health status
    if 'cell_type' in adata.obs.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cell type distribution overall
        cell_type_dist = adata.obs['cell_type'].value_counts().head(10)
        cell_type_dist.plot(kind='bar', ax=ax1)
        ax1.set_title('Top 10 Cell Types (Overall)')
        ax1.set_ylabel('Cell Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Cell type distribution by health status
        cell_type_health = pd.crosstab(adata.obs['cell_type'], adata.obs['health_status'])
        cell_type_health.head(10).plot(kind='bar', ax=ax2)
        ax2.set_title('Top 10 Cell Types by Health Status')
        ax2.set_ylabel('Cell Count')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'celltype_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Cell type visualizations saved to outputs/")


def save_results(adata: ad.AnnData, cell_type_analysis: pd.DataFrame, abnormal_analysis: pd.DataFrame):
    """
    Save annotated results.
    
    Args:
        adata: AnnData object with cell type annotations
        cell_type_analysis: DataFrame with cell type analysis
        abnormal_analysis: DataFrame with abnormal cluster analysis
    """
    logger.info("Saving annotated results...")
    
    # Save annotated data
    adata.write(Config.OUTPUT_DIR / Config.ANNOTATED_DATA_FILE)
    logger.info(f"Annotated data saved to {Config.OUTPUT_DIR / Config.ANNOTATED_DATA_FILE}")
    
    # Save cell type analysis
    cell_type_analysis.to_csv(Config.OUTPUT_DIR / Config.CELLTYPE_SUMMARY_FILE)
    logger.info(f"Cell type analysis saved to {Config.OUTPUT_DIR / Config.CELLTYPE_SUMMARY_FILE}")
    
    # Save abnormal cluster analysis with cell types
    abnormal_analysis.to_csv(Config.OUTPUT_DIR / 'abnormal_clusters_with_celltypes.csv', index=False)
    logger.info("Abnormal clusters analysis with cell types saved")
    
    # Generate summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CELL TYPE ANNOTATION SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append(f"Total cells annotated: {len(adata):,}")
    report_lines.append(f"Unique cell types: {adata.obs['cell_type'].nunique()}")
    report_lines.append("")
    
    report_lines.append("TOP 10 CELL TYPES:")
    cell_type_dist = adata.obs['cell_type'].value_counts().head(10)
    for i, (cell_type, count) in enumerate(cell_type_dist.items(), 1):
        pct = (count / len(adata)) * 100
        report_lines.append(f"{i:2d}. {cell_type}: {count:,} cells ({pct:.1f}%)")
    report_lines.append("")
    
    report_lines.append("ABNORMAL CLUSTERS WITH CELL TYPES:")
    for _, cluster_info in abnormal_analysis.iterrows():
        report_lines.append(f"  Cluster {cluster_info['cluster']}:")
        report_lines.append(f"    Cell type: {cluster_info['dominant_cell_type']} ({cluster_info['cell_type_percentage']:.1f}%)")
        report_lines.append(f"    Leukemia: {cluster_info['leukemia_percentage']:.1f}%")
        report_lines.append("")
    
    report_path = Config.OUTPUT_DIR / 'celltype_annotation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Cell type annotation report saved to {report_path}")


def main():
    """Main analysis pipeline."""
    try:
        logger.info("Starting cell type annotation analysis...")
        
        # Setup environment
        setup_environment()
        
        # Load integrated data
        adata = load_integrated_data()
        
        # Annotate cell types
        adata = annotate_cell_types(adata)
        
        # Analyze cell types by health status
        cell_type_analysis = analyze_cell_types_by_health_status(adata)
        
        # Analyze abnormal clusters with cell types
        abnormal_analysis = analyze_abnormal_clusters_with_cell_types(adata)
        
        # Generate visualizations
        generate_celltype_visualizations(adata)
        
        # Save results
        save_results(adata, cell_type_analysis, abnormal_analysis)
        
        logger.info("Cell type annotation analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 