#!/usr/bin/env python3
"""
Comprehensive Cell Type Annotation Pipeline.

This script runs cell type annotation using multiple methods:
- CellTypist (Python-based)
- Azimuth (Seurat-based, via R)
- SingleR (R-based)

Includes comparison and validation of annotations across methods.
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from leukomap.cell_type_annotation import annotate_cell_types_comprehensive, CellTypeAnnotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning)


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
    Load the integrated data from previous analysis.
    
    Returns:
        AnnData object with integrated data
    """
    logger.info("Loading integrated data...")
    
    # Check for integrated data file
    data_file = Path("results/adata_integrated_healthy_quick.h5ad")
    if not data_file.exists():
        raise FileNotFoundError(f"Integrated data not found: {data_file}")
    
    adata = sc.read_h5ad(data_file)
    logger.info(f"Loaded integrated data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    return adata


def analyze_annotation_quality(adata: ad.AnnData) -> pd.DataFrame:
    """
    Analyze the quality of cell type annotations.
    
    Args:
        adata: AnnData object with annotations
        
    Returns:
        DataFrame with quality metrics
    """
    logger.info("Analyzing annotation quality...")
    
    quality_metrics = []
    
    # Analyze each annotation method
    for method in ['celltypist', 'azimuth', 'singler']:
        cell_type_col = f'{method}_cell_type'
        confidence_col = f'{method}_confidence'
        
        if cell_type_col not in adata.obs.columns:
            continue
        
        # Basic metrics
        cell_types = adata.obs[cell_type_col]
        n_cell_types = cell_types.nunique()
        n_cells = len(cell_types)
        
        # Confidence analysis
        if confidence_col in adata.obs.columns:
            confidence = adata.obs[confidence_col]
            mean_confidence = confidence.mean()
            high_confidence_pct = (confidence > 0.7).mean() * 100
        else:
            mean_confidence = np.nan
            high_confidence_pct = np.nan
        
        # Cell type distribution
        cell_type_dist = cell_types.value_counts()
        dominant_type = cell_type_dist.index[0]
        dominant_pct = (cell_type_dist.iloc[0] / n_cells) * 100
        
        quality_metrics.append({
            'method': method,
            'n_cell_types': n_cell_types,
            'n_cells': n_cells,
            'mean_confidence': mean_confidence,
            'high_confidence_pct': high_confidence_pct,
            'dominant_cell_type': dominant_type,
            'dominant_pct': dominant_pct
        })
    
    quality_df = pd.DataFrame(quality_metrics)
    logger.info(f"Quality analysis complete for {len(quality_df)} methods")
    
    return quality_df


def analyze_cell_types_by_cluster(adata: ad.AnnData) -> pd.DataFrame:
    """
    Analyze cell type distribution by cluster.
    
    Args:
        adata: AnnData object with annotations
        
    Returns:
        DataFrame with cluster analysis
    """
    logger.info("Analyzing cell types by cluster...")
    
    if 'leiden' not in adata.obs.columns:
        logger.warning("No clustering information found")
        return pd.DataFrame()
    
    cluster_analysis = []
    
    # Analyze each annotation method
    for method in ['celltypist', 'azimuth', 'singler']:
        cell_type_col = f'{method}_cell_type'
        
        if cell_type_col not in adata.obs.columns:
            continue
        
        # Create cross-tabulation
        cluster_celltype = pd.crosstab(adata.obs['leiden'], adata.obs[cell_type_col])
        
        # Analyze each cluster
        for cluster in cluster_celltype.index:
            cluster_cells = cluster_celltype.loc[cluster]
            total_cells = cluster_cells.sum()
            
            # Get dominant cell type
            dominant_cell_type = cluster_cells.idxmax()
            dominant_count = cluster_cells.max()
            dominant_pct = (dominant_count / total_cells) * 100
            
            # Get second most common cell type
            if len(cluster_cells) > 1:
                second_cell_type = cluster_cells.nlargest(2).index[1]
                second_count = cluster_cells.nlargest(2).iloc[1]
                second_pct = (second_count / total_cells) * 100
            else:
                second_cell_type = 'None'
                second_count = 0
                second_pct = 0
            
            cluster_analysis.append({
                'method': method,
                'cluster': cluster,
                'total_cells': total_cells,
                'dominant_cell_type': dominant_cell_type,
                'dominant_count': dominant_count,
                'dominant_pct': dominant_pct,
                'second_cell_type': second_cell_type,
                'second_count': second_count,
                'second_pct': second_pct,
                'n_cell_types': len(cluster_cells[cluster_cells > 0])
            })
    
    cluster_df = pd.DataFrame(cluster_analysis)
    logger.info(f"Cluster analysis complete: {len(cluster_df)} cluster-method combinations")
    
    return cluster_df


def generate_annotation_visualizations(adata: ad.AnnData, output_dir: Path):
    """
    Generate comprehensive visualizations for cell type annotations.
    
    Args:
        adata: AnnData object with annotations
        output_dir: Directory to save visualizations
    """
    logger.info("Generating annotation visualizations...")
    
    # Set up plotting
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Get available methods
    available_methods = []
    for method in ['celltypist', 'azimuth', 'singler']:
        if f'{method}_cell_type' in adata.obs.columns:
            available_methods.append(method)
    
    if len(available_methods) == 0:
        logger.warning("No annotation methods available for visualization")
        return
    
    # 1. UMAP plots for each method
    n_methods = len(available_methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(available_methods):
        sc.pl.umap(
            adata, 
            color=f'{method}_cell_type', 
            ax=axes[i], 
            show=False, 
            title=f'{method.upper()} Cell Types',
            legend_loc='right margin'
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_celltype_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cell type distribution plots
    fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(available_methods):
        cell_type_dist = adata.obs[f'{method}_cell_type'].value_counts().head(10)
        cell_type_dist.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{method.upper()} - Top 10 Cell Types')
        axes[i].set_ylabel('Cell Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'celltype_distribution_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cell type by health status
    if 'health_status' in adata.obs.columns:
        fig, axes = plt.subplots(1, n_methods, figsize=(10*n_methods, 6))
        if n_methods == 1:
            axes = [axes]
        
        for i, method in enumerate(available_methods):
            cell_type_health = pd.crosstab(
                adata.obs[f'{method}_cell_type'], 
                adata.obs['health_status']
            )
            cell_type_health.head(10).plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{method.upper()} - Cell Types by Health Status')
            axes[i].set_ylabel('Cell Count')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'celltype_by_health_status.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Confidence distributions
    fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(available_methods):
        confidence_col = f'{method}_confidence'
        if confidence_col in adata.obs.columns:
            axes[i].hist(adata.obs[confidence_col], bins=50, alpha=0.7)
            axes[i].set_title(f'{method.upper()} - Confidence Distribution')
            axes[i].set_xlabel('Confidence Score')
            axes[i].set_ylabel('Cell Count')
            axes[i].axvline(0.7, color='red', linestyle='--', label='High confidence threshold')
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, 'No confidence scores available', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{method.upper()} - No Confidence Data')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Annotation visualizations saved")


def save_detailed_results(adata: ad.AnnData, 
                         quality_metrics: pd.DataFrame,
                         cluster_analysis: pd.DataFrame,
                         output_dir: Path):
    """
    Save detailed analysis results.
    
    Args:
        adata: AnnData object with annotations
        quality_metrics: DataFrame with quality metrics
        cluster_analysis: DataFrame with cluster analysis
        output_dir: Directory to save results
    """
    logger.info("Saving detailed analysis results...")
    
    # Save annotated data
    adata.write(output_dir / 'adata_comprehensive_celltype_annotated.h5ad')
    
    # Save quality metrics
    quality_metrics.to_csv(output_dir / 'annotation_quality_metrics.csv', index=False)
    
    # Save cluster analysis
    if not cluster_analysis.empty:
        cluster_analysis.to_csv(output_dir / 'celltype_cluster_analysis.csv', index=False)
    
    # Save cell type summaries for each method
    for method in ['celltypist', 'azimuth', 'singler']:
        cell_type_col = f'{method}_cell_type'
        if cell_type_col in adata.obs.columns:
            cell_type_summary = adata.obs[cell_type_col].value_counts()
            cell_type_summary.to_csv(output_dir / f'{method}_celltype_summary.csv')
    
    # Generate comprehensive report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE CELL TYPE ANNOTATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total cells analyzed: {len(adata):,}")
    report_lines.append("")
    
    # Quality metrics summary
    report_lines.append("ANNOTATION QUALITY SUMMARY:")
    for _, row in quality_metrics.iterrows():
        report_lines.append(f"  {row['method'].upper()}:")
        report_lines.append(f"    Cell types: {row['n_cell_types']}")
        report_lines.append(f"    Mean confidence: {row['mean_confidence']:.3f}")
        report_lines.append(f"    High confidence cells: {row['high_confidence_pct']:.1f}%")
        report_lines.append(f"    Dominant cell type: {row['dominant_cell_type']} ({row['dominant_pct']:.1f}%)")
        report_lines.append("")
    
    # Cluster analysis summary
    if not cluster_analysis.empty:
        report_lines.append("CLUSTER ANALYSIS SUMMARY:")
        for method in cluster_analysis['method'].unique():
            method_data = cluster_analysis[cluster_analysis['method'] == method]
            report_lines.append(f"  {method.upper()}:")
            report_lines.append(f"    Clusters analyzed: {len(method_data)}")
            report_lines.append(f"    Average dominant cell type percentage: {method_data['dominant_pct'].mean():.1f}%")
            report_lines.append(f"    Average cell types per cluster: {method_data['n_cell_types'].mean():.1f}")
            report_lines.append("")
    
    # Save report
    report_path = output_dir / 'comprehensive_celltype_annotation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Detailed results saved to {output_dir}")


def main():
    """Main analysis pipeline."""
    try:
        logger.info("Starting comprehensive cell type annotation analysis...")
        
        # Setup environment
        setup_environment()
        
        # Load integrated data
        adata = load_integrated_data()
        
        # Run comprehensive annotation
        output_dir = Path("results")
        adata = annotate_cell_types_comprehensive(adata, output_dir)
        
        # Analyze annotation quality
        quality_metrics = analyze_annotation_quality(adata)
        
        # Analyze cell types by cluster
        cluster_analysis = analyze_cell_types_by_cluster(adata)
        
        # Generate visualizations
        generate_annotation_visualizations(adata, output_dir)
        
        # Save detailed results
        save_detailed_results(adata, quality_metrics, cluster_analysis, output_dir)
        
        logger.info("Comprehensive cell type annotation analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 