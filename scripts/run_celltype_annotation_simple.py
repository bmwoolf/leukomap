#!/usr/bin/env python3
"""
Simplified Cell Type Annotation Pipeline.

This script runs cell type annotation focusing on CellTypist,
with fallback mock annotations for testing when real methods fail.
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

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


def annotate_celltypist(adata: ad.AnnData) -> ad.AnnData:
    """
    Annotate cell types using CellTypist.
    
    Args:
        adata: AnnData object
        
    Returns:
        AnnData with CellTypist annotations
    """
    logger.info("Running CellTypist annotation...")
    
    try:
        # Import CellTypist
        import celltypist
        from celltypist.annotate import annotate
        
        # Try different models
        models_to_try = ['blood', 'immune_all', 'immune_low']
        
        for model in models_to_try:
            try:
                logger.info(f"Trying CellTypist model: {model}")
                predictions = annotate(adata, model=model, majority_voting=True)
                
                # Add annotations to adata
                adata.obs['celltypist_cell_type'] = predictions.predicted_labels.majority_voting
                adata.obs['celltypist_confidence'] = predictions.predicted_labels.majority_voting_score
                adata.obs['celltypist_detailed'] = predictions.predicted_labels.predicted_labels
                
                logger.info(f"CellTypist annotation successful with model {model}: {adata.obs['celltypist_cell_type'].nunique()} cell types")
                return adata
                
            except Exception as e:
                logger.warning(f"CellTypist model {model} failed: {e}")
                continue
        
        raise RuntimeError("All CellTypist models failed")
        
    except Exception as e:
        logger.error(f"CellTypist annotation failed: {e}")
        raise


def create_mock_annotations(adata: ad.AnnData) -> ad.AnnData:
    """
    Create mock cell type annotations for testing when real methods fail.
    
    Args:
        adata: AnnData object
        
    Returns:
        AnnData with mock annotations
    """
    logger.info("Creating mock cell type annotations for testing...")
    
    # Define mock cell types based on cluster and health status
    mock_cell_types = []
    mock_confidence = []
    
    for i in range(len(adata)):
        cluster = adata.obs.iloc[i]['leiden'] if 'leiden' in adata.obs.columns else '0'
        health = adata.obs.iloc[i]['health_status'] if 'health_status' in adata.obs.columns else 'unknown'
        
        # Create cell type based on cluster and health status
        if health == 'healthy':
            if cluster in ['0', '1']:
                cell_type = 'B cells'
                confidence = np.random.uniform(0.7, 0.95)
            elif cluster in ['2', '3']:
                cell_type = 'T cells'
                confidence = np.random.uniform(0.7, 0.95)
            else:
                cell_type = 'Monocytes'
                confidence = np.random.uniform(0.6, 0.9)
        else:  # leukemia
            if cluster in ['0', '1']:
                cell_type = 'Leukemic B cells'
                confidence = np.random.uniform(0.6, 0.9)
            elif cluster in ['2', '3']:
                cell_type = 'Leukemic T cells'
                confidence = np.random.uniform(0.6, 0.9)
            else:
                cell_type = 'Leukemic blasts'
                confidence = np.random.uniform(0.5, 0.8)
        
        mock_cell_types.append(cell_type)
        mock_confidence.append(confidence)
    
    # Add to adata
    adata.obs['mock_cell_type'] = mock_cell_types
    adata.obs['mock_confidence'] = mock_confidence
    
    logger.info(f"Mock annotations created: {len(set(mock_cell_types))} cell types")
    
    return adata


def analyze_cell_types_by_condition(adata: ad.AnnData, method: str = 'celltypist') -> pd.DataFrame:
    """
    Analyze cell type distribution by condition.
    
    Args:
        adata: AnnData object
        method: Annotation method to use
        
    Returns:
        DataFrame with cell type analysis
    """
    logger.info(f"Analyzing cell types by health status using {method}")
    
    cell_type_col = f'{method}_cell_type'
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type annotations for {method} not found")
    
    # Create cross-tabulation
    cell_type_health = pd.crosstab(adata.obs[cell_type_col], adata.obs['health_status'])
    
    # Calculate percentages
    cell_type_health_pct = cell_type_health.div(cell_type_health.sum(axis=1), axis=0) * 100
    
    # Add total counts and percentages
    cell_type_health['total_cells'] = cell_type_health.sum(axis=1)
    
    # Add percentage columns for each condition
    for col in cell_type_health_pct.columns:
        cell_type_health[f'{col}_pct'] = cell_type_health_pct[col]
    
    # Sort by total cells
    cell_type_health = cell_type_health.sort_values('total_cells', ascending=False)
    
    return cell_type_health


def generate_visualizations(adata: ad.AnnData, output_dir: Path):
    """
    Generate visualizations for cell type annotations.
    
    Args:
        adata: AnnData object with annotations
        output_dir: Directory to save visualizations
    """
    logger.info("Generating cell type visualizations...")
    
    # Set up plotting
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Find available annotation methods
    available_methods = []
    for method in ['celltypist', 'mock']:
        if f'{method}_cell_type' in adata.obs.columns:
            available_methods.append(method)
    
    if not available_methods:
        logger.warning("No annotation methods available for visualization")
        return
    
    # 1. UMAP plots for each method
    n_methods = len(available_methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 6))
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
    plt.savefig(output_dir / 'umap_celltype_annotations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cell type distribution
    fig, axes = plt.subplots(1, n_methods, figsize=(10*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(available_methods):
        cell_type_dist = adata.obs[f'{method}_cell_type'].value_counts()
        cell_type_dist.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{method.upper()} - Cell Type Distribution')
        axes[i].set_ylabel('Cell Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'celltype_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cell type by health status
    fig, axes = plt.subplots(1, n_methods, figsize=(12*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(available_methods):
        cell_type_health = pd.crosstab(
            adata.obs[f'{method}_cell_type'], 
            adata.obs['health_status']
        )
        cell_type_health.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{method.upper()} - Cell Types by Health Status')
        axes[i].set_ylabel('Cell Count')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'celltype_by_health_status.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confidence distribution (if available)
    confidence_methods = [m for m in available_methods if f'{m}_confidence' in adata.obs.columns]
    if confidence_methods:
        fig, axes = plt.subplots(1, len(confidence_methods), figsize=(8*len(confidence_methods), 5))
        if len(confidence_methods) == 1:
            axes = [axes]
        
        for i, method in enumerate(confidence_methods):
            confidence = adata.obs[f'{method}_confidence']
            axes[i].hist(confidence, bins=30, alpha=0.7)
            axes[i].set_title(f'{method.upper()} - Confidence Distribution')
            axes[i].set_xlabel('Confidence Score')
            axes[i].set_ylabel('Cell Count')
            axes[i].axvline(0.7, color='red', linestyle='--', label='High confidence threshold')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Visualizations saved")


def save_results(adata: ad.AnnData, cell_type_analysis: pd.DataFrame, output_dir: Path):
    """
    Save annotation results.
    
    Args:
        adata: AnnData object with annotations
        cell_type_analysis: DataFrame with cell type analysis
        output_dir: Directory to save results
    """
    logger.info("Saving annotation results...")
    
    # Save annotated data
    adata.write(output_dir / 'adata_celltype_annotated_simple.h5ad')
    
    # Save cell type analysis
    cell_type_analysis.to_csv(output_dir / 'celltype_analysis_simple.csv')
    
    # Save cell type summaries
    for method in ['celltypist', 'mock']:
        cell_type_col = f'{method}_cell_type'
        if cell_type_col in adata.obs.columns:
            cell_type_summary = adata.obs[cell_type_col].value_counts()
            cell_type_summary.to_csv(output_dir / f'{method}_celltype_summary.csv')
    
    # Generate summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CELL TYPE ANNOTATION SUMMARY (SIMPLE)")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append(f"Total cells analyzed: {len(adata):,}")
    report_lines.append("")
    
    # Summary for each method
    for method in ['celltypist', 'mock']:
        cell_type_col = f'{method}_cell_type'
        if cell_type_col in adata.obs.columns:
            cell_types = adata.obs[cell_type_col]
            report_lines.append(f"{method.upper()} ANNOTATION:")
            report_lines.append(f"  Unique cell types: {cell_types.nunique()}")
            report_lines.append(f"  Top 5 cell types:")
            
            top_types = cell_types.value_counts().head(5)
            for cell_type, count in top_types.items():
                pct = (count / len(adata)) * 100
                report_lines.append(f"    {cell_type}: {count:,} cells ({pct:.1f}%)")
            report_lines.append("")
    
    # Cell type by health status summary
    if 'health_status' in adata.obs.columns:
        report_lines.append("CELL TYPES BY HEALTH STATUS:")
        for method in ['celltypist', 'mock']:
            cell_type_col = f'{method}_cell_type'
            if cell_type_col in adata.obs.columns:
                cell_type_health = pd.crosstab(adata.obs[cell_type_col], adata.obs['health_status'])
                report_lines.append(f"  {method.upper()}:")
                for cell_type in cell_type_health.index:
                    healthy_count = cell_type_health.loc[cell_type, 'healthy'] if 'healthy' in cell_type_health.columns else 0
                    leukemia_count = cell_type_health.loc[cell_type, 'leukemia'] if 'leukemia' in cell_type_health.columns else 0
                    report_lines.append(f"    {cell_type}: {healthy_count} healthy, {leukemia_count} leukemia")
                report_lines.append("")
    
    # Save report
    report_path = output_dir / 'celltype_annotation_simple_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main analysis pipeline."""
    try:
        logger.info("Starting simplified cell type annotation analysis...")
        
        # Setup environment
        setup_environment()
        
        # Load integrated data
        adata = load_integrated_data()
        
        # Try CellTypist annotation
        try:
            adata = annotate_celltypist(adata)
            logger.info("CellTypist annotation successful")
        except Exception as e:
            logger.warning(f"CellTypist failed: {e}")
            logger.info("Creating mock annotations for testing")
            adata = create_mock_annotations(adata)
        
        # Analyze cell types by health status
        method = 'celltypist' if 'celltypist_cell_type' in adata.obs.columns else 'mock'
        cell_type_analysis = analyze_cell_types_by_condition(adata, method)
        
        # Generate visualizations
        output_dir = Path("results")
        generate_visualizations(adata, output_dir)
        
        # Save results
        save_results(adata, cell_type_analysis, output_dir)
        
        logger.info("Simplified cell type annotation analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 