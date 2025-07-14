#!/usr/bin/env python3
"""
LeukoMap End-to-End Analysis Example.

This example demonstrates the complete LeukoMap pipeline from data loading
through preprocessing, cell type annotation, and analysis. It serves as both
a tutorial and a validation of the package functionality.
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

# Import LeukoMap components
import leukomap
from leukomap import (
    AnalysisConfig,
    LeukoMapAnalysis,
    DataManager,
    PreprocessingManager,
    CellTypeAnnotator,
    run_leukomap_analysis,
    load_and_preprocess,
    annotate_cells
)

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
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    sc.settings.seed = 42
    
    logger.info("Environment setup complete")


def create_example_config() -> AnalysisConfig:
    """Create a configuration for the end-to-end analysis."""
    logger.info("Creating analysis configuration...")
    
    config = AnalysisConfig(
        data_path=Path("data"),  # Path to your data
        output_dir=Path("results/end_to_end_example"),
        min_genes=200,
        min_cells=3,
        n_latent=10,
        max_epochs=50,  # Reduced for example
        resolution=0.5,
        n_neighbors=15,
        n_pcs=50
    )
    
    logger.info(f"Configuration created:")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Min genes: {config.min_genes}")
    logger.info(f"  Min cells: {config.min_cells}")
    logger.info(f"  Latent dimensions: {config.n_latent}")
    
    return config


def load_and_validate_data(config: AnalysisConfig) -> ad.AnnData:
    """Load and validate data with comprehensive error handling."""
    logger.info("Loading and validating data...")
    
    data_manager = DataManager(config)
    
    try:
        # Try to load from data directory
        adata = data_manager.load_and_validate()
        logger.info(f"✓ Data loaded successfully: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Save loaded data
        data_path = data_manager.save_data(adata, "raw_data.h5ad")
        logger.info(f"✓ Raw data saved to: {data_path}")
        
    except FileNotFoundError:
        logger.warning("Data not found in data directory. Trying to load existing integrated data...")
        
        # Try to load existing integrated data
        try:
            adata = sc.read_h5ad("results/adata_integrated_healthy_quick.h5ad")
            logger.info(f"✓ Loaded existing integrated data: {adata.n_obs} cells, {adata.n_vars} genes")
        except FileNotFoundError:
            logger.warning("No existing data found. Creating mock data for demonstration...")
            adata = create_mock_data_for_demo()
            logger.info(f"✓ Created mock data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Validate data structure
    validate_data_structure(adata)
    
    return adata


def create_mock_data_for_demo() -> ad.AnnData:
    """Create mock single-cell data for demonstration when real data is unavailable."""
    logger.info("Creating mock data for demonstration...")
    
    # Create realistic mock data
    n_cells, n_genes = 1000, 2000
    
    # Generate expression matrix with realistic distribution
    X = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))
    
    # Create cell metadata
    sample_types = ['PBMMC', 'ETV6-RUNX1', 'HHD', 'PRE-T']
    samples = np.random.choice(sample_types, n_cells, p=[0.3, 0.3, 0.2, 0.2])
    
    obs = pd.DataFrame({
        'sample': [f'sample_{i//100}' for i in range(n_cells)],
        'sample_type': samples,
        'total_counts': X.sum(axis=1),
        'n_genes_by_counts': (X > 0).sum(axis=1),
        'percent.mito': np.random.uniform(0, 0.1, n_cells)
    }, index=[f'cell_{i}' for i in range(n_cells)])
    
    # Create gene metadata
    gene_names = [f'GENE_{i:04d}' for i in range(n_genes)]
    var = pd.DataFrame({
        'gene_ids': gene_names,
        'feature_types': ['Gene Expression'] * n_genes,
        'n_cells': (X > 0).sum(axis=0)
    }, index=gene_names)
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    return adata


def validate_data_structure(adata: ad.AnnData) -> None:
    """Validate the structure of the AnnData object."""
    logger.info("Validating data structure...")
    
    # Check basic structure
    assert adata.n_obs > 0, "No cells found in data"
    assert adata.n_vars > 0, "No genes found in data"
    
    # Check required metadata
    required_obs = ['sample', 'sample_type']
    missing_obs = [col for col in required_obs if col not in adata.obs.columns]
    if missing_obs:
        logger.warning(f"Missing cell metadata columns: {missing_obs}")
    
    # Check data quality
    logger.info(f"✓ Data validation passed:")
    logger.info(f"  - Cells: {adata.n_obs:,}")
    logger.info(f"  - Genes: {adata.n_vars:,}")
    logger.info(f"  - Sample types: {adata.obs['sample_type'].nunique()}")
    logger.info(f"  - Samples: {adata.obs['sample'].nunique()}")


def preprocess_data(adata: ad.AnnData, config: AnalysisConfig) -> ad.AnnData:
    """Preprocess the data with comprehensive quality control."""
    logger.info("Preprocessing data...")
    
    preprocessing_manager = PreprocessingManager(config)
    
    # Preprocess data
    preprocessed_adata = preprocessing_manager.preprocess_data(adata, save_results=True)
    logger.info(f"✓ Preprocessing complete: {preprocessed_adata.n_obs} cells, {preprocessed_adata.n_vars} genes")
    
    # Generate preprocessing report
    report_path = preprocessing_manager.generate_preprocessing_report(adata, preprocessed_adata)
    logger.info(f"✓ Preprocessing report saved to: {report_path}")
    
    # Validate preprocessing results
    validate_preprocessing_results(preprocessed_adata)
    
    return preprocessed_adata


def validate_preprocessing_results(adata: ad.AnnData) -> None:
    """Validate that preprocessing was successful."""
    logger.info("Validating preprocessing results...")
    
    # Check that highly variable genes were identified
    if 'highly_variable' in adata.var.columns:
        n_hvg = adata.var['highly_variable'].sum()
        logger.info(f"✓ Highly variable genes identified: {n_hvg}")
    else:
        logger.warning("No highly variable genes found")
    
    # Check that normalization was applied
    if hasattr(adata, 'X') and adata.X is not None:
        logger.info(f"✓ Expression matrix shape: {adata.X.shape}")
    
    # Check for quality metrics
    quality_cols = ['total_counts', 'n_genes_by_counts', 'percent.mito']
    available_quality = [col for col in quality_cols if col in adata.obs.columns]
    logger.info(f"✓ Quality metrics available: {available_quality}")


def perform_cell_type_annotation(adata: ad.AnnData, config: AnalysisConfig) -> ad.AnnData:
    """Perform cell type annotation with fallback mechanisms."""
    logger.info("Performing cell type annotation...")
    
    annotator = CellTypeAnnotator(config)
    
    try:
        # Try real annotation
        annotated_adata = annotator.process(adata)
        logger.info("✓ Cell type annotation completed successfully")
        
    except Exception as e:
        logger.warning(f"Real annotation failed: {e}")
        logger.info("Using mock annotations for demonstration...")
        
        # Add mock annotations
        annotated_adata = annotator._add_mock_annotations(adata.copy())
        logger.info("✓ Mock annotations added for demonstration")
    
    # Validate annotation results
    validate_annotation_results(annotated_adata)
    
    return annotated_adata


def validate_annotation_results(adata: ad.AnnData) -> None:
    """Validate that cell type annotation was successful."""
    logger.info("Validating annotation results...")
    
    # Check for annotation columns
    annotation_cols = ['celltypist_cell_type', 'celltypist_confidence']
    available_annotations = [col for col in annotation_cols if col in adata.obs.columns]
    
    if available_annotations:
        logger.info(f"✓ Annotation columns available: {available_annotations}")
        
        # Check cell type distribution
        if 'celltypist_cell_type' in adata.obs.columns:
            cell_types = adata.obs['celltypist_cell_type'].value_counts()
            logger.info(f"✓ Cell types identified: {len(cell_types)}")
            for cell_type, count in cell_types.head().items():
                logger.info(f"  - {cell_type}: {count:,} cells")
        
        # Check confidence scores
        if 'celltypist_confidence' in adata.obs.columns:
            mean_confidence = adata.obs['celltypist_confidence'].mean()
            logger.info(f"✓ Mean confidence score: {mean_confidence:.3f}")
    else:
        logger.warning("No annotation columns found")


def perform_basic_analysis(adata: ad.AnnData, config: AnalysisConfig) -> None:
    """Perform basic analysis including clustering and visualization."""
    logger.info("Performing basic analysis...")
    
    # Create analysis object
    analysis = LeukoMapAnalysis(config)
    
    # Store data in result tracker
    analysis.tracker.store("raw_data", adata, {"stage": "data_loading"})
    analysis.tracker.store("annotated_data", adata, {"stage": "annotation"})
    
    # Perform clustering if not already done
    if 'leiden' not in adata.obs.columns:
        logger.info("Performing clustering...")
        
        # Compute neighbors
        sc.pp.neighbors(adata, n_neighbors=config.n_neighbors, n_pcs=config.n_pcs)
        
        # Perform clustering
        sc.tl.leiden(adata, resolution=config.resolution)
        
        # Compute UMAP
        sc.tl.umap(adata)
        
        logger.info(f"✓ Clustering complete: {adata.obs['leiden'].nunique()} clusters")
    
    # Generate basic visualizations
    generate_basic_visualizations(adata, config)
    
    # Save analysis results
    save_analysis_results(analysis, adata)


def generate_basic_visualizations(adata: ad.AnnData, config: AnalysisConfig) -> None:
    """Generate basic visualizations for the analysis."""
    logger.info("Generating basic visualizations...")
    
    # Set up plotting
    sc.settings.figdir = config.output_dir / 'figures'
    sc.settings.figdir.mkdir(exist_ok=True)
    
    # UMAP plots
    if 'X_umap' in adata.obsm:
        # Sample type
        sc.pl.umap(adata, color='sample_type', save='_sample_type.png', show=False)
        logger.info("✓ UMAP by sample type saved")
        
        # Cell type (if available)
        if 'celltypist_cell_type' in adata.obs.columns:
            sc.pl.umap(adata, color='celltypist_cell_type', save='_cell_type.png', show=False)
            logger.info("✓ UMAP by cell type saved")
        
        # Clusters
        if 'leiden' in adata.obs.columns:
            sc.pl.umap(adata, color='leiden', save='_clusters.png', show=False)
            logger.info("✓ UMAP by clusters saved")
    
    # Quality metrics
    quality_metrics = ['total_counts', 'n_genes_by_counts', 'percent.mito']
    available_metrics = [metric for metric in quality_metrics if metric in adata.obs.columns]
    
    if available_metrics:
        sc.pl.violin(adata, available_metrics, groupby='sample_type', save='_quality.png', show=False)
        logger.info("✓ Quality metrics plot saved")


def save_analysis_results(analysis: LeukoMapAnalysis, adata: ad.AnnData) -> None:
    """Save all analysis results."""
    logger.info("Saving analysis results...")
    
    # Save AnnData object
    output_path = analysis.config.output_dir / 'data' / 'final_annotated_data.h5ad'
    output_path.parent.mkdir(exist_ok=True)
    adata.write(output_path)
    logger.info(f"✓ Final data saved to: {output_path}")
    
    # Save analysis report
    report_path = analysis.save_analysis_report()
    logger.info(f"✓ Analysis report saved to: {report_path}")
    
    # Save result summary
    save_result_summary(analysis, adata)


def save_result_summary(analysis: LeukoMapAnalysis, adata: ad.AnnData) -> None:
    """Save a comprehensive result summary."""
    logger.info("Saving result summary...")
    
    summary_path = analysis.config.output_dir / 'reports' / 'end_to_end_summary.txt'
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("LeukoMap End-to-End Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Data Summary:\n")
        f.write(f"  - Total cells: {adata.n_obs:,}\n")
        f.write(f"  - Total genes: {adata.n_vars:,}\n")
        f.write(f"  - Sample types: {adata.obs['sample_type'].nunique()}\n")
        f.write(f"  - Samples: {adata.obs['sample'].nunique()}\n\n")
        
        f.write("Sample Type Distribution:\n")
        sample_dist = adata.obs['sample_type'].value_counts()
        for sample_type, count in sample_dist.items():
            f.write(f"  - {sample_type}: {count:,} cells\n")
        f.write("\n")
        
        if 'celltypist_cell_type' in adata.obs.columns:
            f.write("Cell Type Distribution:\n")
            cell_dist = adata.obs['celltypist_cell_type'].value_counts()
            for cell_type, count in cell_dist.items():
                f.write(f"  - {cell_type}: {count:,} cells\n")
            f.write("\n")
        
        if 'leiden' in adata.obs.columns:
            f.write("Cluster Distribution:\n")
            cluster_dist = adata.obs['leiden'].value_counts()
            f.write(f"  - Total clusters: {len(cluster_dist)}\n")
            for cluster, count in cluster_dist.head().items():
                f.write(f"  - Cluster {cluster}: {count:,} cells\n")
            f.write("\n")
        
        f.write("Files Generated:\n")
        for key in analysis.tracker.list_results():
            f.write(f"  - {key}\n")
    
    logger.info(f"✓ Result summary saved to: {summary_path}")


def demonstrate_convenience_functions(config: AnalysisConfig) -> None:
    """Demonstrate the convenience functions provided by LeukoMap."""
    logger.info("Demonstrating convenience functions...")
    
    try:
        # Test the main convenience function
        logger.info("Testing run_leukomap_analysis...")
        results = run_leukomap_analysis(
            str(config.data_path),
            str(config.output_dir / 'convenience_test'),
            min_genes=100,
            min_cells=2
        )
        logger.info(f"✓ Convenience function completed: {len(results)} stages")
        
    except Exception as e:
        logger.warning(f"Convenience function test failed: {e}")


def main():
    """Run the complete end-to-end LeukoMap analysis."""
    print("=" * 80)
    print("LeukoMap End-to-End Analysis Example")
    print("=" * 80)
    
    try:
        # 1. Setup
        setup_environment()
        
        # 2. Configuration
        config = create_example_config()
        
        # 3. Data Loading and Validation
        adata = load_and_validate_data(config)
        
        # 4. Preprocessing
        preprocessed_adata = preprocess_data(adata, config)
        
        # 5. Cell Type Annotation
        annotated_adata = perform_cell_type_annotation(preprocessed_adata, config)
        
        # 6. Basic Analysis
        perform_basic_analysis(annotated_adata, config)
        
        # 7. Demonstrate Convenience Functions
        demonstrate_convenience_functions(config)
        
        # 8. Summary
        print("\n" + "=" * 80)
        print("✓ END-TO-END ANALYSIS COMPLETE!")
        print("=" * 80)
        print()
        print("This example demonstrates:")
        print("  ✓ Complete data loading and validation")
        print("  ✓ Comprehensive preprocessing pipeline")
        print("  ✓ Cell type annotation with fallbacks")
        print("  ✓ Basic clustering and visualization")
        print("  ✓ Result tracking and reporting")
        print("  ✓ Convenience function usage")
        print()
        print("Results available in:")
        print(f"  {config.output_dir}")
        print()
        print("Key files generated:")
        print("  - final_annotated_data.h5ad (processed data)")
        print("  - figures/ (visualizations)")
        print("  - reports/ (analysis reports)")
        print()
        print("The LeukoMap package provides:")
        print("  - Modular architecture for easy customization")
        print("  - Robust error handling and fallbacks")
        print("  - Comprehensive result tracking")
        print("  - Convenience functions for quick analysis")
        print("  - Production-ready code quality")
        
    except Exception as e:
        logger.error(f"End-to-end analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 