"""
LeukoMap - Single-cell RNA-seq analysis for leukemia research.

A modular pipeline for analyzing single-cell RNA-seq data from leukemia samples,
including data loading, preprocessing, cell type annotation, and analysis.
"""

from .core import (
    AnalysisConfig,
    BaseProcessor,
    DataProcessor,
    Pipeline,
    ResultTracker,
    LeukoMapAnalysis,
    AnalysisStage
)

from .data_loading import (
    DataLoader,
    load_data
)

from .preprocessing import (
    PreprocessingPipeline,
    PreprocessingManager
)

from .cell_type_annotation import (
    CellTypeAnnotator,
    annotate_cell_types_simple
)

from .scvi_training import (
    SCVITrainer,
    train_models
)

__version__ = "2.0.0"
__author__ = "LeukoMap Team"

def analyze(scRNA_seq_data, healthy_reference=None, output_dir="results", **kwargs):
    """
    Main analysis function for LeukoMap.
    
    Args:
        scRNA_seq_data: Path to scRNA-seq data or AnnData object
        healthy_reference: Path to healthy reference data (optional)
        output_dir: Output directory for results
        **kwargs: Additional configuration parameters
        
    Returns:
        Dict containing annotated_clusters and druggable_targets
    """
    from pathlib import Path
    import scanpy as sc
    import anndata as ad
    
    # Create configuration
    config = AnalysisConfig(
        output_dir=Path(output_dir),
        **kwargs
    )
    
    # Load data if path provided
    if isinstance(scRNA_seq_data, (str, Path)):
        if Path(scRNA_seq_data).exists():
            adata = sc.read_h5ad(scRNA_seq_data)
        else:
            raise FileNotFoundError(f"Data file not found: {scRNA_seq_data}")
    elif isinstance(scRNA_seq_data, ad.AnnData):
        adata = scRNA_seq_data
    else:
        raise ValueError("scRNA_seq_data must be a file path or AnnData object")
    
    # Initialize analysis
    analysis = LeukoMapAnalysis(config)
    
    # Run complete pipeline
    results = analysis.run_full_analysis_with_data(adata, healthy_reference)
    
    return {
        'annotated_clusters': results.get('annotated_data'),
        'druggable_targets': results.get('druggable_targets'),
        'differential_expression': results.get('differential_expression'),
        'analysis_report': results.get('analysis_report')
    }

# Export main classes and functions
__all__ = [
    # Core classes
    'AnalysisConfig',
    'BaseProcessor', 
    'DataProcessor',
    'Pipeline',
    'ResultTracker',
    'LeukoMapAnalysis',
    'AnalysisStage',
    
    # Data classes
    'DataLoader',
    'load_data',
    
    # Preprocessing classes
    'PreprocessingPipeline',
    'PreprocessingManager',
    
    # Annotation classes
    'CellTypeAnnotator',
    'annotate_cell_types_simple',
    
    # Training classes
    'SCVITrainer',
    'train_models',
    
    # Main function
    'analyze'
] 