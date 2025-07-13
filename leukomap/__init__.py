"""
LeukoMap - Simplified single-cell RNA-seq analysis for leukemia research.

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

from .data import (
    DataLoader,
    DataValidator,
    DataManager
)

from .preprocessing import (
    PreprocessingPipeline,
    PreprocessingManager
)

from .cell_type_annotation import (
    CellTypeAnnotator,
    annotate_cell_types_simple
)

# Legacy imports for backward compatibility
from .data_loading import load_data
from .scvi_training import train_models

__version__ = "2.0.0"
__author__ = "LeukoMap Team"

# Main analysis function
def run_leukomap_analysis(data_path: str, output_dir: str = "results", **kwargs):
    """
    Run complete LeukoMap analysis pipeline.
    
    Args:
        data_path: Path to data directory
        output_dir: Output directory for results
        **kwargs: Additional configuration parameters
        
    Returns:
        Analysis results
    """
    from pathlib import Path
    
    config = AnalysisConfig(
        data_path=Path(data_path),
        output_dir=Path(output_dir),
        **kwargs
    )
    
    analysis = LeukoMapAnalysis(config)
    return analysis.run_full_analysis()

# Convenience functions
def load_and_preprocess(data_path: str, output_dir: str = "results"):
    """Load and preprocess data."""
    from pathlib import Path
    
    config = AnalysisConfig(data_path=Path(data_path), output_dir=Path(output_dir))
    
    # Load data
    loader = DataLoader(config)
    adata = loader.process()
    
    # Preprocess data
    preprocessor = PreprocessingPipeline(config)
    adata = preprocessor.process(adata)
    
    return adata

def annotate_cells(adata, output_dir: str = "results"):
    """Annotate cell types."""
    from pathlib import Path
    
    config = AnalysisConfig(output_dir=Path(output_dir))
    annotator = CellTypeAnnotator(config)
    return annotator.process(adata)

# Export main classes
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
    'DataValidator', 
    'DataManager',
    
    # Preprocessing classes
    'PreprocessingPipeline',
    'PreprocessingManager',
    
    # Annotation classes
    'CellTypeAnnotator',
    'annotate_cell_types_simple',
    
    # Legacy functions
    'load_data',
    'train_models',
    
    # Main functions
    'run_leukomap_analysis',
    'load_and_preprocess',
    'annotate_cells'
] 