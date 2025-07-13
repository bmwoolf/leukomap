"""
Core module for LeukoMap - Simplified base classes and interfaces.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning)


class AnalysisStage(Enum):
    """Analysis stages."""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    ANNOTATION = "annotation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    data_path: Optional[Path] = None
    output_dir: Path = Path("results")
    min_genes: int = 200
    min_cells: int = 3
    max_counts: Optional[int] = None
    max_genes: Optional[int] = None
    target_sum: int = 10000
    n_latent: int = 10
    n_hidden: int = 128
    n_layers: int = 2
    dropout_rate: float = 0.1
    batch_size: int = 128
    max_epochs: int = 400
    learning_rate: float = 1e-3
    annotation_methods: List[str] = field(default_factory=lambda: ['celltypist'])
    confidence_threshold: float = 0.7
    resolution: float = 0.5
    n_neighbors: int = 15
    n_pcs: int = 50
    figure_dpi: int = 300
    figure_format: str = 'png'
    
    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)
        for subdir in ['data', 'models', 'results', 'figures', 'reports']:
            (self.output_dir / subdir).mkdir(exist_ok=True)


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def save_results(self, data: Any, output_path: Path) -> None:
        pass
    
    def validate_input(self, data: Any) -> bool:
        return True
    
    def get_stage(self) -> AnalysisStage:
        return AnalysisStage.ANALYSIS


class DataProcessor(BaseProcessor):
    """Base class for data processing operations."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self._adata: Optional[ad.AnnData] = None
    
    @property
    def adata(self) -> Optional[ad.AnnData]:
        return self._adata
    
    @adata.setter
    def adata(self, value: ad.AnnData):
        self._adata = value
    
    def validate_input(self, data: ad.AnnData) -> bool:
        if not isinstance(data, ad.AnnData):
            self.logger.error("Input must be an AnnData object")
            return False
        if data.n_obs == 0 or data.n_vars == 0:
            self.logger.error("AnnData object is empty")
            return False
        return True
    
    def save_results(self, data: ad.AnnData, output_path: Path) -> None:
        if not self.validate_input(data):
            raise ValueError("Invalid AnnData object")
        data.write(output_path)
        self.logger.info(f"Saved AnnData to {output_path}")


class Pipeline:
    """Main analysis pipeline that orchestrates multiple processors."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processors: List[BaseProcessor] = []
        self.results: Dict[str, Any] = {}
        self.current_stage = AnalysisStage.DATA_LOADING
    
    def add_processor(self, processor: BaseProcessor) -> 'Pipeline':
        self.processors.append(processor)
        return self
    
    def run(self, initial_data: Optional[Any] = None) -> Dict[str, Any]:
        self.logger.info("Starting pipeline execution")
        current_data = initial_data
        
        for i, processor in enumerate(self.processors):
            try:
                self.logger.info(f"Running processor {i+1}/{len(self.processors)}: {processor.__class__.__name__}")
                current_data = processor.process(current_data)
                self.results[processor.__class__.__name__] = current_data
            except Exception as e:
                self.logger.error(f"Processor {processor.__class__.__name__} failed: {e}")
                raise
        
        self.logger.info("Pipeline execution complete")
        return self.results
    
    def get_results(self, stage: Optional[AnalysisStage] = None) -> Any:
        if stage is None:
            return self.results
        return {k: v for k, v in self.results.items() if hasattr(v, 'get_stage') and v.get_stage() == stage}
    
    def save_all_results(self) -> None:
        for processor in self.processors:
            if hasattr(processor, 'adata') and processor.adata is not None:
                output_path = self.config.output_dir / 'data' / f"{processor.__class__.__name__.lower()}_results.h5ad"
                processor.save_results(processor.adata, output_path)


class ResultTracker:
    """Simple result tracking."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def store(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.results[key] = data
        if metadata:
            self.metadata[key] = metadata
    
    def get(self, key: str) -> Any:
        return self.results.get(key)
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        return self.metadata.get(key, {})
    
    def save_to_disk(self, key: str, filename: Optional[str] = None) -> Path:
        if key not in self.results:
            raise KeyError(f"Result '{key}' not found")
        
        if filename is None:
            filename = f"{key}.h5ad"
        
        output_path = self.output_dir / filename
        
        data = self.results[key]
        if isinstance(data, ad.AnnData):
            data.write(output_path)
        else:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        
        return output_path
    
    def list_results(self) -> List[str]:
        return list(self.results.keys())


class LeukoMapAnalysis:
    """Main analysis class."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.pipeline = Pipeline(config)
        self.tracker = ResultTracker(config.output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_full_analysis(self, data_path: Optional[Path] = None) -> Dict[str, Any]:
        if data_path:
            self.config.data_path = data_path
        
        self.logger.info("Starting full LeukoMap analysis")
        
        # Add processors to pipeline
        from .data import DataLoader
        from .preprocessing import PreprocessingPipeline
        
        self.pipeline.add_processor(DataLoader(self.config))
        self.pipeline.add_processor(PreprocessingPipeline(self.config))
        
        # Run pipeline
        results = self.pipeline.run()
        
        # Store results
        for key, value in results.items():
            self.tracker.store(key, value)
        
        self.logger.info("Full analysis complete")
        return results
    
    def get_adata(self) -> Optional[ad.AnnData]:
        return self.tracker.get('PreprocessingPipeline')
    
    def save_analysis_report(self) -> Path:
        report_path = self.config.output_dir / 'reports' / 'analysis_report.txt'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("LeukoMap Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed: {len(self.tracker.list_results())} stages\n")
            f.write(f"Results available: {', '.join(self.tracker.list_results())}\n")
        
        return report_path 