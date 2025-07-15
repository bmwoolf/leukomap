"""
Preprocessing module for LeukoMap - Simplified.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any
import warnings
import json

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

from .core import DataProcessor, AnalysisStage, AnalysisConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")


def _to_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_python_types(v) for v in obj]
    elif hasattr(obj, 'item') and callable(obj.item):
        return obj.item()
    else:
        return obj


class PreprocessingPipeline(DataProcessor):
    """Complete preprocessing pipeline - QC, normalization, feature selection, scaling."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.qc_metrics: Dict[str, Any] = {}
        self.selected_features: List[str] = []
        
    def get_stage(self) -> AnalysisStage:
        return AnalysisStage.PREPROCESSING
    
    def process(self, data: ad.AnnData) -> ad.AnnData:
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        self.logger.info("Starting preprocessing pipeline")
        
        # Calculate QC metrics if not present
        if 'total_counts' not in data.obs.columns:
            sc.pp.calculate_qc_metrics(data, inplace=True)
        
        # Store original metrics
        self.qc_metrics['original'] = {
            'n_cells': data.n_obs,
            'n_genes': data.n_vars,
            'total_counts': data.X.sum() if hasattr(data.X, 'sum') else None
        }
        
        # Step 1: Quality control
        self.logger.info("Step 1: Quality control")
        data = self._filter_cells(data)
        data = self._filter_genes(data)
        
        # Step 2: Normalization (for scVI - keep non-negative)
        self.logger.info("Step 2: Normalization")
        sc.pp.normalize_total(data, target_sum=self.config.target_sum)
        sc.pp.log1p(data)
        
        # Store the log-normalized data for scVI (non-negative) BEFORE feature selection
        data.layers['scvi_input'] = data.X.copy()
        
        # Step 3: Feature selection
        self.logger.info("Step 3: Feature selection")
        sc.pp.highly_variable_genes(data, n_top_genes=2000)
        self.selected_features = data.var_names[data.var['highly_variable']].tolist()
        
        # Store the log-normalized data for scVI (non-negative) BEFORE feature selection
        data.raw = data.copy()
        
        # Now subset to highly variable genes
        data = data[:, data.var['highly_variable']]
        
        # Step 4: Scaling (for PCA/UMAP - can have negative values)
        self.logger.info("Step 4: Scaling")
        sc.pp.scale(data, max_value=10)
        
        # Store final metrics
        self.qc_metrics['final'] = {
            'n_cells': data.n_obs,
            'n_genes': data.n_vars,
            'total_counts': data.X.sum() if hasattr(data.X, 'sum') else None
        }
        
        self.logger.info(f"Preprocessing complete: {data.n_obs} cells, {data.n_vars} genes")
        self.logger.info("Note: Raw (log-normalized) data stored in adata.raw for scVI")
        self.logger.info("Note: Scaled data in adata.X for PCA/UMAP")
        return data
    
    def _filter_cells(self, adata: ad.AnnData) -> ad.AnnData:
        self.logger.info("Filtering cells")
        sc.pp.filter_cells(adata, min_genes=self.config.min_genes)
        
        if self.config.max_counts is not None:
            adata = adata[adata.obs['total_counts'] <= self.config.max_counts]
        
        if self.config.max_genes is not None:
            adata = adata[adata.obs['n_genes_by_counts'] <= self.config.max_genes]
        
        if 'pct_counts_mt' in adata.obs.columns:
            adata = adata[adata.obs['pct_counts_mt'] <= 20]
        
        self.logger.info(f"Cell filtering complete: {adata.n_obs} cells remaining")
        return adata
    
    def _filter_genes(self, adata: ad.AnnData) -> ad.AnnData:
        self.logger.info("Filtering genes")
        sc.pp.filter_genes(adata, min_cells=self.config.min_cells)
        self.logger.info(f"Gene filtering complete: {adata.n_vars} genes remaining")
        return adata
    
    def save_results(self, data: ad.AnnData, output_path: Path) -> None:
        super().save_results(data, output_path)
        
        # Save QC metrics
        qc_path = output_path.parent / f"{output_path.stem}_qc_metrics.json"
        with open(qc_path, 'w') as f:
            json.dump(_to_python_types(self.qc_metrics), f, indent=2)
        
        # Save selected features
        features_path = output_path.parent / f"{output_path.stem}_selected_features.txt"
        with open(features_path, 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        self.logger.info(f"Preprocessing results saved to {output_path}")


class PreprocessingManager:
    """High-level preprocessing management interface."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.pipeline = PreprocessingPipeline(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def preprocess_data(self, adata: ad.AnnData, save_results: bool = True) -> ad.AnnData:
        self.logger.info("Starting data preprocessing")
        preprocessed_adata = self.pipeline.process(adata)
        
        if save_results:
            output_path = self.config.output_dir / 'data' / 'preprocessed_data.h5ad'
            self.pipeline.save_results(preprocessed_adata, output_path)
        
        self.logger.info("Data preprocessing complete")
        return preprocessed_adata
    
    def generate_preprocessing_report(self, original_adata: ad.AnnData, 
                                    preprocessed_adata: ad.AnnData) -> Path:
        report_path = self.config.output_dir / 'reports' / 'preprocessing_report.txt'
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PREPROCESSING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Original data stats
        report_lines.append("ORIGINAL DATA:")
        report_lines.append(f"  Cells: {original_adata.n_obs:,}")
        report_lines.append(f"  Genes: {original_adata.n_vars:,}")
        report_lines.append("")
        
        # Preprocessed data stats
        report_lines.append("PREPROCESSED DATA:")
        report_lines.append(f"  Cells: {preprocessed_adata.n_obs:,}")
        report_lines.append(f"  Genes: {preprocessed_adata.n_vars:,}")
        report_lines.append("")
        
        # Filtering summary
        cells_removed = original_adata.n_obs - preprocessed_adata.n_obs
        genes_removed = original_adata.n_vars - preprocessed_adata.n_vars
        
        report_lines.append("FILTERING SUMMARY:")
        report_lines.append(f"  Cells removed: {cells_removed:,} ({cells_removed/original_adata.n_obs*100:.1f}%)")
        report_lines.append(f"  Genes removed: {genes_removed:,} ({genes_removed/original_adata.n_vars*100:.1f}%)")
        report_lines.append("")
        
        # Preprocessing steps
        report_lines.append("PREPROCESSING STEPS:")
        report_lines.append("  1. Quality control (cell and gene filtering)")
        report_lines.append("  2. Library size normalization")
        report_lines.append("  3. Log transformation")
        report_lines.append("  4. Highly variable gene selection")
        report_lines.append("  5. Scaling to unit variance")
        report_lines.append("")
        
        # Save report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Preprocessing report saved to {report_path}")
        return report_path 