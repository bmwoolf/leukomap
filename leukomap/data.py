"""
Data loading and management module for LeukoMap - Simplified.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any
import warnings

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

from .core import DataProcessor, AnalysisStage, AnalysisConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")


class DataLoader(DataProcessor):
    """Load and manage single-cell RNA-seq data."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.data_path: Optional[Path] = None
        
    def get_stage(self) -> AnalysisStage:
        return AnalysisStage.DATA_LOADING
    
    def process(self, data: Optional[Any] = None) -> ad.AnnData:
        if self.config.data_path is None:
            raise ValueError("No data path specified in configuration")
        
        self.data_path = Path(self.config.data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        self.logger.info(f"Loading data from: {self.data_path}")
        
        if self._is_10x_format(self.data_path):
            self.adata = self._load_10x_data(self.data_path)
        elif self._is_processed_format(self.data_path):
            self.adata = self._load_processed_data(self.data_path)
        else:
            raise ValueError(f"Unsupported data format in: {self.data_path}")
        
        # Ensure AnnData is dense and obs/var columns are string after loading from disk
        self._ensure_dense_and_string(self.adata)
        
        self.adata = self._load_cell_annotations(self.adata, self.data_path)
        self.adata = self._validate_and_clean_data(self.adata)
        
        self.logger.info(f"Successfully loaded data: {self.adata.n_obs} cells, {self.adata.n_vars} genes")
        return self.adata

    def _ensure_dense_and_string(self, adata: ad.AnnData):
        # Convert obs/var columns to string if categorical
        for col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
        for col in adata.var.columns:
            if pd.api.types.is_categorical_dtype(adata.var[col]):
                adata.var[col] = adata.var[col].astype(str)
        # Convert X to dense if needed
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        # Ensure obs/var index is string
        adata.obs.index = adata.obs.index.astype(str)
        adata.var.index = adata.var.index.astype(str)
    
    def _is_10x_format(self, data_dir: Path) -> bool:
        raw_dir = data_dir / "raw"
        if not raw_dir.exists():
            return False
        sample_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        for sample_dir in sample_dirs:
            if (sample_dir / "matrix.mtx").exists() or (sample_dir / "matrix.mtx.gz").exists():
                return True
        return False
    
    def _is_processed_format(self, data_dir: Path) -> bool:
        processed_files = ["data.h5ad", "data.h5", "expression_matrix.csv", "expression_matrix.tsv"]
        for file in processed_files:
            if (data_dir / file).exists():
                return True
        return False
    
    def _load_10x_data(self, data_dir: Path) -> ad.AnnData:
        raw_dir = data_dir / "raw"
        expected_samples = [
            "ETV6-RUNX1_1", "ETV6-RUNX1_2", "ETV6-RUNX1_3", "ETV6-RUNX1_4",
            "HHD_1", "HHD_2", "PRE-T_1", "PRE-T_2", "PBMMC_1", "PBMMC_2", "PBMMC_3"
        ]
        
        adata_list = []
        for sample_name in expected_samples:
            sample_dir = raw_dir / sample_name
            if not sample_dir.exists():
                self.logger.warning(f"Sample directory not found: {sample_dir}")
                continue
            
            def find_file(basename):
                for ext in ["", ".gz"]:
                    f = sample_dir / f"{basename}{ext}"
                    if f.exists():
                        return f
                return None
            
            matrix_file = find_file("matrix.mtx")
            barcodes_file = find_file("barcodes.tsv")
            genes_file = find_file("genes.tsv")
            
            if not (matrix_file and barcodes_file and genes_file):
                self.logger.warning(f"Missing files in: {sample_dir}")
                continue
            
            try:
                sample_adata = sc.read_10x_mtx(sample_dir, var_names='gene_symbols', cache=True)
                sample_adata.obs['sample'] = sample_name
                sample_adata.obs['sample_type'] = self._extract_sample_type(sample_name)
                adata_list.append(sample_adata)
                self.logger.info(f"Loaded sample: {sample_name} ({sample_adata.n_obs} cells)")
            except Exception as e:
                self.logger.error(f"Failed to load sample {sample_name}: {e}")
                continue
        
        if not adata_list:
            raise ValueError("No valid samples found in data directory")
        
        adata = ad.concat(adata_list, join='outer', index_unique=None)
        adata.var_names_make_unique()
        return adata
    
    def _load_processed_data(self, data_dir: Path) -> ad.AnnData:
        if (data_dir / "data.h5ad").exists():
            return sc.read_h5ad(data_dir / "data.h5ad")
        elif (data_dir / "data.h5").exists():
            return sc.read_h5ad(data_dir / "data.h5")
        elif (data_dir / "expression_matrix.csv").exists():
            return sc.read_csv(data_dir / "expression_matrix.csv")
        elif (data_dir / "expression_matrix.tsv").exists():
            return sc.read_csv(data_dir / "expression_matrix.tsv", delimiter='\t')
        else:
            raise FileNotFoundError("No supported processed data files found")
    
    def _extract_sample_type(self, sample_name: str) -> str:
        if sample_name.startswith("ETV6-RUNX1"):
            return "ETV6-RUNX1"
        elif sample_name.startswith("HHD"):
            return "HHD"
        elif sample_name.startswith("PRE-T"):
            return "PRE-T"
        elif sample_name.startswith("PBMMC"):
            return "PBMMC"
        else:
            return "Unknown"
    
    def _load_cell_annotations(self, adata: ad.AnnData, data_dir: Path) -> ad.AnnData:
        annotation_files = ["cell_annotations.csv", "metadata.csv", "annotations.tsv"]
        
        for file_name in annotation_files:
            file_path = data_dir / file_name
            if file_path.exists():
                try:
                    annotations = pd.read_csv(file_path, index_col=0)
                    for col in annotations.columns:
                        if col not in adata.obs.columns:
                            adata.obs[col] = annotations[col]
                    self.logger.info(f"Loaded cell annotations from {file_name}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load annotations from {file_name}: {e}")
        
        return adata
    
    def _validate_and_clean_data(self, adata: ad.AnnData) -> ad.AnnData:
        # Ensure gene names are unique
        adata.var_names_make_unique()
        
        # Add basic metadata if missing
        if 'sample' not in adata.obs.columns:
            adata.obs['sample'] = 'unknown'
        if 'sample_type' not in adata.obs.columns:
            adata.obs['sample_type'] = 'unknown'
        
        # Ensure obs/var columns are string type (not categorical)
        for col in adata.obs.columns:
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)
        for col in adata.var.columns:
            if pd.api.types.is_categorical_dtype(adata.var[col]):
                adata.var[col] = adata.var[col].astype(str)
        
        # Ensure matrix is dense for scanpy QC metrics
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        
        # Calculate basic QC metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        return adata
    
    def save_results(self, data: ad.AnnData, output_path: Path) -> None:
        super().save_results(data, output_path)
        
        # Save data summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Data Summary\n")
            f.write(f"============\n")
            f.write(f"Cells: {data.n_obs:,}\n")
            f.write(f"Genes: {data.n_vars:,}\n")
            f.write(f"Samples: {data.obs['sample'].nunique()}\n")
            f.write(f"Sample types: {data.obs['sample_type'].nunique()}\n")
        
        self.logger.info(f"Data summary saved to {summary_path}")


class DataValidator:
    """Validate data quality and generate reports."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_adata(self, adata: ad.AnnData) -> Dict[str, Any]:
        validation_results = {
            'basic_stats': self._get_basic_stats(adata),
            'quality_metrics': self._get_quality_metrics(adata),
            'sample_distribution': self._get_sample_distribution(adata),
            'issues': self._identify_issues(adata)
        }
        return validation_results
    
    def _get_basic_stats(self, adata: ad.AnnData) -> Dict[str, Any]:
        return {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'n_samples': adata.obs['sample'].nunique() if 'sample' in adata.obs.columns else 0,
            'n_sample_types': adata.obs['sample_type'].nunique() if 'sample_type' in adata.obs.columns else 0
        }
    
    def _get_quality_metrics(self, adata: ad.AnnData) -> Dict[str, Any]:
        if 'total_counts' not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        return {
            'mean_counts_per_cell': float(adata.obs['total_counts'].mean()),
            'median_counts_per_cell': float(adata.obs['total_counts'].median()),
            'mean_genes_per_cell': float(adata.obs['n_genes_by_counts'].mean()),
            'median_genes_per_cell': float(adata.obs['n_genes_by_counts'].median()),
            'total_counts': float(adata.obs['total_counts'].sum())
        }
    
    def _get_sample_distribution(self, adata: ad.AnnData) -> Dict[str, Any]:
        if 'sample' not in adata.obs.columns:
            return {'samples': {}}
        
        sample_counts = adata.obs['sample'].value_counts().to_dict()
        return {'samples': sample_counts}
    
    def _identify_issues(self, adata: ad.AnnData) -> List[str]:
        issues = []
        
        if adata.n_obs == 0:
            issues.append("No cells in dataset")
        if adata.n_vars == 0:
            issues.append("No genes in dataset")
        
        if 'total_counts' in adata.obs.columns:
            if adata.obs['total_counts'].max() > 100000:
                issues.append("Some cells have very high counts (>100k)")
            if adata.obs['total_counts'].min() < 100:
                issues.append("Some cells have very low counts (<100)")
        
        return issues
    
    def generate_validation_report(self, adata: ad.AnnData, output_path: Path) -> None:
        validation_results = self.validate_adata(adata)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Basic stats
        report_lines.append("BASIC STATISTICS:")
        for key, value in validation_results['basic_stats'].items():
            report_lines.append(f"  {key}: {value:,}")
        report_lines.append("")
        
        # Quality metrics
        report_lines.append("QUALITY METRICS:")
        for key, value in validation_results['quality_metrics'].items():
            report_lines.append(f"  {key}: {value:,.2f}")
        report_lines.append("")
        
        # Sample distribution
        if validation_results['sample_distribution']['samples']:
            report_lines.append("SAMPLE DISTRIBUTION:")
            for sample, count in validation_results['sample_distribution']['samples'].items():
                report_lines.append(f"  {sample}: {count:,} cells")
            report_lines.append("")
        
        # Issues
        if validation_results['issues']:
            report_lines.append("IDENTIFIED ISSUES:")
            for issue in validation_results['issues']:
                report_lines.append(f"  - {issue}")
            report_lines.append("")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Validation report saved to {output_path}")


class DataManager:
    """High-level data management interface."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.loader = DataLoader(config)
        self.validator = DataValidator(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_and_validate(self, data_path: Optional[Path] = None) -> ad.AnnData:
        if data_path:
            self.config.data_path = data_path
        
        # Load data
        adata = self.loader.process()
        
        # Validate data
        validation_results = self.validator.validate_adata(adata)
        
        # Generate validation report
        report_path = self.config.output_dir / 'reports' / 'data_validation_report.txt'
        report_path.parent.mkdir(exist_ok=True)
        self.validator.generate_validation_report(adata, report_path)
        
        # Log validation summary
        if validation_results['issues']:
            self.logger.warning(f"Data validation found {len(validation_results['issues'])} issues")
        else:
            self.logger.info("Data validation passed")
        
        return adata
    
    def save_data(self, adata: ad.AnnData, filename: str = "loaded_data.h5ad") -> Path:
        output_path = self.config.output_dir / 'data' / filename
        adata.write(output_path)
        self.logger.info(f"Data saved to {output_path}")
        return output_path 