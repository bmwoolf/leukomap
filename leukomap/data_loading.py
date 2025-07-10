"""
Data loading module for LeukoMap pipeline.

Handles loading of scRNA-seq data from various formats including:
- 10X Genomics format (MTX files)
- Cell annotations (TSV format)
- Multiple sample types from Caron et al. (2020)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import warnings

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress scanpy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")


def load_data(data_dir: Union[str, Path]) -> ad.AnnData:
    """
    Load scRNA-seq data from directory into AnnData format.
    
    This function handles the Caron et al. (2020) pediatric leukemia dataset
    which contains multiple sample types:
    - ETV6-RUNX1 (t(12;21) acute lymphoblastic leukemia)
    - HHD (High hyperdiploid acute lymphoblastic leukemia) 
    - PRE-T (Pre-T acute lymphoblastic leukemia)
    - PBMMC (Healthy pediatric bone marrow mononuclear cells)
    
    Returns
    -------
    ad.AnnData
        AnnData object containing:
        - .X: sparse count matrix
        - .obs: cell metadata (sample, cell_type, etc.)
        - .var: gene metadata
        - .obsm: embeddings (tSNE, UMAP if available)
        - .uns: unstructured metadata
    """
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info(f"Loading scRNA-seq data from: {data_dir}")
    
    # Check for different data formats and load accordingly
    if _is_10x_format(data_dir):
        adata = _load_10x_data(data_dir)
    elif _is_processed_format(data_dir):
        adata = _load_processed_data(data_dir)
    else:
        raise ValueError(f"Unsupported data format in: {data_dir}")
    
    # Load cell annotations if available
    adata = _load_cell_annotations(adata, data_dir)
    
    # Validate and clean the data
    adata = _validate_and_clean_data(adata)
    
    logger.info(f"Successfully loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    return adata


def _is_10x_format(data_dir: Path) -> bool:
    """Check if directory contains 10X Genomics format data."""
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        return False
    
    # Check for at least one sample directory with matrix.mtx (compressed or uncompressed)
    sample_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    for sample_dir in sample_dirs:
        if (sample_dir / "matrix.mtx").exists() or (sample_dir / "matrix.mtx.gz").exists():
            return True
    
    return False


def _is_processed_format(data_dir: Path) -> bool:
    """Check if directory contains processed data files."""
    processed_files = [
        "data.h5ad",
        "data.h5",
        "expression_matrix.csv",
        "expression_matrix.tsv"
    ]
    
    for file in processed_files:
        if (data_dir / file).exists():
            return True
    
    return False


def _load_10x_data(data_dir: Path) -> ad.AnnData:
    """Load 10X Genomics format data from multiple samples."""
    raw_dir = data_dir / "raw"
    
    # Expected sample names from Caron et al. (2020)
    expected_samples = [
        "ETV6-RUNX1_1", "ETV6-RUNX1_2", "ETV6-RUNX1_3", "ETV6-RUNX1_4",
        "HHD_1", "HHD_2", 
        "PRE-T_1", "PRE-T_2",
        "PBMMC_1", "PBMMC_2", "PBMMC_3"
    ]
    
    adata_list = []
    
    for sample_name in expected_samples:
        sample_dir = raw_dir / sample_name
        
        if not sample_dir.exists():
            logger.warning(f"Sample directory not found: {sample_dir}")
            continue
        
        # Accept both compressed and uncompressed files
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
            logger.warning(f"Missing files in: {sample_dir}")
            continue
        
        try:
            # Load individual sample - handle both old (genes.tsv) and new (features.tsv) formats
            sample_adata = sc.read_10x_mtx(
                sample_dir,
                var_names='gene_symbols',  # Use gene symbol (second column)
                cache=True
            )
            # Add sample information
            sample_adata.obs['sample'] = sample_name
            sample_adata.obs['sample_type'] = _extract_sample_type(sample_name)
            adata_list.append(sample_adata)
            logger.info(f"Loaded sample: {sample_name} ({sample_adata.n_obs} cells)")
        except Exception as e:
            logger.error(f"Failed to load sample {sample_name}: {e}")
            continue
    
    if not adata_list:
        raise ValueError("No valid samples found in data directory")
    
    # Concatenate all samples
    adata = ad.concat(adata_list, join='outer', index_unique=None)
    
    # Ensure gene names are unique
    adata.var_names_make_unique()
    
    return adata


def _load_processed_data(data_dir: Path) -> ad.AnnData:
    """Load pre-processed data files."""
    
    # Try different file formats
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


def _extract_sample_type(sample_name: str) -> str:
    """Extract sample type from sample name."""
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


def _load_cell_annotations(adata: ad.AnnData, data_dir: Path) -> ad.AnnData:
    """Load cell annotations from TSV file if available, harmonizing cell IDs."""
    import re
    # Look for annotation files
    annotation_files = [
        "GSE132509_cell_annotations.tsv",
        "cell_annotations.tsv",
        "annotations.tsv"
    ]
    for ann_file in annotation_files:
        ann_path = data_dir / "annotations" / ann_file
        if ann_path.exists():
            try:
                annotations = pd.read_csv(ann_path, sep='\t', index_col=0)
                # Harmonize annotation cell IDs to match AnnData barcodes
                def harmonize(cell_id):
                    # Remove sample prefix (up to last underscore), add -1
                    return cell_id.split('_')[-1] + '-1'
                annotations['barcode'] = annotations.index.map(harmonize)
                # Drop duplicate barcodes, keeping the first occurrence
                annotations = annotations[~annotations['barcode'].duplicated(keep='first')]
                annotations = annotations.set_index('barcode')
                # Match annotations to cells
                common_cells = adata.obs.index.intersection(annotations.index)
                if len(common_cells) > 0:
                    # Add annotations to adata.obs
                    for col in annotations.columns:
                        if col not in adata.obs.columns:
                            adata.obs.loc[common_cells, col] = annotations.loc[common_cells, col]
                    logger.info(f"Loaded annotations for {len(common_cells)} cells")
                else:
                    logger.warning("No matching cell IDs found in annotations after harmonization")
            except Exception as e:
                logger.error(f"Failed to load annotations from {ann_path}: {e}")
    return adata


def _validate_and_clean_data(adata: ad.AnnData) -> ad.AnnData:
    """Validate and clean the loaded data."""
    
    # Basic validation
    if adata.n_obs == 0:
        raise ValueError("No cells found in data")
    
    if adata.n_vars == 0:
        raise ValueError("No genes found in data")
    
    # Remove cells with no counts
    sc.pp.filter_cells(adata, min_genes=1)
    
    # Remove genes with no counts
    sc.pp.filter_genes(adata, min_cells=1)
    
    # Ensure gene names are strings
    adata.var_names = adata.var_names.astype(str)
    
    # Ensure cell names are strings
    adata.obs_names = adata.obs_names.astype(str)
    
    # Add basic metadata
    adata.uns['data_source'] = 'Caron et al. (2020) - GSE132509'
    adata.uns['n_cells'] = adata.n_obs
    adata.uns['n_genes'] = adata.n_vars
    
    logger.info(f"Data validation complete: {adata.n_obs} cells, {adata.n_vars} genes")
    
    return adata


def download_caron_data(output_dir: Union[str, Path], force: bool = False) -> Path:
    """
    Download Caron et al. (2020) data from GEO.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save downloaded data
    force : bool, default False
        Force re-download if data already exists
    
    Returns
    -------
    Path
        Path to downloaded data directory
    """
    # This would implement actual download logic
    # For now, return the expected structure
    output_dir = Path(output_dir)
    
    if output_dir.exists() and not force:
        logger.info(f"Data already exists at: {output_dir}")
        return output_dir
    
    logger.info("Download functionality not yet implemented")
    logger.info("Please manually download GSE132509 from GEO and extract to:")
    logger.info(f"  {output_dir}")
    
    return output_dir 