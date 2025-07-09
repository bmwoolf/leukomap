"""
Preprocessing module for LeukoMap pipeline.

This module contains the preprocess() function and related preprocessing utilities.
"""

import logging
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_genes: int = 6000,
    max_counts: int = 50000,
    max_mito: float = 0.2,
    save_path: Optional[str] = "cache/adata_preprocessed.h5ad"
) -> ad.AnnData:
    """
    Preprocess AnnData for scVI/scANVI and downstream analysis.
    Steps:
      1. Filter cells and genes by quality control metrics.
      2. Annotate mitochondrial gene content.
      3. Remove cells with high mitochondrial content.
      4. Annotate/correct batch/sample columns.
      5. (Optional) Compute highly variable genes (for visualization).
      6. Save checkpoint of preprocessed AnnData.
    Returns:
      Preprocessed AnnData object (raw counts, filtered, annotated).
    """
    logger.info("Starting preprocessing...")

    # 1. Annotate mitochondrial gene content first
    mito_genes = adata.var_names.str.upper().str.startswith('MT-')
    mito_counts = np.asarray(adata[:, mito_genes].X.sum(axis=1)).flatten()
    total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    adata.obs['pct_counts_mt'] = mito_counts / total_counts
    logger.info("Annotated mitochondrial gene content.")

    # 2. Filter cells and genes by quality control metrics
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info(f"After initial filtering: {adata.n_obs} cells, {adata.n_vars} genes")

    # 3. Apply additional QC filters
    adata = adata[adata.obs['n_genes'] < max_genes, :]
    adata = adata[adata.obs['n_genes'] > min_genes, :]
    adata = adata[adata.obs['pct_counts_mt'] < max_mito, :]
    if 'n_counts' in adata.obs.columns:
        adata = adata[adata.obs['n_counts'] < max_counts, :]
    logger.info(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")

    # Check for empty AnnData after filtering
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData is empty after filtering (no cells or genes remain). Adjust QC thresholds or check input data.")

    # 4. Annotate/correct batch/sample columns
    for col in ['sample', 'sample_type']:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype('category')
    logger.info("Annotated batch/sample columns.")

    # 5. Compute highly variable genes (for visualization)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', subset=False)
    logger.info("Computed highly variable genes.")

    # 6. Save checkpoint
    if save_path:
        adata.write(save_path)
        logger.info(f"Saved preprocessed AnnData to {save_path}")

    logger.info("Preprocessing complete.")
    return adata 