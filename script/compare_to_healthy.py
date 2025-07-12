#!/usr/bin/env python3
"""
Compare leukemia samples to healthy bone marrow reference using scVI.
- Preprocess all PBMMC (healthy) and leukemia .h5ad files in cache/
- Integrate datasets, add sample_type labels
- Train scVI (GPU)
- Visualize UMAP colored by healthy/leukemia
- Perform differential expression
- Output UMAP and DE results
"""
import os
from pathlib import Path
import scanpy as sc
import anndata
import numpy as np
from scvi.model import SCVI
import torch

# --- Config ---
CACHE_DIR = Path("cache")
PBMMC_PREFIX = "data-raw-PBMMC"
LEUKEMIA_PREFIXES = [
    "data-raw-ETV6-RUNX1",
    "data-raw-PRE-T",
    "data-raw-HHD"
]
PREPROCESS_MIN_GENES = 200
PREPROCESS_MIN_CELLS = 3
N_HVG = 2000
SCVI_LATENT_DIM = 20
SCVI_EPOCHS = 100

# --- Helper: Preprocess AnnData ---
def preprocess_adata(adata):
    sc.pp.filter_cells(adata, min_genes=PREPROCESS_MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=PREPROCESS_MIN_CELLS)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=True, flavor="seurat_v3")
    return adata

# --- 1. Load all healthy and leukemia .h5ad files ---
healthy_files = sorted(CACHE_DIR.glob(f"{PBMMC_PREFIX}*_matrix.h5ad"))
leukemia_files = []
for prefix in LEUKEMIA_PREFIXES:
    leukemia_files.extend(sorted(CACHE_DIR.glob(f"{prefix}*_matrix.h5ad")))

print("Detected healthy files:")
for f in healthy_files:
    print(f"  {f}")
print("Detected leukemia files:")
for f in leukemia_files:
    print(f"  {f}")

if not healthy_files and not leukemia_files:
    print("ERROR: No healthy or leukemia .h5ad files found in cache/. Check file names and paths.")
    exit(1)

adatas = []
labels = []

for f in healthy_files:
    ad = sc.read_h5ad(f)
    ad.obs["sample_type"] = "healthy"
    ad.obs["sample_id"] = f.stem
    adatas.append(preprocess_adata(ad))
    labels.append("healthy")

for f in leukemia_files:
    ad = sc.read_h5ad(f)
    ad.obs["sample_type"] = "leukemia"
    ad.obs["sample_id"] = f.stem
    adatas.append(preprocess_adata(ad))
    labels.append("leukemia")

# --- 2. Integrate datasets ---
adata_combined = anndata.concat(adatas, join="outer", label="sample_id", keys=[ad.obs["sample_id"][0] for ad in adatas], fill_value=0)
adata_combined.obs["sample_type"] = adata_combined.obs["sample_type"].astype("category")

# --- 3. Train scVI (GPU if available) ---
scvi_device = "cuda" if torch.cuda.is_available() else "cpu"
SCVI.setup_anndata(adata_combined, batch_key="sample_type")
model = SCVI(adata_combined, n_latent=SCVI_LATENT_DIM)
model.train(max_epochs=SCVI_EPOCHS, use_gpu=(scvi_device=="cuda"))
adata_combined.obsm["X_scvi"] = model.get_latent_representation()

# --- 4. UMAP visualization ---
sc.pp.neighbors(adata_combined, use_rep="X_scvi")
sc.tl.umap(adata_combined)
sc.pl.umap(adata_combined, color="sample_type", save="_healthy_vs_leukemia.png", show=False)

# --- 5. Differential expression (leukemia vs healthy) ---
# Use scVI's built-in DE
de_df = model.differential_expression(groupby="sample_type", group1="leukemia", group2="healthy")
de_df.to_csv(CACHE_DIR / "de_leukemia_vs_healthy_scvi.csv")

print("\nAnalysis complete!")
print("- UMAP saved to: figures/umap_healthy_vs_leukemia.png")
print("- Differential expression results: cache/de_leukemia_vs_healthy_scvi.csv") 