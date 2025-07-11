#!/usr/bin/env python3
"""
Demo script to load Caron et al. (2020) scRNA-seq data using LeukoMap pipeline.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from leukomap.data_loading import load_data

if __name__ == "__main__":
    print("Loading data from ./data ...")
    adata = load_data("data")
    print("\n=== AnnData object ===")
    print(adata)
    print("\n=== .obs (cell metadata) ===")
    print(adata.obs.head())
    print("\n=== .var (gene metadata) ===")
    print(adata.var.head())
    print("\n=== .uns (unstructured metadata) ===")
    print(adata.uns)
    print("\nDone.") 