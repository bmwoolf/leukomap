#!/usr/bin/env python3
"""
Example script demonstrating the load_data function.

This script shows how to use the LeukoMap data loading functionality
to load scRNA-seq data from the Caron et al. (2020) dataset.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import leukomap
sys.path.append(str(Path(__file__).parent.parent))

from leukomap.data_loading import load_data, download_caron_data


def main():
    """Main function demonstrating data loading."""
    
    print("LeukoMap Data Loading Example")
    print("=" * 40)
    
    # Example 1: Load data from a directory
    data_dir = Path("data/caron_2020")
    
    if data_dir.exists():
        print(f"\n1. Loading data from: {data_dir}")
        try:
            adata = load_data(data_dir)
            print(f"   Successfully loaded: {adata.n_obs} cells, {adata.n_vars} genes")
            print(f"   Sample types: {adata.obs['sample_type'].unique()}")
            print(f"   Data source: {adata.uns.get('data_source', 'Unknown')}")
        except Exception as e:
            print(f"   Error loading data: {e}")
    else:
        print(f"\n1. Data directory not found: {data_dir}")
        print("   To use this example, download GSE132509 from GEO and extract to:")
        print(f"   {data_dir}")
    
    # Example 2: Show expected data structure
    print(f"\n2. Expected data structure:")
    print(f"   {data_dir}/")
    print(f"   ├── raw/")
    print(f"   │   ├── ETV6-RUNX1_1/")
    print(f"   │   │   ├── matrix.mtx")
    print(f"   │   │   ├── features.tsv")
    print(f"   │   │   └── barcodes.tsv")
    print(f"   │   ├── HHD_1/")
    print(f"   │   ├── PRE-T_1/")
    print(f"   │   └── PBMMC_1/")
    print(f"   └── annotations/")
    print(f"       └── GSE132509_cell_annotations.tsv")
    
    # Example 3: Download functionality (placeholder)
    print(f"\n3. Download functionality:")
    download_caron_data("data/downloads")
    
    print(f"\nExample completed!")


if __name__ == "__main__":
    main() 