#!/usr/bin/env python3
"""
Fix 10X Genomics format by creating expected file names.
"""

import os
from pathlib import Path

def fix_10x_format():
    """Create expected file names for 10X format."""
    data_dir = Path("data/raw")
    
    # Sample directories
    samples = [
        "ETV6-RUNX1_1", "ETV6-RUNX1_2", "ETV6-RUNX1_3", "ETV6-RUNX1_4",
        "HHD_1", "HHD_2", 
        "PRE-T_1", "PRE-T_2",
        "PBMMC_1", "PBMMC_2", "PBMMC_3"
    ]
    
    for sample in samples:
        sample_dir = data_dir / sample
        if not sample_dir.exists():
            print(f"Sample directory not found: {sample_dir}")
            continue
        
        # Change to the sample directory to create relative symlinks
        os.chdir(sample_dir)
        
        # Create symbolic links for expected file names
        genes_file = "genes.tsv.gz"
        features_file = "features.tsv.gz"
        
        if os.path.exists(genes_file) and not os.path.exists(features_file):
            # Create symbolic link from genes.tsv.gz to features.tsv.gz
            os.symlink(genes_file, features_file)
            print(f"Created symlink: {sample}/features.tsv.gz -> genes.tsv.gz")
        
        # Also create uncompressed versions if needed
        genes_uncompressed = "genes.tsv"
        features_uncompressed = "features.tsv"
        
        if os.path.exists(genes_file) and not os.path.exists(genes_uncompressed):
            # Create symbolic link for uncompressed version
            os.symlink(genes_file, genes_uncompressed)
            print(f"Created symlink: {sample}/genes.tsv -> genes.tsv.gz")
        
        if os.path.exists(features_file) and not os.path.exists(features_uncompressed):
            # Create symbolic link for uncompressed version
            os.symlink(features_file, features_uncompressed)
            print(f"Created symlink: {sample}/features.tsv -> features.tsv.gz")
        
        # Go back to the original directory
        os.chdir("/home/bradley/Desktop/bioinformatics/leukomap")

if __name__ == "__main__":
    fix_10x_format() 