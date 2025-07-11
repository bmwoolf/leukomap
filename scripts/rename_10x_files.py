#!/usr/bin/env python3
"""
Rename 10X Genomics files to standard format for the pipeline.
"""

import os
from pathlib import Path

def rename_10x_files():
    """Rename all 10X files to standard format."""
    data_dir = Path("data/raw")
    
    # Sample mappings
    samples = [
        "ETV6-RUNX1_1", "ETV6-RUNX1_2", "ETV6-RUNX1_3", "ETV6-RUNX1_4",
        "HHD_1", "HHD_2", 
        "PRE-T_1", "PRE-T_2",
        "PBMMC_1", "PBMMC_2", "PBMMC_3"
    ]
    
    # GSM ID mappings
    gsm_mappings = {
        "ETV6-RUNX1_1": "GSM3872434",
        "ETV6-RUNX1_2": "GSM3872435", 
        "ETV6-RUNX1_3": "GSM3872436",
        "ETV6-RUNX1_4": "GSM3872437",
        "HHD_1": "GSM3872438",
        "HHD_2": "GSM3872439",
        "PRE-T_1": "GSM3872440",
        "PRE-T_2": "GSM3872441",
        "PBMMC_1": "GSM3872442",
        "PBMMC_2": "GSM3872443",
        "PBMMC_3": "GSM3872444"
    }
    
    for sample in samples:
        sample_dir = data_dir / sample
        if not sample_dir.exists():
            print(f"Sample directory not found: {sample_dir}")
            continue
            
        gsm_id = gsm_mappings[sample]
        
        # Rename files
        old_files = {
            f"{gsm_id}_{sample}.barcodes.tsv.gz": "barcodes.tsv.gz",
            f"{gsm_id}_{sample}.genes.tsv.gz": "genes.tsv.gz", 
            f"{gsm_id}_{sample}.matrix.mtx.gz": "matrix.mtx.gz"
        }
        
        for old_name, new_name in old_files.items():
            old_path = sample_dir / old_name
            new_path = sample_dir / new_name
            
            if old_path.exists():
                old_path.rename(new_path)
                print(f"Renamed {old_name} -> {new_name} in {sample}")
            else:
                print(f"File not found: {old_path}")

if __name__ == "__main__":
    rename_10x_files() 