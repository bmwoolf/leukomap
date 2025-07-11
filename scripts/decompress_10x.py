#!/usr/bin/env python3
"""
Decompress 10X Genomics files for easier loading.
"""

import gzip
import shutil
from pathlib import Path

def decompress_10x_files():
    """Decompress all 10X files."""
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
        
        print(f"Decompressing files for {sample}...")
        
        # Files to decompress
        files_to_decompress = [
            "barcodes.tsv.gz",
            "genes.tsv.gz", 
            "matrix.mtx.gz"
        ]
        
        for compressed_file in files_to_decompress:
            compressed_path = sample_dir / compressed_file
            uncompressed_path = sample_dir / compressed_file.replace('.gz', '')
            
            if compressed_path.exists() and not uncompressed_path.exists():
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(uncompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  Decompressed {compressed_file} -> {compressed_file.replace('.gz', '')}")

if __name__ == "__main__":
    decompress_10x_files() 