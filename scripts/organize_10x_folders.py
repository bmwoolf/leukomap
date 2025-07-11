#!/usr/bin/env python3
"""
Fix and rerun: Organize flat 10X files into per-sample directories for Scanpy compatibility.
"""
import os
import shutil

RAW_DIR = "data/raw"

# Find all unique sample names from GSM* files
files = os.listdir(RAW_DIR)
sample_names = set()
for f in files:
    if f.startswith('GSM') and (f.endswith('.barcodes.tsv') or f.endswith('.genes.tsv') or f.endswith('.matrix.mtx')):
        sample = f.split('_', 1)[1].rsplit('.', 2)[0]
        sample_names.add(sample)

for sample in sample_names:
    sample_dir = os.path.join(RAW_DIR, sample)
    os.makedirs(sample_dir, exist_ok=True)
    for t, ext in [('barcodes', 'tsv'), ('genes', 'tsv'), ('matrix', 'mtx')]:
        pattern = f"_{sample}.{t}.{ext}"
        srcs = [f for f in files if pattern in f]
        for src in srcs:
            src_path = os.path.join(RAW_DIR, src)
            dst_path = os.path.join(sample_dir, f"{t}.{ext}")
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} -> {dst_path}")
print("Done fixing 10X sample folders.") 