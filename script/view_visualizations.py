#!/usr/bin/env python3
"""
Display the generated UMAP visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def display_visualizations():
    """Display all generated UMAP visualizations."""
    cache_dir = Path("cache")
    
    # List of visualization files
    viz_files = [
        "umap_sample_type.png",
        "umap_celltype.png", 
        "umap_sample_type_enhanced.png",
        "umap_quality_metrics.png"
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('LeukoMap: scVI Latent Space UMAP Visualizations', fontsize=16, fontweight='bold')
    
    for i, filename in enumerate(viz_files):
        filepath = cache_dir / filename
        if filepath.exists():
            row = i // 2
            col = i % 2
            img = mpimg.imread(filepath)
            axes[row, col].imshow(img)
            axes[row, col].set_title(filename.replace('.png', '').replace('_', ' ').title(), fontsize=12)
            axes[row, col].axis('off')
        else:
            print(f"File not found: {filepath}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_visualizations() 