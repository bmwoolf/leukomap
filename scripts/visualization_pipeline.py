#!/usr/bin/env python3
"""
Enhanced Visualization Pipeline for LeukoMap

Creates publication-ready visualizations using auto cell type annotations.
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeukoMapVisualizer:
    """Enhanced visualization pipeline for LeukoMap results."""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sc.settings.set_figure_params(dpi=300, frameon=False)
        sc.settings.verbosity = 1
        
        # Color palettes
        self.cell_type_colors = {
            'B_cell': '#1f77b4',
            'T_cell': '#ff7f0e', 
            'NK_cell': '#2ca02c',
            'Monocyte': '#d62728',
            'Neutrophil': '#9467bd',
            'Dendritic_cell': '#8c564b',
            'Erythrocyte': '#e377c2',
            'Platelet': '#7f7f7f',
            'Stem_cell': '#bcbd22',
            'Progenitor_cell': '#17becf',
            'Blast_cell': '#ff9896',
            'Unknown': '#cccccc'
        }
    
    def create_umap_plots(self, adata: sc.AnnData, save_prefix: str = "umap") -> Dict[str, Path]:
        """Create comprehensive UMAP visualizations."""
        logger.info("Creating UMAP visualizations...")
        
        # Ensure UMAP is computed
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_scVI' if 'X_scVI' in adata.obsm else 'X_pca')
            sc.tl.umap(adata)
        
        plots = {}
        
        # 1. Cell type UMAP
        if 'predicted_celltype' in adata.obs.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sc.pl.umap(adata, color='predicted_celltype', ax=ax, show=False, 
                      palette=self.cell_type_colors, legend_loc='on data')
            plt.title('Cell Type Distribution (UMAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            celltype_plot = self.output_dir / f"{save_prefix}_celltype.png"
            plt.savefig(celltype_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['celltype_umap'] = celltype_plot
        
        # 2. Sample type UMAP
        if 'sample_type' in adata.obs.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sc.pl.umap(adata, color='sample_type', ax=ax, show=False)
            plt.title('Sample Type Distribution (UMAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            sample_plot = self.output_dir / f"{save_prefix}_sample_type.png"
            plt.savefig(sample_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['sample_type_umap'] = sample_plot
        
        # 3. Annotation confidence UMAP
        if 'annotation_confidence' in adata.obs.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sc.pl.umap(adata, color='annotation_confidence', ax=ax, show=False, 
                      color_map='viridis', vmin=0, vmax=1)
            plt.title('Annotation Confidence (UMAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            conf_plot = self.output_dir / f"{save_prefix}_confidence.png"
            plt.savefig(conf_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['confidence_umap'] = conf_plot
        
        # 4. Clustering UMAP
        if 'leiden' in adata.obs.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sc.pl.umap(adata, color='leiden', ax=ax, show=False)
            plt.title('Clustering Results (UMAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            cluster_plot = self.output_dir / f"{save_prefix}_clusters.png"
            plt.savefig(cluster_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['clusters_umap'] = cluster_plot
        
        logger.info(f"Created {len(plots)} UMAP plots")
        return plots
    
    def create_cell_type_summary(self, adata: sc.AnnData, save_prefix: str = "celltype") -> Dict[str, Path]:
        """Create cell type summary visualizations."""
        logger.info("Creating cell type summary visualizations...")
        
        if 'predicted_celltype' not in adata.obs.columns:
            logger.warning("No cell type annotations found")
            return {}
        
        plots = {}
        
        # 1. Cell type distribution bar plot
        cell_counts = adata.obs['predicted_celltype'].value_counts()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        colors = [self.cell_type_colors.get(ct, '#cccccc') for ct in cell_counts.index]
        bars = ax.bar(range(len(cell_counts)), cell_counts.values, color=colors)
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Number of Cells', fontsize=12)
        ax.set_title('Cell Type Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(cell_counts)))
        ax.set_xticklabels(cell_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, cell_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        dist_plot = self.output_dir / f"{save_prefix}_distribution.png"
        plt.savefig(dist_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plots['celltype_distribution'] = dist_plot
        
        # 2. Cell type proportions pie chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        colors = [self.cell_type_colors.get(ct, '#cccccc') for ct in cell_counts.index]
        wedges, texts, autotexts = ax.pie(cell_counts.values, labels=cell_counts.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Cell Type Proportions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        pie_plot = self.output_dir / f"{save_prefix}_proportions.png"
        plt.savefig(pie_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plots['celltype_proportions'] = pie_plot
        
        # 3. Sample type vs Cell type heatmap
        if 'sample_type' in adata.obs.columns:
            ct_sample_table = pd.crosstab(adata.obs['predicted_celltype'], 
                                        adata.obs['sample_type'], normalize='index')
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sns.heatmap(ct_sample_table, annot=True, fmt='.2f', cmap='Blues', ax=ax)
            ax.set_title('Cell Type Distribution by Sample Type', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sample Type', fontsize=12)
            ax.set_ylabel('Cell Type', fontsize=12)
            
            plt.tight_layout()
            
            heatmap_plot = self.output_dir / f"{save_prefix}_sample_heatmap.png"
            plt.savefig(heatmap_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['celltype_sample_heatmap'] = heatmap_plot
        
        logger.info(f"Created {len(plots)} cell type summary plots")
        return plots
    
    def create_quality_metrics(self, adata: sc.AnnData, save_prefix: str = "quality") -> Dict[str, Path]:
        """Create quality metrics visualizations."""
        logger.info("Creating quality metrics visualizations...")
        
        plots = {}
        
        # 1. Annotation confidence distribution
        if 'annotation_confidence' in adata.obs.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(adata.obs['annotation_confidence'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Annotation Confidence', fontsize=12)
            ax1.set_ylabel('Number of Cells', fontsize=12)
            ax1.set_title('Annotation Confidence Distribution', fontsize=14, fontweight='bold')
            ax1.axvline(adata.obs['annotation_confidence'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {adata.obs["annotation_confidence"].mean():.3f}')
            ax1.legend()
            
            # Box plot by cell type
            if 'predicted_celltype' in adata.obs.columns:
                cell_types = adata.obs['predicted_celltype'].unique()
                conf_by_type = [adata.obs[adata.obs['predicted_celltype'] == ct]['annotation_confidence'].values 
                               for ct in cell_types]
                ax2.boxplot(conf_by_type, labels=cell_types)
                ax2.set_xlabel('Cell Type', fontsize=12)
                ax2.set_ylabel('Annotation Confidence', fontsize=12)
                ax2.set_title('Confidence by Cell Type', fontsize=14, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            conf_plot = self.output_dir / f"{save_prefix}_confidence.png"
            plt.savefig(conf_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['confidence_metrics'] = conf_plot
        
        # 2. Gene expression quality metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Genes per cell
        ax1.hist(adata.obs['n_genes_by_counts'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.set_xlabel('Number of Genes', fontsize=12)
        ax1.set_ylabel('Number of Cells', fontsize=12)
        ax1.set_title('Genes per Cell', fontsize=14, fontweight='bold')
        
        # UMIs per cell
        ax2.hist(adata.obs['total_counts'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Total UMIs', fontsize=12)
        ax2.set_ylabel('Number of Cells', fontsize=12)
        ax2.set_title('UMIs per Cell', fontsize=14, fontweight='bold')
        
        # Mitochondrial fraction
        if 'pct_counts_mt' in adata.obs.columns:
            ax3.hist(adata.obs['pct_counts_mt'], bins=50, alpha=0.7, color='gold', edgecolor='black')
            ax3.set_xlabel('Mitochondrial %', fontsize=12)
            ax3.set_ylabel('Number of Cells', fontsize=12)
            ax3.set_title('Mitochondrial Fraction', fontsize=14, fontweight='bold')
        
        # Gene expression distribution
        ax4.hist(np.log1p(adata.X.toarray().flatten()), bins=100, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Log(Expression + 1)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Gene Expression Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        quality_plot = self.output_dir / f"{save_prefix}_metrics.png"
        plt.savefig(quality_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plots['quality_metrics'] = quality_plot
        
        logger.info(f"Created {len(plots)} quality metrics plots")
        return plots
    
    def create_marker_genes_heatmap(self, adata: sc.AnnData, save_prefix: str = "markers") -> Dict[str, Path]:
        """Create marker genes heatmap."""
        logger.info("Creating marker genes heatmap...")
        
        if 'predicted_celltype' not in adata.obs.columns:
            logger.warning("No cell type annotations found for marker analysis")
            return {}
        
        plots = {}
        
        try:
            # Find marker genes for each cell type
            sc.tl.rank_genes_groups(adata, 'predicted_celltype', method='wilcoxon')
            
            # Get top markers for each cell type
            n_markers = 5  # Top 5 markers per cell type
            cell_types = adata.obs['predicted_celltype'].unique()
            
            # Collect top markers
            top_markers = []
            for ct in cell_types:
                if ct != 'Unknown':
                    markers = sc.get.rank_genes_groups_df(adata, group=ct)
                    top_markers.extend(markers.head(n_markers)['names'].tolist())
            
            # Remove duplicates
            top_markers = list(set(top_markers))
            
            if len(top_markers) > 0:
                # Create heatmap
                sc.pl.rank_genes_groups_heatmap(adata, groups=cell_types, n_genes=len(top_markers), 
                                              gene_symbols=top_markers, show=False, 
                                              save=f"_{save_prefix}_heatmap.png")
                
                heatmap_plot = self.output_dir / f"{save_prefix}_heatmap.png"
                plots['marker_heatmap'] = heatmap_plot
                
                logger.info(f"Created marker genes heatmap with {len(top_markers)} genes")
            else:
                logger.warning("No marker genes found")
                
        except Exception as e:
            logger.error(f"Error creating marker genes heatmap: {e}")
        
        return plots
    
    def create_comprehensive_report(self, adata: sc.AnnData, save_prefix: str = "comprehensive") -> Dict[str, Path]:
        """Create a comprehensive visualization report."""
        logger.info("Creating comprehensive visualization report...")
        
        all_plots = {}
        
        # Create all visualization types
        all_plots.update(self.create_umap_plots(adata, f"{save_prefix}_umap"))
        all_plots.update(self.create_cell_type_summary(adata, f"{save_prefix}_celltype"))
        all_plots.update(self.create_quality_metrics(adata, f"{save_prefix}_quality"))
        all_plots.update(self.create_marker_genes_heatmap(adata, f"{save_prefix}_markers"))
        
        # Create summary report
        self._create_summary_report(adata, all_plots, save_prefix)
        
        logger.info(f"Created comprehensive report with {len(all_plots)} plots")
        return all_plots
    
    def _create_summary_report(self, adata: sc.AnnData, plots: Dict[str, Path], save_prefix: str):
        """Create a text summary report."""
        report_path = self.output_dir / f"{save_prefix}_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LEUKOMAP VISUALIZATION SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Dataset Summary:\n")
            f.write(f"- Total cells: {adata.n_obs:,}\n")
            f.write(f"- Total genes: {adata.n_vars:,}\n")
            f.write(f"- Sparsity: {1 - (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])):.2%}\n\n")
            
            if 'predicted_celltype' in adata.obs.columns:
                f.write(f"Cell Type Annotations:\n")
                cell_counts = adata.obs['predicted_celltype'].value_counts()
                for ct, count in cell_counts.items():
                    f.write(f"- {ct}: {count:,} cells ({count/adata.n_obs:.1%})\n")
                f.write(f"- Total cell types: {len(cell_counts)}\n\n")
            
            if 'annotation_confidence' in adata.obs.columns:
                f.write(f"Annotation Quality:\n")
                f.write(f"- Mean confidence: {adata.obs['annotation_confidence'].mean():.3f}\n")
                f.write(f"- Median confidence: {adata.obs['annotation_confidence'].median():.3f}\n")
                f.write(f"- High confidence cells (>0.7): {(adata.obs['annotation_confidence'] > 0.7).sum():,}\n\n")
            
            f.write(f"Generated Plots:\n")
            for plot_name, plot_path in plots.items():
                f.write(f"- {plot_name}: {plot_path.name}\n")
            
            f.write(f"\nReport generated: {pd.Timestamp.now()}\n")
        
        logger.info(f"Summary report saved to {report_path}")

def main():
    """Main function for standalone visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LeukoMap Visualization Pipeline")
    parser.add_argument("--input-file", "-i", required=True, help="Input AnnData file (.h5ad)")
    parser.add_argument("--output-dir", "-o", default="results/figures", help="Output directory")
    parser.add_argument("--prefix", "-p", default="leukomap", help="Output file prefix")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    adata = sc.read_h5ad(args.input_file)
    
    # Create visualizations
    visualizer = LeukoMapVisualizer(args.output_dir)
    plots = visualizer.create_comprehensive_report(adata, args.prefix)
    
    print(f"\nVisualization complete! Generated {len(plots)} plots in {args.output_dir}")
    print("Files created:")
    for plot_name, plot_path in plots.items():
        print(f"  - {plot_name}: {plot_path}")

if __name__ == "__main__":
    main() 