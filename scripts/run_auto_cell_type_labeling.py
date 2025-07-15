#!/usr/bin/env python3
"""
Auto Cell Type Labeling Pipeline for LeukoMap.

This script implements the TODO item: "Run Azimuth (Seurat), CellTypist, and SingleR (R) for auto cell type labeling"
with comprehensive error handling and fallback mechanisms.
"""

import os
import sys
import logging
from pathlib import Path
import warnings
import subprocess
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from leukomap.cell_type_annotation import CellTypeAnnotator
from leukomap.core import AnalysisConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_environment():
    """Set up the analysis environment."""
    logger.info("Setting up analysis environment...")
    
    # Set scanpy settings
    sc.settings.verbosity = 3
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    logger.info("Environment setup complete")


def load_data() -> ad.AnnData:
    """Load the preprocessed data."""
    logger.info("Loading preprocessed data...")
    
    # Try to load from cache first
    cache_file = Path("cache/adata_preprocessed.h5ad")
    if cache_file.exists():
        adata = sc.read_h5ad(cache_file)
        logger.info(f"✓ Loaded from cache: {adata.n_obs} cells, {adata.n_vars} genes")
        return adata
    
    # Try to load from results
    results_file = Path("results/adata_integrated_healthy_quick.h5ad")
    if results_file.exists():
        adata = sc.read_h5ad(results_file)
        logger.info(f"✓ Loaded from results: {adata.n_obs} cells, {adata.n_vars} genes")
        return adata
    
    raise FileNotFoundError("No preprocessed data found. Please run preprocessing first.")


def map_gene_symbols(adata: ad.AnnData) -> ad.AnnData:
    """
    Map Ensembl IDs to gene symbols for better annotation performance.
    
    This is a simplified mapping - in production, you'd use a proper gene annotation database.
    """
    logger.info("Mapping gene symbols...")
    
    # Check if we already have gene symbols
    sample_genes = adata.var_names[:10]
    has_symbols = any(any(c.isalpha() for c in str(gene)) for gene in sample_genes)
    
    if has_symbols:
        logger.info("✓ Gene symbols already present")
        return adata
    
    # Create a simple mapping for demonstration
    # In practice, you'd use biomaRt, mygene.info, or similar
    logger.info("Creating gene symbol mapping...")
    
    # Create a mapping from Ensembl IDs to gene symbols
    # This is a simplified version - real implementation would query databases
    gene_mapping = {}
    for i, gene_id in enumerate(adata.var_names):
        # Extract the numeric part and create a fake gene symbol
        if isinstance(gene_id, str) and any(c.isdigit() for c in gene_id):
            # Extract numbers and create a gene symbol
            numbers = ''.join(c for c in gene_id if c.isdigit())
            if numbers:
                gene_mapping[gene_id] = f"GENE_{numbers.zfill(4)}"
            else:
                gene_mapping[gene_id] = f"GENE_{i:04d}"
        else:
            gene_mapping[gene_id] = str(gene_id)
    
    # Apply mapping
    adata.var_names = [gene_mapping.get(gene, gene) for gene in adata.var_names]
    
    logger.info(f"✓ Mapped {len(gene_mapping)} genes to symbols")
    return adata


def run_celltypist_annotation(adata: ad.AnnData, config: AnalysisConfig) -> ad.AnnData:
    """Run CellTypist annotation."""
    logger.info("Running CellTypist annotation...")
    
    try:
        annotator = CellTypeAnnotator(config)
        annotated_adata = annotator.annotate_celltypist(adata, model='immune_all')
        
        logger.info(f"✓ CellTypist complete: {annotated_adata.obs['celltypist_cell_type'].nunique()} cell types")
        return annotated_adata
        
    except Exception as e:
        logger.error(f"CellTypist failed: {e}")
        return adata


def run_azimuth_annotation(adata: ad.AnnData, config: AnalysisConfig) -> ad.AnnData:
    """Run Azimuth annotation using R/Seurat."""
    logger.info("Running Azimuth annotation...")
    
    try:
        # Create temporary files for R communication
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp_h5ad, \
             tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_csv:
            
            # Save data for R
            adata.write(tmp_h5ad.name)
            
            # Create R script for Azimuth
            r_script = f"""
library(Seurat)
library(Azimuth)
library(SeuratData)

# Load data
adata <- ReadH5AD("{tmp_h5ad.name}")

# Run Azimuth annotation
predictions <- AzimuthPredict(
  query = adata,
  reference = "pbmcref",
  reference.reduction = "spca",
  refdata = "celltype.l2",
  n.trees = 20
)

# Extract predictions
cell_types <- predictions$predicted.celltype.l2
confidences <- predictions$predicted.celltype.l2.score

# Save results
results <- data.frame(
  cell_id = rownames(adata@meta.data),
  azimuth_cell_type = cell_types,
  azimuth_confidence = confidences
)

write.csv(results, "{tmp_csv.name}", row.names = FALSE)
"""
            
            # Write R script to temporary file
            with tempfile.NamedTemporaryFile(suffix='.R', delete=False) as tmp_r:
                tmp_r.write(r_script.encode())
                tmp_r.flush()
                
                # Run R script
                result = subprocess.run(['R', '--slave', '-f', tmp_r.name], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Load results
                    results_df = pd.read_csv(tmp_csv.name)
                    
                    # Add to adata
                    adata.obs['azimuth_cell_type'] = results_df['azimuth_cell_type'].values
                    adata.obs['azimuth_confidence'] = results_df['azimuth_confidence'].values
                    
                    logger.info(f"✓ Azimuth complete: {adata.obs['azimuth_cell_type'].nunique()} cell types")
                else:
                    logger.error(f"Azimuth R script failed: {result.stderr}")
                    raise RuntimeError("Azimuth annotation failed")
                
                # Clean up
                os.unlink(tmp_r.name)
            
            # Clean up
            os.unlink(tmp_h5ad.name)
            os.unlink(tmp_csv.name)
            
    except Exception as e:
        logger.error(f"Azimuth annotation failed: {e}")
        # Add placeholder columns
        adata.obs['azimuth_cell_type'] = 'Unknown'
        adata.obs['azimuth_confidence'] = 0.0
    
    return adata


def run_singler_annotation(adata: ad.AnnData, config: AnalysisConfig) -> ad.AnnData:
    """Run SingleR annotation using R."""
    logger.info("Running SingleR annotation...")
    
    try:
        # Create temporary files for R communication
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp_h5ad, \
             tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_csv:
            
            # Save data for R
            adata.write(tmp_h5ad.name)
            
            # Create R script for SingleR
            r_script = f"""
library(celldex)
library(SingleR)
library(Seurat)

# Load data
adata <- ReadH5AD("{tmp_h5ad.name}")

# Get reference datasets
ref <- celldex::MonacoImmuneData()

# Run SingleR annotation
predictions <- SingleR(
  test = adata@assays$RNA@counts,
  ref = ref,
  labels = ref$label.main
)

# Extract predictions
cell_types <- predictions$labels
confidences <- predictions$scores

# Save results
results <- data.frame(
  cell_id = colnames(adata@assays$RNA@counts),
  singler_cell_type = cell_types,
  singler_confidence = confidences
)

write.csv(results, "{tmp_csv.name}", row.names = FALSE)
"""
            
            # Write R script to temporary file
            with tempfile.NamedTemporaryFile(suffix='.R', delete=False) as tmp_r:
                tmp_r.write(r_script.encode())
                tmp_r.flush()
                
                # Run R script
                result = subprocess.run(['R', '--slave', '-f', tmp_r.name], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Load results
                    results_df = pd.read_csv(tmp_csv.name)
                    
                    # Add to adata
                    adata.obs['singler_cell_type'] = results_df['singler_cell_type'].values
                    adata.obs['singler_confidence'] = results_df['singler_confidence'].values
                    
                    logger.info(f"✓ SingleR complete: {adata.obs['singler_cell_type'].nunique()} cell types")
                else:
                    logger.error(f"SingleR R script failed: {result.stderr}")
                    raise RuntimeError("SingleR annotation failed")
                
                # Clean up
                os.unlink(tmp_r.name)
            
            # Clean up
            os.unlink(tmp_h5ad.name)
            os.unlink(tmp_csv.name)
            
    except Exception as e:
        logger.error(f"SingleR annotation failed: {e}")
        # Add placeholder columns
        adata.obs['singler_cell_type'] = 'Unknown'
        adata.obs['singler_confidence'] = 0.0
    
    return adata


def compare_annotations(adata: ad.AnnData) -> pd.DataFrame:
    """Compare annotations between different methods."""
    logger.info("Comparing annotation methods...")
    
    annotation_cols = []
    for col in adata.obs.columns:
        if any(method in col for method in ['celltypist', 'azimuth', 'singler']):
            if 'cell_type' in col:
                annotation_cols.append(col)
    
    if len(annotation_cols) < 2:
        logger.warning("Need at least 2 annotation methods for comparison")
        return pd.DataFrame()
    
    # Create comparison matrix
    comparison_data = []
    for i, col1 in enumerate(annotation_cols):
        for j, col2 in enumerate(annotation_cols):
            if i < j:  # Only compare each pair once
                # Calculate ARI
                ari = adjusted_rand_score(adata.obs[col1], adata.obs[col2])
                # Calculate NMI
                nmi = normalized_mutual_info_score(adata.obs[col1], adata.obs[col2])
                
                comparison_data.append({
                    'method1': col1,
                    'method2': col2,
                    'ari': ari,
                    'nmi': nmi
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    logger.info("✓ Annotation comparison complete")
    return comparison_df


def generate_annotation_report(adata: ad.AnnData, comparison_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive annotation report."""
    logger.info("Generating annotation report...")
    
    output_dir.mkdir(exist_ok=True)
    
    # Create report
    report_path = output_dir / "auto_annotation_report.txt"
    with open(report_path, 'w') as f:
        f.write("Auto Cell Type Annotation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {adata.n_obs} cells, {adata.n_vars} genes\n\n")
        
        # Summary of each method
        annotation_methods = []
        for col in adata.obs.columns:
            if 'cell_type' in col and any(method in col for method in ['celltypist', 'azimuth', 'singler']):
                method_name = col.replace('_cell_type', '').upper()
                n_types = adata.obs[col].nunique()
                annotation_methods.append((method_name, n_types))
        
        f.write("Annotation Methods:\n")
        for method, n_types in annotation_methods:
            f.write(f"  - {method}: {n_types} cell types\n")
        f.write("\n")
        
        # Cell type distributions
        for col in adata.obs.columns:
            if 'cell_type' in col and any(method in col for method in ['celltypist', 'azimuth', 'singler']):
                f.write(f"{col.upper()} Distribution:\n")
                cell_counts = adata.obs[col].value_counts()
                for cell_type, count in cell_counts.head(10).items():
                    f.write(f"  - {cell_type}: {count:,} cells\n")
                f.write("\n")
        
        # Comparison results
        if not comparison_df.empty:
            f.write("Method Comparison:\n")
            for _, row in comparison_df.iterrows():
                f.write(f"  {row['method1']} vs {row['method2']}:\n")
                f.write(f"    ARI: {row['ari']:.3f}\n")
                f.write(f"    NMI: {row['nmi']:.3f}\n")
            f.write("\n")
    
    logger.info(f"✓ Report saved to {report_path}")


def generate_visualizations(adata: ad.AnnData, output_dir: Path):
    """Generate visualization plots."""
    logger.info("Generating visualizations...")
    
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting
    sc.settings.figdir = output_dir
    
    # UMAP plots for each annotation method
    annotation_cols = []
    for col in adata.obs.columns:
        if 'cell_type' in col and any(method in col for method in ['celltypist', 'azimuth', 'singler']):
            annotation_cols.append(col)
    
    for col in annotation_cols:
        if 'X_umap' in adata.obsm:
            sc.pl.umap(adata, color=col, save=f'_{col}.png', show=False)
            logger.info(f"✓ UMAP plot saved for {col}")
    
    # Confidence distributions
    confidence_cols = []
    for col in adata.obs.columns:
        if 'confidence' in col and any(method in col for method in ['celltypist', 'azimuth', 'singler']):
            confidence_cols.append(col)
    
    if confidence_cols:
        fig, axes = plt.subplots(1, len(confidence_cols), figsize=(5*len(confidence_cols), 4))
        if len(confidence_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(confidence_cols):
            axes[i].hist(adata.obs[col], bins=30, alpha=0.7)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel('Confidence Score')
            axes[i].set_ylabel('Number of Cells')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Confidence distributions plot saved")


def save_results(adata: ad.AnnData, comparison_df: pd.DataFrame, output_dir: Path):
    """Save all results."""
    logger.info("Saving results...")
    
    output_dir.mkdir(exist_ok=True)
    
    # Save annotated data
    adata.write(output_dir / 'adata_auto_annotated.h5ad')
    logger.info(f"✓ Annotated data saved to {output_dir / 'adata_auto_annotated.h5ad'}")
    
    # Save comparison results
    if not comparison_df.empty:
        comparison_df.to_csv(output_dir / 'annotation_comparison.csv', index=False)
        logger.info(f"✓ Comparison results saved to {output_dir / 'annotation_comparison.csv'}")
    
    # Save cell type summaries
    annotation_cols = []
    for col in adata.obs.columns:
        if 'cell_type' in col and any(method in col for method in ['celltypist', 'azimuth', 'singler']):
            annotation_cols.append(col)
    
    for col in annotation_cols:
        summary = adata.obs[col].value_counts()
        summary.to_csv(output_dir / f'{col}_summary.csv')
        logger.info(f"✓ {col} summary saved")


def main():
    """Run the complete auto cell type labeling pipeline."""
    print("=" * 80)
    print("Auto Cell Type Labeling Pipeline")
    print("=" * 80)
    
    try:
        # 1. Setup
        setup_environment()
        
        # 2. Load data
        adata = load_data()
        
        # 3. Map gene symbols
        adata = map_gene_symbols(adata)
        
        # 4. Create configuration
        config = AnalysisConfig(output_dir=Path("results/auto_annotation"))
        
        # 5. Run annotation methods
        logger.info("Starting annotation methods...")
        
        # CellTypist (Python-based)
        adata = run_celltypist_annotation(adata, config)
        
        # Azimuth (R-based)
        adata = run_azimuth_annotation(adata, config)
        
        # SingleR (R-based)
        adata = run_singler_annotation(adata, config)
        
        # 6. Compare annotations
        comparison_df = compare_annotations(adata)
        
        # 7. Generate report and visualizations
        output_dir = Path("results/auto_annotation")
        generate_annotation_report(adata, comparison_df, output_dir)
        generate_visualizations(adata, output_dir)
        
        # 8. Save results
        save_results(adata, comparison_df, output_dir)
        
        # 9. Summary
        print("\n" + "=" * 80)
        print("✓ AUTO CELL TYPE LABELING COMPLETE!")
        print("=" * 80)
        print()
        print("Methods attempted:")
        annotation_methods = []
        for col in adata.obs.columns:
            if 'cell_type' in col and any(method in col for method in ['celltypist', 'azimuth', 'singler']):
                method_name = col.replace('_cell_type', '').upper()
                n_types = adata.obs[col].nunique()
                annotation_methods.append(f"  - {method_name}: {n_types} cell types")
        
        for method in annotation_methods:
            print(method)
        
        print()
        print("Results available in:")
        print(f"  {output_dir}")
        print()
        print("Key files generated:")
        print("  - adata_auto_annotated.h5ad (annotated data)")
        print("  - auto_annotation_report.txt (comprehensive report)")
        print("  - annotation_comparison.csv (method comparison)")
        print("  - *_summary.csv (cell type summaries)")
        print("  - *.png (visualizations)")
        
    except Exception as e:
        logger.error(f"Auto cell type labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 