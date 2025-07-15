#!/usr/bin/env python3
"""
Auto Cell Type Labeling Script for LeukoMap

This script provides automated cell type annotation using multiple methods:
1. CellTypist (Python) - Primary method
2. Fallback mock annotations when R packages are unavailable

Usage:
    python scripts/auto_celltype_labeling.py [--input-file INPUT] [--output-dir OUTPUT]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import celltypist
    from celltypist import annotate
    CELLTYPIST_AVAILABLE = True
except ImportError:
    CELLTYPIST_AVAILABLE = False
    print("Warning: CellTypist not available")

# R integration removed - using Python-only approach
R_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoCellTypeLabeler:
    """Automated cell type labeling using multiple methods."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Mock cell types for fallback
        self.mock_cell_types = [
            'B_cell', 'T_cell', 'NK_cell', 'Monocyte', 'Neutrophil',
            'Dendritic_cell', 'Erythrocyte', 'Platelet', 'Stem_cell',
            'Progenitor_cell', 'Blast_cell', 'Unknown'
        ]
        
        # CellTypist models to try
        self.celltypist_models = [
            'Immune_All_Low.pkl',  # General immune cells
            'Blood_All_Low.pkl',   # Blood cells
            'Human_Lung_Atlas.pkl' # Human reference
        ]
    
    def load_data(self, input_file: str) -> sc.AnnData:
        """Load single-cell data from various formats."""
        logger.info(f"Loading data from {input_file}")
        
        if input_file.endswith('.h5ad'):
            adata = sc.read_h5ad(input_file)
        elif input_file.endswith('.csv'):
            # Assume gene expression matrix
            data = pd.read_csv(input_file, index_col=0)
            adata = sc.AnnData(X=data.T)
            adata.var_names = data.index
            adata.obs_names = data.columns
        else:
            raise ValueError(f"Unsupported file format: {input_file}")
        
        logger.info(f"Loaded data with shape: {adata.shape}")
        return adata
    
    def run_celltypist(self, adata: sc.AnnData) -> Optional[pd.DataFrame]:
        """Run CellTypist annotation."""
        if not CELLTYPIST_AVAILABLE:
            logger.warning("CellTypist not available")
            return None
        
        try:
            logger.info("Running CellTypist annotation...")
            
            # Try different models
            for model_name in self.celltypist_models:
                try:
                    logger.info(f"Trying model: {model_name}")
                    
                    # Run annotation
                    predictions = annotate(adata, model=model_name, majority_voting=True)
                    
                    # Extract results
                    results = predictions.predicted_labels
                    confidence = predictions.probability_matrix.max(axis=1)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'cell_id': adata.obs_names,
                        'predicted_celltype': results,
                        'confidence': confidence,
                        'method': 'CellTypist',
                        'model': model_name
                    })
                    
                    logger.info(f"CellTypist completed with model {model_name}")
                    return results_df
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    continue
            
            logger.warning("All CellTypist models failed")
            return None
            
        except Exception as e:
            logger.error(f"CellTypist annotation failed: {e}")
            return None
    
    def run_r_annotation(self, adata: sc.AnnData) -> Optional[pd.DataFrame]:
        """R-based annotation removed - using Python-only approach."""
        logger.info("R-based annotation not available - using Python alternatives")
        return None
    
    def create_mock_annotations(self, adata: sc.AnnData) -> pd.DataFrame:
        """Create mock cell type annotations for testing."""
        logger.info("Creating mock cell type annotations...")
        
        n_cells = adata.n_obs
        
        # Create realistic mock annotations based on typical blood cell distributions
        cell_type_probs = {
            'B_cell': 0.15,
            'T_cell': 0.25,
            'NK_cell': 0.08,
            'Monocyte': 0.12,
            'Neutrophil': 0.20,
            'Dendritic_cell': 0.05,
            'Erythrocyte': 0.03,
            'Platelet': 0.02,
            'Stem_cell': 0.03,
            'Progenitor_cell': 0.04,
            'Blast_cell': 0.02,
            'Unknown': 0.01
        }
        
        # Generate cell types based on probabilities
        cell_types = np.random.choice(
            list(cell_type_probs.keys()),
            size=n_cells,
            p=list(cell_type_probs.values())
        )
        
        # Generate confidence scores (higher for common cell types)
        confidences = []
        for cell_type in cell_types:
            base_conf = cell_type_probs[cell_type]
            # Add some noise
            conf = np.clip(base_conf + np.random.normal(0, 0.1), 0.1, 0.95)
            confidences.append(conf)
        
        results_df = pd.DataFrame({
            'cell_id': adata.obs_names,
            'predicted_celltype': cell_types,
            'confidence': confidences,
            'method': 'Mock_Annotation',
            'model': 'Simulated_Distribution'
        })
        
        logger.info("Mock annotations created")
        return results_df
    
    def combine_annotations(self, annotations: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple annotation results."""
        if not annotations:
            logger.warning("No annotations to combine")
            return pd.DataFrame()
        
        if len(annotations) == 1:
            return annotations[0]
        
        # For now, just return the first successful annotation
        # In a full implementation, you might want to:
        # 1. Compare confidence scores
        # 2. Use voting mechanisms
        # 3. Resolve conflicts
        logger.info("Using first available annotation method")
        return annotations[0]
    
    def save_results(self, results_df: pd.DataFrame, prefix: str = "auto_celltype"):
        """Save annotation results."""
        # Save main results
        output_file = self.output_dir / f"{prefix}_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Save summary statistics
        summary = results_df.groupby('predicted_celltype').agg({
            'confidence': ['count', 'mean', 'std']
        }).round(3)
        
        summary_file = self.output_dir / f"{prefix}_summary.csv"
        summary.to_csv(summary_file)
        logger.info(f"Summary saved to {summary_file}")
        
        # Create confidence distribution plot
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data=results_df, x='confidence', bins=20)
            plt.title('Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            
            plot_file = self.output_dir / f"{prefix}_confidence_dist.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confidence plot saved to {plot_file}")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
    
    def run_annotation_pipeline(self, input_file: str) -> pd.DataFrame:
        """Run the complete annotation pipeline."""
        logger.info("Starting auto cell type annotation pipeline...")
        
        # Load data
        adata = self.load_data(input_file)
        
        # Collect annotations from different methods
        annotations = []
        
        # Try CellTypist
        celltypist_results = self.run_celltypist(adata)
        if celltypist_results is not None:
            annotations.append(celltypist_results)
        
        # Try R-based methods
        r_results = self.run_r_annotation(adata)
        if r_results is not None:
            annotations.append(r_results)
        
        # If no methods worked, use mock annotations
        if not annotations:
            logger.warning("No annotation methods succeeded, using mock annotations")
            mock_results = self.create_mock_annotations(adata)
            annotations.append(mock_results)
        
        # Combine results
        final_results = self.combine_annotations(annotations)
        
        # Save results
        self.save_results(final_results)
        
        logger.info("Annotation pipeline completed")
        return final_results

def main():
    parser = argparse.ArgumentParser(description="Auto cell type labeling for LeukoMap")
    parser.add_argument("--input-file", "-i", required=True, help="Input file path")
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Run annotation
    labeler = AutoCellTypeLabeler(output_dir=args.output_dir)
    results = labeler.run_annotation_pipeline(args.input_file)
    
    # Print summary
    print("\n" + "="*50)
    print("ANNOTATION SUMMARY")
    print("="*50)
    print(f"Total cells annotated: {len(results)}")
    print(f"Method used: {results['method'].iloc[0]}")
    print(f"Model used: {results['model'].iloc[0]}")
    print(f"Average confidence: {results['confidence'].mean():.3f}")
    print(f"Unique cell types: {results['predicted_celltype'].nunique()}")
    
    print("\nCell type distribution:")
    cell_counts = results['predicted_celltype'].value_counts()
    for cell_type, count in cell_counts.head(10).items():
        print(f"  {cell_type}: {count} cells")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 