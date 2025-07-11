#!/usr/bin/env python3
"""
Local GPU training script for LeukoMap scVI models.
Run this on your Ubuntu machine with 4090.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

# Add leukomap to path
sys.path.append(str(Path(__file__).parent.parent))

from leukomap.data_loading import load_data
from leukomap.preprocessing import preprocess
from leukomap.scvi_training import train_models

def main():
    parser = argparse.ArgumentParser(description="Train scVI/scANVI models on local GPU")
    parser.add_argument("--data_path", default="data", help="Path to data directory")
    parser.add_argument("--max_epochs", type=int, default=400, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--n_latent", type=int, default=10, help="Number of latent dimensions")
    parser.add_argument("--train_scanvi", action="store_true", help="Also train scANVI model")
    parser.add_argument("--output_dir", default="cache", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.output_dir}/training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU detected! Training will be very slow on CPU.")
    
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        adata = load_data(args.data_path)
        logger.info(f"Loaded data: {adata.shape}")
        
        logger.info("Preprocessing data...")
        processed = preprocess(
            adata.copy(),
            min_genes=200,
            min_cells=3,
            max_genes=6000,
            max_counts=50000,
            max_mito=0.2
        )
        logger.info(f"Preprocessed data: {processed.shape}")
        
        # Train models
        logger.info("Starting model training...")
        models = train_models(
            processed,
            do_train_scvi=True,
            do_train_scanvi=args.train_scanvi,
            save_models=True,
            scvi_params={
                "n_latent": args.n_latent,
                "max_epochs": args.max_epochs,
                "save_path": f"{args.output_dir}/scvi_model"
            },
            scanvi_params={
                "n_latent": args.n_latent,
                "max_epochs": args.max_epochs,
                "save_path": f"{args.output_dir}/scanvi_model"
            } if args.train_scanvi else None
        )
        
        # Save AnnData with latent representations
        logger.info("Saving AnnData with latent representations...")
        processed.write(f"{args.output_dir}/adata_with_latent.h5ad")
        
        # Print summary
        logger.info("Training complete!")
        logger.info(f"Models saved to: {args.output_dir}/")
        logger.info(f"AnnData with embeddings: {args.output_dir}/adata_with_latent.h5ad")
        
        if "scvi" in models:
            logger.info("scVI model trained successfully")
        if "scanvi" in models:
            logger.info("scANVI model trained successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 