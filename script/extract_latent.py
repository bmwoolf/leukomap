#!/usr/bin/env python3
"""
Extract latent embeddings from trained scVI model.
"""

import scanpy as sc
import numpy as np
from pathlib import Path
import logging
import sys

# Add leukomap to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_latent_embeddings():
    """Extract latent embeddings from trained scVI model."""
    
    # Load the trained model and data
    model_path = Path("cache/scvi_model")
    adata_path = Path("cache/adata_preprocessed.h5ad")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    
    if not adata_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {adata_path}")
    
    logger.info("Loading preprocessed data...")
    adata = sc.read_h5ad(adata_path)
    
    logger.info("Loading trained scVI model...")
    from scvi.model import SCVI
    model = SCVI.load(model_path, adata=adata)
    
    logger.info("Extracting latent representations...")
    # Extract latent embeddings directly
    latent_rep = model.get_latent_representation(adata, give_mean=True)
    logger.info(f"Extracted latent representation: {latent_rep.shape}")
    
    # Add to AnnData
    adata.obsm['X_scvi'] = latent_rep
    logger.info(f"Added latent representation to adata.obsm['X_scvi']: {adata.obsm['X_scvi'].shape}")
    
    # Save updated AnnData
    output_path = "cache/adata_with_latent.h5ad"
    adata.write(output_path)
    logger.info(f"Saved AnnData with latent embeddings to: {output_path}")
    
    return adata

if __name__ == "__main__":
    extract_latent_embeddings() 