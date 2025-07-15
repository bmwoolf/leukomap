"""
scVI/scANVI training module for LeukoMap pipeline.

This module contains functions for training scVI and scANVI models on preprocessed data.
"""

import logging
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import os

# Import scVI modules
try:
    import scvi
    from scvi.model import SCVI, SCANVI
    from scvi.data import AnnDataManager
    from scvi.data.fields import LayerField, CategoricalObsField, NumericalObsField
except ImportError:
    raise ImportError("scVI is not installed. Please install with: pip install scvi-tools")

from .core import DataProcessor, AnalysisStage

logger = logging.getLogger(__name__)


class SCVITrainer(DataProcessor):
    """scVI model trainer for single-cell data."""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.training_history = None
    
    def get_stage(self) -> AnalysisStage:
        return AnalysisStage.TRAINING
    
    def process(self, data: ad.AnnData) -> Any:
        """Train scVI model on data."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        self.logger.info("Setting up scVI data")
        # Use raw data (log-normalized, non-negative) for scVI instead of scaled data
        if hasattr(data, 'layers') and 'scvi_input' in data.layers:
            self.logger.info("Using adata.layers['scvi_input'] for scVI (log-normalized, non-negative data)")
            adata = setup_scvi_data(data, batch_key=self.config.batch_key, layer='scvi_input')
        else:
            self.logger.warning("No adata.layers['scvi_input'] found, using adata.X (may contain negative values)")
            adata = setup_scvi_data(data, batch_key=self.config.batch_key)
        
        self.logger.info("Training scVI model")
        self.model = train_scvi_model(
            adata,
            n_latent=self.config.n_latent,
            n_hidden=self.config.n_hidden,
            n_layers=self.config.n_layers,
            dropout_rate=self.config.dropout_rate,
            batch_size=self.config.batch_size,
            max_epochs=self.config.max_epochs,
            learning_rate=self.config.learning_rate,
            gpu_id=getattr(self.config, 'gpu_id', 0)
        )
        
        self.logger.info("scVI training completed")
        return self.model
    
    def generate_embeddings(self, model, adata: ad.AnnData) -> ad.AnnData:
        """Generate latent embeddings from trained model."""
        if model is None:
            raise ValueError("No trained model available")
        
        self.logger.info("Generating latent embeddings")
        
        # Get latent representation
        latent = model.get_latent_representation(adata)
        
        # Store in AnnData
        adata.obsm['X_scVI'] = latent
        
        # Compute neighbors and UMAP
        sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=self.config.n_neighbors)
        sc.tl.umap(adata)
        
        # Add clustering
        sc.tl.leiden(adata, resolution=self.config.resolution)
        
        self.logger.info(f"Generated embeddings: {latent.shape}")
        return adata


def setup_scvi_data(
    adata: ad.AnnData,
    batch_key: str = "sample_type",
    layer: Optional[str] = None
) -> ad.AnnData:
    """
    Set up AnnData for scVI training.
    
    Args:
        adata: Preprocessed AnnData object
        batch_key: Column in adata.obs to use as batch variable
        layer: Layer to use (None for .X)
    
    Returns:
        AnnData with scVI setup
    """
    logger.info(f"Setting up scVI data with batch key: {batch_key}")
    
    # Ensure batch key exists
    if batch_key not in adata.obs.columns:
        logger.warning(f"Batch key '{batch_key}' not found, using 'sample'")
        batch_key = 'sample'
        if batch_key not in adata.obs.columns:
            # Create a dummy batch key
            adata.obs[batch_key] = 'batch1'
    
    # Set up scVI data
    SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        layer=layer
    )
    
    logger.info(f"scVI data setup complete. Batches: {adata.obs[batch_key].unique()}")
    return adata


def train_scvi_model(
    adata: ad.AnnData,
    n_latent: int = 10,
    n_hidden: int = 128,
    n_layers: int = 2,
    dropout_rate: float = 0.1,
    batch_size: int = 128,
    max_epochs: int = 400,
    learning_rate: float = 1e-3,
    gpu_id: int = 0
) -> SCVI:
    """
    Train scVI model on AnnData.
    
    Args:
        adata: AnnData object with scVI setup
        n_latent: Number of latent dimensions
        n_hidden: Number of hidden units
        n_layers: Number of layers
        dropout_rate: Dropout rate
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Trained SCVI model
    """
    logger.info(f"Training scVI model with {n_latent} latent dimensions")
    
    # Set device - GPU only
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required but not available. Please ensure CUDA is installed and a GPU is available.")
    
    device = f"cuda:{gpu_id}"
    logger.info(f"Using GPU: {device}")
    
    # Create model
    model = SCVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    )
    
    # Train model
    model.train(
        batch_size=batch_size,
        max_epochs=max_epochs,
        plan_kwargs={"lr": learning_rate},
        accelerator="gpu",
        devices=[gpu_id]
    )
    
    logger.info("scVI model training completed")
    return model


def train_scanvi_model(
    adata: ad.AnnData,
    labels_key: str = "cell_type",
    n_latent: int = 10,
    n_hidden: int = 128,
    n_layers: int = 2,
    dropout_rate: float = 0.1,
    batch_size: int = 128,
    max_epochs: int = 400,
    learning_rate: float = 1e-3
) -> SCANVI:
    """
    Train scANVI model on AnnData with cell type labels.
    
    Args:
        adata: AnnData object with scVI setup and cell type labels
        labels_key: Column in adata.obs containing cell type labels
        n_latent: Number of latent dimensions
        n_hidden: Number of hidden units
        n_layers: Number of layers
        dropout_rate: Dropout rate
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Trained SCANVI model
    """
    logger.info(f"Training scANVI model with {n_latent} latent dimensions")
    
    # Set up scANVI data
    SCANVI.setup_anndata(adata, labels_key=labels_key)
    
    # Create model
    model = SCANVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    )
    
    # Train model
    model.train(
        batch_size=batch_size,
        max_epochs=max_epochs,
        plan_kwargs={"lr": learning_rate}
    )
    
    logger.info("scANVI model training completed")
    return model


def train_models(
    adata: ad.AnnData,
    batch_key: str = "sample_type",
    labels_key: Optional[str] = None,
    n_latent: int = 10,
    n_hidden: int = 128,
    n_layers: int = 2,
    dropout_rate: float = 0.1,
    batch_size: int = 128,
    max_epochs: int = 400,
    learning_rate: float = 1e-3
) -> Dict[str, Any]:
    """
    Train both scVI and scANVI models.
    
    Args:
        adata: Preprocessed AnnData object
        batch_key: Column in adata.obs to use as batch variable
        labels_key: Column in adata.obs containing cell type labels (for scANVI)
        n_latent: Number of latent dimensions
        n_hidden: Number of hidden units
        n_layers: Number of layers
        dropout_rate: Dropout rate
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Dictionary containing trained models
    """
    logger.info("Training scVI and scANVI models")
    
    # Set up data
    adata = setup_scvi_data(adata, batch_key=batch_key)
    
    # Train scVI model
    scvi_model = train_scvi_model(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate
    )
    
    models = {'scvi': scvi_model}
    
    # Train scANVI model if labels are available
    if labels_key and labels_key in adata.obs.columns:
        try:
            scanvi_model = train_scanvi_model(
                adata,
                labels_key=labels_key,
                n_latent=n_latent,
                n_hidden=n_hidden,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                batch_size=batch_size,
                max_epochs=max_epochs,
                learning_rate=learning_rate
            )
            models['scanvi'] = scanvi_model
            logger.info("Both scVI and scANVI models trained successfully")
        except Exception as e:
            logger.warning(f"scANVI training failed: {e}")
    else:
        logger.info("No cell type labels available, skipping scANVI training")
    
    return models


def save_model(model: Any, output_path: Path) -> None:
    """Save trained model to disk."""
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")


def load_model(model_path: Path) -> Any:
    """Load trained model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine model type and load
    if 'scanvi' in str(model_path).lower():
        model = SCANVI.load(model_path)
    else:
        model = SCVI.load(model_path)
    
    logger.info(f"Model loaded from {model_path}")
    return model 