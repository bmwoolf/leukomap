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

logger = logging.getLogger(__name__)


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
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.columns")
    
    # Set up scVI data
    SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        layer=layer
    )
    
    logger.info(f"scVI data setup complete. Batches: {adata.obs[batch_key].cat.categories.tolist()}")
    return adata


def train_scvi(
    adata: ad.AnnData,
    n_latent: int = 10,
    n_layers: int = 2,
    n_hidden: int = 128,
    dropout_rate: float = 0.1,
    max_epochs: int = 400,
    learning_rate: float = 0.001,
    train_size: float = 0.9,
    random_state: int = 42,
    use_gpu: bool = True,
    save_path: Optional[str] = "cache/scvi_model"
) -> SCVI:
    """
    Train scVI model on preprocessed data.
    
    Args:
        adata: Preprocessed AnnData with scVI setup
        n_latent: Number of latent dimensions
        n_layers: Number of hidden layers in encoder/decoder
        n_hidden: Number of hidden units per layer
        dropout_rate: Dropout rate for regularization
        max_epochs: Maximum training epochs
        learning_rate: Learning rate for optimization
        train_size: Fraction of data for training
        random_state: Random seed for reproducibility
        use_gpu: Whether to use GPU if available
        save_path: Path to save trained model
    
    Returns:
        Trained SCVI model
    """
    logger.info("Starting scVI training...")
    
    # Set random seed
    scvi.settings.seed = random_state
    
    # Create model
    model = SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate
    )
    
    logger.info(f"Created SCVI model with {n_latent} latent dimensions")
    
    # Train model
    model.train(
        max_epochs=max_epochs,
        train_size=train_size
    )
    
    logger.info("scVI training complete")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"Saved scVI model to {save_path}")
    
    return model


def setup_scanvi_data(
    adata: ad.AnnData,
    batch_key: str = "sample_type",
    labels_key: str = "celltype",
    unlabeled_category: str = "Unknown",
    layer: Optional[str] = None
) -> ad.AnnData:
    """
    Set up AnnData for scANVI training.
    
    Args:
        adata: Preprocessed AnnData object
        batch_key: Column in adata.obs to use as batch variable
        labels_key: Column in adata.obs to use as cell type labels
        unlabeled_category: Category to use for unlabeled cells
        layer: Layer to use (None for .X)
    
    Returns:
        AnnData with scANVI setup
    """
    logger.info(f"Setting up scANVI data with batch key: {batch_key}, labels key: {labels_key}")
    
    # Ensure keys exist
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.columns")
    if labels_key not in adata.obs.columns:
        raise ValueError(f"Labels key '{labels_key}' not found in adata.obs.columns")
    
    # Set up scANVI data
    SCANVI.setup_anndata(
        adata,
        batch_key=batch_key,
        labels_key=labels_key,
        layer=layer
    )
    
    logger.info(f"scANVI data setup complete. Cell types: {adata.obs[labels_key].cat.categories.tolist()}")
    return adata


def train_scanvi(
    adata: ad.AnnData,
    n_latent: int = 10,
    n_layers: int = 2,
    n_hidden: int = 128,
    dropout_rate: float = 0.1,
    max_epochs: int = 400,
    learning_rate: float = 0.001,
    train_size: float = 0.9,
    random_state: int = 42,
    use_gpu: bool = True,
    save_path: Optional[str] = "cache/scanvi_model"
) -> SCANVI:
    """
    Train scANVI model on preprocessed data.
    
    Args:
        adata: Preprocessed AnnData with scANVI setup
        n_latent: Number of latent dimensions
        n_layers: Number of hidden layers in encoder/decoder
        n_hidden: Number of hidden units per layer
        dropout_rate: Dropout rate for regularization
        max_epochs: Maximum training epochs
        learning_rate: Learning rate for optimization
        train_size: Fraction of data for training
        random_state: Random seed for reproducibility
        use_gpu: Whether to use GPU if available
        save_path: Path to save trained model
    
    Returns:
        Trained SCANVI model
    """
    logger.info("Starting scANVI training...")
    
    # Set random seed
    scvi.settings.seed = random_state
    
    # Create model
    model = SCANVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate
    )
    
    logger.info(f"Created SCANVI model with {n_latent} latent dimensions")
    
    # Train model
    model.train(
        max_epochs=max_epochs,
        train_size=train_size
    )
    
    logger.info("scANVI training complete")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"Saved scANVI model to {save_path}")
    
    return model


def get_latent_representation(
    model: Any,
    adata: ad.AnnData,
    give_mean: bool = True,
    batch_size: int = 128
) -> np.ndarray:
    """
    Extract latent representation from trained model.
    
    Args:
        model: Trained scVI or scANVI model
        adata: AnnData object
        give_mean: Whether to return mean of latent distribution
        batch_size: Batch size for inference
    
    Returns:
        Latent representation array
    """
    logger.info("Extracting latent representation...")
    
    latent = model.get_latent_representation(
        adata,
        give_mean=give_mean,
        batch_size=batch_size
    )
    
    logger.info(f"Extracted latent representation: {latent.shape}")
    return latent


def add_latent_to_adata(
    model: Any,
    adata: ad.AnnData,
    key: str = "X_scvi",
    give_mean: bool = True,
    batch_size: int = 128
) -> ad.AnnData:
    """
    Add latent representation to AnnData object.
    
    Args:
        model: Trained scVI or scANVI model
        adata: AnnData object
        key: Key to store latent representation in adata.obsm
        give_mean: Whether to return mean of latent distribution
        batch_size: Batch size for inference
    
    Returns:
        AnnData with latent representation added
    """
    logger.info(f"Adding latent representation to adata.obsm['{key}']")
    
    latent = get_latent_representation(model, adata, give_mean, batch_size)
    adata.obsm[key] = latent
    
    logger.info(f"Added latent representation: {latent.shape}")
    return adata


def train_models(
    adata: ad.AnnData,
    do_train_scvi: bool = True,
    do_train_scanvi: bool = True,
    scvi_params: Optional[Dict[str, Any]] = None,
    scanvi_params: Optional[Dict[str, Any]] = None,
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Train both scVI and scANVI models with default parameters.
    
    Args:
        adata: Preprocessed AnnData object
        train_scvi: Whether to train scVI model
        train_scanvi: Whether to train scANVI model
        scvi_params: Parameters for scVI training
        scanvi_params: Parameters for scANVI training
        save_models: Whether to save trained models
    
    Returns:
        Dictionary containing trained models
    """
    logger.info("Starting model training pipeline...")
    
    # Default parameters
    default_scvi_params = {
        "n_latent": 10,
        "n_layers": 2,
        "n_hidden": 128,
        "dropout_rate": 0.1,
        "max_epochs": 400,
        "learning_rate": 0.001,
        "train_size": 0.9,
        "random_state": 42,
        "use_gpu": True,
        "save_path": "cache/scvi_model" if save_models else None
    }
    
    default_scanvi_params = {
        "n_latent": 10,
        "n_layers": 2,
        "n_hidden": 128,
        "dropout_rate": 0.1,
        "max_epochs": 400,
        "learning_rate": 0.001,
        "train_size": 0.9,
        "random_state": 42,
        "use_gpu": True,
        "save_path": "cache/scanvi_model" if save_models else None
    }
    
    # Update with provided parameters
    if scvi_params:
        default_scvi_params.update(scvi_params)
    if scanvi_params:
        default_scanvi_params.update(scanvi_params)
    
    models = {}
    
    # Train scVI
    if do_train_scvi:
        logger.info("Training scVI model...")
        adata_scvi = adata.copy()
        setup_scvi_data(adata_scvi)
        models["scvi"] = train_scvi(adata_scvi, **default_scvi_params)
        
        # Add latent representation
        add_latent_to_adata(models["scvi"], adata_scvi, key="X_scvi")
    
    # Train scANVI
    if do_train_scanvi:
        logger.info("Training scANVI model...")
        adata_scanvi = adata.copy()
        setup_scanvi_data(adata_scanvi)
        models["scanvi"] = train_scanvi(adata_scanvi, **default_scanvi_params)
        
        # Add latent representation
        add_latent_to_adata(models["scanvi"], adata_scanvi, key="X_scanvi")
    
    logger.info("Model training pipeline complete")
    return models 