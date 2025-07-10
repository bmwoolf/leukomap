"""
LeukoMap: Enhanced scRNA-seq Pipeline for Pediatric Leukemia

A modern re-analysis of the Caron et al. (2020) pediatric leukemia 
single-cell RNA-seq dataset with deeper embeddings, clinical references, 
richer annotations, and functional drug mapping.
"""

__version__ = "0.1.0"
__author__ = "LeukoMap Team"

from .data_loading import load_data
from .preprocessing import preprocess
from .scvi_training import (
    setup_scvi_data,
    train_scvi,
    setup_scanvi_data,
    train_scanvi,
    embed,
    get_latent_representation,
    add_latent_to_adata,
    train_models
)

__all__ = [
    "load_data", 
    "preprocess",
    "setup_scvi_data",
    "train_scvi", 
    "setup_scanvi_data",
    "train_scanvi",
    "embed",
    "get_latent_representation",
    "add_latent_to_adata",
    "train_models"
] 