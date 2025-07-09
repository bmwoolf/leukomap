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

__all__ = ["load_data", "preprocess"] 