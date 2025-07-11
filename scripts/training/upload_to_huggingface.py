#!/usr/bin/env python3
"""
Upload trained scVI model to Hugging Face Hub.

This script uploads the trained model weights and metadata to Hugging Face
for easy sharing and access across different machines.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import LocalEntryNotFoundError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model_card(model_info: Dict[str, Any]) -> str:
    """Create a model card for the Hugging Face repository."""
    
    card = f"""---
language: en
tags:
- single-cell
- rna-seq
- leukemia
- scvi
- bioinformatics
license: mit
---

# LeukoMap scVI Model

This is a trained scVI (single-cell Variational Inference) model for pediatric leukemia single-cell RNA-seq analysis.

## Model Details

- **Model Type**: scVI (single-cell Variational Inference)
- **Training Data**: Caron et al. (2020) pediatric leukemia dataset
- **Architecture**: Variational Autoencoder for single-cell data
- **Latent Dimensions**: {model_info.get('latent_dim', 'Unknown')}
- **Training Epochs**: {model_info.get('max_epochs', 'Unknown')}

## Usage

```python
from scvi.model import SCVI
import scanpy as sc

# Load your AnnData object
adata = sc.read_h5ad("your_data.h5ad")

# Load the model
model = SCVI.load("your-username/leukomap-scvi", adata)

# Get latent representation
latent = model.get_latent_representation()
```

## Dataset

This model was trained on the Caron et al. (2020) pediatric leukemia dataset:
- **GEO Accession**: GSE132509
- **Paper**: https://doi.org/10.1038/s41598-020-64929-x
- **Original Analysis**: https://github.com/CBC-UCONN/Single-Cell-Transcriptomics

## Citation

If you use this model, please cite:
- Caron et al. (2020) Single-cell analysis of childhood leukemia reveals a link between developmental states and ribosomal protein expression as a source of intra-individual heterogeneity
- Lopez et al. (2018) Deep generative modeling for single-cell transcriptomics
"""
    
    return card


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """Extract basic information about the model."""
    
    info = {
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "model_path": str(model_path),
    }
    
    # Try to load the model to get more info
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            # Extract some basic info from the checkpoint
            if 'model_state_dict' in checkpoint:
                info['has_state_dict'] = True
            if 'optimizer_state_dict' in checkpoint:
                info['has_optimizer_state'] = True
            if 'epoch' in checkpoint:
                info['last_epoch'] = checkpoint['epoch']
    except Exception as e:
        logger.warning(f"Could not load model for detailed info: {e}")
    
    return info


def upload_model_to_hf(
    model_path: Path,
    repo_name: str,
    username: str,
    private: bool = False,
    token: str = None
) -> str:
    """
    Upload the trained model to Hugging Face Hub.
    
    Args:
        model_path: Path to the model file
        repo_name: Name for the repository
        username: Hugging Face username
        private: Whether the repository should be private
        token: Hugging Face token (if not provided, will use login)
    
    Returns:
        URL of the uploaded repository
    """
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create repository name
    repo_id = f"{username}/{repo_name}"
    
    # Get model info
    model_info = get_model_info(model_path)
    logger.info(f"Model info: {model_info}")
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            token=token,
            exist_ok=True
        )
        logger.info(f"Repository created/accessed: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise
    
    # Create model card
    model_card = create_model_card(model_info)
    
    # Upload files
    api = HfApi(token=token)
    
    # Upload model file
    logger.info(f"Uploading model file: {model_path}")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="model.pt",
        repo_id=repo_id,
        commit_message="Add trained scVI model"
    )
    
    # Upload model card
    logger.info("Uploading model card")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card"
    )
    
    # Upload model info as JSON
    logger.info("Uploading model metadata")
    api.upload_file(
        path_or_fileobj=json.dumps(model_info, indent=2).encode(),
        path_in_repo="model_info.json",
        repo_id=repo_id,
        commit_message="Add model metadata"
    )
    
    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Model successfully uploaded to: {repo_url}")
    
    return repo_url


def main():
    parser = argparse.ArgumentParser(description="Upload trained scVI model to Hugging Face")
    parser.add_argument(
        "--model-path",
        default="cache/scvi_model/model.pt",
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--repo-name",
        default="leukomap-scvi",
        help="Name for the Hugging Face repository"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token (optional, will use login if not provided)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Available model files:")
        cache_dir = Path("cache")
        if cache_dir.exists():
            for file in cache_dir.rglob("*.pt"):
                logger.info(f"  {file}")
        return 1
    
    try:
        repo_url = upload_model_to_hf(
            model_path=model_path,
            repo_name=args.repo_name,
            username=args.username,
            private=args.private,
            token=args.token
        )
        
        print(f"\n✅ Model successfully uploaded!")
        print(f"Repository URL: {repo_url}")
        print(f"\nTo use this model on another computer:")
        print(f"from scvi.model import SCVI")
        print(f"model = SCVI.load('{args.username}/{args.repo_name}', adata)")
        
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 