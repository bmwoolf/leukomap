"""
Tests for scVI training and embedding functionality.
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from unittest.mock import Mock, patch
import tempfile
import os

from leukomap.scvi_training import (
    setup_scvi_data,
    train_scvi,
    setup_scanvi_data,
    train_scanvi,
    embed,
    get_latent_representation,
    add_latent_to_adata,
    train_models
)


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    # Create synthetic data
    n_cells = 100
    n_genes = 50
    
    # Create sparse count matrix
    X = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)
    
    # Create cell metadata
    obs = pd.DataFrame({
        'sample_type': np.random.choice(['ETV6-RUNX1', 'HHD', 'PRE-T', 'PBMMC'], n_cells),
        'celltype': np.random.choice(['B-cell', 'T-cell', 'Myeloid', 'Unknown'], n_cells),
        'n_genes': np.random.randint(200, 1000, n_cells),
        'n_counts': np.random.randint(1000, 10000, n_cells)
    })
    
    # Create gene metadata
    var = pd.DataFrame({
        'gene_name': [f'Gene_{i}' for i in range(n_genes)],
        'highly_variable': np.random.choice([True, False], n_genes)
    })
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def mock_scvi_model():
    """Create a mock scVI model for testing."""
    model = Mock()
    model.get_latent_representation.return_value = np.random.randn(100, 10)
    return model


class TestSetupScviData:
    """Test scVI data setup functionality."""
    
    @patch('leukomap.scvi_training.SCVI')
    def test_setup_scvi_data_success(self, mock_scvi, mock_adata):
        """Test successful scVI data setup."""
        result = setup_scvi_data(mock_adata, batch_key="sample_type")
        
        assert result is mock_adata
        mock_scvi.setup_anndata.assert_called_once_with(
            mock_adata, batch_key="sample_type", layer=None
        )
    
    def test_setup_scvi_data_missing_batch_key(self, mock_adata):
        """Test scVI data setup with missing batch key."""
        # Remove batch key
        mock_adata.obs = mock_adata.obs.drop(columns=['sample_type'])
        
        with pytest.raises(ValueError, match="Batch key 'sample_type' not found"):
            setup_scvi_data(mock_adata, batch_key="sample_type")


class TestSetupScanviData:
    """Test scANVI data setup functionality."""
    
    @patch('leukomap.scvi_training.SCANVI')
    def test_setup_scanvi_data_success(self, mock_scanvi, mock_adata):
        """Test successful scANVI data setup."""
        result = setup_scanvi_data(mock_adata, batch_key="sample_type", labels_key="celltype")
        
        assert result is mock_adata
        mock_scanvi.setup_anndata.assert_called_once_with(
            mock_adata, batch_key="sample_type", labels_key="celltype", layer=None
        )
    
    def test_setup_scanvi_data_missing_keys(self, mock_adata):
        """Test scANVI data setup with missing keys."""
        # Remove required keys
        mock_adata.obs = mock_adata.obs.drop(columns=['sample_type', 'celltype'])
        
        with pytest.raises(ValueError, match="Batch key 'sample_type' not found"):
            setup_scanvi_data(mock_adata, batch_key="sample_type", labels_key="celltype")


class TestTrainScvi:
    """Test scVI training functionality."""
    
    @patch('leukomap.scvi_training.SCVI')
    @patch('leukomap.scvi_training.scvi.settings')
    def test_train_scvi_success(self, mock_settings, mock_scvi_class, mock_adata):
        """Test successful scVI training."""
        # Setup mock model
        mock_model = Mock()
        mock_scvi_class.return_value = mock_model
        
        # Setup data
        setup_scvi_data(mock_adata)
        
        result = train_scvi(mock_adata, max_epochs=10)
        
        assert result is mock_model
        mock_model.train.assert_called_once_with(max_epochs=10, train_size=0.9)
        mock_settings.seed.assert_called_once_with(42)


class TestTrainScanvi:
    """Test scANVI training functionality."""
    
    @patch('leukomap.scvi_training.SCANVI')
    @patch('leukomap.scvi_training.scvi.settings')
    def test_train_scanvi_success(self, mock_settings, mock_scanvi_class, mock_adata):
        """Test successful scANVI training."""
        # Setup mock model
        mock_model = Mock()
        mock_scanvi_class.return_value = mock_model
        
        # Setup data
        setup_scanvi_data(mock_adata)
        
        result = train_scanvi(mock_adata, max_epochs=10)
        
        assert result is mock_model
        mock_model.train.assert_called_once_with(max_epochs=10, train_size=0.9)
        mock_settings.seed.assert_called_once_with(42)


class TestGetLatentRepresentation:
    """Test latent representation extraction."""
    
    def test_get_latent_representation_success(self, mock_scvi_model, mock_adata):
        """Test successful latent representation extraction."""
        result = get_latent_representation(mock_scvi_model, mock_adata)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 10)
        mock_scvi_model.get_latent_representation.assert_called_once_with(
            mock_adata, give_mean=True, batch_size=128
        )


class TestAddLatentToAdata:
    """Test adding latent representation to AnnData."""
    
    def test_add_latent_to_adata_success(self, mock_scvi_model, mock_adata):
        """Test successfully adding latent representation to AnnData."""
        result = add_latent_to_adata(mock_scvi_model, mock_adata, key="X_test")
        
        assert result is mock_adata
        assert "X_test" in result.obsm
        assert result.obsm["X_test"].shape == (100, 10)


class TestEmbed:
    """Test the main embed function."""
    
    def test_embed_success(self, mock_scvi_model, mock_adata):
        """Test successful embedding extraction."""
        result = embed(mock_scvi_model, mock_adata, key="X_embed")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 10)
        assert "X_embed" in mock_adata.obsm
        assert mock_adata.obsm["X_embed"].shape == (100, 10)
    
    def test_embed_custom_parameters(self, mock_scvi_model, mock_adata):
        """Test embedding with custom parameters."""
        result = embed(
            mock_scvi_model, 
            mock_adata, 
            give_mean=False, 
            batch_size=64, 
            key="X_custom"
        )
        
        assert isinstance(result, np.ndarray)
        mock_scvi_model.get_latent_representation.assert_called_with(
            mock_adata, give_mean=False, batch_size=64
        )


class TestTrainModels:
    """Test the train_models pipeline function."""
    
    @patch('leukomap.scvi_training.train_scvi')
    @patch('leukomap.scvi_training.train_scanvi')
    @patch('leukomap.scvi_training.setup_scvi_data')
    @patch('leukomap.scvi_training.setup_scanvi_data')
    def test_train_models_scvi_only(
        self, mock_setup_scanvi, mock_setup_scvi, mock_train_scanvi, mock_train_scvi, mock_adata
    ):
        """Test training only scVI model."""
        mock_scvi_model = Mock()
        mock_train_scvi.return_value = mock_scvi_model
        
        result = train_models(mock_adata, do_train_scvi=True, do_train_scanvi=False)
        
        assert "scvi" in result
        assert result["scvi"] is mock_scvi_model
        assert "scanvi" not in result
        mock_train_scvi.assert_called_once()
        mock_train_scanvi.assert_not_called()
    
    @patch('leukomap.scvi_training.train_scvi')
    @patch('leukomap.scvi_training.train_scanvi')
    @patch('leukomap.scvi_training.setup_scvi_data')
    @patch('leukomap.scvi_training.setup_scanvi_data')
    def test_train_models_scanvi_only(
        self, mock_setup_scanvi, mock_setup_scvi, mock_train_scanvi, mock_train_scvi, mock_adata
    ):
        """Test training only scANVI model."""
        mock_scanvi_model = Mock()
        mock_train_scanvi.return_value = mock_scanvi_model
        
        result = train_models(mock_adata, do_train_scvi=False, do_train_scanvi=True)
        
        assert "scanvi" in result
        assert result["scanvi"] is mock_scanvi_model
        assert "scvi" not in result
        mock_train_scanvi.assert_called_once()
        mock_train_scvi.assert_not_called()
    
    @patch('leukomap.scvi_training.train_scvi')
    @patch('leukomap.scvi_training.train_scanvi')
    @patch('leukomap.scvi_training.setup_scvi_data')
    @patch('leukomap.scvi_training.setup_scanvi_data')
    def test_train_models_both(
        self, mock_setup_scanvi, mock_setup_scvi, mock_train_scanvi, mock_train_scvi, mock_adata
    ):
        """Test training both scVI and scANVI models."""
        mock_scvi_model = Mock()
        mock_scanvi_model = Mock()
        mock_train_scvi.return_value = mock_scvi_model
        mock_train_scanvi.return_value = mock_scanvi_model
        
        result = train_models(mock_adata, do_train_scvi=True, do_train_scanvi=True)
        
        assert "scvi" in result
        assert "scanvi" in result
        assert result["scvi"] is mock_scvi_model
        assert result["scanvi"] is mock_scanvi_model
        mock_train_scvi.assert_called_once()
        mock_train_scanvi.assert_called_once()


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @patch('leukomap.scvi_training.SCVI')
    @patch('leukomap.scvi_training.scvi.settings')
    def test_complete_embedding_pipeline(self, mock_settings, mock_scvi_class, mock_adata):
        """Test the complete embedding pipeline from setup to embedding."""
        # Setup mock model
        mock_model = Mock()
        mock_model.get_latent_representation.return_value = np.random.randn(100, 10)
        mock_scvi_class.return_value = mock_model
        
        # Setup data
        setup_scvi_data(mock_adata)
        
        # Train model
        trained_model = train_scvi(mock_adata, max_epochs=10)
        
        # Extract embeddings
        latent_space = embed(trained_model, mock_adata, key="X_scvi")
        
        # Verify results
        assert isinstance(latent_space, np.ndarray)
        assert latent_space.shape == (100, 10)
        assert "X_scvi" in mock_adata.obsm
        assert mock_adata.obsm["X_scvi"].shape == (100, 10) 