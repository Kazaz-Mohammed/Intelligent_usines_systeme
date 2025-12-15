"""
Tests pour le service de transfer learning
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from app.services.transfer_learning_service import TransferLearningService
from app.config import settings


@pytest.fixture
def transfer_learning_service():
    """Fixture pour créer une instance du service"""
    with patch('app.services.transfer_learning_service.settings') as mock_settings:
        mock_settings.transfer_learning_enabled = True
        mock_settings.transfer_learning_pretrained_path = None
        mock_settings.transfer_learning_freeze_layers = False
        service = TransferLearningService()
        return service


@pytest.fixture
def sample_model():
    """Fixture pour créer un modèle PyTorch simple"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.lstm = nn.LSTM(5, 64, 2, batch_first=True)
            self.fc = nn.Linear(64, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    
    return SimpleModel()


@pytest.fixture
def sample_checkpoint(sample_model, tmp_path):
    """Fixture pour créer un checkpoint de modèle"""
    checkpoint_path = tmp_path / "pretrained_lstm.pt"
    checkpoint = {
        'model_state_dict': sample_model.state_dict(),
        'config': {'input_size': 5, 'hidden_size': 64}
    }
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


class TestTransferLearningService:
    """Tests pour TransferLearningService"""
    
    def test_init(self, transfer_learning_service):
        """Test initialisation"""
        assert transfer_learning_service.transfer_learning_enabled is True
        assert len(transfer_learning_service.pretrained_models) == 0
    
    def test_load_pretrained_model_success(self, transfer_learning_service, sample_checkpoint):
        """Test chargement d'un modèle pré-entraîné"""
        result = transfer_learning_service.load_pretrained_model(
            "lstm",
            model_path=sample_checkpoint
        )
        
        assert result is not None
        assert "lstm" in transfer_learning_service.pretrained_models
        assert isinstance(result, dict)  # state_dict
    
    def test_load_pretrained_model_not_found(self, transfer_learning_service):
        """Test chargement avec fichier inexistant"""
        result = transfer_learning_service.load_pretrained_model(
            "lstm",
            model_path="/nonexistent/path/model.pt"
        )
        
        assert result is None
        assert "lstm" not in transfer_learning_service.pretrained_models
    
    def test_load_pretrained_model_disabled(self):
        """Test avec transfer learning désactivé"""
        with patch('app.services.transfer_learning_service.settings') as mock_settings:
            mock_settings.transfer_learning_enabled = False
            service = TransferLearningService()
            
            result = service.load_pretrained_model("lstm", model_path="test.pt")
            assert result is None
    
    def test_apply_transfer_learning(self, transfer_learning_service, sample_model, sample_checkpoint):
        """Test application du transfer learning"""
        # Charger le modèle pré-entraîné
        transfer_learning_service.load_pretrained_model("lstm", model_path=sample_checkpoint)
        
        # Appliquer transfer learning
        modified_model = transfer_learning_service.apply_transfer_learning(
            sample_model,
            "lstm"
        )
        
        assert modified_model is not None
        assert modified_model == sample_model  # Même instance
    
    def test_apply_transfer_learning_no_pretrained(self, transfer_learning_service, sample_model):
        """Test application sans modèle pré-entraîné"""
        modified_model = transfer_learning_service.apply_transfer_learning(
            sample_model,
            "lstm"
        )
        
        assert modified_model == sample_model
    
    def test_freeze_encoder_layers(self, transfer_learning_service, sample_model, sample_checkpoint):
        """Test gel des couches de l'encodeur"""
        transfer_learning_service.load_pretrained_model("lstm", model_path=sample_checkpoint)
        transfer_learning_service.apply_transfer_learning(
            sample_model,
            "lstm",
            freeze_encoder=True
        )
        
        # Vérifier que les paramètres LSTM sont gelés
        for param in sample_model.lstm.parameters():
            assert param.requires_grad is False
        
        # Vérifier que les paramètres FC ne sont pas gelés
        for param in sample_model.fc.parameters():
            assert param.requires_grad is True
    
    def test_fine_tune_model(self, transfer_learning_service, sample_model):
        """Test fine-tuning d'un modèle"""
        # Données d'entraînement
        np.random.seed(42)
        X_train = np.random.randn(50, 10, 5)  # (samples, sequence, features)
        y_train = np.random.rand(50) * 200
        
        result = transfer_learning_service.fine_tune_model(
            sample_model,
            X_train,
            y_train,
            epochs=2,
            learning_rate=0.001,
            batch_size=8
        )
        
        assert result["status"] == "success"
        assert "mae" in result
        assert "rmse" in result
        assert result["epochs"] == 2
    
    def test_save_pretrained_model(self, transfer_learning_service, sample_model, tmp_path):
        """Test sauvegarde d'un modèle pré-entraîné"""
        save_path = tmp_path / "saved_model.pt"
        
        result = transfer_learning_service.save_pretrained_model(
            sample_model,
            "lstm",
            str(save_path),
            config={"input_size": 5}
        )
        
        assert result is True
        assert save_path.exists()
        
        # Vérifier que le fichier peut être chargé
        checkpoint = torch.load(save_path)
        assert "model_state_dict" in checkpoint
        assert checkpoint["model_name"] == "lstm"
    
    def test_get_pretrained_info(self, transfer_learning_service, sample_checkpoint):
        """Test récupération des informations"""
        transfer_learning_service.load_pretrained_model("lstm", model_path=sample_checkpoint)
        
        info = transfer_learning_service.get_pretrained_info()
        
        assert "enabled" in info
        assert "freeze_layers" in info
        assert "pretrained_models" in info
        assert "lstm" in info["pretrained_models"]

