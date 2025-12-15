"""
Service de Transfer Learning pour les modèles RUL
Pré-entraînement sur NASA C-MAPSS et fine-tuning sur données usine
"""
import logging
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class TransferLearningService:
    """Service pour gérer le transfer learning depuis NASA C-MAPSS"""
    
    def __init__(self):
        """Initialise le service de transfer learning"""
        self.transfer_learning_enabled = settings.transfer_learning_enabled
        self.pretrained_path = settings.transfer_learning_pretrained_path
        self.freeze_layers = settings.transfer_learning_freeze_layers
        
        # Modèles pré-entraînés chargés
        self.pretrained_models: Dict[str, Any] = {}
        
        logger.info(f"TransferLearningService initialisé (enabled: {self.transfer_learning_enabled})")
    
    def load_pretrained_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Optional[torch.nn.Module]:
        """
        Charge un modèle pré-entraîné depuis un fichier
        
        Args:
            model_name: Nom du modèle ('lstm', 'gru', 'tcn')
            model_path: Chemin vers le modèle (optionnel, utilise config si None)
            device: Device PyTorch (optionnel)
        
        Returns:
            Modèle PyTorch chargé ou None
        """
        if not self.transfer_learning_enabled:
            logger.warning("Transfer learning désactivé dans la configuration")
            return None
        
        model_path = model_path or self.pretrained_path
        
        if model_path is None:
            logger.warning(f"Aucun chemin de modèle pré-entraîné fourni pour {model_name}")
            return None
        
        # Construire le chemin complet
        if not os.path.isabs(model_path):
            # Chemin relatif, chercher dans le dossier du projet
            base_path = Path(__file__).parent.parent.parent.parent
            model_path = base_path / "models" / "pretrained" / model_path
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"Modèle pré-entraîné non trouvé: {model_path}")
            return None
        
        try:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Charger le modèle
            if model_name in ['lstm', 'gru']:
                # Pour LSTM/GRU, on charge l'état du modèle
                checkpoint = torch.load(model_path, map_location=device)
                
                # Vérifier si c'est un checkpoint complet ou juste les poids
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        model_config = checkpoint.get('config', {})
                    else:
                        state_dict = checkpoint
                        model_config = {}
                else:
                    state_dict = checkpoint
                    model_config = {}
                
                # Stocker pour utilisation ultérieure
                self.pretrained_models[model_name] = {
                    'state_dict': state_dict,
                    'config': model_config,
                    'path': str(model_path)
                }
                
                logger.info(f"Modèle pré-entraîné {model_name} chargé depuis {model_path}")
                return state_dict  # Retourner state_dict pour application au modèle
                
            elif model_name == 'tcn':
                # TCN similaire
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        model_config = checkpoint.get('config', {})
                    else:
                        state_dict = checkpoint
                        model_config = {}
                else:
                    state_dict = checkpoint
                    model_config = {}
                
                self.pretrained_models[model_name] = {
                    'state_dict': state_dict,
                    'config': model_config,
                    'path': str(model_path)
                }
                
                logger.info(f"Modèle pré-entraîné {model_name} chargé depuis {model_path}")
                return state_dict
                
            else:
                logger.warning(f"Type de modèle non supporté pour transfer learning: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle pré-entraîné {model_name}: {e}", exc_info=True)
            return None
    
    def apply_transfer_learning(
        self,
        model: torch.nn.Module,
        model_name: str,
        freeze_encoder: bool = None
    ) -> torch.nn.Module:
        """
        Applique le transfer learning à un modèle
        
        Args:
            model: Modèle PyTorch à modifier
            model_name: Nom du modèle ('lstm', 'gru', 'tcn')
            freeze_encoder: Si True, freeze les couches de l'encodeur (override config)
        
        Returns:
            Modèle modifié avec transfer learning
        """
        if model_name not in self.pretrained_models:
            logger.warning(f"Aucun modèle pré-entraîné disponible pour {model_name}")
            return model
        
        freeze_encoder = freeze_encoder if freeze_encoder is not None else self.freeze_layers
        
        try:
            state_dict = self.pretrained_models[model_name]['state_dict']
            
            # Charger les poids pré-entraînés
            model_dict = model.state_dict()
            
            # Filtrer les poids compatibles
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            
            if not pretrained_dict:
                logger.warning(f"Aucun poids compatible trouvé pour {model_name}")
                return model
            
            # Charger les poids compatibles
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            logger.info(f"Transfer learning appliqué à {model_name}: {len(pretrained_dict)}/{len(state_dict)} poids chargés")
            
            # Freeze les couches si demandé
            if freeze_encoder:
                self._freeze_encoder_layers(model, model_name)
                logger.info(f"Couches de l'encodeur gelées pour {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Erreur lors de l'application du transfer learning: {e}", exc_info=True)
            return model
    
    def _freeze_encoder_layers(self, model: torch.nn.Module, model_name: str):
        """Gèle les couches de l'encodeur pour le fine-tuning"""
        if model_name in ['lstm', 'gru']:
            # Pour LSTM/GRU, geler les couches LSTM/GRU
            if hasattr(model, 'lstm'):
                for param in model.lstm.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'gru'):
                for param in model.gru.parameters():
                    param.requires_grad = False
        elif model_name == 'tcn':
            # Pour TCN, geler les premières couches
            if hasattr(model, 'tcn'):
                for i, layer in enumerate(model.tcn.layers):
                    if i < len(model.tcn.layers) // 2:  # Geler la moitié des couches
                        for param in layer.parameters():
                            param.requires_grad = False
    
    def fine_tune_model(
        self,
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        learning_rate: float = 0.0001,  # Learning rate plus faible pour fine-tuning
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune un modèle pré-entraîné sur de nouvelles données
        
        Args:
            model: Modèle avec transfer learning appliqué
            X_train: Données d'entraînement
            y_train: Targets d'entraînement
            epochs: Nombre d'epochs pour fine-tuning
            learning_rate: Learning rate (plus faible que l'entraînement initial)
            batch_size: Taille de batch
            device: Device PyTorch
        
        Returns:
            Dictionnaire avec métriques de fine-tuning
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Optimizer avec learning rate réduit
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()
        
        # Dataset et DataLoader
        from torch.utils.data import DataLoader, TensorDataset
        
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Fine-tuning loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Fine-tuning Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Calculer métriques finales
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy().flatten()
            actuals = y_train
            
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return {
            "status": "success",
            "epochs": epochs,
            "final_loss": train_losses[-1],
            "mae": float(mae),
            "rmse": float(rmse),
            "train_losses": train_losses
        }
    
    def save_pretrained_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        save_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Sauvegarde un modèle pré-entraîné
        
        Args:
            model: Modèle PyTorch à sauvegarder
            model_name: Nom du modèle
            save_path: Chemin de sauvegarde
            config: Configuration du modèle (optionnel)
        
        Returns:
            True si succès, False sinon
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'config': config or {}
            }
            
            torch.save(checkpoint, save_path)
            logger.info(f"Modèle pré-entraîné {model_name} sauvegardé dans {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {e}", exc_info=True)
            return False
    
    def get_pretrained_info(self) -> Dict[str, Any]:
        """Retourne des informations sur les modèles pré-entraînés chargés"""
        return {
            "enabled": self.transfer_learning_enabled,
            "freeze_layers": self.freeze_layers,
            "pretrained_models": {
                name: {
                    "path": info['path'],
                    "config": info.get('config', {})
                }
                for name, info in self.pretrained_models.items()
            }
        }

