"""
Service GRU pour la prédiction RUL
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MLflow imports are optional
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from app.config import settings

# Import optionnel pour transfer learning
try:
    from app.services.transfer_learning_service import TransferLearningService
    TRANSFER_LEARNING_AVAILABLE = True
except ImportError:
    TRANSFER_LEARNING_AVAILABLE = False
    TransferLearningService = None

logger = logging.getLogger(__name__)


class GRUModel(nn.Module):
    """Modèle GRU pour prédiction RUL"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Nombre de features d'entrée
            hidden_size: Taille de la couche cachée
            num_layers: Nombre de couches GRU
            dropout: Taux de dropout
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor de forme (batch_size, sequence_length, input_size)
        
        Returns:
            Tensor de forme (batch_size, 1) - prédiction RUL
        """
        # GRU forward
        gru_out, _ = self.gru(x)
        
        # Prendre la dernière sortie de la séquence
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # RUL doit être positive
        out = torch.relu(out)
        
        return out


class GRUService:
    """Service pour la prédiction RUL avec GRU"""
    
    def __init__(self, transfer_learning_service: Optional[Any] = None):
        """Initialise le service GRU"""
        self.model: Optional[GRUModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size: Optional[int] = None
        self.sequence_length = settings.gru_sequence_length
        self.is_trained = False
        self.transfer_learning_service = transfer_learning_service
        
        logger.info(f"GRU Service initialisé sur device: {self.device}")
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences temporelles à partir des données
        
        Args:
            data: Données de forme (n_samples, n_features)
            sequence_length: Longueur de la séquence
        
        Returns:
            X: Séquences de forme (n_sequences, sequence_length, n_features)
            y: Targets de forme (n_sequences,)
        """
        if len(data) < sequence_length:
            raise ValueError(f"Données insuffisantes: {len(data)} < {sequence_length}")
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, -1])  # Dernière colonne = RUL
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle GRU
        
        Args:
            X_train: Données d'entraînement (n_samples, n_features) ou (n_samples, sequence_length, n_features)
            y_train: Targets d'entraînement (n_samples,)
            X_val: Données de validation (optionnel)
            y_val: Targets de validation (optionnel)
            feature_names: Noms des features
        
        Returns:
            Dict avec métriques d'entraînement
        """
        logger.info("Début de l'entraînement GRU")
        
        # Préparer les données
        if len(X_train.shape) == 2:
            X_seq, y_seq = self._create_sequences(
                np.column_stack([X_train, y_train]),
                self.sequence_length
            )
        else:
            X_seq = X_train
            y_seq = y_train
        
        self.input_size = X_seq.shape[2]
        
        # Créer le modèle
        self.model = GRUModel(
            input_size=self.input_size,
            hidden_size=settings.gru_hidden_size,
            num_layers=settings.gru_num_layers,
            dropout=settings.gru_dropout
        ).to(self.device)
        
        # Paramètres d'entraînement (override si fournis)
        train_epochs = epochs if epochs is not None else settings.gru_epochs
        train_batch_size = batch_size if batch_size is not None else settings.gru_batch_size
        
        # Optimizer et loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=settings.gru_learning_rate
        )
        criterion = nn.MSELoss()
        
        # Dataset et DataLoader
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True
        )
        
        # Validation data
        val_dataloader = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val_seq, y_val_seq = self._create_sequences(
                    np.column_stack([X_val, y_val]),
                    self.sequence_length
                )
            else:
                X_val_seq = X_val
                y_val_seq = y_val
            
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).reshape(-1, 1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=train_batch_size,
                shuffle=False
            )
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(train_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(dataloader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_dataloader is not None:
                self.model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        predictions = self.model(batch_X)
                        loss = criterion(predictions, batch_y)
                        epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{train_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{train_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}"
                    )
        
        # Évaluation finale
        self.model.eval()
        with torch.no_grad():
            train_predictions = self.model(X_tensor).cpu().numpy().flatten()
            train_mae = mean_absolute_error(y_seq, train_predictions)
            train_rmse = np.sqrt(mean_squared_error(y_seq, train_predictions))
            train_r2 = r2_score(y_seq, train_predictions)
        
        metrics = {
            "train_mae": float(train_mae),
            "train_rmse": float(train_rmse),
            "train_r2": float(train_r2),
            "final_train_loss": float(train_losses[-1])
        }
        
        if val_dataloader is not None:
            val_predictions = self.model(X_val_tensor).cpu().numpy().flatten()
            metrics["val_mae"] = float(mean_absolute_error(y_val_seq, val_predictions))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val_seq, val_predictions)))
            metrics["val_r2"] = float(r2_score(y_val_seq, val_predictions))
            metrics["best_val_loss"] = float(best_val_loss)
        
        self.is_trained = True
        logger.info(f"Entraînement GRU terminé. MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
        
        # MLflow logging
        if settings.mlflow_enabled and MLFLOW_AVAILABLE:
            try:
                mlflow.log_params({
                    "model": "gru",
                    "hidden_size": settings.gru_hidden_size,
                    "num_layers": settings.gru_num_layers,
                    "dropout": settings.gru_dropout,
                    "sequence_length": self.sequence_length,
                    "batch_size": settings.gru_batch_size,
                    "epochs": settings.gru_epochs,
                    "learning_rate": settings.gru_learning_rate
                })
                mlflow.log_metrics(metrics)
                mlflow.pytorch.log_model(self.model, "gru_model")
                logger.info("Modèle GRU enregistré dans MLflow")
            except Exception as e:
                logger.warning(f"Erreur lors du logging MLflow: {e}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la RUL
        
        Args:
            X: Données de forme (n_samples, n_features) ou (n_samples, sequence_length, n_features)
        
        Returns:
            Prédictions RUL (n_samples,)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez train() d'abord.")
        
        self.model.eval()
        
        # Préparer les données
        if len(X.shape) == 2:
            if len(X) < self.sequence_length:
                padding = np.tile(X[0:1], (self.sequence_length - len(X), 1))
                X = np.vstack([padding, X])
            
            X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        else:
            X_seq = X
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "model_type": "gru",
            "is_trained": self.is_trained,
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "hidden_size": settings.gru_hidden_size,
            "num_layers": settings.gru_num_layers,
            "device": str(self.device)
        }

