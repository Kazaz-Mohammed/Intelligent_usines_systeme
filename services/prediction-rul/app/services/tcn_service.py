"""
Service TCN (Temporal Convolutional Network) pour la prédiction RUL
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
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


class Chomp1d(nn.Module):
    """Module pour enlever le padding à droite"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Bloc temporel pour TCN"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Réseau de convolution temporelle"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    """Modèle TCN pour prédiction RUL"""
    
    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Nombre de features d'entrée
            num_channels: Liste des canaux pour chaque couche
            kernel_size: Taille du kernel de convolution
            dropout: Taux de dropout
        """
        super(TCNModel, self).__init__()
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1] // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_channels[-1] // 2, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor de forme (batch_size, sequence_length, input_size)
        
        Returns:
            Tensor de forme (batch_size, 1) - prédiction RUL
        """
        # TCN attend (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # TCN forward
        tcn_out = self.tcn(x)
        
        # Prendre la dernière sortie de la séquence
        last_output = tcn_out[:, :, -1]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # RUL doit être positive
        out = torch.relu(out)
        
        return out


class TCNService:
    """Service pour la prédiction RUL avec TCN"""
    
    def __init__(self, transfer_learning_service: Optional[Any] = None):
        """Initialise le service TCN"""
        self.model: Optional[TCNModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size: Optional[int] = None
        self.sequence_length = settings.tcn_sequence_length
        self.is_trained = False
        self.transfer_learning_service = transfer_learning_service
        
        logger.info(f"TCN Service initialisé sur device: {self.device}")
    
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
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle TCN
        
        Args:
            X_train: Données d'entraînement (n_samples, n_features) ou (n_samples, sequence_length, n_features)
            y_train: Targets d'entraînement (n_samples,)
            X_val: Données de validation (optionnel)
            y_val: Targets de validation (optionnel)
            feature_names: Noms des features
        
        Returns:
            Dict avec métriques d'entraînement
        """
        logger.info("Début de l'entraînement TCN")
        
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
        self.model = TCNModel(
            input_size=self.input_size,
            num_channels=settings.tcn_num_channels,
            kernel_size=settings.tcn_kernel_size,
            dropout=settings.tcn_dropout
        ).to(self.device)
        
        # Optimizer et loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=settings.tcn_learning_rate
        )
        criterion = nn.MSELoss()
        
        # Dataset et DataLoader
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=settings.tcn_batch_size,
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
                batch_size=settings.tcn_batch_size,
                shuffle=False
            )
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(settings.tcn_epochs):
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
                        f"Epoch {epoch + 1}/{settings.tcn_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{settings.tcn_epochs} - "
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
        logger.info(f"Entraînement TCN terminé. MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
        
        # MLflow logging
        if settings.mlflow_enabled and MLFLOW_AVAILABLE:
            try:
                mlflow.log_params({
                    "model": "tcn",
                    "num_channels": str(settings.tcn_num_channels),
                    "kernel_size": settings.tcn_kernel_size,
                    "dropout": settings.tcn_dropout,
                    "sequence_length": self.sequence_length,
                    "batch_size": settings.tcn_batch_size,
                    "epochs": settings.tcn_epochs,
                    "learning_rate": settings.tcn_learning_rate
                })
                mlflow.log_metrics(metrics)
                mlflow.pytorch.log_model(self.model, "tcn_model")
                logger.info("Modèle TCN enregistré dans MLflow")
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
            "model_type": "tcn",
            "is_trained": self.is_trained,
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "num_channels": settings.tcn_num_channels,
            "kernel_size": settings.tcn_kernel_size,
            "device": str(self.device)
        }

