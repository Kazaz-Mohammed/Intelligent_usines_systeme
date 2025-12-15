"""
Service LSTM Autoencoder pour la détection d'anomalies
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple
from torch.utils.data import DataLoader, TensorDataset
from app.config import settings
from app.services.mlflow_service import MLflowService

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """Architecture LSTM Autoencoder"""
    
    def __init__(self, input_dim: int, encoder_layers: List[int], decoder_layers: List[int]):
        """
        Initialise l'autoencodeur LSTM
        
        Args:
            input_dim: Dimension des features d'entrée
            encoder_layers: Liste des dimensions des couches LSTM de l'encodeur
            decoder_layers: Liste des dimensions des couches LSTM du décodeur
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i, hidden_dim in enumerate(encoder_layers):
            input_size = input_dim if i == 0 else encoder_layers[i - 1]
            self.encoder.append(nn.LSTM(input_size, hidden_dim, batch_first=True))
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i, hidden_dim in enumerate(decoder_layers):
            input_size = decoder_layers[i - 1] if i > 0 else encoder_layers[-1]
            output_size = input_dim if i == len(decoder_layers) - 1 else decoder_layers[i]
            self.decoder.append(nn.LSTM(input_size, hidden_dim, batch_first=True))
        
        # Couche de sortie pour reconstruire les features
        self.output_layer = nn.Linear(decoder_layers[-1], input_dim)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode l'input
        
        Args:
            x: Tensor de forme (batch_size, sequence_length, input_dim)
        
        Returns:
            Encoded representation
        """
        h = x
        for lstm in self.encoder:
            h, (hidden, cell) = lstm(h)
        return h
    
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Décode l'encoded representation
        
        Args:
            encoded: Encoded representation
        
        Returns:
            Reconstructed output
        """
        h = encoded
        for lstm in self.decoder:
            h, (hidden, cell) = lstm(h)
        # Appliquer la couche de sortie
        output = self.output_layer(h)
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor de forme (batch_size, sequence_length, input_dim)
        
        Returns:
            Reconstructed output
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


class LSTMAutoencoderService:
    """Service pour la détection d'anomalies avec LSTM Autoencoder"""
    
    def __init__(self):
        """Initialise le service LSTM Autoencoder"""
        self.model: Optional[LSTMAutoencoder] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names: Optional[List[str]] = None
        self.is_trained: bool = False
        self.threshold: Optional[float] = None
        self.mlflow_service = MLflowService()
        
        # Paramètres depuis la config
        self.encoder_layers = settings.lstm_autoencoder_encoder_layers
        self.decoder_layers = settings.lstm_autoencoder_decoder_layers
        self.sequence_length = settings.lstm_autoencoder_sequence_length
        self.batch_size = settings.lstm_autoencoder_batch_size
        self.epochs = settings.lstm_autoencoder_epochs
        self.learning_rate = settings.lstm_autoencoder_learning_rate
        self.threshold_percentile = settings.lstm_autoencoder_threshold_percentile
        
        logger.info(
            f"LSTMAutoencoderService initialisé sur device={self.device}, "
            f"encoder_layers={self.encoder_layers}, decoder_layers={self.decoder_layers}, "
            f"sequence_length={self.sequence_length}"
        )
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Crée des séquences à partir des données
        
        Args:
            X: Données de forme (n_samples, n_features)
        
        Returns:
            Séquences de forme (n_sequences, sequence_length, n_features)
        """
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)
    
    def train(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Entraîne le modèle LSTM Autoencoder
        
        Args:
            X: Données d'entraînement (n_samples, n_features)
            feature_names: Noms des features (optionnel)
        
        Returns:
            Dict avec les métriques d'entraînement
        """
        try:
            logger.info(f"Entraînement LSTM Autoencoder sur {X.shape[0]} échantillons, {X.shape[1]} features")
            
            # Sauvegarder les noms de features
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Créer des séquences
            sequences = self._create_sequences(X)
            logger.info(f"Créé {len(sequences)} séquences de longueur {self.sequence_length}")
            
            # Convertir en Tensor
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            
            # Créer le DataLoader
            dataset = TensorDataset(sequences_tensor, sequences_tensor)  # Input = Target (autoencoder)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Initialiser le modèle
            input_dim = X.shape[1]
            self.model = LSTMAutoencoder(
                input_dim=input_dim,
                encoder_layers=self.encoder_layers,
                decoder_layers=self.decoder_layers
            ).to(self.device)
            
            # Optimizer et loss
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # Logging MLflow - démarrer le run
            run = self.mlflow_service.start_run(run_name=f"lstm_autoencoder_{X.shape[0]}_samples")
            
            # Entraînement
            self.model.train()
            train_losses = []
            
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_inputs, batch_targets in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    reconstructed = self.model(batch_inputs)
                    
                    # Loss
                    loss = criterion(reconstructed, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                train_losses.append(avg_loss)
                
                # Log metrics à chaque epoch
                if run:
                    self.mlflow_service.log_metrics({"train_loss": avg_loss}, step=epoch)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            # Calculer le seuil basé sur les erreurs de reconstruction
            self.model.eval()
            reconstruction_errors = []
            
            with torch.no_grad():
                for batch_inputs, batch_targets in dataloader:
                    reconstructed = self.model(batch_inputs)
                    errors = torch.mean((reconstructed - batch_targets) ** 2, dim=(1, 2))
                    reconstruction_errors.extend(errors.cpu().numpy())
            
            reconstruction_errors = np.array(reconstruction_errors)
            self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
            
            self.is_trained = True
            
            metrics = {
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_sequences": len(sequences),
                "sequence_length": self.sequence_length,
                "final_loss": float(train_losses[-1]),
                "min_loss": float(np.min(train_losses)),
                "max_loss": float(np.max(train_losses)),
                "threshold": float(self.threshold),
                "threshold_percentile": self.threshold_percentile,
                "mean_reconstruction_error": float(np.mean(reconstruction_errors)),
                "std_reconstruction_error": float(np.std(reconstruction_errors))
            }
            
            # Logging MLflow - paramètres et métriques finales
            if run:
                try:
                    # Log parameters
                    self.mlflow_service.log_params({
                        "model_type": "lstm_autoencoder",
                        "encoder_layers": str(self.encoder_layers),
                        "decoder_layers": str(self.decoder_layers),
                        "sequence_length": self.sequence_length,
                        "batch_size": self.batch_size,
                        "epochs": self.epochs,
                        "learning_rate": self.learning_rate,
                        "threshold_percentile": self.threshold_percentile,
                        "device": str(self.device),
                        "n_samples": X.shape[0],
                        "n_features": X.shape[1]
                    })
                    
                    # Log final metrics
                    self.mlflow_service.log_metrics({
                        "final_loss": metrics["final_loss"],
                        "min_loss": metrics["min_loss"],
                        "max_loss": metrics["max_loss"],
                        "threshold": metrics["threshold"],
                        "mean_reconstruction_error": metrics["mean_reconstruction_error"],
                        "std_reconstruction_error": metrics["std_reconstruction_error"]
                    })
                    
                    # Log model
                    self.mlflow_service.log_model_pytorch(
                        self.model,
                        artifact_path="lstm_autoencoder_model",
                        registered_model_name="lstm_autoencoder"
                    )
                finally:
                    self.mlflow_service.end_run()
            
            logger.info(f"LSTM Autoencoder entraîné avec succès. Métriques: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement LSTM Autoencoder: {e}", exc_info=True)
            raise
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les scores d'anomalie basés sur l'erreur de reconstruction
        
        Args:
            X: Données à scorer (n_samples, n_features)
        
        Returns:
            Scores d'anomalie normalisés entre 0 et 1
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            # Créer des séquences
            sequences = self._create_sequences(X)
            
            if len(sequences) == 0:
                # Si pas assez de données pour créer une séquence, retourner un score par défaut
                return np.array([0.5] * len(X))
            
            # Convertir en Tensor
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            
            # Calculer les erreurs de reconstruction
            self.model.eval()
            reconstruction_errors = []
            
            with torch.no_grad():
                reconstructed = self.model(sequences_tensor)
                errors = torch.mean((reconstructed - sequences_tensor) ** 2, dim=(1, 2))
                reconstruction_errors = errors.cpu().numpy()
            
            # Normaliser les scores entre 0 et 1
            # Plus l'erreur est élevée, plus le score est élevé
            if len(reconstruction_errors) > 0:
                min_error = np.min(reconstruction_errors)
                max_error = np.max(reconstruction_errors)
                if max_error > min_error:
                    scores = (reconstruction_errors - min_error) / (max_error - min_error)
                else:
                    scores = np.zeros_like(reconstruction_errors)
            else:
                scores = np.array([0.5])
            
            # Étendre les scores pour correspondre au nombre d'échantillons
            # Les premiers échantillons n'ont pas de séquence complète
            extended_scores = np.zeros(len(X))
            extended_scores[self.sequence_length - 1:] = scores
            
            # Remplir les premiers échantillons avec le premier score disponible
            if len(scores) > 0:
                extended_scores[:self.sequence_length - 1] = scores[0]
            
            return extended_scores
            
        except Exception as e:
            logger.error(f"Erreur lors du scoring LSTM Autoencoder: {e}", exc_info=True)
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les labels d'anomalie (0=normal, 1=anomalie)
        
        Args:
            X: Données à prédire (n_samples, n_features)
        
        Returns:
            Labels (0 ou 1)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            scores = self.predict_scores(X)
            
            # Utiliser le seuil pour déterminer les anomalies
            if self.threshold is not None:
                # Convertir le seuil en score normalisé (approximation)
                predictions = (scores >= 0.5).astype(int)  # Seuil à 0.5 pour les scores normalisés
            else:
                predictions = (scores >= 0.5).astype(int)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction LSTM Autoencoder: {e}", exc_info=True)
            raise
    
    def detect_anomaly(self, features: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Détecte une anomalie à partir d'un dictionnaire de features
        
        Note: Cette méthode nécessite un historique de features pour créer une séquence.
        Pour une détection en temps-réel, il faut maintenir un buffer de séquences.
        
        Args:
            features: Dictionnaire {nom_feature: valeur}
            threshold: Seuil personnalisé (optionnel)
        
        Returns:
            Dict avec score, is_anomaly, etc.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        try:
            # Convertir le dictionnaire en array numpy
            if self.feature_names is None:
                raise ValueError("Les noms de features ne sont pas définis")
            
            # Créer un array avec les features dans le bon ordre
            # Pour une détection ponctuelle, on répète la séquence
            feature_array = np.array([[features.get(name, 0.0) for name in self.feature_names]])
            
            # Répéter pour créer une séquence
            sequence = np.repeat(feature_array, self.sequence_length, axis=0)
            sequence = sequence.reshape(1, self.sequence_length, -1)
            
            # Convertir en Tensor
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # Calculer l'erreur de reconstruction
            self.model.eval()
            with torch.no_grad():
                reconstructed = self.model(sequence_tensor)
                error = torch.mean((reconstructed - sequence_tensor) ** 2).item()
            
            # Normaliser le score
            # Utiliser le seuil pour normaliser
            if self.threshold is not None:
                score = min(error / (self.threshold + 1e-10), 1.0)
            else:
                score = min(error, 1.0)
            
            # Utiliser le seuil fourni ou celui du modèle
            if threshold is None:
                threshold = 0.5  # Seuil par défaut pour les scores normalisés
            
            is_anomaly = score >= threshold
            
            return {
                "score": float(score),
                "is_anomaly": bool(is_anomaly),
                "threshold": float(threshold),
                "reconstruction_error": float(error)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalie: {e}", exc_info=True)
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            "model_type": "lstm_autoencoder",
            "is_trained": self.is_trained,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "sequence_length": self.sequence_length,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "device": str(self.device)
        }

