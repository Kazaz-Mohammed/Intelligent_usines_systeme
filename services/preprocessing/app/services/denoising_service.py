"""
Service de débruitage des données
"""
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import signal

from app.models.sensor_data import PreprocessedData
from app.config import settings

logger = logging.getLogger(__name__)


class DenoisingService:
    """Service pour débruitage des signaux capteurs"""
    
    def __init__(self):
        self.enable_denoising = settings.enable_denoising
    
    def denoise_single_sensor(
        self,
        data: List[PreprocessedData],
        method: str = "butterworth",
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        order: int = 4
    ) -> List[PreprocessedData]:
        """
        Débruite les données d'un capteur
        
        Args:
            data: Liste de données prétraitées
            method: Méthode de débruitage ("butterworth", "moving_average", "savgol")
            lowcut: Fréquence de coupure basse (Hz)
            highcut: Fréquence de coupure haute (Hz)
            order: Ordre du filtre (pour butterworth)
            
        Returns:
            Liste de données débruitées
        """
        if not self.enable_denoising or not data:
            return data
        
        if len(data) < 3:
            return data
        
        # Extraire les valeurs
        values = np.array([d.value for d in data])
        timestamps = [d.timestamp for d in data]
        
        # Calculer la fréquence d'échantillonnage
        if len(timestamps) > 1:
            time_diffs = np.diff([ts.timestamp() for ts in timestamps])
            sampling_rate = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 1.0
        else:
            sampling_rate = 1.0
        
        # Appliquer le débruitage
        if method == "butterworth":
            denoised_values = self._butterworth_filter(
                values, sampling_rate, lowcut, highcut, order
            )
        elif method == "moving_average":
            denoised_values = self._moving_average_filter(values, window_size=5)
        elif method == "savgol":
            denoised_values = self._savgol_filter(values, window_length=5, polyorder=2)
        else:
            logger.warning(f"Méthode de débruitage inconnue: {method}, pas de débruitage")
            return data
        
        # Créer les nouvelles données
        result = []
        for i, original_data in enumerate(data):
            metadata = original_data.preprocessing_metadata.copy()
            metadata['denoised'] = True
            metadata['denoising_method'] = method
            
            result.append(PreprocessedData(
                timestamp=original_data.timestamp,
                asset_id=original_data.asset_id,
                sensor_id=original_data.sensor_id,
                value=float(denoised_values[i]),
                unit=original_data.unit,
                quality=original_data.quality,
                source_type=original_data.source_type,
                preprocessing_metadata=metadata
            ))
        
        logger.debug(f"Débruitage appliqué: {method}, {len(data)} points")
        return result
    
    def _butterworth_filter(
        self,
        values: np.ndarray,
        sampling_rate: float,
        lowcut: Optional[float],
        highcut: Optional[float],
        order: int = 4
    ) -> np.ndarray:
        """
        Applique un filtre Butterworth passe-bande
        
        Args:
            values: Valeurs à filtrer
            sampling_rate: Fréquence d'échantillonnage (Hz)
            lowcut: Fréquence de coupure basse (Hz)
            highcut: Fréquence de coupure haute (Hz)
            order: Ordre du filtre
            
        Returns:
            Valeurs filtrées
        """
        nyquist = sampling_rate / 2.0
        
        if lowcut is None and highcut is None:
            # Pas de filtre
            return values
        elif lowcut is None:
            # Filtre passe-bas
            b, a = signal.butter(order, highcut/nyquist, btype='low')
        elif highcut is None:
            # Filtre passe-haut
            b, a = signal.butter(order, lowcut/nyquist, btype='high')
        else:
            # Filtre passe-bande
            b, a = signal.butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
        
        filtered = signal.filtfilt(b, a, values)
        return filtered
    
    def _moving_average_filter(
        self,
        values: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Applique un filtre moyenne mobile
        
        Args:
            values: Valeurs à filtrer
            window_size: Taille de la fenêtre
            
        Returns:
            Valeurs filtrées
        """
        if len(values) < window_size:
            return values
        
        # Utiliser convolution pour moyenne mobile
        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(values, kernel, mode='same')
        
        return filtered
    
    def _savgol_filter(
        self,
        values: np.ndarray,
        window_length: int = 5,
        polyorder: int = 2
    ) -> np.ndarray:
        """
        Applique un filtre Savitzky-Golay
        
        Args:
            values: Valeurs à filtrer
            window_length: Longueur de la fenêtre (doit être impair)
            polyorder: Ordre du polynôme
            
        Returns:
            Valeurs filtrées
        """
        if len(values) < window_length:
            return values
        
        # S'assurer que window_length est impair
        if window_length % 2 == 0:
            window_length += 1
        
        # S'assurer que polyorder < window_length
        if polyorder >= window_length:
            polyorder = window_length - 1
        
        filtered = signal.savgol_filter(values, window_length, polyorder)
        return filtered
    
    def denoise_dataframe(
        self,
        df: pd.DataFrame,
        value_column: str = "value",
        method: str = "butterworth"
    ) -> pd.DataFrame:
        """
        Débruite un DataFrame
        
        Args:
            df: DataFrame avec colonne de valeurs
            value_column: Nom de la colonne de valeurs
            method: Méthode de débruitage
            
        Returns:
            DataFrame débruité
        """
        if not self.enable_denoising or value_column not in df.columns:
            return df
        
        df = df.copy()
        values = df[value_column].values
        
        if method == "moving_average":
            denoised = self._moving_average_filter(values)
        elif method == "savgol":
            denoised = self._savgol_filter(values)
        else:
            # Par défaut, pas de débruitage pour DataFrame
            return df
        
        df[value_column] = denoised
        return df

