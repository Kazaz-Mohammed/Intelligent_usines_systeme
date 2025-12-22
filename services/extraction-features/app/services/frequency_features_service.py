"""
Service de calcul de features fréquentielles
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq

from app.config import settings
from app.models.feature_data import PreprocessedDataReference, ExtractedFeature

logger = logging.getLogger(__name__)


class FrequencyFeaturesService:
    """Service pour calculer des caractéristiques fréquentielles"""
    
    def __init__(self):
        self.enable_frequency_features = settings.enable_frequency_features
        self.frequency_features_list = settings.frequency_features_list
    
    def calculate_frequency_features(
        self,
        data: List[PreprocessedDataReference],
        sampling_rate: Optional[float] = None,
        feature_names: Optional[List[str]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques fréquentielles pour une série de données
        
        Args:
            data: Liste de données prétraitées
            sampling_rate: Fréquence d'échantillonnage en Hz (optionnel)
            feature_names: Liste de noms de features à calculer (optionnel)
        
        Returns:
            Liste de features extraites
        """
        if not self.enable_frequency_features or not data:
            return []
        
        if len(data) < 2:
            logger.warning("Pas assez de données pour calculer des features fréquentielles")
            return []
        
        # Extraire les valeurs
        # PreprocessedDataReference a 'value' (singular)
        values = np.array([d.value for d in data])
        timestamps = [d.timestamp for d in data]
        asset_id = data[0].asset_id
        sensor_id = data[0].sensor_id
        
        # Calculer la fréquence d'échantillonnage si non fournie
        if sampling_rate is None:
            sampling_rate = self._calculate_sampling_rate(timestamps)
        
        # Utiliser la liste de features configurée ou celle fournie
        features_to_calculate = feature_names if feature_names else self.frequency_features_list
        
        # Calculer les features
        features = []
        base_timestamp = timestamps[-1]  # Utiliser le dernier timestamp
        
        # Calculer la FFT une seule fois
        fft_values, frequencies = self._calculate_fft(values, sampling_rate)
        
        for feature_name in features_to_calculate:
            try:
                feature_value = self._calculate_feature(
                    values,
                    fft_values,
                    frequencies,
                    sampling_rate,
                    feature_name
                )
                
                if feature_value is not None and not np.isnan(feature_value) and not np.isinf(feature_value):
                    features.append(ExtractedFeature(
                        timestamp=base_timestamp,
                        asset_id=asset_id,
                        sensor_id=sensor_id,
                        feature_name=feature_name,
                        feature_value=float(feature_value),
                        feature_type="frequency",
                        metadata={
                            "window_size": len(data),
                            "sampling_rate": sampling_rate,
                            "calculation_method": "frequency_features_service"
                        }
                    ))
            except Exception as e:
                logger.error(f"Erreur lors du calcul de la feature {feature_name}: {e}", exc_info=True)
                continue
        
        logger.debug(f"Calculé {len(features)} features fréquentielles pour {asset_id}/{sensor_id}")
        return features
    
    def calculate_frequency_features_batch(
        self,
        data_dict: Dict[str, List[PreprocessedDataReference]],
        sampling_rate: Optional[float] = None,
        feature_names: Optional[List[str]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques fréquentielles pour plusieurs séries de données
        
        Args:
            data_dict: Dictionnaire de données prétraitées par sensor_id
            sampling_rate: Fréquence d'échantillonnage en Hz (optionnel)
            feature_names: Liste de noms de features à calculer (optionnel)
        
        Returns:
            Liste de features extraites
        """
        all_features = []
        
        for sensor_id, data_list in data_dict.items():
            if not data_list:
                continue
            
            features = self.calculate_frequency_features(
                data_list,
                sampling_rate,
                feature_names
            )
            all_features.extend(features)
        
        logger.debug(f"Calculé {len(all_features)} features fréquentielles au total")
        return all_features
    
    def _calculate_sampling_rate(self, timestamps: List[datetime]) -> float:
        """
        Calcule la fréquence d'échantillonnage à partir des timestamps
        
        Args:
            timestamps: Liste de timestamps
        
        Returns:
            Fréquence d'échantillonnage en Hz
        """
        if len(timestamps) < 2:
            return 1.0  # Default to 1 Hz
        
        # Calculer les intervalles de temps
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if diff > 0:
                time_diffs.append(diff)
        
        if not time_diffs:
            return 1.0  # Default to 1 Hz
        
        # Utiliser la moyenne des intervalles
        avg_interval = np.mean(time_diffs)
        sampling_rate = 1.0 / avg_interval if avg_interval > 0 else 1.0
        
        return float(sampling_rate)
    
    def _calculate_fft(self, values: np.ndarray, sampling_rate: float) -> tuple:
        """
        Calcule la FFT du signal
        
        Args:
            values: Array de valeurs
            sampling_rate: Fréquence d'échantillonnage en Hz
        
        Returns:
            Tuple (fft_values, frequencies)
        """
        # Calculer la FFT
        fft_values = fft(values)
        
        # Calculer les fréquences
        frequencies = fftfreq(len(values), 1.0 / sampling_rate)
        
        # Ne garder que les fréquences positives
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        fft_values = fft_values[positive_freq_mask]
        
        # Calculer la magnitude
        magnitude = np.abs(fft_values)
        
        return magnitude, frequencies
    
    def _calculate_feature(
        self,
        values: np.ndarray,
        fft_values: np.ndarray,
        frequencies: np.ndarray,
        sampling_rate: float,
        feature_name: str
    ) -> Optional[float]:
        """
        Calcule une feature fréquentielle spécifique
        
        Args:
            values: Array de valeurs (domaine temporel)
            fft_values: Array de valeurs FFT (magnitude)
            frequencies: Array de fréquences
            sampling_rate: Fréquence d'échantillonnage en Hz
            feature_name: Nom de la feature à calculer
        
        Returns:
            Valeur de la feature ou None si erreur
        """
        try:
            if feature_name == "spectral_centroid":
                return self._calculate_spectral_centroid(fft_values, frequencies)
            elif feature_name == "spectral_rolloff":
                return self._calculate_spectral_rolloff(fft_values, frequencies)
            elif feature_name == "spectral_bandwidth":
                return self._calculate_spectral_bandwidth(fft_values, frequencies)
            elif feature_name == "zero_crossing_rate":
                return self._calculate_zero_crossing_rate(values)
            elif feature_name == "spectral_flatness":
                return self._calculate_spectral_flatness(fft_values)
            else:
                logger.warning(f"Feature fréquentielle inconnue: {feature_name}")
                return None
        except Exception as e:
            logger.error(f"Erreur lors du calcul de {feature_name}: {e}", exc_info=True)
            return None
    
    def _calculate_spectral_centroid(self, fft_values: np.ndarray, frequencies: np.ndarray) -> float:
        """
        Calcule le spectral centroid (fréquence moyenne pondérée par l'amplitude)
        
        Args:
            fft_values: Magnitude de la FFT
            frequencies: Fréquences
        
        Returns:
            Spectral centroid en Hz
        """
        if len(fft_values) == 0 or np.sum(fft_values) == 0:
            return 0.0
        
        # Calculer le centroid
        centroid = np.sum(frequencies * fft_values) / np.sum(fft_values)
        
        return float(centroid)
    
    def _calculate_spectral_rolloff(self, fft_values: np.ndarray, frequencies: np.ndarray, rolloff_percentile: float = 0.85) -> float:
        """
        Calcule le spectral rolloff (fréquence en dessous de laquelle 85% de l'énergie est contenue)
        
        Args:
            fft_values: Magnitude de la FFT
            frequencies: Fréquences
            rolloff_percentile: Percentile pour le rolloff (default: 0.85)
        
        Returns:
            Spectral rolloff en Hz
        """
        if len(fft_values) == 0:
            return 0.0
        
        # Calculer l'énergie cumulative
        energy = fft_values ** 2
        cumulative_energy = np.cumsum(energy)
        total_energy = cumulative_energy[-1]
        
        if total_energy == 0:
            return 0.0
        
        # Trouver la fréquence de rolloff
        threshold = total_energy * rolloff_percentile
        rolloff_idx = np.where(cumulative_energy >= threshold)[0]
        
        if len(rolloff_idx) == 0:
            return float(frequencies[-1])
        
        return float(frequencies[rolloff_idx[0]])
    
    def _calculate_spectral_bandwidth(self, fft_values: np.ndarray, frequencies: np.ndarray) -> float:
        """
        Calcule le spectral bandwidth (écart-type autour du centroid)
        
        Args:
            fft_values: Magnitude de la FFT
            frequencies: Fréquences
        
        Returns:
            Spectral bandwidth en Hz
        """
        if len(fft_values) == 0 or np.sum(fft_values) == 0:
            return 0.0
        
        # Calculer le centroid
        centroid = self._calculate_spectral_centroid(fft_values, frequencies)
        
        # Calculer le bandwidth
        bandwidth = np.sqrt(np.sum(((frequencies - centroid) ** 2) * fft_values) / np.sum(fft_values))
        
        return float(bandwidth)
    
    def _calculate_zero_crossing_rate(self, values: np.ndarray) -> float:
        """
        Calcule le zero crossing rate (taux de passage par zéro)
        
        Args:
            values: Array de valeurs
        
        Returns:
            Zero crossing rate (nombre de passages par zéro par seconde)
        """
        if len(values) < 2:
            return 0.0
        
        # Calculer les passages par zéro
        zero_crossings = np.where(np.diff(np.signbit(values)))[0]
        zcr = len(zero_crossings) / len(values)
        
        return float(zcr)
    
    def _calculate_spectral_flatness(self, fft_values: np.ndarray) -> float:
        """
        Calcule le spectral flatness (platitude spectrale)
        
        Args:
            fft_values: Magnitude de la FFT
        
        Returns:
            Spectral flatness (0 = tonal, 1 = bruit)
        """
        if len(fft_values) == 0:
            return 0.0
        
        # Éviter les valeurs nulles
        fft_values = fft_values + 1e-10
        
        # Calculer la moyenne géométrique
        geometric_mean = np.exp(np.mean(np.log(fft_values)))
        
        # Calculer la moyenne arithmétique
        arithmetic_mean = np.mean(fft_values)
        
        if arithmetic_mean == 0:
            return 0.0
        
        # Calculer le flatness
        flatness = geometric_mean / arithmetic_mean
        
        return float(flatness)
    
    def calculate_band_energy(
        self,
        data: List[PreprocessedDataReference],
        sampling_rate: Optional[float] = None,
        bands: Optional[List[tuple]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule l'énergie dans différentes bandes de fréquences
        
        Args:
            data: Liste de données prétraitées
            sampling_rate: Fréquence d'échantillonnage en Hz (optionnel)
            bands: Liste de tuples (low_freq, high_freq) pour les bandes (optionnel)
        
        Returns:
            Liste de features extraites
        """
        if not self.enable_frequency_features or not data:
            return []
        
        if len(data) < 2:
            logger.warning("Pas assez de données pour calculer l'énergie de bande")
            return []
        
        # Extraire les valeurs
        # PreprocessedDataReference a 'value' (singular)
        values = np.array([d.value for d in data])
        timestamps = [d.timestamp for d in data]
        asset_id = data[0].asset_id
        sensor_id = data[0].sensor_id
        
        # Calculer la fréquence d'échantillonnage si non fournie
        if sampling_rate is None:
            sampling_rate = self._calculate_sampling_rate(timestamps)
        
        # Utiliser des bandes par défaut si non fournies
        if bands is None:
            nyquist = sampling_rate / 2.0
            bands = [
                (0, nyquist * 0.25),  # Basses fréquences
                (nyquist * 0.25, nyquist * 0.5),  # Moyennes fréquences
                (nyquist * 0.5, nyquist * 0.75),  # Hautes fréquences
                (nyquist * 0.75, nyquist)  # Très hautes fréquences
            ]
        
        # Calculer la FFT
        fft_values, frequencies = self._calculate_fft(values, sampling_rate)
        
        # Calculer l'énergie pour chaque bande
        features = []
        base_timestamp = timestamps[-1]
        
        for i, (low_freq, high_freq) in enumerate(bands):
            try:
                # Trouver les fréquences dans la bande
                band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                band_energy = np.sum(fft_values[band_mask] ** 2)
                
                # Normaliser par la largeur de la bande
                band_width = high_freq - low_freq
                normalized_energy = band_energy / band_width if band_width > 0 else band_energy
                
                features.append(ExtractedFeature(
                    timestamp=base_timestamp,
                    asset_id=asset_id,
                    sensor_id=sensor_id,
                    feature_name=f"band_energy_{i}_{int(low_freq)}_{int(high_freq)}",
                    feature_value=float(normalized_energy),
                    feature_type="frequency",
                    metadata={
                        "window_size": len(data),
                        "sampling_rate": sampling_rate,
                        "band_low": low_freq,
                        "band_high": high_freq,
                        "calculation_method": "band_energy"
                    }
                ))
            except Exception as e:
                logger.error(f"Erreur lors du calcul de l'énergie de bande {i}: {e}", exc_info=True)
                continue
        
        logger.debug(f"Calculé {len(features)} features d'énergie de bande pour {asset_id}/{sensor_id}")
        return features

