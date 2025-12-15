"""
Service d'analyse fréquentielle
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

from app.models.sensor_data import PreprocessedData
from app.config import settings

logger = logging.getLogger(__name__)


class FrequencyAnalysisService:
    """Service pour l'analyse fréquentielle des signaux"""
    
    def __init__(self):
        self.enable_frequency_analysis = settings.enable_frequency_analysis
    
    def analyze_frequency(
        self,
        data: List[PreprocessedData],
        method: str = "fft",
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyse fréquentielle d'un signal
        
        Args:
            data: Liste de données prétraitées
            method: Méthode ("fft" ou "stft")
            window_size: Taille de fenêtre pour STFT
            
        Returns:
            Dictionnaire avec résultats de l'analyse
        """
        if not self.enable_frequency_analysis or not data:
            return {}
        
        if len(data) < 10:
            logger.warning("Pas assez de données pour analyse fréquentielle")
            return {}
        
        # Extraire les valeurs et timestamps
        values = np.array([d.value for d in data])
        timestamps = [d.timestamp for d in data]
        
        # Calculer la fréquence d'échantillonnage
        if len(timestamps) > 1:
            time_diffs = np.diff([ts.timestamp() for ts in timestamps])
            sampling_rate = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 1.0
        else:
            sampling_rate = 1.0
        
        if method == "fft":
            return self._fft_analysis(values, sampling_rate)
        elif method == "stft":
            return self._stft_analysis(values, sampling_rate, window_size)
        else:
            logger.warning(f"Méthode d'analyse inconnue: {method}")
            return {}
    
    def _fft_analysis(
        self,
        values: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, Any]:
        """
        Analyse FFT
        
        Args:
            values: Valeurs du signal
            sampling_rate: Fréquence d'échantillonnage (Hz)
            
        Returns:
            Résultats de l'analyse FFT
        """
        # Appliquer FFT
        fft_values = fft(values)
        frequencies = fftfreq(len(values), 1.0/sampling_rate)
        
        # Prendre seulement les fréquences positives
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = np.abs(fft_values[positive_freq_idx])
        
        # Trouver les fréquences dominantes
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Ignorer DC (0 Hz)
        dominant_frequency = frequencies[dominant_freq_idx]
        dominant_magnitude = magnitude[dominant_freq_idx]
        
        # Calculer la puissance totale
        power = np.sum(magnitude ** 2)
        
        # Trouver les bandes de fréquences importantes
        # Top 5 fréquences
        top_freq_indices = np.argsort(magnitude[1:])[-5:][::-1] + 1
        top_frequencies = frequencies[top_freq_indices].tolist()
        top_magnitudes = magnitude[top_freq_indices].tolist()
        
        return {
            "method": "fft",
            "sampling_rate": float(sampling_rate),
            "dominant_frequency": float(dominant_frequency),
            "dominant_magnitude": float(dominant_magnitude),
            "total_power": float(power),
            "top_frequencies": top_frequencies,
            "top_magnitudes": top_magnitudes,
            "frequency_bands": self._calculate_frequency_bands(frequencies, magnitude)
        }
    
    def _stft_analysis(
        self,
        values: np.ndarray,
        sampling_rate: float,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyse STFT (Short-Time Fourier Transform)
        
        Args:
            values: Valeurs du signal
            sampling_rate: Fréquence d'échantillonnage (Hz)
            window_size: Taille de la fenêtre (None = auto)
            
        Returns:
            Résultats de l'analyse STFT
        """
        if window_size is None:
            window_size = min(256, len(values) // 4)
        
        # Appliquer STFT
        frequencies, times, Zxx = signal.stft(
            values,
            sampling_rate,
            nperseg=window_size,
            noverlap=window_size // 2
        )
        
        # Calculer la magnitude
        magnitude = np.abs(Zxx)
        
        # Trouver la fréquence dominante moyenne
        mean_magnitude = np.mean(magnitude, axis=1)
        dominant_freq_idx = np.argmax(mean_magnitude)
        dominant_frequency = frequencies[dominant_freq_idx]
        
        # Calculer l'énergie par bande de fréquences
        frequency_bands = self._calculate_frequency_bands_stft(frequencies, magnitude)
        
        return {
            "method": "stft",
            "sampling_rate": float(sampling_rate),
            "window_size": window_size,
            "dominant_frequency": float(dominant_frequency),
            "frequency_bands": frequency_bands,
            "time_frequency_energy": float(np.sum(magnitude ** 2))
        }
    
    def _calculate_frequency_bands(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcule l'énergie par bande de fréquences
        
        Args:
            frequencies: Fréquences
            magnitude: Magnitudes
            
        Returns:
            Dict {bande: énergie}
        """
        # Définir les bandes de fréquences (Hz)
        bands = {
            "low": (0, 10),
            "medium": (10, 50),
            "high": (50, None)
        }
        
        band_energy = {}
        for band_name, (low, high) in bands.items():
            if high is None:
                mask = frequencies >= low
            else:
                mask = (frequencies >= low) & (frequencies < high)
            
            if mask.any():
                energy = np.sum(magnitude[mask] ** 2)
                band_energy[band_name] = float(energy)
            else:
                band_energy[band_name] = 0.0
        
        return band_energy
    
    def _calculate_frequency_bands_stft(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray
    ) -> Dict[str, float]:
        """Calcule l'énergie par bande pour STFT"""
        # Même logique que _calculate_frequency_bands mais pour STFT
        bands = {
            "low": (0, 10),
            "medium": (10, 50),
            "high": (50, None)
        }
        
        band_energy = {}
        for band_name, (low, high) in bands.items():
            if high is None:
                mask = frequencies >= low
            else:
                mask = (frequencies >= low) & (frequencies < high)
            
            if mask.any():
                # Somme sur toutes les fenêtres temporelles
                energy = np.sum(magnitude[mask, :] ** 2)
                band_energy[band_name] = float(energy)
            else:
                band_energy[band_name] = 0.0
        
        return band_energy
    
    def add_frequency_analysis_to_data(
        self,
        data: List[PreprocessedData],
        analysis_result: Dict[str, Any]
    ) -> List[PreprocessedData]:
        """
        Ajoute les résultats d'analyse fréquentielle aux données
        
        Args:
            data: Liste de données prétraitées
            analysis_result: Résultats de l'analyse
            
        Returns:
            Liste de données avec analyse fréquentielle ajoutée
        """
        if not analysis_result:
            return data
        
        result = []
        for item in data:
            new_item = item.model_copy()
            new_item.frequency_analysis = analysis_result
            result.append(new_item)
        
        return result

