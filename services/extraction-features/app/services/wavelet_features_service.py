"""
Service de calcul de features avec transformées ondelettes
"""
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyWavelets n'est pas disponible, les features ondelettes ne seront pas calculées")

from app.config import settings
from app.models.feature_data import PreprocessedDataReference, ExtractedFeature

logger = logging.getLogger(__name__)


class WaveletFeaturesService:
    """Service pour calculer des caractéristiques avec transformées ondelettes"""
    
    def __init__(self):
        self.enable_wavelet_features = PYWT_AVAILABLE and settings.enable_wavelet_features
        self.wavelet_family = 'db4'  # Daubechies 4
        self.max_level = 4  # Niveau maximum de décomposition
        
        if not PYWT_AVAILABLE:
            logger.warning("PyWavelets n'est pas disponible, les features ondelettes ne seront pas calculées")
    
    def calculate_wavelet_features(
        self,
        data: List[PreprocessedDataReference],
        wavelet: str = 'db4',
        max_level: Optional[int] = None,
        feature_names: Optional[List[str]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques avec transformées ondelettes
        
        Args:
            data: Liste de données prétraitées
            wavelet: Famille d'ondelettes (default: 'db4')
            max_level: Niveau maximum de décomposition (optionnel)
            feature_names: Liste de noms de features à calculer (optionnel)
        
        Returns:
            Liste de features extraites
        """
        if not self.enable_wavelet_features or not data:
            return []
        
        if len(data) < 2:
            logger.warning("Pas assez de données pour calculer des features ondelettes")
            return []
        
        # Extraire les valeurs
        # PreprocessedDataReference peut avoir 'value' (singular) ou 'values' (plural)
        values_list = []
        for d in data:
            if hasattr(d, 'value'):
                values_list.append(d.value)
            elif hasattr(d, 'values') and isinstance(d.values, list):
                values_list.extend(d.values)
            else:
                continue
        values = np.array(values_list)
        timestamps = [d.timestamp for d in data]
        asset_id = data[0].asset_id
        sensor_id = data[0].sensor_id
        
        # Utiliser les paramètres par défaut si non spécifiés
        if max_level is None:
            max_level = self.max_level
        
        # Calculer la décomposition en ondelettes
        try:
            coeffs = self._decompose_wavelet(values, wavelet, max_level)
        except Exception as e:
            logger.error(f"Erreur lors de la décomposition en ondelettes: {e}", exc_info=True)
            return []
        
        # Calculer les features
        features = []
        base_timestamp = timestamps[-1]  # Utiliser le dernier timestamp
        
        # Features par défaut si non spécifiées
        if feature_names is None:
            feature_names = ['energy', 'entropy', 'variance', 'mean', 'std']
        
        for feature_name in feature_names:
            try:
                if feature_name == 'energy':
                    # Énergie par niveau
                    energy_features = self._calculate_energy_by_level(coeffs)
                    for level, energy in energy_features.items():
                        features.append(ExtractedFeature(
                            timestamp=base_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=f"wavelet_energy_level_{level}",
                            feature_value=float(energy),
                            feature_type="wavelet",
                            metadata={
                                "window_size": len(data),
                                "wavelet": wavelet,
                                "level": level,
                                "calculation_method": "wavelet_energy"
                            }
                        ))
                
                elif feature_name == 'entropy':
                    # Entropie par niveau
                    entropy_features = self._calculate_entropy_by_level(coeffs)
                    for level, entropy in entropy_features.items():
                        features.append(ExtractedFeature(
                            timestamp=base_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=f"wavelet_entropy_level_{level}",
                            feature_value=float(entropy),
                            feature_type="wavelet",
                            metadata={
                                "window_size": len(data),
                                "wavelet": wavelet,
                                "level": level,
                                "calculation_method": "wavelet_entropy"
                            }
                        ))
                
                elif feature_name == 'variance':
                    # Variance par niveau
                    variance_features = self._calculate_variance_by_level(coeffs)
                    for level, variance in variance_features.items():
                        features.append(ExtractedFeature(
                            timestamp=base_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=f"wavelet_variance_level_{level}",
                            feature_value=float(variance),
                            feature_type="wavelet",
                            metadata={
                                "window_size": len(data),
                                "wavelet": wavelet,
                                "level": level,
                                "calculation_method": "wavelet_variance"
                            }
                        ))
                
                elif feature_name == 'mean':
                    # Moyenne par niveau
                    mean_features = self._calculate_mean_by_level(coeffs)
                    for level, mean_val in mean_features.items():
                        features.append(ExtractedFeature(
                            timestamp=base_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=f"wavelet_mean_level_{level}",
                            feature_value=float(mean_val),
                            feature_type="wavelet",
                            metadata={
                                "window_size": len(data),
                                "wavelet": wavelet,
                                "level": level,
                                "calculation_method": "wavelet_mean"
                            }
                        ))
                
                elif feature_name == 'std':
                    # Écart-type par niveau
                    std_features = self._calculate_std_by_level(coeffs)
                    for level, std_val in std_features.items():
                        features.append(ExtractedFeature(
                            timestamp=base_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=f"wavelet_std_level_{level}",
                            feature_value=float(std_val),
                            feature_type="wavelet",
                            metadata={
                                "window_size": len(data),
                                "wavelet": wavelet,
                                "level": level,
                                "calculation_method": "wavelet_std"
                            }
                        ))
                
                else:
                    logger.warning(f"Feature ondelette inconnue: {feature_name}")
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul de la feature {feature_name}: {e}", exc_info=True)
                continue
        
        logger.debug(f"Calculé {len(features)} features ondelettes pour {asset_id}/{sensor_id}")
        return features
    
    def calculate_wavelet_features_batch(
        self,
        data_dict: Dict[str, List[PreprocessedDataReference]],
        wavelet: str = 'db4',
        max_level: Optional[int] = None,
        feature_names: Optional[List[str]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques avec transformées ondelettes pour plusieurs séries
        
        Args:
            data_dict: Dictionnaire de données prétraitées par sensor_id
            wavelet: Famille d'ondelettes (default: 'db4')
            max_level: Niveau maximum de décomposition (optionnel)
            feature_names: Liste de noms de features à calculer (optionnel)
        
        Returns:
            Liste de features extraites
        """
        all_features = []
        
        for sensor_id, data_list in data_dict.items():
            if not data_list:
                continue
            
            features = self.calculate_wavelet_features(
                data_list,
                wavelet,
                max_level,
                feature_names
            )
            all_features.extend(features)
        
        logger.debug(f"Calculé {len(all_features)} features ondelettes au total")
        return all_features
    
    def _decompose_wavelet(
        self,
        values: np.ndarray,
        wavelet: str = 'db4',
        max_level: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Décompose un signal en ondelettes
        
        Args:
            values: Array de valeurs
            wavelet: Famille d'ondelettes
            max_level: Niveau maximum de décomposition
        
        Returns:
            Liste de coefficients (cA, cD1, cD2, ..., cDmax_level)
        """
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets n'est pas disponible")
        
        # Calculer le niveau maximum possible
        if max_level is None:
            max_level = pywt.dwt_max_level(len(values), wavelet)
            max_level = min(max_level, self.max_level)
        
        # Décomposition en ondelettes
        coeffs = pywt.wavedec(values, wavelet, level=max_level)
        
        return coeffs
    
    def _calculate_energy_by_level(self, coeffs: List[np.ndarray]) -> Dict[int, float]:
        """
        Calcule l'énergie par niveau de décomposition
        
        Args:
            coeffs: Coefficients de décomposition
        
        Returns:
            Dictionnaire {niveau: énergie}
        """
        energy_by_level = {}
        
        for level, coeff in enumerate(coeffs):
            energy = np.sum(coeff ** 2)
            energy_by_level[level] = float(energy)
        
        return energy_by_level
    
    def _calculate_entropy_by_level(self, coeffs: List[np.ndarray]) -> Dict[int, float]:
        """
        Calcule l'entropie par niveau de décomposition
        
        Args:
            coeffs: Coefficients de décomposition
        
        Returns:
            Dictionnaire {niveau: entropie}
        """
        entropy_by_level = {}
        
        for level, coeff in enumerate(coeffs):
            # Normaliser les coefficients pour l'entropie
            coeff_abs = np.abs(coeff)
            if np.sum(coeff_abs) == 0:
                entropy_by_level[level] = 0.0
                continue
            
            # Normaliser pour avoir une distribution de probabilité
            p = coeff_abs / np.sum(coeff_abs)
            
            # Éviter les valeurs nulles pour le log
            p = p[p > 0]
            
            # Calculer l'entropie de Shannon
            entropy = -np.sum(p * np.log2(p + 1e-10))
            entropy_by_level[level] = float(entropy)
        
        return entropy_by_level
    
    def _calculate_variance_by_level(self, coeffs: List[np.ndarray]) -> Dict[int, float]:
        """
        Calcule la variance par niveau de décomposition
        
        Args:
            coeffs: Coefficients de décomposition
        
        Returns:
            Dictionnaire {niveau: variance}
        """
        variance_by_level = {}
        
        for level, coeff in enumerate(coeffs):
            variance = np.var(coeff)
            variance_by_level[level] = float(variance)
        
        return variance_by_level
    
    def _calculate_mean_by_level(self, coeffs: List[np.ndarray]) -> Dict[int, float]:
        """
        Calcule la moyenne par niveau de décomposition
        
        Args:
            coeffs: Coefficients de décomposition
        
        Returns:
            Dictionnaire {niveau: moyenne}
        """
        mean_by_level = {}
        
        for level, coeff in enumerate(coeffs):
            mean_val = np.mean(coeff)
            mean_by_level[level] = float(mean_val)
        
        return mean_by_level
    
    def _calculate_std_by_level(self, coeffs: List[np.ndarray]) -> Dict[int, float]:
        """
        Calcule l'écart-type par niveau de décomposition
        
        Args:
            coeffs: Coefficients de décomposition
        
        Returns:
            Dictionnaire {niveau: écart-type}
        """
        std_by_level = {}
        
        for level, coeff in enumerate(coeffs):
            std_val = np.std(coeff)
            std_by_level[level] = float(std_val)
        
        return std_by_level
    
    def calculate_wavelet_energy_ratio(
        self,
        data: List[PreprocessedDataReference],
        wavelet: str = 'db4',
        max_level: Optional[int] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule le ratio d'énergie entre différents niveaux de décomposition
        
        Args:
            data: Liste de données prétraitées
            wavelet: Famille d'ondelettes
            max_level: Niveau maximum de décomposition
        
        Returns:
            Liste de features extraites
        """
        if not self.enable_wavelet_features or not data:
            return []
        
        if len(data) < 2:
            logger.warning("Pas assez de données pour calculer les ratios d'énergie")
            return []
        
        # Extraire les valeurs
        # PreprocessedDataReference peut avoir 'value' (singular) ou 'values' (plural)
        values_list = []
        for d in data:
            if hasattr(d, 'value'):
                values_list.append(d.value)
            elif hasattr(d, 'values') and isinstance(d.values, list):
                values_list.extend(d.values)
            else:
                continue
        values = np.array(values_list)
        timestamps = [d.timestamp for d in data]
        asset_id = data[0].asset_id
        sensor_id = data[0].sensor_id
        
        # Utiliser les paramètres par défaut si non spécifiés
        if max_level is None:
            max_level = self.max_level
        
        try:
            # Décomposer en ondelettes
            coeffs = self._decompose_wavelet(values, wavelet, max_level)
            
            # Calculer l'énergie par niveau
            energy_by_level = self._calculate_energy_by_level(coeffs)
            
            # Calculer l'énergie totale
            total_energy = sum(energy_by_level.values())
            
            if total_energy == 0:
                return []
            
            # Calculer les ratios d'énergie
            features = []
            base_timestamp = timestamps[-1]
            
            for level, energy in energy_by_level.items():
                ratio = energy / total_energy
                features.append(ExtractedFeature(
                    timestamp=base_timestamp,
                    asset_id=asset_id,
                    sensor_id=sensor_id,
                    feature_name=f"wavelet_energy_ratio_level_{level}",
                    feature_value=float(ratio),
                    feature_type="wavelet",
                    metadata={
                        "window_size": len(data),
                        "wavelet": wavelet,
                        "level": level,
                        "calculation_method": "wavelet_energy_ratio"
                    }
                ))
            
            logger.debug(f"Calculé {len(features)} ratios d'énergie ondelettes pour {asset_id}/{sensor_id}")
            return features
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des ratios d'énergie: {e}", exc_info=True)
            return []
    
    def get_available_wavelets(self) -> List[str]:
        """
        Retourne la liste des ondelettes disponibles
        
        Returns:
            Liste des ondelettes disponibles
        """
        if not PYWT_AVAILABLE:
            return []
        
        return pywt.wavelist()
    
    def is_available(self) -> bool:
        """Vérifie si PyWavelets est disponible"""
        return self.enable_wavelet_features

