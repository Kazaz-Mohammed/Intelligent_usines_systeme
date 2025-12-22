"""
Service de calcul de features temporelles
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

from app.config import settings
from app.models.feature_data import PreprocessedDataReference, ExtractedFeature

logger = logging.getLogger(__name__)


class TemporalFeaturesService:
    """Service pour calculer des caractéristiques temporelles"""
    
    def __init__(self):
        self.enable_temporal_features = settings.enable_temporal_features
        self.temporal_features_list = settings.temporal_features_list
    
    def calculate_temporal_features(
        self,
        data: List[PreprocessedDataReference],
        feature_names: Optional[List[str]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques temporelles pour une série de données
        
        Args:
            data: Liste de données prétraitées
            feature_names: Liste de noms de features à calculer (optionnel)
        
        Returns:
            Liste de features extraites
        """
        if not self.enable_temporal_features or not data:
            return []
        
        if len(data) < 2:
            logger.warning("Pas assez de données pour calculer des features temporelles")
            return []
        
        # Extraire les valeurs
        # PreprocessedDataReference a 'value' (singular) qui peut être converti en liste via la propriété 'values'
        # PreprocessedDataReference a 'value' (singular)
        values = np.array([d.value for d in data])
        timestamps = [d.timestamp for d in data]
        asset_id = data[0].asset_id
        sensor_id = data[0].sensor_id
        
        # Utiliser la liste de features configurée ou celle fournie
        features_to_calculate = feature_names if feature_names else self.temporal_features_list
        
        # Calculer les features
        features = []
        base_timestamp = timestamps[-1]  # Utiliser le dernier timestamp
        
        for feature_name in features_to_calculate:
            try:
                feature_value = self._calculate_feature(values, feature_name)
                
                if feature_value is not None and not np.isnan(feature_value) and not np.isinf(feature_value):
                    try:
                        feature = ExtractedFeature(
                            timestamp=base_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=feature_name,
                            feature_value=float(feature_value),
                            feature_type="temporal",
                            metadata={
                                "window_size": len(data),
                                "calculation_method": "temporal_features_service"
                            }
                        )
                        features.append(feature)
                    except Exception as create_error:
                        logger.error(
                            f"Erreur lors de la création de ExtractedFeature pour {feature_name}: {create_error}. "
                            f"Values: timestamp={base_timestamp}, asset_id={asset_id}, sensor_id={sensor_id}, "
                            f"feature_name={feature_name}, feature_value={feature_value}",
                            exc_info=True
                        )
                        continue
            except Exception as e:
                logger.error(f"Erreur lors du calcul de la feature {feature_name}: {e}", exc_info=True)
                continue
        
        logger.debug(f"Calculé {len(features)} features temporelles pour {asset_id}/{sensor_id}")
        return features
    
    def calculate_temporal_features_batch(
        self,
        data_dict: Dict[str, List[PreprocessedDataReference]],
        feature_names: Optional[List[str]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques temporelles pour plusieurs séries de données
        
        Args:
            data_dict: Dictionnaire de données prétraitées par sensor_id
            feature_names: Liste de noms de features à calculer (optionnel)
        
        Returns:
            Liste de features extraites
        """
        all_features = []
        
        for sensor_id, data_list in data_dict.items():
            if not data_list:
                continue
            
            features = self.calculate_temporal_features(data_list, feature_names)
            all_features.extend(features)
        
        logger.debug(f"Calculé {len(all_features)} features temporelles au total")
        return all_features
    
    def _calculate_feature(self, values: np.ndarray, feature_name: str) -> Optional[float]:
        """
        Calcule une feature temporelle spécifique
        
        Args:
            values: Array de valeurs
            feature_name: Nom de la feature à calculer
        
        Returns:
            Valeur de la feature ou None si erreur
        """
        try:
            if feature_name == "mean":
                return np.mean(values)
            elif feature_name == "std":
                return np.std(values)
            elif feature_name == "var":
                return np.var(values)
            elif feature_name == "min":
                return np.min(values)
            elif feature_name == "max":
                return np.max(values)
            elif feature_name == "median":
                return np.median(values)
            elif feature_name == "rms":
                return self._calculate_rms(values)
            elif feature_name == "kurtosis":
                return self._calculate_kurtosis(values)
            elif feature_name == "skewness":
                return self._calculate_skewness(values)
            elif feature_name == "crest_factor":
                return self._calculate_crest_factor(values)
            elif feature_name == "peak_to_peak":
                return self._calculate_peak_to_peak(values)
            elif feature_name == "form_factor":
                return self._calculate_form_factor(values)
            else:
                logger.warning(f"Feature temporelle inconnue: {feature_name}")
                return None
        except Exception as e:
            logger.error(f"Erreur lors du calcul de {feature_name}: {e}", exc_info=True)
            return None
    
    def _calculate_rms(self, values: np.ndarray) -> float:
        """Calcule la valeur RMS (Root Mean Square)"""
        return np.sqrt(np.mean(values ** 2))
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calcule le kurtosis"""
        if len(values) < 4:
            return 0.0
        return float(stats.kurtosis(values))
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calcule le skewness (asymétrie)"""
        if len(values) < 3:
            return 0.0
        return float(stats.skew(values))
    
    def _calculate_crest_factor(self, values: np.ndarray) -> float:
        """Calcule le crest factor (rapport pic/RMS)"""
        rms = self._calculate_rms(values)
        if rms == 0:
            return 0.0
        peak = np.max(np.abs(values))
        return float(peak / rms)
    
    def _calculate_peak_to_peak(self, values: np.ndarray) -> float:
        """Calcule la valeur peak-to-peak"""
        return float(np.max(values) - np.min(values))
    
    def _calculate_form_factor(self, values: np.ndarray) -> float:
        """Calcule le form factor (RMS/mean)"""
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        rms = self._calculate_rms(values)
        return float(rms / mean)
    
    def calculate_rolling_features(
        self,
        data: List[PreprocessedDataReference],
        window_size: int = 10,
        feature_names: Optional[List[str]] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des features temporelles avec fenêtres glissantes
        
        Args:
            data: Liste de données prétraitées
            window_size: Taille de la fenêtre glissante
            feature_names: Liste de noms de features à calculer (optionnel)
        
        Returns:
            Liste de features extraites
        """
        if not self.enable_temporal_features or not data:
            return []
        
        if len(data) < window_size:
            logger.warning(f"Pas assez de données pour fenêtre glissante (need {window_size}, got {len(data)})")
            return []
        
        # Extraire les valeurs
        # PreprocessedDataReference a 'value' (singular)
        values = np.array([d.value for d in data])
        timestamps = [d.timestamp for d in data]
        asset_id = data[0].asset_id
        sensor_id = data[0].sensor_id
        
        # Utiliser la liste de features configurée ou celle fournie
        features_to_calculate = feature_names if feature_names else self.temporal_features_list
        
        # Créer un DataFrame pour les calculs de fenêtres glissantes
        df = pd.DataFrame({
            'value': values,
            'timestamp': timestamps
        })
        
        # Calculer les features avec fenêtres glissantes
        features = []
        
        for i in range(window_size - 1, len(df)):
            window_values = df['value'].iloc[i - window_size + 1:i + 1].values
            window_timestamp = df['timestamp'].iloc[i]
            
            for feature_name in features_to_calculate:
                try:
                    feature_value = self._calculate_feature(window_values, feature_name)
                    
                    if feature_value is not None and not np.isnan(feature_value) and not np.isinf(feature_value):
                        features.append(ExtractedFeature(
                            timestamp=window_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=f"{feature_name}_rolling_{window_size}",
                            feature_value=float(feature_value),
                            feature_type="temporal",
                            metadata={
                                "window_size": window_size,
                                "calculation_method": "rolling_features",
                                "base_feature": feature_name
                            }
                        ))
                except Exception as e:
                    logger.error(f"Erreur lors du calcul de la feature {feature_name}: {e}", exc_info=True)
                    continue
        
        logger.debug(f"Calculé {len(features)} features temporelles avec fenêtres glissantes pour {asset_id}/{sensor_id}")
        return features

