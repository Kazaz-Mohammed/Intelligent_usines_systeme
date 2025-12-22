"""
Service de standardisation par type d'actif
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

from app.config import settings
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector

logger = logging.getLogger(__name__)


class StandardizationService:
    """Service pour standardiser les features par type d'actif"""
    
    def __init__(self):
        self.enable_standardization = settings.enable_standardization
        self.standardization_method = settings.standardization_method
        
        # Templates de standardisation par type d'actif
        self.asset_type_templates = {
            "pump": {
                "method": "z-score",
                "features": {
                    "rms": {"mean": 25.0, "std": 5.0},
                    "kurtosis": {"mean": 3.0, "std": 1.0},
                    "crest_factor": {"mean": 4.0, "std": 1.0},
                    "spectral_centroid": {"mean": 150.0, "std": 30.0},
                }
            },
            "motor": {
                "method": "z-score",
                "features": {
                    "rms": {"mean": 30.0, "std": 6.0},
                    "kurtosis": {"mean": 2.5, "std": 0.8},
                    "crest_factor": {"mean": 3.5, "std": 0.9},
                    "spectral_centroid": {"mean": 200.0, "std": 40.0},
                }
            },
            "conveyor": {
                "method": "z-score",
                "features": {
                    "rms": {"mean": 20.0, "std": 4.0},
                    "kurtosis": {"mean": 2.0, "std": 0.7},
                    "crest_factor": {"mean": 3.0, "std": 0.8},
                    "spectral_centroid": {"mean": 100.0, "std": 20.0},
                }
            },
            "cnc": {
                "method": "z-score",
                "features": {
                    "rms": {"mean": 35.0, "std": 7.0},
                    "kurtosis": {"mean": 3.5, "std": 1.2},
                    "crest_factor": {"mean": 4.5, "std": 1.1},
                    "spectral_centroid": {"mean": 250.0, "std": 50.0},
                }
            },
            "default": {
                "method": "z-score",
                "features": {}
            }
        }
    
    def standardize_features(
        self,
        features: List[ExtractedFeature],
        asset_type: Optional[str] = None,
        method: Optional[str] = None
    ) -> List[ExtractedFeature]:
        """
        Standardise une liste de features selon le type d'actif
        
        Args:
            features: Liste de features à standardiser
            asset_type: Type d'actif (pump, motor, conveyor, cnc, etc.)
            method: Méthode de standardisation (z-score, min-max, robust)
        
        Returns:
            Liste de features standardisées
        """
        if not self.enable_standardization or not features:
            return features
        
        # Utiliser le type d'actif par défaut si non spécifié
        asset_type = asset_type or "default"
        
        # Utiliser la méthode configurée si non spécifiée
        method = method or self.standardization_method
        
        # Obtenir le template pour ce type d'actif
        template = self.asset_type_templates.get(asset_type, self.asset_type_templates["default"])
        
        # Standardiser chaque feature
        standardized_features = []
        
        for feature in features:
            try:
                # Obtenir les paramètres de standardisation pour cette feature
                feature_params = template.get("features", {}).get(feature.feature_name)
                
                if feature_params:
                    # Standardiser avec les paramètres du template
                    standardized_value = self._standardize_value(
                        feature.feature_value,
                        method,
                        feature_params
                    )
                else:
                    # Standardiser avec la méthode globale (sans template spécifique)
                    standardized_value = self._standardize_value_global(
                        feature.feature_value,
                        method
                    )
                
                # Créer la feature standardisée
                standardized_feature = ExtractedFeature(
                    timestamp=feature.timestamp,
                    asset_id=feature.asset_id,
                    sensor_id=feature.sensor_id,
                    feature_name=f"{feature.feature_name}_standardized",
                    feature_value=float(standardized_value),
                    feature_type=feature.feature_type,
                    metadata={
                        **(feature.metadata or {}),
                        "original_value": feature.feature_value,
                        "standardization_method": method,
                        "asset_type": asset_type,
                        "standardized": True
                    }
                )
                
                standardized_features.append(standardized_feature)
                
            except Exception as e:
                logger.error(f"Erreur lors de la standardisation de la feature {feature.feature_name}: {e}", exc_info=True)
                # Garder la feature originale en cas d'erreur
                standardized_features.append(feature)
        
        logger.debug(f"Standardisé {len(standardized_features)} features pour asset_type={asset_type}")
        return standardized_features
    
    def standardize_feature_vector(
        self,
        feature_vector: ExtractedFeaturesVector,
        asset_type: Optional[str] = None,
        method: Optional[str] = None
    ) -> ExtractedFeaturesVector:
        """
        Standardise un vecteur de features selon le type d'actif
        
        Args:
            feature_vector: Vecteur de features à standardiser
            asset_type: Type d'actif (pump, motor, conveyor, cnc, etc.)
            method: Méthode de standardisation (z-score, min-max, robust)
        
        Returns:
            Vecteur de features standardisé
        """
        if not self.enable_standardization:
            return feature_vector
        
        # Utiliser le type d'actif depuis les métadonnées ou par défaut
        asset_type = asset_type or feature_vector.feature_metadata.get("asset_type") if feature_vector.feature_metadata else None
        asset_type = asset_type or "default"
        
        # Utiliser la méthode configurée si non spécifiée
        method = method or self.standardization_method
        
        # Obtenir le template pour ce type d'actif
        template = self.asset_type_templates.get(asset_type, self.asset_type_templates["default"])
        
        # Standardiser les features
        standardized_features = {}
        original_features = feature_vector.features.copy()
        
        for feature_name, feature_value in feature_vector.features.items():
            try:
                # Obtenir les paramètres de standardisation pour cette feature
                feature_params = template.get("features", {}).get(feature_name)
                
                if feature_params:
                    # Standardiser avec les paramètres du template
                    standardized_value = self._standardize_value(
                        feature_value,
                        method,
                        feature_params
                    )
                else:
                    # Standardiser avec la méthode globale
                    standardized_value = self._standardize_value_global(
                        feature_value,
                        method
                    )
                
                standardized_features[feature_name] = float(standardized_value)
                
            except Exception as e:
                logger.error(f"Erreur lors de la standardisation de la feature {feature_name}: {e}", exc_info=True)
                # Garder la valeur originale en cas d'erreur
                standardized_features[feature_name] = feature_value
        
        # Créer le vecteur standardisé
        standardized_vector = ExtractedFeaturesVector(
            feature_vector_id=feature_vector.feature_vector_id,
            timestamp=feature_vector.timestamp,
            asset_id=feature_vector.asset_id,
            start_time=feature_vector.start_time,
            end_time=feature_vector.end_time,
            features=standardized_features,
            feature_metadata={
                **(feature_vector.feature_metadata or {}),
                "original_features": original_features,
                "standardization_method": method,
                "asset_type": asset_type,
                "standardized": True
            }
        )
        
        logger.debug(f"Vecteur de features standardisé pour asset_type={asset_type}")
        return standardized_vector
    
    def _standardize_value(
        self,
        value: float,
        method: str,
        params: Dict[str, float]
    ) -> float:
        """
        Standardise une valeur avec des paramètres spécifiques
        
        Args:
            value: Valeur à standardiser
            method: Méthode de standardisation (z-score, min-max, robust)
            params: Paramètres de standardisation (mean, std, min, max, median, iqr)
        
        Returns:
            Valeur standardisée
        """
        if method == "z-score":
            # Z-score: (x - mean) / std
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            if std == 0:
                return 0.0
            return (value - mean) / std
        
        elif method == "min-max":
            # Min-Max: (x - min) / (max - min)
            min_val = params.get("min", 0.0)
            max_val = params.get("max", 1.0)
            if max_val == min_val:
                return 0.0
            return (value - min_val) / (max_val - min_val)
        
        elif method == "robust":
            # Robust: (x - median) / IQR
            median = params.get("median", 0.0)
            iqr = params.get("iqr", 1.0)
            if iqr == 0:
                return 0.0
            return (value - median) / iqr
        
        else:
            logger.warning(f"Méthode de standardisation inconnue: {method}, retour de la valeur originale")
            return value
    
    def _standardize_value_global(
        self,
        value: float,
        method: str
    ) -> float:
        """
        Standardise une valeur avec la méthode globale (sans template spécifique)
        
        Args:
            value: Valeur à standardiser
            method: Méthode de standardisation (z-score, min-max, robust)
        
        Returns:
            Valeur standardisée (peut être la valeur originale si pas de template)
        """
        # Pour une standardisation globale sans template, on peut utiliser des statistiques globales
        # Pour l'instant, on retourne la valeur originale si pas de template
        logger.debug(f"Standardisation globale avec méthode {method} (pas de template spécifique)")
        return value
    
    def update_asset_type_template(
        self,
        asset_type: str,
        method: str,
        features: Dict[str, Dict[str, float]]
    ):
        """
        Met à jour ou crée un template de standardisation pour un type d'actif
        
        Args:
            asset_type: Type d'actif
            method: Méthode de standardisation
            features: Dictionnaire de features avec leurs paramètres
        """
        self.asset_type_templates[asset_type] = {
            "method": method,
            "features": features
        }
        
        logger.info(f"Template de standardisation mis à jour pour asset_type={asset_type}")
    
    def get_asset_type_template(self, asset_type: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le template de standardisation pour un type d'actif
        
        Args:
            asset_type: Type d'actif
        
        Returns:
            Template de standardisation ou None
        """
        return self.asset_type_templates.get(asset_type)
    
    def calculate_statistics_from_data(
        self,
        features_list: List[List[ExtractedFeature]],
        asset_type: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcule les statistiques (mean, std, etc.) à partir de données historiques
        
        Args:
            features_list: Liste de listes de features (historique)
            asset_type: Type d'actif
        
        Returns:
            Dictionnaire de features avec leurs statistiques
        """
        # Grouper les features par nom
        features_by_name: Dict[str, List[float]] = {}
        
        for features in features_list:
            for feature in features:
                if feature.feature_name not in features_by_name:
                    features_by_name[feature.feature_name] = []
                features_by_name[feature.feature_name].append(feature.feature_value)
        
        # Calculer les statistiques pour chaque feature
        statistics = {}
        
        for feature_name, values in features_by_name.items():
            if len(values) < 2:
                continue
            
            values_array = np.array(values)
            
            statistics[feature_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "median": float(np.median(values_array)),
                "iqr": float(np.percentile(values_array, 75) - np.percentile(values_array, 25))
            }
        
        # Mettre à jour le template
        self.update_asset_type_template(
            asset_type,
            self.standardization_method,
            statistics
        )
        
        logger.info(f"Statistiques calculées pour asset_type={asset_type}: {len(statistics)} features")
        return statistics

