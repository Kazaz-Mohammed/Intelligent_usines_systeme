"""
Service de calcul de features temporelles avec tsfresh
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("tsfresh n'est pas disponible, utilisation des features manuelles")

from app.config import settings
from app.models.feature_data import PreprocessedDataReference, ExtractedFeature

logger = logging.getLogger(__name__)


class TSFreshFeaturesService:
    """Service pour calculer des caractéristiques temporelles avec tsfresh"""
    
    def __init__(self):
        self.enable_tsfresh = TSFRESH_AVAILABLE and settings.enable_temporal_features
        if not TSFRESH_AVAILABLE:
            logger.warning("tsfresh n'est pas disponible, les features tsfresh ne seront pas calculées")
    
    def calculate_tsfresh_features(
        self,
        data: List[PreprocessedDataReference],
        kind_to_fc_parameters: Optional[Dict[str, Any]] = None,
        impute_function: Optional[callable] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques temporelles avec tsfresh
        
        Args:
            data: Liste de données prétraitées
            kind_to_fc_parameters: Dictionnaire de paramètres pour tsfresh (optionnel)
            impute_function: Fonction d'imputation pour tsfresh (optionnel)
        
        Returns:
            Liste de features extraites
        """
        if not self.enable_tsfresh or not data:
            return []
        
        if len(data) < 2:
            logger.warning("Pas assez de données pour calculer des features tsfresh")
            return []
        
        try:
            # Créer un DataFrame pour tsfresh
            df = pd.DataFrame({
                'id': [data[0].asset_id] * len(data),
                'time': range(len(data)),
                'value': [d.value for d in data]  # PreprocessedDataReference a 'value' (singular)
            })
            
            # Utiliser les paramètres par défaut si non spécifiés
            if kind_to_fc_parameters is None:
                # Utiliser les paramètres minimaux pour des performances rapides
                kind_to_fc_parameters = MinimalFCParameters()
            
            # Extraire les features
            extracted_features_df = extract_features(
                df,
                column_id='id',
                column_sort='time',
                column_value='value',
                kind_to_fc_parameters=kind_to_fc_parameters,
                impute_function=impute_function if impute_function else impute
            )
            
            # Convertir en liste de ExtractedFeature
            features = []
            base_timestamp = data[-1].timestamp  # Utiliser le dernier timestamp
            asset_id = data[0].asset_id
            sensor_id = data[0].sensor_id
            
            # Check if dataframe is empty
            if extracted_features_df.empty or len(extracted_features_df) == 0:
                logger.warning(f"tsfresh n'a pas produit de features pour {sensor_id}")
                return features
            
            for feature_name, feature_value in extracted_features_df.iloc[0].items():
                try:
                    if pd.notna(feature_value) and not np.isinf(feature_value):
                        features.append(ExtractedFeature(
                            timestamp=base_timestamp,
                            asset_id=asset_id,
                            sensor_id=sensor_id,
                            feature_name=f"tsfresh_{feature_name}",
                            feature_value=float(feature_value),
                            feature_type="temporal",
                            metadata={
                                "window_size": len(data),
                                "calculation_method": "tsfresh",
                                "base_feature": feature_name
                            }
                        ))
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion de la feature {feature_name}: {e}", exc_info=True)
                    continue
            
            logger.debug(f"Calculé {len(features)} features tsfresh pour {asset_id}/{sensor_id}")
            return features
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des features tsfresh: {e}", exc_info=True)
            return []
    
    def calculate_tsfresh_features_batch(
        self,
        data_dict: Dict[str, List[PreprocessedDataReference]],
        kind_to_fc_parameters: Optional[Dict[str, Any]] = None,
        impute_function: Optional[callable] = None
    ) -> List[ExtractedFeature]:
        """
        Calcule des caractéristiques temporelles avec tsfresh pour plusieurs séries
        
        Args:
            data_dict: Dictionnaire de données prétraitées par sensor_id
            kind_to_fc_parameters: Dictionnaire de paramètres pour tsfresh (optionnel)
            impute_function: Fonction d'imputation pour tsfresh (optionnel)
        
        Returns:
            Liste de features extraites
        """
        all_features = []
        
        for sensor_id, data_list in data_dict.items():
            if not data_list:
                continue
            
            features = self.calculate_tsfresh_features(
                data_list,
                kind_to_fc_parameters,
                impute_function
            )
            all_features.extend(features)
        
        logger.debug(f"Calculé {len(all_features)} features tsfresh au total")
        return all_features
    
    def is_available(self) -> bool:
        """Vérifie si tsfresh est disponible"""
        return self.enable_tsfresh

