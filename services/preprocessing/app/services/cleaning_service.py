"""
Service de nettoyage des données
"""
import logging
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats

from app.models.sensor_data import SensorData, PreprocessedData
from app.config import settings

logger = logging.getLogger(__name__)


class CleaningService:
    """Service pour nettoyer les données capteurs"""
    
    def __init__(self):
        self.outlier_threshold = settings.outlier_threshold
        
    def clean_single_value(
        self, 
        sensor_data: SensorData,
        historical_values: Optional[List[float]] = None
    ) -> PreprocessedData:
        """
        Nettoie une valeur unique
        
        Args:
            sensor_data: Donnée capteur à nettoyer
            historical_values: Valeurs historiques pour détection outliers (optionnel)
            
        Returns:
            Données prétraitées
        """
        preprocessing_metadata: Dict[str, Any] = {
            "outlier_removed": False,
            "missing_value_imputed": False,
            "quality_improved": False,
        }
        
        value = sensor_data.value
        quality = sensor_data.quality
        
        # Vérifier qualité initiale
        if quality == 0:  # Bad quality
            logger.warning(f"Donnée de mauvaise qualité ignorée: asset={sensor_data.asset_id}, sensor={sensor_data.sensor_id}")
            # On peut choisir d'imputer ou de rejeter
            # Pour l'instant, on garde mais on marque
            preprocessing_metadata["quality_improved"] = False
        
        # Détection d'outliers si valeurs historiques disponibles
        if historical_values and len(historical_values) > 10:
            is_outlier = self._detect_outlier(value, historical_values)
            if is_outlier:
                logger.warning(f"Outlier détecté: value={value}, asset={sensor_data.asset_id}, sensor={sensor_data.sensor_id}")
                preprocessing_metadata["outlier_removed"] = True
                # Imputer avec la médiane des valeurs historiques
                value = np.median(historical_values)
                preprocessing_metadata["missing_value_imputed"] = True
        
        # Vérifier valeurs NaN ou infinies
        if not np.isfinite(value):
            logger.warning(f"Valeur non finie détectée: {value}, imputation avec 0")
            value = 0.0
            preprocessing_metadata["missing_value_imputed"] = True
        
        return PreprocessedData(
            timestamp=sensor_data.timestamp,
            asset_id=sensor_data.asset_id,
            sensor_id=sensor_data.sensor_id,
            value=float(value),
            unit=sensor_data.unit,
            quality=quality,
            source_type=sensor_data.source_type,
            preprocessing_metadata=preprocessing_metadata
        )
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        value_column: str = "value",
        quality_column: Optional[str] = "quality"
    ) -> pd.DataFrame:
        """
        Nettoie un DataFrame de données capteurs
        
        Args:
            df: DataFrame avec colonnes timestamp, asset_id, sensor_id, value, etc.
            value_column: Nom de la colonne contenant les valeurs
            quality_column: Nom de la colonne qualité (optionnel)
            
        Returns:
            DataFrame nettoyé avec colonne 'preprocessing_metadata'
        """
        df = df.copy()
        
        # Créer colonne metadata si elle n'existe pas
        if 'preprocessing_metadata' not in df.columns:
            df['preprocessing_metadata'] = None
        
        # 1. Détection et suppression des outliers (Z-score)
        if len(df) > 10:
            z_scores = np.abs(stats.zscore(df[value_column]))
            outliers = z_scores > self.outlier_threshold
            
            if outliers.any():
                logger.info(f"Outliers détectés: {outliers.sum()}/{len(df)}")
                # Marquer les outliers dans metadata
                for idx in df[outliers].index:
                    metadata = {"outlier_removed": True}
                    df.at[idx, 'preprocessing_metadata'] = metadata
                    # Imputer avec la médiane
                    df.at[idx, value_column] = df[value_column].median()
        
        # 2. Gestion des valeurs manquantes
        missing_count = df[value_column].isna().sum()
        if missing_count > 0:
            logger.info(f"Valeurs manquantes détectées: {missing_count}")
            # Imputation avec interpolation linéaire
            df[value_column] = df[value_column].interpolate(method='linear')
            # Si encore des NaN au début/fin, remplir avec forward/backward fill
            df[value_column] = df[value_column].fillna(method='ffill').fillna(method='bfill')
            
            # Marquer dans metadata
            for idx in df[df[value_column].isna()].index:
                metadata = df.at[idx, 'preprocessing_metadata'] or {}
                metadata["missing_value_imputed"] = True
                df.at[idx, 'preprocessing_metadata'] = metadata
        
        # 3. Vérifier valeurs infinies
        inf_count = np.isinf(df[value_column]).sum()
        if inf_count > 0:
            logger.warning(f"Valeurs infinies détectées: {inf_count}")
            df[value_column] = df[value_column].replace([np.inf, -np.inf], np.nan)
            df[value_column] = df[value_column].fillna(df[value_column].median())
        
        # 4. Filtrer par qualité si colonne qualité existe
        if quality_column and quality_column in df.columns:
            initial_count = len(df)
            df = df[df[quality_column] >= 1]  # Garder seulement quality >= 1 (uncertain ou good)
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.info(f"Données de mauvaise qualité filtrées: {removed_count}")
        
        return df
    
    def _detect_outlier(self, value: float, historical_values: List[float]) -> bool:
        """
        Détecte si une valeur est un outlier
        
        Args:
            value: Valeur à vérifier
            historical_values: Valeurs historiques
            
        Returns:
            True si outlier
        """
        if len(historical_values) < 10:
            return False
        
        # Méthode Z-score
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return False
        
        z_score = abs((value - mean) / std)
        return z_score > self.outlier_threshold
    
    def detect_outliers_iqr(self, values: pd.Series) -> pd.Series:
        """
        Détecte les outliers avec la méthode IQR (Interquartile Range)
        
        Args:
            values: Série de valeurs
            
        Returns:
            Série booléenne (True = outlier)
        """
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (values < lower_bound) | (values > upper_bound)

