"""
Service de rééchantillonnage et synchronisation
"""
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import interpolate

from app.models.sensor_data import PreprocessedData
from app.config import settings

logger = logging.getLogger(__name__)


class ResamplingService:
    """Service pour rééchantillonner et synchroniser les données multi-capteurs"""
    
    def __init__(self):
        self.resampling_rate = settings.resampling_rate  # Hz (None = pas de rééchantillonnage)
    
    def resample_single_sensor(
        self,
        data: List[PreprocessedData],
        target_rate: Optional[float] = None
    ) -> List[PreprocessedData]:
        """
        Rééchantillonne les données d'un seul capteur
        
        Args:
            data: Liste de données prétraitées
            target_rate: Fréquence cible en Hz (None = utiliser config)
            
        Returns:
            Liste de données rééchantillonnées
        """
        if not data:
            return []
        
        if len(data) < 2:
            return data
        
        target_rate = target_rate or self.resampling_rate
        if target_rate is None:
            # Pas de rééchantillonnage demandé
            return data
        
        # Convertir en DataFrame
        df = self._preprocessed_data_to_dataframe(data)
        
        # Rééchantillonner
        df_resampled = self._resample_dataframe(df, target_rate)
        
        # Convertir en liste de PreprocessedData
        return self._dataframe_to_preprocessed_data(df_resampled, data[0])
    
    def synchronize_multiple_sensors(
        self,
        sensor_data_dict: Dict[str, List[PreprocessedData]],
        target_rate: Optional[float] = None
    ) -> Dict[str, List[PreprocessedData]]:
        """
        Synchronise plusieurs capteurs à la même fréquence
        
        Args:
            sensor_data_dict: Dict {sensor_id: [PreprocessedData]}
            target_rate: Fréquence cible en Hz
            
        Returns:
            Dict {sensor_id: [PreprocessedData]} synchronisé
        """
        if not sensor_data_dict:
            return {}
        
        target_rate = target_rate or self.resampling_rate
        
        # Trouver la plage temporelle commune
        all_timestamps = []
        for sensor_data_list in sensor_data_dict.values():
            all_timestamps.extend([d.timestamp for d in sensor_data_list])
        
        if not all_timestamps:
            return {}
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Créer index temporel cible
        if target_rate:
            # Fréquence fixe
            time_index = pd.date_range(
                start=min_time,
                end=max_time,
                freq=pd.Timedelta(seconds=1.0/target_rate)
            )
        else:
            # Utiliser la fréquence la plus élevée
            time_index = self._create_unified_time_index(sensor_data_dict)
        
        # Rééchantillonner chaque capteur
        synchronized = {}
        for sensor_id, sensor_data_list in sensor_data_dict.items():
            if not sensor_data_list:
                continue
            
            # Convertir en DataFrame
            df = self._preprocessed_data_to_dataframe(sensor_data_list)
            
            # Rééchantillonner sur l'index temporel cible
            df_resampled = df.reindex(time_index, method='nearest')
            df_resampled = df_resampled.interpolate(method='linear')
            
            # Convertir en liste
            synchronized[sensor_id] = self._dataframe_to_preprocessed_data(
                df_resampled,
                sensor_data_list[0]
            )
        
        logger.info(f"Synchronisation de {len(sensor_data_dict)} capteurs sur {len(time_index)} points")
        return synchronized
    
    def _resample_dataframe(
        self,
        df: pd.DataFrame,
        target_rate: float
    ) -> pd.DataFrame:
        """
        Rééchantillonne un DataFrame à une fréquence cible
        
        Args:
            df: DataFrame avec index datetime
            target_rate: Fréquence cible en Hz
            
        Returns:
            DataFrame rééchantillonné
        """
        # Calculer la période cible
        target_period = pd.Timedelta(seconds=1.0/target_rate)
        
        # Rééchantillonner
        df_resampled = df.resample(target_period).mean()
        
        # Interpoler les valeurs manquantes
        df_resampled = df_resampled.interpolate(method='linear')
        
        return df_resampled
    
    def _preprocessed_data_to_dataframe(
        self,
        data: List[PreprocessedData]
    ) -> pd.DataFrame:
        """Convertit une liste de PreprocessedData en DataFrame"""
        records = []
        for item in data:
            records.append({
                'timestamp': item.timestamp,
                'value': item.value,
                'quality': item.quality,
            })
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _dataframe_to_preprocessed_data(
        self,
        df: pd.DataFrame,
        template: PreprocessedData
    ) -> List[PreprocessedData]:
        """Convertit un DataFrame en liste de PreprocessedData"""
        result = []
        
        for timestamp, row in df.iterrows():
            # Mettre à jour les métadonnées
            metadata = template.preprocessing_metadata.copy()
            metadata['resampled'] = True
            
            result.append(PreprocessedData(
                timestamp=timestamp if isinstance(timestamp, datetime) else timestamp.to_pydatetime(),
                asset_id=template.asset_id,
                sensor_id=template.sensor_id,
                value=float(row['value']),
                unit=template.unit,
                quality=int(row.get('quality', template.quality)),
                source_type=template.source_type,
                preprocessing_metadata=metadata
            ))
        
        return result
    
    def _create_unified_time_index(
        self,
        sensor_data_dict: Dict[str, List[PreprocessedData]]
    ) -> pd.DatetimeIndex:
        """Crée un index temporel unifié à partir de tous les capteurs"""
        all_timestamps = set()
        
        for sensor_data_list in sensor_data_dict.values():
            for data in sensor_data_list:
                all_timestamps.add(data.timestamp)
        
        if not all_timestamps:
            return pd.DatetimeIndex([])
        
        timestamps = sorted(all_timestamps)
        return pd.DatetimeIndex(timestamps)

