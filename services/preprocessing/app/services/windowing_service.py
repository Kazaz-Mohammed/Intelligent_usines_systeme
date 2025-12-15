"""
Service de fenêtrage glissant
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

from app.models.sensor_data import PreprocessedData, WindowedData
from app.config import settings

logger = logging.getLogger(__name__)


class WindowingService:
    """Service pour créer des fenêtres glissantes pour ML"""
    
    def __init__(self):
        self.window_size = settings.window_size
        self.window_overlap = settings.window_overlap
    
    def create_windows(
        self,
        sensor_data_dict: Dict[str, List[PreprocessedData]],
        window_size: Optional[int] = None,
        overlap: Optional[float] = None
    ) -> List[WindowedData]:
        """
        Crée des fenêtres glissantes à partir de données multi-capteurs
        
        Args:
            sensor_data_dict: Dict {sensor_id: [PreprocessedData]}
            window_size: Taille de fenêtre (None = utiliser config)
            overlap: Chevauchement (0.0-1.0, None = utiliser config)
            
        Returns:
            Liste de WindowedData
        """
        if not sensor_data_dict:
            return []
        
        window_size = window_size or self.window_size
        overlap = overlap if overlap is not None else self.window_overlap
        
        # Vérifier que toutes les listes ont la même longueur (après synchronisation)
        lengths = [len(data) for data in sensor_data_dict.values()]
        if not lengths:
            return []
        
        min_length = min(lengths)
        if min_length < window_size:
            logger.warning(f"Pas assez de données pour créer des fenêtres: {min_length} < {window_size}")
            return []
        
        # Calculer le pas (step)
        step = int(window_size * (1 - overlap))
        if step < 1:
            step = 1
        
        windows = []
        asset_id = None
        
        # Créer les fenêtres
        for start_idx in range(0, min_length - window_size + 1, step):
            end_idx = start_idx + window_size
            
            # Extraire les données de la fenêtre pour chaque capteur
            window_sensor_data = {}
            timestamps = []
            
            for sensor_id, sensor_data_list in sensor_data_dict.items():
                window_data = sensor_data_list[start_idx:end_idx]
                
                if not window_data:
                    continue
                
                # Stocker les valeurs
                window_sensor_data[sensor_id] = [d.value for d in window_data]
                
                # Stocker les timestamps (première fois)
                if not timestamps:
                    timestamps = [d.timestamp for d in window_data]
                
                # Récupérer asset_id (première fois)
                if asset_id is None:
                    asset_id = window_data[0].asset_id
            
            if not window_sensor_data or not timestamps:
                continue
            
            # Créer la fenêtre
            window_id = f"WINDOW_{uuid.uuid4().hex[:8].upper()}"
            
            window = WindowedData(
                window_id=window_id,
                asset_id=asset_id or "UNKNOWN",
                start_timestamp=timestamps[0],
                end_timestamp=timestamps[-1],
                sensor_data=window_sensor_data,
                metadata={
                    "window_size": window_size,
                    "overlap": overlap,
                    "step": step,
                    "sensor_count": len(window_sensor_data),
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            windows.append(window)
        
        logger.info(f"Créé {len(windows)} fenêtres de taille {window_size} avec overlap {overlap}")
        return windows
    
    def create_windows_from_single_sensor(
        self,
        data: List[PreprocessedData],
        window_size: Optional[int] = None,
        overlap: Optional[float] = None
    ) -> List[WindowedData]:
        """
        Crée des fenêtres à partir d'un seul capteur
        
        Args:
            data: Liste de données prétraitées
            window_size: Taille de fenêtre
            overlap: Chevauchement
            
        Returns:
            Liste de WindowedData
        """
        if not data:
            return []
        
        # Convertir en format multi-capteurs
        sensor_id = data[0].sensor_id
        sensor_data_dict = {sensor_id: data}
        
        return self.create_windows(sensor_data_dict, window_size, overlap)
    
    def create_windows_with_metadata(
        self,
        sensor_data_dict: Dict[str, List[PreprocessedData]],
        additional_metadata: Dict[str, Any] = None,
        window_size: Optional[int] = None,
        overlap: Optional[float] = None
    ) -> List[WindowedData]:
        """
        Crée des fenêtres avec métadonnées supplémentaires
        
        Args:
            sensor_data_dict: Dict {sensor_id: [PreprocessedData]}
            additional_metadata: Métadonnées supplémentaires à ajouter
            window_size: Taille de fenêtre
            overlap: Chevauchement
            
        Returns:
            Liste de WindowedData avec métadonnées
        """
        windows = self.create_windows(sensor_data_dict, window_size, overlap)
        
        if additional_metadata:
            for window in windows:
                window.metadata.update(additional_metadata)
        
        return windows

