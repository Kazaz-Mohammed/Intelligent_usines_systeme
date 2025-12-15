"""
Service pour définir les entités et features dans Feast
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from feast import Entity, Feature, FeatureView, ValueType
    from feast.types import Float32, Int64, String
    from feast.data_source import FileSource
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Feast n'est pas disponible")

from app.config import settings
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector

logger = logging.getLogger(__name__)


class FeastEntityService:
    """Service pour définir les entités et features dans Feast"""
    
    def __init__(self):
        self.enable_feast = FEAST_AVAILABLE and settings.feast_enable
        
        if not FEAST_AVAILABLE:
            logger.warning("Feast n'est pas disponible")
        elif self.enable_feast:
            self._define_entities()
            self._define_features()
    
    def _define_entities(self):
        """Définit les entités dans Feast"""
        try:
            # Entité Asset
            self.asset_entity = Entity(
                name="asset",
                value_type=ValueType.STRING,
                description="Asset ID"
            )
            
            # Entité Sensor (optionnelle)
            self.sensor_entity = Entity(
                name="sensor",
                value_type=ValueType.STRING,
                description="Sensor ID"
            )
            
            logger.info("Entités Feast définies")
            
        except Exception as e:
            logger.error(f"Erreur lors de la définition des entités: {e}", exc_info=True)
    
    def _define_features(self):
        """Définit les features dans Feast"""
        try:
            # Features temporelles
            self.temporal_features = [
                Feature(name="mean", dtype=Float32),
                Feature(name="std", dtype=Float32),
                Feature(name="var", dtype=Float32),
                Feature(name="min", dtype=Float32),
                Feature(name="max", dtype=Float32),
                Feature(name="median", dtype=Float32),
                Feature(name="rms", dtype=Float32),
                Feature(name="kurtosis", dtype=Float32),
                Feature(name="skewness", dtype=Float32),
                Feature(name="crest_factor", dtype=Float32),
                Feature(name="peak_to_peak", dtype=Float32),
                Feature(name="form_factor", dtype=Float32),
            ]
            
            # Features fréquentielles
            self.frequency_features = [
                Feature(name="spectral_centroid", dtype=Float32),
                Feature(name="spectral_rolloff", dtype=Float32),
                Feature(name="spectral_bandwidth", dtype=Float32),
                Feature(name="zero_crossing_rate", dtype=Float32),
                Feature(name="spectral_flatness", dtype=Float32),
            ]
            
            # Features ondelettes (exemple pour niveau 0)
            self.wavelet_features = [
                Feature(name="wavelet_energy_level_0", dtype=Float32),
                Feature(name="wavelet_entropy_level_0", dtype=Float32),
                Feature(name="wavelet_variance_level_0", dtype=Float32),
                Feature(name="wavelet_mean_level_0", dtype=Float32),
                Feature(name="wavelet_std_level_0", dtype=Float32),
            ]
            
            logger.info("Features Feast définies")
            
        except Exception as e:
            logger.error(f"Erreur lors de la définition des features: {e}", exc_info=True)
    
    def create_feature_view(
        self,
        name: str,
        features: List[Feature],
        source: Any,
        entities: List[Entity]
    ) -> Optional[Any]:
        """
        Crée une FeatureView dans Feast
        
        Args:
            name: Nom de la FeatureView
            features: Liste de features
            source: Source de données
            entities: Liste d'entités
        
        Returns:
            FeatureView ou None si erreur
        """
        if not self.enable_feast:
            return None
        
        try:
            feature_view = FeatureView(
                name=name,
                entities=entities,
                features=features,
                source=source,
                ttl=None  # Pas de TTL pour les features historiques
            )
            
            logger.info(f"FeatureView créée: {name}")
            return feature_view
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la FeatureView: {e}", exc_info=True)
            return None
    
    def get_all_features(self) -> List[Feature]:
        """
        Retourne toutes les features définies
        
        Returns:
            Liste de toutes les features
        """
        if not self.enable_feast:
            return []
        
        all_features = []
        all_features.extend(self.temporal_features)
        all_features.extend(self.frequency_features)
        all_features.extend(self.wavelet_features)
        
        return all_features
    
    def is_available(self) -> bool:
        """Vérifie si Feast est disponible"""
        return self.enable_feast

