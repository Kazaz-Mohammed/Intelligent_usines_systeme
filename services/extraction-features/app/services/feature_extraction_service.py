"""
Service principal d'orchestration de l'extraction de features
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
import uuid

from app.models.feature_data import (
    PreprocessedDataReference,
    WindowedDataReference,
    ExtractedFeature,
    ExtractedFeaturesVector
)
from app.services.kafka_consumer import KafkaConsumerService
from app.services.kafka_producer import KafkaProducerService
from app.services.temporal_features_service import TemporalFeaturesService
from app.services.tsfresh_features_service import TSFreshFeaturesService
from app.services.frequency_features_service import FrequencyFeaturesService
from app.services.wavelet_features_service import WaveletFeaturesService
from app.services.standardization_service import StandardizationService
from app.services.feast_service import FeastService
from app.services.asset_service import AssetService
from app.database.timescaledb import TimescaleDBService
from app.config import settings

logger = logging.getLogger(__name__)


class FeatureExtractionService:
    """
    Orchestre l'extraction de features depuis les données prétraitées.
    Peut fonctionner en mode streaming (traitement immédiat)
    ou en mode batch (traitement par fenêtres).
    """
    
    def __init__(self):
        import sys
        print("  Initialisation FeatureExtractionService...", file=sys.stderr)
        logger.info("Initialisation FeatureExtractionService...")
        
        try:
            print("    Création Kafka Consumer...", file=sys.stderr)
            self.kafka_consumer = KafkaConsumerService()
            print("    ✓ Kafka Consumer créé", file=sys.stderr)
            
            print("    Création Kafka Producer...", file=sys.stderr)
            self.kafka_producer = KafkaProducerService()
            print("    ✓ Kafka Producer créé", file=sys.stderr)
            
            print("    Création Temporal Features Service...", file=sys.stderr)
            self.temporal_features_service = TemporalFeaturesService()
            print("    ✓ Temporal Features Service créé", file=sys.stderr)
            
            print("    Création TSFresh Features Service...", file=sys.stderr)
            self.tsfresh_features_service = TSFreshFeaturesService()
            print("    ✓ TSFresh Features Service créé", file=sys.stderr)
            
            print("    Création Frequency Features Service...", file=sys.stderr)
            self.frequency_features_service = FrequencyFeaturesService()
            print("    ✓ Frequency Features Service créé", file=sys.stderr)
            
            print("    Création Wavelet Features Service...", file=sys.stderr)
            self.wavelet_features_service = WaveletFeaturesService()
            print("    ✓ Wavelet Features Service créé", file=sys.stderr)
            
            print("    Création Standardization Service...", file=sys.stderr)
            self.standardization_service = StandardizationService()
            print("    ✓ Standardization Service créé", file=sys.stderr)
            
            print("    Création Feast Service...", file=sys.stderr)
            self.feast_service = FeastService()
            print("    ✓ Feast Service créé", file=sys.stderr)
            
            print("    Création Asset Service...", file=sys.stderr)
            self.asset_service = AssetService()
            print("    ✓ Asset Service créé", file=sys.stderr)
            
            print("    Création TimescaleDB Service...", file=sys.stderr)
            self.timescale_db_service = TimescaleDBService()
            print("    ✓ TimescaleDB Service créé", file=sys.stderr)
            
            # Buffer pour les données par asset
            self.data_buffer: Dict[str, List[ExtractedFeature]] = defaultdict(list)
            self.last_processed_time: Dict[str, datetime] = defaultdict(lambda: datetime.min)
            
            print("  ✓ FeatureExtractionService initialisé", file=sys.stderr)
            logger.info("✓ FeatureExtractionService initialisé")
        except Exception as e:
            print(f"  ✗ Erreur lors de l'initialisation FeatureExtractionService: {e}", file=sys.stderr)
            logger.error(f"Erreur lors de l'initialisation FeatureExtractionService: {e}", exc_info=True)
            raise
    
    async def process_preprocessed_data(
        self,
        preprocessed_data: List[PreprocessedDataReference],
        mode: str = "streaming"
    ):
        """
        Traite des données prétraitées et extrait les features
        
        Args:
            preprocessed_data: Liste de données prétraitées
            mode: "streaming" pour traitement immédiat, "batch" pour accumulation
        """
        logger.info(f"🔵 process_preprocessed_data appelé avec {len(preprocessed_data)} messages")
        if not preprocessed_data:
            logger.warning("⚠ Aucune donnée prétraitée reçue")
            return
        
        # Grouper par asset_id et sensor_id
        grouped_by_asset: Dict[str, Dict[str, List[PreprocessedDataReference]]] = defaultdict(lambda: defaultdict(list))
        
        for data in preprocessed_data:
            grouped_by_asset[data.asset_id][data.sensor_id].append(data)
        
        logger.info(f"📊 Données groupées: {len(grouped_by_asset)} assets, {sum(len(sensors) for sensors in grouped_by_asset.values())} capteurs")
        
        # Traiter chaque asset
        for asset_id, sensors_data in grouped_by_asset.items():
            logger.info(f"🔧 Traitement de l'asset {asset_id} avec {len(sensors_data)} capteurs")
            try:
                # Obtenir le type d'actif (depuis la base de données ou les métadonnées)
                asset_type = await self._get_asset_type(asset_id)
                
                # Calculer les features pour chaque capteur
                all_features: List[ExtractedFeature] = []
                
                for sensor_id, sensor_data in sensors_data.items():
                    # Trier par timestamp
                    sensor_data.sort(key=lambda x: x.timestamp)
                    logger.info(f"  📡 Capteur {sensor_id}: {len(sensor_data)} points de données")
                    
                    # Calculer les features pour ce capteur
                    sensor_features: List[ExtractedFeature] = []
                    
                    # Calculer les features temporelles
                    if settings.enable_temporal_features:
                        logger.debug(f"    Calcul des features temporelles pour {sensor_id}...")
                        temporal_features = self.temporal_features_service.calculate_temporal_features(
                            sensor_data
                        )
                        logger.info(f"    ✓ {len(temporal_features)} features temporelles calculées pour {sensor_id}")
                        sensor_features.extend(temporal_features)
                        
                        # Calculer les features tsfresh (optionnel)
                        if self.tsfresh_features_service.is_available():
                            try:
                                tsfresh_features = self.tsfresh_features_service.calculate_tsfresh_features(
                                    sensor_data
                                )
                                sensor_features.extend(tsfresh_features)
                            except Exception as e:
                                logger.warning(f"Erreur lors du calcul des features tsfresh: {e}")
                    
                    # Calculer les features fréquentielles
                    if settings.enable_frequency_features:
                        frequency_features = self.frequency_features_service.calculate_frequency_features(
                            sensor_data
                        )
                        sensor_features.extend(frequency_features)
                        
                        # Calculer l'énergie par bande
                        try:
                            band_energy_features = self.frequency_features_service.calculate_band_energy(
                                sensor_data
                            )
                            sensor_features.extend(band_energy_features)
                        except Exception as e:
                            logger.warning(f"Erreur lors du calcul de l'énergie de bande: {e}")
                    
                    # Calculer les features ondelettes
                    if settings.enable_wavelet_features and self.wavelet_features_service.is_available():
                        try:
                            wavelet_features = self.wavelet_features_service.calculate_wavelet_features(
                                sensor_data
                            )
                            sensor_features.extend(wavelet_features)
                        except Exception as e:
                            logger.warning(f"Erreur lors du calcul des features ondelettes: {e}")
                    
                    # En mode streaming, publier immédiatement après chaque capteur pour éviter les timeouts
                    if mode == "streaming" and sensor_features:
                        # Publier les features de ce capteur immédiatement
                        try:
                            self._publish_features_grouped(sensor_features, asset_id)
                            logger.info(f"  ✓ Publié {len(sensor_features)} features pour capteur {sensor_id}")
                        except Exception as e:
                            logger.error(f"  ✗ Erreur lors de la publication des features pour capteur {sensor_id}: {e}", exc_info=True)
                    
                    # Ajouter aux features totales pour stockage
                    all_features.extend(sensor_features)
                
                # Standardiser les features par type d'actif
                if settings.enable_standardization and all_features:
                    standardized_features = self.standardization_service.standardize_features(
                        all_features,
                        asset_type
                    )
                    all_features.extend(standardized_features)
                
                # Stocker dans TimescaleDB
                if all_features:
                    self.timescale_db_service.insert_extracted_features_batch(all_features)
                    logger.debug(f"Features stockées dans TimescaleDB pour asset={asset_id}: {len(all_features)} features")
                
                # Stocker dans Feast (si activé)
                if self.feast_service.is_available() and all_features:
                    # Grouper les features par timestamp pour Feast (timestamp is in metadata)
                    features_by_timestamp = defaultdict(list)
                    for feature in all_features:
                        # Get timestamp from metadata or use current time
                        timestamp_str = feature.metadata.get("timestamp")
                        if timestamp_str:
                            from datetime import datetime
                            timestamp = datetime.fromisoformat(timestamp_str)
                        else:
                            timestamp = datetime.now()
                        features_by_timestamp[timestamp].append(feature)
                    
                    for timestamp, features in features_by_timestamp.items():
                        self.feast_service.store_features(features, asset_id, timestamp)
                    
                    logger.debug(f"Features stockées dans Feast pour asset={asset_id}: {len(all_features)} features")
                
                # En mode streaming, les features ont déjà été publiées par capteur
                # En mode batch, publier maintenant
                if mode == "batch":
                    self._publish_features_grouped(all_features, asset_id)
                    logger.info(f"Publié {len(all_features)} features en batch pour asset={asset_id}")
                elif mode == "streaming":
                    logger.info(f"✓ Traitement terminé: {len(all_features)} features totales pour asset={asset_id} (déjà publiées par capteur)")
                elif mode == "batch":
                    # Mode batch: accumuler et traiter par fenêtres
                    self._accumulate_data(asset_id, all_features)
                    feature_vector = await self._process_and_publish_batch(asset_id, asset_type)
                    if feature_vector:
                        self.timescale_db_service.insert_feature_vector(feature_vector)
                        logger.debug(f"Vecteur de features stocké dans TimescaleDB pour asset={asset_id}")
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement des données pour asset={asset_id}: {e}", exc_info=True)
                continue
    
    async def process_windowed_data(
        self,
        windowed_data: WindowedDataReference
    ):
        """
        Traite des fenêtres de données et extrait les features
        
        Args:
            windowed_data: Fenêtre de données prétraitées
        """
        try:
            # Obtenir le type d'actif
            asset_type = await self._get_asset_type(windowed_data.asset_id)
            
            # Calculer les features pour chaque capteur
            all_features: List[ExtractedFeature] = []
            
            for sensor_id, sensor_data in windowed_data.sensor_data.items():
                # Trier par timestamp
                sensor_data.sort(key=lambda x: x.timestamp)
                
                # Calculer les features temporelles
                if settings.enable_temporal_features:
                    temporal_features = self.temporal_features_service.calculate_temporal_features(
                        sensor_data
                    )
                    all_features.extend(temporal_features)
                
                # Calculer les features fréquentielles
                if settings.enable_frequency_features:
                    frequency_features = self.frequency_features_service.calculate_frequency_features(
                        sensor_data
                    )
                    all_features.extend(frequency_features)
                
                # Calculer les features ondelettes
                if settings.enable_wavelet_features and self.wavelet_features_service.is_available():
                    wavelet_features = self.wavelet_features_service.calculate_wavelet_features(
                        sensor_data
                    )
                    all_features.extend(wavelet_features)
            
            # Créer un vecteur de features
            feature_vector = self._create_feature_vector(
                windowed_data,
                all_features,
                asset_type
            )
            
            # Standardiser le vecteur de features
            if settings.enable_standardization:
                feature_vector = self.standardization_service.standardize_feature_vector(
                    feature_vector,
                    asset_type
                )
            
            # Stocker dans TimescaleDB
            self.timescale_db_service.insert_feature_vector(feature_vector)
            logger.debug(f"Vecteur de features stocké dans TimescaleDB: {feature_vector.vector_id}")
            
            # Stocker dans Feast (si activé)
            if self.feast_service.is_available():
                self.feast_service.store_feature_vector(feature_vector, windowed_data.asset_id)
                logger.debug(f"Vecteur de features stocké dans Feast: {feature_vector.vector_id}")
            
            # Publier sur Kafka
            self.kafka_producer.publish_feature_vector(feature_vector)
            logger.info(f"Vecteur de features publié: {feature_vector.vector_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la fenêtre {windowed_data.window_id}: {e}", exc_info=True)
            raise
    
    def _create_feature_vector(
        self,
        windowed_data: WindowedDataReference,
        features: List[ExtractedFeature],
        asset_type: Optional[str] = None
    ) -> ExtractedFeaturesVector:
        """
        Crée un vecteur de features à partir d'une liste de features
        
        Args:
            windowed_data: Fenêtre de données
            features: Liste de features
            asset_type: Type d'actif
        
        Returns:
            Vecteur de features
        """
        # Convertir les features en dictionnaire
        features_dict = {feature.feature_name: feature.feature_value for feature in features}
        
        # Créer le vecteur de features
        # Get sensor_id from first feature or use default
        sensor_id = features[0].sensor_id if features else "unknown"
        feature_vector = ExtractedFeaturesVector(
            vector_id=f"fv_{windowed_data.window_id}",
            timestamp=windowed_data.end_timestamp,
            asset_id=windowed_data.asset_id,
            sensor_id=sensor_id,
            window_id=windowed_data.window_id,
            features=[f for f in features],  # List of ExtractedFeature objects
            feature_values=features_dict,  # Dict for quick lookup
            metadata={
                "window_id": windowed_data.window_id,
                "asset_type": asset_type,
                "num_features": len(features),
                "feature_types": list(set(f.feature_type for f in features)),
                "standardized": False
            }
        )
        
        return feature_vector
    
    def _accumulate_data(self, asset_id: str, features: List[ExtractedFeature]):
        """Accumule les features dans un buffer pour le traitement par lots"""
        # Pour le mode batch, on accumule les features par timestamp (from metadata)
        self.data_buffer[asset_id].extend(features)
        self.data_buffer[asset_id].sort(key=lambda x: x.metadata.get("timestamp", ""))
    
    async def _process_and_publish_batch(
        self,
        asset_id: str,
        asset_type: Optional[str] = None
    ) -> Optional[ExtractedFeaturesVector]:
        """
        Traite les features accumulées par lots et crée un vecteur de features
        """
        try:
            current_buffer = self.data_buffer[asset_id]
            if not current_buffer:
                return None
            
            # Créer un vecteur de features à partir du buffer
            features_dict = {feature.feature_name: feature.feature_value for feature in current_buffer}
            
            # Créer le vecteur de features
            vector_id = f"fv_{asset_id}_{uuid.uuid4().hex[:8]}"
            # Get timestamps and sensor_id from features
            timestamps = []
            sensor_ids = set()
            for f in current_buffer:
                if f.timestamp:
                    timestamps.append(f.timestamp)
                if f.sensor_id:
                    sensor_ids.add(f.sensor_id)
            min_timestamp = min(timestamps) if timestamps else datetime.now()
            max_timestamp = max(timestamps) if timestamps else datetime.now()
            sensor_id = list(sensor_ids)[0] if sensor_ids else "unknown"
            
            feature_vector = ExtractedFeaturesVector(
                vector_id=vector_id,
                timestamp=max_timestamp,
                asset_id=asset_id,
                sensor_id=sensor_id,
                features=[f for f in current_buffer],  # List of ExtractedFeature objects
                feature_values=features_dict,  # Dict for quick lookup
                metadata={
                    "asset_type": asset_type,
                    "num_features": len(current_buffer),
                    "feature_types": list(set(f.feature_type for f in current_buffer)),
                    "standardized": False
                }
            )
            
            # Standardiser le vecteur de features
            if settings.enable_standardization:
                feature_vector = self.standardization_service.standardize_feature_vector(
                    feature_vector,
                    asset_type
                )
            
            # Publier sur Kafka
            self.kafka_producer.publish_feature_vector(feature_vector)
            logger.info(f"Vecteur de features publié en batch: {vector_id}")
            
            # Nettoyer le buffer
            self.data_buffer[asset_id].clear()
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Erreur lors de l'accumulation et traitement batch: {e}", exc_info=True)
            return None
    
    def _publish_features_grouped(self, features: List[ExtractedFeature], asset_id: str):
        """
        Groupe les features par sensor_id et timestamp, puis publie sur Kafka
        dans le format attendu par les services downstream (detection-anomalies, prediction-rul)
        
        Args:
            features: Liste de features à publier
            asset_id: ID de l'actif
        """
        if not features:
            return
        
        # Grouper par sensor_id et timestamp
        grouped: Dict[str, Dict[str, List[ExtractedFeature]]] = defaultdict(lambda: defaultdict(list))
        
        for feature in features:
            sensor_id = feature.sensor_id
            timestamp_str = feature.timestamp.isoformat() if feature.timestamp else ""
            # Use timestamp as key for grouping
            grouped[sensor_id][timestamp_str].append(feature)
        
        # Publier un message par groupe (sensor_id + timestamp)
        for sensor_id, timestamp_groups in grouped.items():
            for timestamp_str, feature_group in timestamp_groups.items():
                # Convertir les features en dictionnaire {feature_name: feature_value}
                features_dict = {f.feature_name: f.feature_value for f in feature_group}
                
                # Extraire timestamp
                try:
                    if timestamp_str:
                        if isinstance(timestamp_str, str):
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = timestamp_str
                    else:
                        timestamp = datetime.now()
                except:
                    timestamp = datetime.now()
                
                # Créer le message dans le format attendu par downstream services
                message = {
                    "asset_id": asset_id,
                    "sensor_id": sensor_id,
                    "features": features_dict,
                    "timestamp": timestamp.isoformat(),
                    "metadata": {
                        "num_features": len(feature_group),
                        "feature_types": list(set(f.feature_type for f in feature_group)),
                        "source": "extraction-features-service"
                    }
                }
                
                # Publier le message
                self.kafka_producer.publish_feature_message(message)
    
    async def _get_asset_type(self, asset_id: str) -> Optional[str]:
        """
        Récupère le type d'actif depuis la base de données
        
        Args:
            asset_id: ID de l'actif
        
        Returns:
            Type d'actif ou None
        """
        try:
            # Récupérer le type d'actif depuis la base de données
            asset_type = self.asset_service.get_asset_type(asset_id)
            return asset_type
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du type d'actif: {e}", exc_info=True)
            return None
    
    def get_buffer_size(self, asset_id: str) -> int:
        """Retourne la taille actuelle du buffer pour un asset"""
        return len(self.data_buffer[asset_id])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du service
        
        Returns:
            Dictionnaire avec les statistiques
        """
        return {
            "buffers": {
                asset_id: len(features) for asset_id, features in self.data_buffer.items()
            },
            "last_processed": {
                asset_id: last_time.isoformat() for asset_id, last_time in self.last_processed_time.items()
            },
            "services": {
                "temporal_features": settings.enable_temporal_features,
                "frequency_features": settings.enable_frequency_features,
                "wavelet_features": settings.enable_wavelet_features,
                "standardization": settings.enable_standardization,
                "feast": self.feast_service.is_available()
            }
        }

