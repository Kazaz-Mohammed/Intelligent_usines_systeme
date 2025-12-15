"""
Service principal d'orchestration du prétraitement
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

from app.models.sensor_data import SensorData, PreprocessedData, WindowedData
from app.services.kafka_consumer import KafkaConsumerService
from app.services.kafka_producer import KafkaProducerService
from app.services.cleaning_service import CleaningService
from app.services.resampling_service import ResamplingService
from app.services.denoising_service import DenoisingService
from app.services.frequency_analysis_service import FrequencyAnalysisService
from app.services.windowing_service import WindowingService
from app.config import settings

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service principal qui orchestre tout le pipeline de prétraitement"""
    
    def __init__(self):
        self.kafka_consumer = KafkaConsumerService()
        self.kafka_producer = KafkaProducerService()
        self.cleaning_service = CleaningService()
        self.resampling_service = ResamplingService()
        self.denoising_service = DenoisingService()
        self.frequency_analysis_service = FrequencyAnalysisService()
        self.windowing_service = WindowingService()
        
        # Buffer pour accumuler les données par asset/sensor
        self.data_buffer: Dict[str, Dict[str, List[PreprocessedData]]] = defaultdict(lambda: defaultdict(list))
        self.buffer_size = 1000  # Taille max du buffer par capteur
        
    def process_single_sensor_data(self, sensor_data: SensorData) -> Optional[PreprocessedData]:
        """
        Traite une donnée capteur unique
        
        Args:
            sensor_data: Donnée capteur brute
            
        Returns:
            Donnée prétraitées ou None si rejetée
        """
        try:
            # 1. Nettoyage
            cleaned_data = self.cleaning_service.clean_single_value(sensor_data)
            
            # Vérifier si la donnée a été rejetée (qualité trop faible)
            if cleaned_data.quality == 0:
                logger.debug(f"Donnée rejetée (qualité 0): asset={sensor_data.asset_id}, sensor={sensor_data.sensor_id}")
                return None
            
            # 2. Débruitage (si activé)
            if settings.enable_denoising:
                denoised_list = self.denoising_service.denoise_single_sensor(
                    [cleaned_data],
                    method="butterworth"
                )
                if denoised_list:
                    cleaned_data = denoised_list[0]
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement d'une donnée: {e}", exc_info=True)
            return None
    
    def process_and_publish(self, sensor_data: SensorData):
        """
        Traite une donnée et la publie immédiatement (mode streaming)
        
        Args:
            sensor_data: Donnée capteur brute
        """
        try:
            # Traiter la donnée
            preprocessed = self.process_single_sensor_data(sensor_data)
            
            if preprocessed is None:
                return
            
            # Analyse fréquentielle (optionnel, nécessite plusieurs points)
            # On saute pour l'instant en mode streaming
            
            # Publier sur Kafka
            self.kafka_producer.publish_preprocessed_data(preprocessed)
            
            logger.debug(f"Donnée prétraitée et publiée: asset={preprocessed.asset_id}, sensor={preprocessed.sensor_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement et publication: {e}", exc_info=True)
    
    def accumulate_and_process_batch(
        self,
        sensor_data: SensorData,
        min_batch_size: int = 100
    ) -> Optional[List[WindowedData]]:
        """
        Accumule les données et crée des fenêtres quand le batch est suffisant
        
        Args:
            sensor_data: Donnée capteur brute
            min_batch_size: Taille minimale du batch pour créer des fenêtres
            
        Returns:
            Liste de fenêtres créées ou None
        """
        try:
            # Traiter la donnée
            preprocessed = self.process_single_sensor_data(sensor_data)
            
            if preprocessed is None:
                return None
            
            # Ajouter au buffer
            asset_id = preprocessed.asset_id
            sensor_id = preprocessed.sensor_id
            
            self.data_buffer[asset_id][sensor_id].append(preprocessed)
            
            # Limiter la taille du buffer
            if len(self.data_buffer[asset_id][sensor_id]) > self.buffer_size:
                self.data_buffer[asset_id][sensor_id] = self.data_buffer[asset_id][sensor_id][-self.buffer_size:]
            
            # Vérifier si on a assez de données pour créer des fenêtres
            total_points = sum(len(sensor_list) for sensor_list in self.data_buffer[asset_id].values())
            
            if total_points >= min_batch_size:
                # Créer les fenêtres
                windows = self._create_windows_from_buffer(asset_id)
                
                # Publier les fenêtres
                for window in windows:
                    self.kafka_producer.publish_windowed_data(window)
                
                # Nettoyer le buffer (garder seulement les dernières données pour overlap)
                self._clean_buffer(asset_id, keep_last=settings.window_size)
                
                logger.info(f"Créé et publié {len(windows)} fenêtres pour asset={asset_id}")
                return windows
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de l'accumulation et traitement batch: {e}", exc_info=True)
            return None
    
    def _create_windows_from_buffer(self, asset_id: str) -> List[WindowedData]:
        """
        Crée des fenêtres à partir du buffer pour un asset
        
        Args:
            asset_id: ID de l'asset
            
        Returns:
            Liste de fenêtres
        """
        if asset_id not in self.data_buffer:
            return []
        
        sensor_data_dict = self.data_buffer[asset_id]
        
        if not sensor_data_dict:
            return []
        
        # Synchroniser les capteurs si nécessaire
        if settings.resampling_rate:
            synchronized = self.resampling_service.synchronize_multiple_sensors(sensor_data_dict)
        else:
            synchronized = sensor_data_dict
        
        # Débruitage (si activé)
        if settings.enable_denoising:
            for sensor_id, sensor_data_list in synchronized.items():
                synchronized[sensor_id] = self.denoising_service.denoise_single_sensor(
                    sensor_data_list,
                    method="butterworth"
                )
        
        # Analyse fréquentielle (si activé)
        if settings.enable_frequency_analysis:
            for sensor_id, sensor_data_list in synchronized.items():
                if len(sensor_data_list) >= 10:
                    analysis_result = self.frequency_analysis_service.analyze_frequency(
                        sensor_data_list,
                        method="fft"
                    )
                    if analysis_result:
                        synchronized[sensor_id] = self.frequency_analysis_service.add_frequency_analysis_to_data(
                            sensor_data_list,
                            analysis_result
                        )
        
        # Créer les fenêtres
        windows = self.windowing_service.create_windows(synchronized)
        
        return windows
    
    def _clean_buffer(self, asset_id: str, keep_last: int = 100):
        """
        Nettoie le buffer en gardant seulement les dernières données
        
        Args:
            asset_id: ID de l'asset
            keep_last: Nombre de points à garder
        """
        if asset_id not in self.data_buffer:
            return
        
        for sensor_id in self.data_buffer[asset_id]:
            sensor_list = self.data_buffer[asset_id][sensor_id]
            if len(sensor_list) > keep_last:
                self.data_buffer[asset_id][sensor_id] = sensor_list[-keep_last:]
    
    def process_batch(
        self,
        sensor_data_list: List[SensorData]
    ) -> List[PreprocessedData]:
        """
        Traite un batch de données
        
        Args:
            sensor_data_list: Liste de données capteurs brutes
            
        Returns:
            Liste de données prétraitées
        """
        result = []
        
        # Grouper par asset et sensor
        grouped: Dict[str, Dict[str, List[SensorData]]] = defaultdict(lambda: defaultdict(list))
        
        for sensor_data in sensor_data_list:
            grouped[sensor_data.asset_id][sensor_data.sensor_id].append(sensor_data)
        
        # Traiter chaque groupe
        for asset_id, sensors_dict in grouped.items():
            for sensor_id, sensor_list in sensors_dict.items():
                # Nettoyer
                cleaned_list = []
                for data in sensor_list:
                    cleaned = self.cleaning_service.clean_single_value(data)
                    if cleaned.quality > 0:  # Garder seulement les données de qualité acceptable
                        cleaned_list.append(cleaned)
                
                if not cleaned_list:
                    continue
                
                # Rééchantillonnage (si nécessaire)
                if settings.resampling_rate:
                    cleaned_list = self.resampling_service.resample_single_sensor(cleaned_list)
                
                # Débruitage (si activé)
                if settings.enable_denoising:
                    cleaned_list = self.denoising_service.denoise_single_sensor(
                        cleaned_list,
                        method="butterworth"
                    )
                
                # Analyse fréquentielle (si activé et assez de données)
                if settings.enable_frequency_analysis and len(cleaned_list) >= 10:
                    analysis_result = self.frequency_analysis_service.analyze_frequency(
                        cleaned_list,
                        method="fft"
                    )
                    if analysis_result:
                        cleaned_list = self.frequency_analysis_service.add_frequency_analysis_to_data(
                            cleaned_list,
                            analysis_result
                        )
                
                result.extend(cleaned_list)
        
        return result
    
    def start_processing_loop(self, mode: str = "streaming"):
        """
        Démarre la boucle de traitement principale
        
        Args:
            mode: "streaming" (publie immédiatement) ou "batch" (accumule et fenêtre)
        """
        logger.info(f"Démarrage du pipeline de prétraitement en mode: {mode}")
        
        def message_handler(sensor_data: SensorData):
            """Handler appelé pour chaque message Kafka"""
            try:
                if mode == "streaming":
                    self.process_and_publish(sensor_data)
                elif mode == "batch":
                    self.accumulate_and_process_batch(sensor_data)
                else:
                    logger.error(f"Mode inconnu: {mode}")
            except Exception as e:
                logger.error(f"Erreur dans le handler de message: {e}", exc_info=True)
        
        # Démarrer la consommation Kafka
        self.kafka_consumer.start(message_handler)
    
    def stop(self):
        """Arrête le service"""
        logger.info("Arrêt du service de prétraitement")
        self.kafka_consumer.stop()
        self.kafka_producer.flush()
        self.kafka_producer.close()

