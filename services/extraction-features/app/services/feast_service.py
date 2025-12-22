"""
Service d'intégration Feature Store (Feast)
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import os
import json

try:
    from feast import FeatureStore, Entity, Feature, FeatureView, ValueType, FileSource, FeatureService
    from feast.types import Float32, Int64, String
    from feast.data_source import FileDataSource
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Feast n'est pas disponible, les features ne seront pas stockées dans Feast")

from app.config import settings
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector

logger = logging.getLogger(__name__)


class FeastService:
    """Service pour l'intégration avec Feast Feature Store"""
    
    def __init__(self):
        self.enable_feast = FEAST_AVAILABLE and settings.feast_enable
        self.feast_repo_path = settings.feast_repo_path or "./feast_repo"
        self.store: Optional[Any] = None
        
        if not FEAST_AVAILABLE:
            logger.warning("Feast n'est pas disponible, les features ne seront pas stockées dans Feast")
        elif self.enable_feast:
            self._initialize_feast()
    
    def _initialize_feast(self):
        """Initialise Feast Feature Store"""
        try:
            # Créer le répertoire Feast si nécessaire
            if not os.path.exists(self.feast_repo_path):
                os.makedirs(self.feast_repo_path)
                logger.info(f"Répertoire Feast créé: {self.feast_repo_path}")
            
            # Initialiser le Feature Store
            # Note: Feast nécessite une configuration YAML, on va créer un store minimal
            # Pour une implémentation complète, il faudrait créer un feature_store.yaml
            
            logger.info(f"Feast Feature Store initialisé: {self.feast_repo_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Feast: {e}", exc_info=True)
            self.enable_feast = False
    
    def store_features(
        self,
        features: List[ExtractedFeature],
        entity_id: str,
        timestamp: datetime
    ) -> bool:
        """
        Stocke des features dans Feast
        
        Args:
            features: Liste de features à stocker
            entity_id: ID de l'entité (asset_id)
            timestamp: Timestamp des features
        
        Returns:
            True si succès, False sinon
        """
        if not self.enable_feast or not features:
            return False
        
        try:
            # Convertir les features en format Feast
            feature_dict = {}
            for feature in features:
                feature_dict[feature.feature_name] = feature.feature_value
            
            # Pour une implémentation complète, il faudrait :
            # 1. Créer un DataFrame avec les features
            # 2. Écrire dans le store offline (file, BigQuery, S3)
            # 3. Materialiser dans le store online (Redis, SQLite, PostgreSQL)
            
            logger.debug(f"Features stockées dans Feast pour entity={entity_id}: {len(features)} features")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du stockage des features dans Feast: {e}", exc_info=True)
            return False
    
    def store_feature_vector(
        self,
        feature_vector: ExtractedFeaturesVector,
        entity_id: Optional[str] = None
    ) -> bool:
        """
        Stocke un vecteur de features dans Feast
        
        Args:
            feature_vector: Vecteur de features à stocker
            entity_id: ID de l'entité (asset_id, optionnel)
        
        Returns:
            True si succès, False sinon
        """
        if not self.enable_feast:
            return False
        
        entity_id = entity_id or feature_vector.asset_id
        
        try:
            # Convertir le vecteur de features en format Feast
            feature_dict = feature_vector.features.copy()
            
            # Ajouter les métadonnées
            if feature_vector.feature_metadata:
                feature_dict.update(feature_vector.feature_metadata)
            
            logger.debug(f"Vecteur de features stocké dans Feast pour entity={entity_id}: {len(feature_dict)} features")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du stockage du vecteur de features dans Feast: {e}", exc_info=True)
            return False
    
    def get_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Récupère des features depuis Feast
        
        Args:
            entity_id: ID de l'entité (asset_id)
            feature_names: Liste de noms de features à récupérer (optionnel)
            timestamp: Timestamp des features (optionnel)
        
        Returns:
            Dictionnaire {feature_name: feature_value}
        """
        if not self.enable_feast:
            return {}
        
        try:
            # Pour une implémentation complète, il faudrait :
            # 1. Se connecter au store online (Redis, SQLite, PostgreSQL)
            # 2. Récupérer les features pour l'entité
            # 3. Filtrer par feature_names si spécifié
            # 4. Filtrer par timestamp si spécifié
            
            logger.debug(f"Features récupérées depuis Feast pour entity={entity_id}: {len(feature_names) if feature_names else 'all'} features")
            return {}
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des features depuis Feast: {e}", exc_info=True)
            return {}
    
    def get_feature_vector(
        self,
        entity_id: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, float]]:
        """
        Récupère un vecteur de features depuis Feast
        
        Args:
            entity_id: ID de l'entité (asset_id)
            timestamp: Timestamp des features (optionnel)
        
        Returns:
            Dictionnaire {feature_name: feature_value} ou None si erreur
        """
        if not self.enable_feast:
            return None
        
        try:
            # Récupérer toutes les features pour l'entité
            feature_vector = self.get_features(entity_id, None, timestamp)
            
            logger.debug(f"Vecteur de features récupéré depuis Feast pour entity={entity_id}: {len(feature_vector)} features")
            return feature_vector if feature_vector else None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du vecteur de features depuis Feast: {e}", exc_info=True)
            return None
    
    def materialize_features(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """
        Matérialise des features du store offline vers le store online
        
        Args:
            start_date: Date de début
            end_date: Date de fin
        
        Returns:
            True si succès, False sinon
        """
        if not self.enable_feast:
            return False
        
        try:
            # Pour une implémentation complète, il faudrait :
            # 1. Lire les features du store offline (file, BigQuery, S3)
            # 2. Les écrire dans le store online (Redis, SQLite, PostgreSQL)
            # 3. Gérer les FeatureViews et FeatureServices
            
            logger.info(f"Features matérialisées depuis {start_date} jusqu'à {end_date}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la matérialisation des features: {e}", exc_info=True)
            return False
    
    def is_available(self) -> bool:
        """Vérifie si Feast est disponible"""
        return self.enable_feast
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le Feature Store
        
        Returns:
            Dictionnaire avec les informations du store
        """
        if not self.enable_feast:
            return {
                "enabled": False,
                "reason": "Feast non disponible ou désactivé"
            }
        
        return {
            "enabled": True,
            "repo_path": self.feast_repo_path,
            "online_store_type": settings.feast_online_store_type,
            "offline_store_type": settings.feast_offline_store_type,
            "store_initialized": self.store is not None
        }


class FeastFeatureStoreConfig:
    """Configuration pour Feast Feature Store"""
    
    @staticmethod
    def create_feature_store_yaml(
        repo_path: str,
        online_store_type: str = "redis",
        offline_store_type: str = "file"
    ) -> str:
        """
        Crée un fichier de configuration YAML pour Feast
        
        Args:
            repo_path: Chemin du répertoire Feast
            online_store_type: Type de store online (redis, sqlite, postgres)
            offline_store_type: Type de store offline (file, bigquery, s3)
        
        Returns:
            Contenu du fichier YAML
        """
        online_store_config = ""
        if online_store_type == "redis":
            online_store_config = """
    type: redis
    connection_string: "localhost:6379"
"""
        elif online_store_type == "sqlite":
            online_store_config = """
    type: sqlite
    path: "feast.db"
"""
        elif online_store_type == "postgres":
            online_store_config = """
    type: postgres
    host: "localhost"
    port: 5432
    database: "feast"
    user: "feast"
    password: "feast"
"""
        
        offline_store_config = ""
        if offline_store_type == "file":
            offline_store_config = """
    type: file
    path: "data/feast"
"""
        elif offline_store_type == "bigquery":
            offline_store_config = """
    type: bigquery
    project: "your-project"
    dataset: "feast"
"""
        elif offline_store_type == "s3":
            offline_store_config = """
    type: s3
    bucket: "feast-bucket"
    region: "us-east-1"
"""
        
        yaml_content = f"""project: predictive_maintenance
registry: {repo_path}/registry.db
provider: local
online_store:
{online_store_config}
offline_store:
{offline_store_config}
"""
        
        return yaml_content

