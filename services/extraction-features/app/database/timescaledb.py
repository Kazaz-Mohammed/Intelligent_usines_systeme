"""
Service pour l'interaction avec TimescaleDB
"""
import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import json
import sys
import os
import uuid

from app.config import settings
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector

logger = logging.getLogger(__name__)


class TimescaleDBService:
    """Gère la connexion et les opérations avec TimescaleDB"""
    
    def __init__(self):
        logger.info("Initialisation TimescaleDB Service...")
        self.pool: Optional[SimpleConnectionPool] = None
        try:
            self._create_pool()
            if self.pool:
                logger.info("✓ TimescaleDB Service initialisé")
            else:
                logger.warning("TimescaleDB pool non initialisé, le service fonctionnera sans base de données")
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation TimescaleDB: {e}")
            logger.warning("Le service continuera sans base de données")
            self.pool = None
    
    def _create_pool(self):
        """Crée le pool de connexions"""
        try:
            # Forcer l'encodage UTF-8 pour éviter les problèmes sur Windows
            if sys.platform == 'win32':
                os.environ['PGCLIENTENCODING'] = 'UTF8'
            
            logger.debug(f"Connexion à TimescaleDB: {settings.database_host}:{settings.database_port}/{settings.database_name}")
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.database_host,
                port=settings.database_port,
                database=settings.database_name,
                user=settings.database_user,
                password=settings.database_password,
                connect_timeout=5  # 5 second timeout
            )
            logger.info(f"Pool de connexions TimescaleDB créé: {settings.database_host}:{settings.database_port}/{settings.database_name}")
        except Exception as e:
            logger.warning(f"Erreur lors de la création du pool TimescaleDB: {e}")
            logger.warning("Le service continuera sans base de données")
            self.pool = None
    
    @contextmanager
    def get_connection(self):
        """Context manager pour obtenir une connexion du pool"""
        if not self.pool:
            logger.error("Le pool de connexions TimescaleDB n'est pas initialisé.")
            raise ConnectionError("TimescaleDB connection pool not initialized.")
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def insert_extracted_feature(self, feature: ExtractedFeature):
        """
        Insère une seule feature extraite dans la table extracted_features.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO extracted_features
                        (timestamp, asset_id, sensor_id, feature_name, feature_value, feature_type, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (timestamp, asset_id, sensor_id, feature_name) DO UPDATE
                        SET feature_value = EXCLUDED.feature_value,
                            metadata = EXCLUDED.metadata
                    """
                    
                    # Use direct fields from ExtractedFeature model
                    timestamp = feature.timestamp
                    asset_id = feature.asset_id
                    sensor_id = feature.sensor_id
                    
                    metadata_json = json.dumps(feature.metadata) if feature.metadata else None
                    
                    # Get feature_type value (handle enum)
                    feature_type_value = feature.feature_type.value if hasattr(feature.feature_type, 'value') else str(feature.feature_type)
                    
                    cur.execute(query, (
                        timestamp,
                        asset_id,
                        sensor_id,
                        feature.feature_name,
                        feature.feature_value,
                        feature_type_value,
                        metadata_json
                    ))
                conn.commit()
                logger.debug(f"Feature extraite insérée: {feature.feature_name} pour {asset_id}/{sensor_id}")
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion de la feature: {e}", exc_info=True)
            raise
    
    def insert_extracted_features_batch(self, features: List[ExtractedFeature]):
        """
        Insère un lot de features extraites dans la table extracted_features.
        """
        if not features:
            return
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO extracted_features
                        (timestamp, asset_id, sensor_id, feature_name, feature_value, feature_type, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (timestamp, asset_id, sensor_id, feature_name) DO UPDATE
                        SET feature_value = EXCLUDED.feature_value,
                            metadata = EXCLUDED.metadata
                    """
                    
                    values = []
                    for feature in features:
                        # Use direct fields from ExtractedFeature model
                        timestamp = feature.timestamp
                        asset_id = feature.asset_id
                        sensor_id = feature.sensor_id
                        
                        metadata_json = json.dumps(feature.metadata) if feature.metadata else None
                        
                        # Get feature_type value (handle enum)
                        feature_type_value = feature.feature_type.value if hasattr(feature.feature_type, 'value') else str(feature.feature_type)
                        
                        values.append((
                            timestamp,
                            asset_id,
                            sensor_id,
                            feature.feature_name,
                            feature.feature_value,
                            feature_type_value,
                            metadata_json
                        ))
                    
                    cur.executemany(query, values)
                conn.commit()
                logger.debug(f"Lot de {len(features)} features extraites inséré.")
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion du lot de features: {e}", exc_info=True)
            raise
    
    def insert_feature_vector(self, feature_vector: ExtractedFeaturesVector):
        """
        Insère un vecteur de features dans la table extracted_feature_vectors.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO extracted_feature_vectors
                        (feature_vector_id, timestamp, asset_id, start_time, end_time, features, feature_metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (feature_vector_id) DO UPDATE
                        SET features = EXCLUDED.features,
                            feature_metadata = EXCLUDED.feature_metadata
                    """
                    
                    features_json = json.dumps(feature_vector.features)
                    metadata_json = json.dumps(feature_vector.feature_metadata) if feature_vector.feature_metadata else None
                    
                    cur.execute(query, (
                        feature_vector.feature_vector_id,
                        feature_vector.timestamp,
                        feature_vector.asset_id,
                        feature_vector.start_time,
                        feature_vector.end_time,
                        features_json,
                        metadata_json
                    ))
                conn.commit()
                logger.debug(f"Vecteur de features inséré: {feature_vector.feature_vector_id}")
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion du vecteur de features: {e}", exc_info=True)
            raise
    
    def insert_feature_vectors_batch(self, feature_vectors: List[ExtractedFeaturesVector]):
        """
        Insère un lot de vecteurs de features dans la table extracted_feature_vectors.
        """
        if not feature_vectors:
            return
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO extracted_feature_vectors
                        (feature_vector_id, timestamp, asset_id, start_time, end_time, features, feature_metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (feature_vector_id) DO UPDATE
                        SET features = EXCLUDED.features,
                            feature_metadata = EXCLUDED.feature_metadata
                    """
                    
                    values = []
                    for feature_vector in feature_vectors:
                        features_json = json.dumps(feature_vector.features)
                        metadata_json = json.dumps(feature_vector.feature_metadata) if feature_vector.feature_metadata else None
                        values.append((
                            feature_vector.feature_vector_id,
                            feature_vector.timestamp,
                            feature_vector.asset_id,
                            feature_vector.start_time,
                            feature_vector.end_time,
                            features_json,
                            metadata_json
                        ))
                    
                    cur.executemany(query, values)
                conn.commit()
                logger.debug(f"Lot de {len(feature_vectors)} vecteurs de features inséré.")
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion du lot de vecteurs de features: {e}", exc_info=True)
            raise
    
    def get_features_by_asset(
        self,
        asset_id: str,
        limit: int = 100
    ) -> List[ExtractedFeature]:
        """
        Récupère les features pour un asset donné.
        
        Args:
            asset_id: ID de l'asset
            limit: Nombre maximum de features à récupérer
        
        Returns:
            Liste de features extraites
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT timestamp, asset_id, sensor_id, feature_name, feature_value, feature_type, metadata
                        FROM extracted_features
                        WHERE asset_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """
                    
                    cur.execute(query, (asset_id, limit))
                    rows = cur.fetchall()
                    
                    features = []
                    for row in rows:
                        metadata = json.loads(row[6]) if row[6] else {}
                        
                        features.append(ExtractedFeature(
                            timestamp=row[0],
                            asset_id=row[1],
                            sensor_id=row[2],
                            feature_name=row[3],
                            feature_value=float(row[4]),
                            feature_type=row[5],
                            metadata=metadata
                        ))
                    
                    return features
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des features: {e}", exc_info=True)
            return []
    
    def close(self):
        """Ferme le pool de connexions"""
        if self.pool:
            try:
                self.pool.closeall()
                logger.info("Pool de connexions TimescaleDB fermé")
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du pool: {e}", exc_info=True)
            finally:
                self.pool = None

