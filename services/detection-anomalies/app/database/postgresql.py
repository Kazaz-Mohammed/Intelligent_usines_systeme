"""
Service de base de données PostgreSQL pour la journalisation des anomalies
"""
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)


class PostgreSQLService:
    """Service pour gérer la connexion et les opérations PostgreSQL"""
    
    def __init__(self):
        """Initialise le service PostgreSQL"""
        self.connection_pool: Optional[ThreadedConnectionPool] = None
        self._init_connection_pool()
        self._create_tables()
    
    def _init_connection_pool(self):
        """Initialise le pool de connexions"""
        try:
            self.connection_pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.database_host,
                port=settings.database_port,
                database=settings.database_name,
                user=settings.database_user,
                password=settings.database_password
            )
            logger.info(f"Pool de connexions PostgreSQL initialisé: {settings.database_host}:{settings.database_port}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du pool PostgreSQL: {e}", exc_info=True)
            self.connection_pool = None
    
    @contextmanager
    def get_connection(self):
        """Context manager pour obtenir une connexion du pool"""
        if not self.connection_pool:
            raise RuntimeError("Pool de connexions PostgreSQL non initialisé")
        
        conn = self.connection_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur lors de l'exécution de la requête: {e}", exc_info=True)
            raise
        finally:
            self.connection_pool.putconn(conn)
    
    def _create_tables(self):
        """Crée les tables nécessaires si elles n'existent pas"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS anomaly_detections (
            id SERIAL PRIMARY KEY,
            asset_id VARCHAR(255) NOT NULL,
            sensor_id VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            final_score DECIMAL(5, 4) NOT NULL,
            is_anomaly BOOLEAN NOT NULL,
            criticality VARCHAR(20) NOT NULL,
            scores JSONB NOT NULL,
            features JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_asset_id (asset_id),
            INDEX idx_sensor_id (sensor_id),
            INDEX idx_timestamp (timestamp),
            INDEX idx_is_anomaly (is_anomaly),
            INDEX idx_criticality (criticality)
        );
        
        CREATE INDEX IF NOT EXISTS idx_asset_timestamp ON anomaly_detections(asset_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON anomaly_detections(sensor_id, timestamp DESC);
        """
        
        # Note: PostgreSQL n'utilise pas INDEX dans CREATE TABLE comme MySQL
        # On doit créer les index séparément
        create_table_sql_fixed = """
        CREATE TABLE IF NOT EXISTS anomaly_detections (
            id SERIAL PRIMARY KEY,
            asset_id VARCHAR(255) NOT NULL,
            sensor_id VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            final_score DECIMAL(5, 4) NOT NULL,
            is_anomaly BOOLEAN NOT NULL,
            criticality VARCHAR(20) NOT NULL,
            scores JSONB NOT NULL,
            features JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_asset_id ON anomaly_detections(asset_id);
        CREATE INDEX IF NOT EXISTS idx_sensor_id ON anomaly_detections(sensor_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON anomaly_detections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_is_anomaly ON anomaly_detections(is_anomaly);
        CREATE INDEX IF NOT EXISTS idx_criticality ON anomaly_detections(criticality);
        CREATE INDEX IF NOT EXISTS idx_asset_timestamp ON anomaly_detections(asset_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON anomaly_detections(sensor_id, timestamp DESC);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Exécuter chaque instruction séparément
                    for statement in create_table_sql_fixed.split(';'):
                        statement = statement.strip()
                        if statement:
                            cursor.execute(statement)
                    logger.info("Tables PostgreSQL créées/vérifiées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création des tables: {e}", exc_info=True)
            # Ne pas lever l'exception pour permettre au service de démarrer même si la DB n'est pas disponible
    
    def insert_anomaly(self, anomaly_data: Dict[str, Any]) -> Optional[int]:
        """
        Insère une anomalie détectée dans la base de données
        
        Args:
            anomaly_data: Données de l'anomalie (AnomalyDetectionResult ou dict)
        
        Returns:
            ID de l'anomalie insérée ou None en cas d'erreur
        """
        try:
            # Extraire les données
            if hasattr(anomaly_data, 'model_dump'):
                # Pydantic model
                data = anomaly_data.model_dump()
            else:
                data = anomaly_data
            
            # Préparer les données pour l'insertion
            asset_id = data.get('asset_id')
            sensor_id = data.get('sensor_id')
            timestamp = data.get('timestamp')
            final_score = data.get('final_score')
            is_anomaly = data.get('is_anomaly')
            criticality = data.get('criticality')
            scores = data.get('scores', [])
            features = data.get('features', {})
            metadata = data.get('metadata', {})
            
            # Convertir timestamp si nécessaire
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            # Convertir les scores en JSON
            scores_json = []
            if isinstance(scores, list):
                for score in scores:
                    if hasattr(score, 'model_dump'):
                        scores_json.append(score.model_dump())
                    else:
                        scores_json.append(score)
            
            insert_sql = """
            INSERT INTO anomaly_detections (
                asset_id, sensor_id, timestamp, final_score, is_anomaly,
                criticality, scores, features, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            RETURNING id
            """
            
            # Préparer les valeurs
            criticality_str = str(criticality.value) if hasattr(criticality, 'value') else str(criticality)
            scores_json_str = json.dumps(scores_json) if scores_json else '[]'
            features_json_str = json.dumps(features) if features else '{}'
            metadata_json_str = json.dumps(metadata) if metadata else None
            
            logger.debug(f"Inserting anomaly: asset={asset_id}, sensor={sensor_id}, score={final_score}, criticality={criticality_str}")
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        insert_sql,
                        (
                            asset_id,
                            sensor_id,
                            timestamp,
                            float(final_score),
                            bool(is_anomaly),
                            criticality_str,
                            scores_json_str,
                            features_json_str,
                            metadata_json_str
                        )
                    )
                    result = cursor.fetchone()
                    anomaly_id = result[0] if result else None
                    if anomaly_id:
                        logger.info(f"Anomaly inserted successfully with ID: {anomaly_id}")
                    else:
                        logger.warning(f"Insert executed but no ID returned")
                    return anomaly_id
                    
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion de l'anomalie: {e}", exc_info=True)
            # Re-raise to let caller know about the error
            raise
    
    def get_anomalies(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        is_anomaly: Optional[bool] = None,
        criticality: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Récupère les anomalies depuis la base de données
        
        Args:
            asset_id: Filtrer par asset_id
            sensor_id: Filtrer par sensor_id
            start_date: Date de début
            end_date: Date de fin
            is_anomaly: Filtrer par is_anomaly
            criticality: Filtrer par criticité
            limit: Nombre maximum de résultats
            offset: Offset pour la pagination
        
        Returns:
            Liste des anomalies
        """
        try:
            conditions = []
            params = []
            param_counter = 1
            
            if asset_id:
                conditions.append("asset_id = %s")
                params.append(asset_id)
            
            if sensor_id:
                conditions.append("sensor_id = %s")
                params.append(sensor_id)
            
            if start_date:
                conditions.append("timestamp >= %s")
                params.append(start_date)
            
            if end_date:
                conditions.append("timestamp <= %s")
                params.append(end_date)
            
            if is_anomaly is not None:
                conditions.append("is_anomaly = %s")
                params.append(is_anomaly)
            
            if criticality:
                conditions.append("criticality = %s")
                params.append(criticality)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query_sql = f"""
            SELECT 
                id, asset_id, sensor_id, timestamp, final_score, is_anomaly,
                criticality, scores, features, metadata, created_at
            FROM anomaly_detections
            WHERE {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query_sql, params)
                    rows = cursor.fetchall()
                    
                    # Convertir en liste de dictionnaires
                    anomalies = []
                    for row in rows:
                        anomaly = dict(row)
                        # Convertir les timestamps en ISO format
                        if anomaly.get('timestamp'):
                            anomaly['timestamp'] = anomaly['timestamp'].isoformat()
                        if anomaly.get('created_at'):
                            anomaly['created_at'] = anomaly['created_at'].isoformat()
                        anomalies.append(anomaly)
                    
                    return anomalies
                    
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des anomalies: {e}", exc_info=True)
            return []
    
    def get_anomaly_count(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        is_anomaly: Optional[bool] = None,
        criticality: Optional[str] = None
    ) -> int:
        """
        Compte le nombre d'anomalies avec les filtres donnés
        
        Args:
            asset_id: Filtrer par asset_id
            sensor_id: Filtrer par sensor_id
            start_date: Date de début
            end_date: Date de fin
            is_anomaly: Filtrer par is_anomaly
            criticality: Filtrer par criticité
        
        Returns:
            Nombre d'anomalies
        """
        try:
            conditions = []
            params = []
            param_counter = 1
            
            if asset_id:
                conditions.append("asset_id = %s")
                params.append(asset_id)
            
            if sensor_id:
                conditions.append("sensor_id = %s")
                params.append(sensor_id)
            
            if start_date:
                conditions.append("timestamp >= %s")
                params.append(start_date)
            
            if end_date:
                conditions.append("timestamp <= %s")
                params.append(end_date)
            
            if is_anomaly is not None:
                conditions.append("is_anomaly = %s")
                params.append(is_anomaly)
            
            if criticality:
                conditions.append("criticality = %s")
                params.append(criticality)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query_sql = f"""
            SELECT COUNT(*) as count
            FROM anomaly_detections
            WHERE {where_clause}
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query_sql, params)
                    result = cursor.fetchone()
                    return result[0] if result else 0
                    
        except Exception as e:
            logger.error(f"Erreur lors du comptage des anomalies: {e}", exc_info=True)
            return 0
    
    def close(self):
        """Ferme le pool de connexions"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None
            logger.info("Pool de connexions PostgreSQL fermé")
