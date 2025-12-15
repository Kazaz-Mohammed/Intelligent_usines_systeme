"""
Service PostgreSQL pour la journalisation des prédictions RUL
"""
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

from app.config import settings
from app.models.rul_data import RULPredictionResult
from typing import Tuple

logger = logging.getLogger(__name__)


class PostgreSQLService:
    """Service pour interagir avec PostgreSQL pour la journalisation des prédictions RUL"""
    
    def __init__(self):
        logger.info("Initialisation PostgreSQL Service...")
        self.pool: Optional[ThreadedConnectionPool] = None
        try:
            self._initialize_pool()
            if self.pool:
                self._create_tables()
                logger.info("✓ PostgreSQL Service initialisé")
            else:
                logger.warning("PostgreSQL pool non initialisé, le service fonctionnera sans base de données")
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation PostgreSQL: {e}")
            logger.warning("Le service continuera sans base de données")
            self.pool = None
    
    def _initialize_pool(self):
        """Initialise le pool de connexions PostgreSQL"""
        try:
            logger.debug(f"Connexion à PostgreSQL: {settings.database_host}:{settings.database_port}/{settings.database_name}")
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.database_host,
                port=settings.database_port,
                database=settings.database_name,
                user=settings.database_user,
                password=settings.database_password,
                connect_timeout=5  # 5 second timeout
            )
            logger.info("Pool de connexions PostgreSQL initialisé avec succès")
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation du pool PostgreSQL: {e}")
            logger.warning("Le service continuera sans base de données")
            self.pool = None
    
    def get_connection(self):
        """Obtient une connexion du pool"""
        if not self.pool:
            raise ConnectionError("Le pool de connexions PostgreSQL n'est pas initialisé")
        try:
            return self.pool.getconn()
        except Exception as e:
            logger.error(f"Erreur lors de l'obtention d'une connexion du pool: {e}", exc_info=True)
            raise
    
    def put_connection(self, conn):
        """Remet une connexion dans le pool"""
        if self.pool and conn:
            self.pool.putconn(conn)
    
    def close(self):
        """Ferme le pool de connexions"""
        if self.pool:
            try:
                self.pool.closeall()
                logger.info("Pool de connexions PostgreSQL fermé")
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du pool: {e}", exc_info=True)
            finally:
                self.pool = None
    
    def _create_tables(self):
        """Crée la table rul_predictions si elle n'existe pas"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Create table if not exists
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS rul_predictions (
                    id SERIAL PRIMARY KEY,
                    asset_id VARCHAR(255) NOT NULL,
                    sensor_id VARCHAR(255),
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    rul_prediction DECIMAL(10, 2) NOT NULL,
                    confidence_interval_lower DECIMAL(10, 2),
                    confidence_interval_upper DECIMAL(10, 2),
                    confidence_level DECIMAL(3, 2) DEFAULT 0.95,
                    uncertainty DECIMAL(10, 2),
                    model_used VARCHAR(50) NOT NULL,
                    model_scores JSONB,
                    features JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(create_table_sql)
                
                # Add sensor_id column if it doesn't exist (migration for existing tables)
                try:
                    cursor.execute("""
                        DO $$ 
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name='rul_predictions' AND column_name='sensor_id'
                            ) THEN
                                ALTER TABLE rul_predictions ADD COLUMN sensor_id VARCHAR(255);
                            END IF;
                        END $$;
                    """)
                except Exception as e:
                    logger.debug(f"Column sensor_id check: {e}")
                
                # Create indexes (only if they don't exist)
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_rul_asset_id ON rul_predictions(asset_id);",
                    "CREATE INDEX IF NOT EXISTS idx_rul_sensor_id ON rul_predictions(sensor_id);",
                    "CREATE INDEX IF NOT EXISTS idx_rul_timestamp ON rul_predictions(timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_rul_model_used ON rul_predictions(model_used);",
                    "CREATE INDEX IF NOT EXISTS idx_rul_asset_timestamp ON rul_predictions(asset_id, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_rul_sensor_timestamp ON rul_predictions(sensor_id, timestamp DESC);"
                ]
                
                for index_sql in indexes:
                    try:
                        cursor.execute(index_sql)
                    except Exception as e:
                        logger.warning(f"Could not create index (may already exist): {e}")
                
            conn.commit()
            logger.info("Tables PostgreSQL créées/vérifiées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la création des tables: {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.put_connection(conn)
    
    def insert_rul_prediction(self, rul_result: RULPredictionResult) -> Optional[int]:
        """
        Insère une prédiction RUL dans la base de données
        
        Args:
            rul_result: Objet RULPredictionResult
        
        Returns:
            L'ID de la prédiction insérée ou None en cas d'erreur
        """
        # Use ACTUAL column names from the existing database table
        insert_sql = """
        INSERT INTO rul_predictions (
            asset_id, sensor_id, timestamp, rul_mean,
            rul_lower, rul_upper, confidence_interval,
            model_used, model_version, features, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        RETURNING id;
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        rul_result.asset_id,
                        getattr(rul_result, 'sensor_id', None),
                        rul_result.timestamp,
                        rul_result.rul_prediction,  # maps to rul_mean
                        rul_result.confidence_interval_lower,  # maps to rul_lower
                        rul_result.confidence_interval_upper,  # maps to rul_upper
                        rul_result.confidence_level,  # maps to confidence_interval
                        rul_result.model_used,
                        'v1.0',  # model_version
                        json.dumps(rul_result.features),
                        json.dumps(rul_result.metadata) if rul_result.metadata else None,
                    )
                )
                prediction_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Prédiction RUL {prediction_id} journalisée pour l'actif {rul_result.asset_id}")
            return prediction_id
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion de la prédiction RUL: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self.put_connection(conn)
    
    def get_rul_predictions(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_used: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Récupère les prédictions RUL de la base de données avec des filtres et pagination
        
        Args:
            asset_id: Filtrer par asset_id
            sensor_id: Filtrer par sensor_id
            start_date: Date de début
            end_date: Date de fin
            model_used: Filtrer par modèle utilisé
            limit: Nombre maximum de résultats
            offset: Offset pour la pagination
        
        Returns:
            Dict avec 'predictions' (liste), 'total', 'limit', 'offset', 'filters'
        """
        query_parts = []
        params = []
        
        if asset_id:
            query_parts.append("asset_id = %s")
            params.append(asset_id)
        if sensor_id:
            query_parts.append("sensor_id = %s")
            params.append(sensor_id)
        if start_date:
            query_parts.append("timestamp >= %s")
            params.append(start_date)
        if end_date:
            query_parts.append("timestamp <= %s")
            params.append(end_date)
        if model_used:
            query_parts.append("model_used = %s")
            params.append(model_used)
        
        where_clause = "WHERE " + " AND ".join(query_parts) if query_parts else ""
        
        count_sql = f"SELECT COUNT(*) FROM rul_predictions {where_clause};"
        
        # Use actual column names and alias to expected API format
        select_sql = f"""
        SELECT 
            id, asset_id, sensor_id, timestamp, 
            rul_mean as rul_prediction,
            rul_lower as confidence_interval_lower, 
            rul_upper as confidence_interval_upper, 
            confidence_interval as confidence_level,
            model_used, model_version, features, metadata
        FROM rul_predictions
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT %s OFFSET %s;
        """
        params.extend([limit, offset])
        
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(count_sql, params[:-2])  # Exclure limit et offset pour le count
                total_count = cursor.fetchone()['count']
                
                cursor.execute(select_sql, params)
                predictions_data = cursor.fetchall()
            
            # Convertir les résultats en dictionnaires sérialisables
            predictions_results = []
            for row in predictions_data:
                try:
                    row_dict = dict(row)
                    # Convertir les champs JSONB
                    if row_dict.get('model_scores'):
                        row_dict['model_scores'] = json.loads(row_dict['model_scores']) if isinstance(row_dict['model_scores'], str) else row_dict['model_scores']
                    if row_dict.get('features'):
                        row_dict['features'] = json.loads(row_dict['features']) if isinstance(row_dict['features'], str) else row_dict['features']
                    if row_dict.get('metadata'):
                        row_dict['metadata'] = json.loads(row_dict['metadata']) if isinstance(row_dict['metadata'], str) else row_dict['metadata']
                    
                    # Assurer que timestamp est un datetime avec timezone
                    if isinstance(row_dict.get('timestamp'), datetime) and row_dict['timestamp'].tzinfo is None:
                        row_dict['timestamp'] = row_dict['timestamp'].replace(tzinfo=timezone.utc)
                    if isinstance(row_dict.get('created_at'), datetime) and row_dict['created_at'].tzinfo is None:
                        row_dict['created_at'] = row_dict['created_at'].replace(tzinfo=timezone.utc)
                    
                    # Return the dict directly - column names already match the API response format
                    predictions_results.append(row_dict)
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion d'une ligne de prédiction: {e} - Données: {row}", exc_info=True)
                    predictions_results.append(dict(row))  # Retourner le dict brut en cas d'erreur
            
            return {
                "predictions": predictions_results,
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "filters": {
                    "asset_id": asset_id,
                    "sensor_id": sensor_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "model_used": model_used
                }
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prédictions RUL: {e}", exc_info=True)
            raise
        finally:
            if conn:
                self.put_connection(conn)
    
    def get_rul_prediction_count(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_used: Optional[str] = None
    ) -> int:
        """
        Compte le nombre de prédictions RUL avec des filtres
        
        Args:
            asset_id: Filtrer par asset_id
            sensor_id: Filtrer par sensor_id
            start_date: Date de début
            end_date: Date de fin
            model_used: Filtrer par modèle utilisé
        
        Returns:
            Nombre de prédictions
        """
        query_parts = []
        params = []
        
        if asset_id:
            query_parts.append("asset_id = %s")
            params.append(asset_id)
        if sensor_id:
            query_parts.append("sensor_id = %s")
            params.append(sensor_id)
        if start_date:
            query_parts.append("timestamp >= %s")
            params.append(start_date)
        if end_date:
            query_parts.append("timestamp <= %s")
            params.append(end_date)
        if model_used:
            query_parts.append("model_used = %s")
            params.append(model_used)
        
        where_clause = "WHERE " + " AND ".join(query_parts) if query_parts else ""
        count_sql = f"SELECT COUNT(*) FROM rul_predictions {where_clause};"
        
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(count_sql, params)
                count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Erreur lors du comptage des prédictions RUL: {e}", exc_info=True)
            raise
        finally:
            if conn:
                self.put_connection(conn)
    
    def get_latest_rul_prediction(
        self,
        asset_id: str,
        sensor_id: Optional[str] = None
    ) -> Optional[RULPredictionResult]:
        """
        Récupère la dernière prédiction RUL pour un actif
        
        Args:
            asset_id: ID de l'actif
            sensor_id: ID du capteur (optionnel)
        
        Returns:
            Dernière prédiction RUL ou None
        """
        query_parts = ["asset_id = %s"]
        params = [asset_id]
        
        if sensor_id:
            query_parts.append("sensor_id = %s")
            params.append(sensor_id)
        
        where_clause = "WHERE " + " AND ".join(query_parts)
        
        # Use actual column names and alias to expected API format
        select_sql = f"""
        SELECT 
            id, asset_id, sensor_id, timestamp, 
            rul_mean as rul_prediction,
            rul_lower as confidence_interval_lower, 
            rul_upper as confidence_interval_upper, 
            confidence_interval as confidence_level,
            model_used, model_version, features, metadata
        FROM rul_predictions
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_sql, params)
                row = cursor.fetchone()
            
            if row:
                row_dict = dict(row)
                # Convertir les champs JSONB
                row_dict['model_scores'] = json.loads(row_dict['model_scores']) if isinstance(row_dict['model_scores'], str) else row_dict['model_scores']
                row_dict['features'] = json.loads(row_dict['features']) if isinstance(row_dict['features'], str) else row_dict['features']
                row_dict['metadata'] = json.loads(row_dict['metadata']) if isinstance(row_dict['metadata'], str) else row_dict['metadata']
                
                # Assurer que timestamp est un datetime avec timezone
                if isinstance(row_dict['timestamp'], datetime) and row_dict['timestamp'].tzinfo is None:
                    row_dict['timestamp'] = row_dict['timestamp'].replace(tzinfo=timezone.utc)
                
                # Le modèle RULPredictionResult utilise déjà rul_prediction, confidence_interval_lower, etc.
                return RULPredictionResult(**row_dict)
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la dernière prédiction: {e}", exc_info=True)
            return None
        finally:
            if conn:
                self.put_connection(conn)

