"""
Service d'accès à TimescaleDB
"""
import logging
from typing import List, Optional
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager

from app.config import settings
from app.models.sensor_data import PreprocessedData, WindowedData

logger = logging.getLogger(__name__)


class TimescaleDBService:
    """Service pour stocker les données dans TimescaleDB"""
    
    def __init__(self):
        self.pool: Optional[SimpleConnectionPool] = None
        self._create_pool()
    
    def _create_pool(self):
        """Crée le pool de connexions"""
        try:
            import sys
            import os
            # Forcer l'encodage UTF-8 pour éviter les problèmes sur Windows
            if sys.platform == 'win32':
                os.environ['PGCLIENTENCODING'] = 'UTF8'
            
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.database_host,
                port=settings.database_port,
                database=settings.database_name,
                user=settings.database_user,
                password=settings.database_password,
                connect_timeout=5
            )
            logger.info(f"Pool de connexions TimescaleDB créé: {settings.database_host}:{settings.database_port}/{settings.database_name}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du pool: {e}", exc_info=True)
            logger.error(f"Configuration: host={settings.database_host}, port={settings.database_port}, db={settings.database_name}, user={settings.database_user}")
            self.pool = None
            # Ne pas lancer l'exception, laisser les tests gérer
    
    @contextmanager
    def get_connection(self):
        """Context manager pour obtenir une connexion du pool"""
        if not self.pool:
            raise RuntimeError("Pool de connexions non initialisé")
        
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def insert_preprocessed_data(self, data: PreprocessedData):
        """
        Insère une donnée prétraitée dans TimescaleDB
        
        Args:
            data: Donnée prétraitée
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO preprocessed_sensor_data 
                        (time, asset_id, sensor_id, value, unit, quality, source_type, preprocessing_metadata, frequency_analysis)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (time, asset_id, sensor_id) DO NOTHING
                    """
                    
                    import json
                    metadata_json = json.dumps(data.preprocessing_metadata) if data.preprocessing_metadata else None
                    frequency_json = json.dumps(data.frequency_analysis) if data.frequency_analysis else None
                    
                    cur.execute(query, (
                        data.timestamp,
                        data.asset_id,
                        data.sensor_id,
                        data.value,
                        data.unit,
                        data.quality,
                        data.source_type,
                        metadata_json,
                        frequency_json
                    ))
                    
                    conn.commit()
                    logger.debug(f"Donnée prétraitée insérée: asset={data.asset_id}, sensor={data.sensor_id}")
                    
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion: {e}", exc_info=True)
            raise
    
    def insert_preprocessed_batch(self, data_list: List[PreprocessedData]):
        """
        Insère un batch de données prétraitées
        
        Args:
            data_list: Liste de données prétraitées
        """
        if not data_list:
            return
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO preprocessed_sensor_data 
                        (time, asset_id, sensor_id, value, unit, quality, source_type, preprocessing_metadata, frequency_analysis)
                        VALUES %s
                        ON CONFLICT DO NOTHING
                    """
                    
                    import json
                    values = []
                    for data in data_list:
                        metadata_json = json.dumps(data.preprocessing_metadata) if data.preprocessing_metadata else None
                        frequency_json = json.dumps(data.frequency_analysis) if data.frequency_analysis else None
                        
                        values.append((
                            data.timestamp,
                            data.asset_id,
                            data.sensor_id,
                            data.value,
                            data.unit,
                            data.quality,
                            data.source_type,
                            metadata_json,
                            frequency_json
                        ))
                    
                    execute_values(cur, query, values)
                    conn.commit()
                    logger.info(f"Batch de {len(data_list)} données prétraitées inséré")
                    
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion batch: {e}", exc_info=True)
            raise
    
    def insert_windowed_data(self, window: WindowedData):
        """
        Insère une fenêtre dans TimescaleDB
        
        Args:
            window: Données fenêtrées
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO windowed_sensor_data 
                        (window_id, asset_id, start_time, end_time, sensor_data, metadata)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (window_id) DO UPDATE SET
                            sensor_data = EXCLUDED.sensor_data,
                            metadata = EXCLUDED.metadata
                    """
                    
                    import json
                    sensor_data_json = json.dumps(window.sensor_data)
                    metadata_json = json.dumps(window.metadata) if window.metadata else None
                    
                    cur.execute(query, (
                        window.window_id,
                        window.asset_id,
                        window.start_timestamp,
                        window.end_timestamp,
                        sensor_data_json,
                        metadata_json
                    ))
                    
                    conn.commit()
                    logger.debug(f"Fenêtre insérée: window_id={window.window_id}, asset={window.asset_id}")
                    
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion de fenêtre: {e}", exc_info=True)
            raise
    
    def insert_windows_batch(self, windows: List[WindowedData]):
        """
        Insère un batch de fenêtres
        
        Args:
            windows: Liste de fenêtres
        """
        if not windows:
            return
        
        for window in windows:
            self.insert_windowed_data(window)
        
        logger.info(f"Batch de {len(windows)} fenêtres inséré")
    
    def close(self):
        """Ferme le pool de connexions"""
        if self.pool:
            self.pool.closeall()
            logger.info("Pool de connexions TimescaleDB fermé")

