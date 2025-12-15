"""
Service pour récupérer les informations sur les actifs
"""
import logging
from typing import Optional
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import sys
import os

from app.config import settings
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AssetService:
    """Service pour récupérer les informations sur les actifs depuis la base de données"""
    
    def __init__(self):
        logger.info("Initialisation Asset Service...")
        self.pool: Optional[SimpleConnectionPool] = None
        try:
            self._create_pool()
            if self.pool:
                logger.info("✓ Asset Service initialisé")
            else:
                logger.warning("Asset Service pool non initialisé, le service fonctionnera sans base de données")
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation Asset Service: {e}")
            logger.warning("Le service continuera sans base de données")
            self.pool = None
    
    def _create_pool(self):
        """Crée le pool de connexions"""
        try:
            # Forcer l'encodage UTF-8 pour éviter les problèmes sur Windows
            if sys.platform == 'win32':
                os.environ['PGCLIENTENCODING'] = 'UTF8'
            
            logger.debug(f"Connexion à TimescaleDB pour AssetService: {settings.database_host}:{settings.database_port}/{settings.database_name}")
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
            logger.info(f"Pool de connexions TimescaleDB créé pour AssetService: {settings.database_host}:{settings.database_port}/{settings.database_name}")
        except Exception as e:
            logger.warning(f"Erreur lors de la création du pool AssetService: {e}")
            logger.warning("Le service continuera sans base de données")
            self.pool = None
    
    def get_asset_type(self, asset_id: str) -> Optional[str]:
        """
        Récupère le type d'actif depuis la base de données
        
        Args:
            asset_id: ID de l'actif
        
        Returns:
            Type d'actif ou None
        """
        if not self.pool:
            return None
        
        try:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    query = "SELECT type FROM assets WHERE id = %s"
                    cur.execute(query, (asset_id,))
                    row = cur.fetchone()
                    
                    if row:
                        return row[0]
                    else:
                        logger.warning(f"Actif {asset_id} non trouvé dans la base de données")
                        return None
            finally:
                self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du type d'actif: {e}", exc_info=True)
            return None
    
    def get_asset_info(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations complètes sur un actif
        
        Args:
            asset_id: ID de l'actif
        
        Returns:
            Dictionnaire avec les informations de l'actif ou None
        """
        if not self.pool:
            return None
        
        try:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, name, type, location, line_id, criticity, manufacturer, installation_date, metadata
                        FROM assets
                        WHERE id = %s
                    """
                    cur.execute(query, (asset_id,))
                    row = cur.fetchone()
                    
                    if row:
                        import json
                        return {
                            "id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "location": row[3],
                            "line_id": row[4],
                            "criticity": row[5],
                            "manufacturer": row[6],
                            "installation_date": row[7].isoformat() if row[7] else None,
                            "metadata": json.loads(row[8]) if row[8] else None
                        }
                    else:
                        return None
            finally:
                self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations de l'actif: {e}", exc_info=True)
            return None
    
    def close(self):
        """Ferme le pool de connexions"""
        if self.pool:
            self.pool.closeall()
            logger.info("Pool de connexions AssetService fermé")

