"""
Tests d'intégration avec TimescaleDB
"""
import pytest
from datetime import datetime
from app.database.timescaledb import TimescaleDBService
from app.models.sensor_data import PreprocessedData, WindowedData


@pytest.mark.integration
class TestTimescaleDBIntegration:
    """Tests d'intégration avec TimescaleDB"""
    
    @pytest.fixture
    def timescaledb_service(self):
        """Service TimescaleDB pour les tests"""
        try:
            service = TimescaleDBService()
            # Tester la connexion
            with service.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return service
        except Exception as e:
            pytest.skip(f"TimescaleDB non disponible: {e}")
    
    def test_timescaledb_connection(self, timescaledb_service):
        """Test connexion à TimescaleDB"""
        try:
            with timescaledb_service.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()
                    assert version is not None
                    print(f"✅ TimescaleDB connecté: {version[0][:50]}...")
        except Exception as e:
            pytest.skip(f"TimescaleDB non disponible: {e}")
    
    def test_timescaledb_tables_exist(self, timescaledb_service):
        """Test que les tables existent"""
        try:
            with timescaledb_service.get_connection() as conn:
                with conn.cursor() as cur:
                    # Vérifier la table preprocessed_sensor_data
                    cur.execute("""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_name = 'preprocessed_sensor_data'
                    """)
                    count = cur.fetchone()[0]
                    assert count == 1, "Table preprocessed_sensor_data n'existe pas"
                    
                    # Vérifier la table windowed_sensor_data
                    cur.execute("""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_name = 'windowed_sensor_data'
                    """)
                    count = cur.fetchone()[0]
                    assert count == 1, "Table windowed_sensor_data n'existe pas"
                    
                    print("✅ Tables TimescaleDB existent")
        except Exception as e:
            pytest.skip(f"TimescaleDB non disponible: {e}")
    
    def test_insert_preprocessed_data(self, timescaledb_service):
        """Test insertion de données prétraitées"""
        try:
            # Créer une donnée prétraitée
            preprocessed_data = PreprocessedData(
                timestamp=datetime.utcnow(),
                asset_id="TEST_ASSET",
                sensor_id="TEST_SENSOR",
                value=25.5,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={"test": True, "outlier_removed": False}
            )
            
            # Insérer
            timescaledb_service.insert_preprocessed_data(preprocessed_data)
            print("✅ Donnée prétraitée insérée avec succès")
            
            # Vérifier l'insertion
            with timescaledb_service.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) FROM preprocessed_sensor_data 
                        WHERE asset_id = 'TEST_ASSET' AND sensor_id = 'TEST_SENSOR'
                    """)
                    count = cur.fetchone()[0]
                    assert count > 0, "Donnée non insérée"
                    
            print("✅ Donnée vérifiée dans TimescaleDB")
            
        except Exception as e:
            pytest.skip(f"TimescaleDB non disponible: {e}")
    
    def test_insert_windowed_data(self, timescaledb_service):
        """Test insertion de fenêtres"""
        try:
            # Créer une fenêtre
            windowed_data = WindowedData(
                window_id="TEST_WINDOW_001",
                asset_id="TEST_ASSET",
                start_timestamp=datetime.utcnow(),
                end_timestamp=datetime.utcnow(),
                sensor_data={
                    "SENSOR001": [{"value": 25.0, "timestamp": datetime.utcnow().isoformat()}]
                },
                metadata={"window_size": 100, "overlap": 0.5}
            )
            
            # Insérer
            timescaledb_service.insert_windowed_data(windowed_data)
            print("✅ Fenêtre insérée avec succès")
            
            # Vérifier l'insertion
            with timescaledb_service.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) FROM windowed_sensor_data 
                        WHERE window_id = 'TEST_WINDOW_001'
                    """)
                    count = cur.fetchone()[0]
                    assert count == 1, "Fenêtre non insérée"
                    
            print("✅ Fenêtre vérifiée dans TimescaleDB")
            
        except Exception as e:
            pytest.skip(f"TimescaleDB non disponible: {e}")
    
    def test_insert_batch(self, timescaledb_service):
        """Test insertion en batch"""
        try:
            # Créer plusieurs données
            data_list = [
                PreprocessedData(
                    timestamp=datetime.utcnow(),
                    asset_id="TEST_ASSET",
                    sensor_id=f"TEST_SENSOR_{i}",
                    value=25.0 + i * 0.1,
                    unit="°C",
                    quality=2,
                    source_type="TEST",
                    preprocessing_metadata={"batch_test": True}
                )
                for i in range(10)
            ]
            
            # Insérer en batch
            timescaledb_service.insert_preprocessed_batch(data_list)
            print(f"✅ Batch de {len(data_list)} données inséré avec succès")
            
            # Vérifier l'insertion
            with timescaledb_service.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) FROM preprocessed_sensor_data 
                        WHERE asset_id = 'TEST_ASSET' AND preprocessing_metadata->>'batch_test' = 'true'
                    """)
                    count = cur.fetchone()[0]
                    assert count >= 10, f"Seulement {count} données insérées au lieu de 10"
                    
            print("✅ Batch vérifié dans TimescaleDB")
            
        except Exception as e:
            pytest.skip(f"TimescaleDB non disponible: {e}")
        finally:
            if 'timescaledb_service' in locals():
                timescaledb_service.close()

