"""
Tests pour le service de rééchantillonnage
"""
import pytest
from datetime import datetime, timedelta
from app.services.resampling_service import ResamplingService
from app.models.sensor_data import PreprocessedData


class TestResamplingService:
    """Tests pour ResamplingService"""
    
    @pytest.fixture
    def resampling_service(self):
        return ResamplingService()
    
    @pytest.fixture
    def sample_data(self):
        """Créer des données d'exemple"""
        base_time = datetime.utcnow()
        data = []
        for i in range(10):
            data.append(PreprocessedData(
                timestamp=base_time + timedelta(seconds=i*2),  # 2 secondes entre chaque point
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + i * 0.1,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            ))
        return data
    
    def test_resample_single_sensor_no_resampling(self, resampling_service, sample_data):
        """Test sans rééchantillonnage (resampling_rate = None)"""
        result = resampling_service.resample_single_sensor(sample_data, target_rate=None)
        
        assert len(result) == len(sample_data)
        assert result[0].value == sample_data[0].value
    
    def test_resample_single_sensor_with_rate(self, resampling_service, sample_data):
        """Test avec rééchantillonnage à 1 Hz"""
        result = resampling_service.resample_single_sensor(sample_data, target_rate=1.0)
        
        assert len(result) > 0
        # Vérifier que les métadonnées indiquent le rééchantillonnage
        assert result[0].preprocessing_metadata.get("resampled", False) is True
    
    def test_synchronize_multiple_sensors(self, resampling_service):
        """Test synchronisation de plusieurs capteurs"""
        base_time = datetime.utcnow()
        
        # Capteur 1 : 1 point par seconde
        sensor1_data = [
            PreprocessedData(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + i,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            )
            for i in range(10)
        ]
        
        # Capteur 2 : 1 point toutes les 2 secondes
        sensor2_data = [
            PreprocessedData(
                timestamp=base_time + timedelta(seconds=i*2),
                asset_id="ASSET001",
                sensor_id="SENSOR002",
                value=100.0 + i,
                unit="RPM",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            )
            for i in range(5)
        ]
        
        sensor_dict = {
            "SENSOR001": sensor1_data,
            "SENSOR002": sensor2_data
        }
        
        result = resampling_service.synchronize_multiple_sensors(sensor_dict, target_rate=1.0)
        
        assert "SENSOR001" in result
        assert "SENSOR002" in result
        # Les deux capteurs devraient avoir le même nombre de points après synchronisation
        assert len(result["SENSOR001"]) == len(result["SENSOR002"])

