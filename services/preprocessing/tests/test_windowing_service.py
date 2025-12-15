"""
Tests pour le service de fenêtrage
"""
import pytest
from datetime import datetime, timedelta

from app.services.windowing_service import WindowingService
from app.models.sensor_data import PreprocessedData


class TestWindowingService:
    """Tests pour WindowingService"""
    
    @pytest.fixture
    def windowing_service(self):
        return WindowingService()
    
    @pytest.fixture
    def sample_data(self):
        """Créer des données d'exemple"""
        base_time = datetime.utcnow()
        data = []
        
        for i in range(200):
            data.append(PreprocessedData(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + i * 0.01,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            ))
        return data
    
    def test_create_windows_single_sensor(self, windowing_service, sample_data):
        """Test création de fenêtres pour un seul capteur"""
        windows = windowing_service.create_windows_from_single_sensor(
            sample_data,
            window_size=50,
            overlap=0.5
        )
        
        assert len(windows) > 0
        assert windows[0].window_id is not None
        assert windows[0].asset_id == "ASSET001"
        assert len(windows[0].sensor_data["SENSOR001"]) == 50
        assert "window_size" in windows[0].metadata
    
    def test_create_windows_multiple_sensors(self, windowing_service):
        """Test création de fenêtres pour plusieurs capteurs"""
        base_time = datetime.utcnow()
        
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
            for i in range(100)
        ]
        
        sensor2_data = [
            PreprocessedData(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR002",
                value=100.0 + i,
                unit="RPM",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            )
            for i in range(100)
        ]
        
        sensor_dict = {
            "SENSOR001": sensor1_data,
            "SENSOR002": sensor2_data
        }
        
        windows = windowing_service.create_windows(
            sensor_dict,
            window_size=50,
            overlap=0.5
        )
        
        assert len(windows) > 0
        assert "SENSOR001" in windows[0].sensor_data
        assert "SENSOR002" in windows[0].sensor_data
        assert len(windows[0].sensor_data["SENSOR001"]) == 50
        assert len(windows[0].sensor_data["SENSOR002"]) == 50
    
    def test_window_overlap(self, windowing_service, sample_data):
        """Test que le chevauchement fonctionne correctement"""
        windows_no_overlap = windowing_service.create_windows_from_single_sensor(
            sample_data,
            window_size=50,
            overlap=0.0
        )
        
        windows_with_overlap = windowing_service.create_windows_from_single_sensor(
            sample_data,
            window_size=50,
            overlap=0.5
        )
        
        # Avec overlap, on devrait avoir plus de fenêtres
        assert len(windows_with_overlap) > len(windows_no_overlap)
    
    def test_window_metadata(self, windowing_service, sample_data):
        """Test que les métadonnées de fenêtres sont correctes"""
        windows = windowing_service.create_windows_from_single_sensor(
            sample_data,
            window_size=50,
            overlap=0.5
        )
        
        assert len(windows) > 0
        window = windows[0]
        assert window.metadata["window_size"] == 50
        assert window.metadata["overlap"] == 0.5
        assert "created_at" in window.metadata

