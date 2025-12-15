"""
Tests pour le service d'analyse fréquentielle
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from app.services.frequency_analysis_service import FrequencyAnalysisService
from app.models.sensor_data import PreprocessedData


class TestFrequencyAnalysisService:
    """Tests pour FrequencyAnalysisService"""
    
    @pytest.fixture
    def frequency_service(self):
        return FrequencyAnalysisService()
    
    @pytest.fixture
    def sinusoidal_data(self):
        """Créer des données sinusoïdales (signal périodique)"""
        base_time = datetime.utcnow()
        data = []
        sampling_rate = 100  # 100 Hz
        frequency = 10  # 10 Hz
        
        for i in range(200):
            # Signal sinusoïdal à 10 Hz
            value = 25.0 + 5 * np.sin(2 * np.pi * frequency * i / sampling_rate)
            
            data.append(PreprocessedData(
                timestamp=base_time + timedelta(seconds=i/sampling_rate),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=value,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            ))
        return data
    
    def test_fft_analysis(self, frequency_service, sinusoidal_data):
        """Test analyse FFT"""
        result = frequency_service.analyze_frequency(sinusoidal_data, method="fft")
        
        assert result is not None
        assert result.get("method") == "fft"
        assert "dominant_frequency" in result
        assert "dominant_magnitude" in result
        assert "total_power" in result
        assert "top_frequencies" in result
        assert "frequency_bands" in result
        
        # La fréquence dominante devrait être proche de 10 Hz
        dominant_freq = result["dominant_frequency"]
        assert 8 <= dominant_freq <= 12  # Tolérance
    
    def test_stft_analysis(self, frequency_service, sinusoidal_data):
        """Test analyse STFT"""
        result = frequency_service.analyze_frequency(
            sinusoidal_data,
            method="stft",
            window_size=64
        )
        
        assert result is not None
        assert result.get("method") == "stft"
        assert "dominant_frequency" in result
        assert "window_size" in result
        assert "frequency_bands" in result
    
    def test_frequency_analysis_disabled(self, frequency_service):
        """Test que l'analyse est désactivée si enable_frequency_analysis = False"""
        frequency_service.enable_frequency_analysis = False
        
        data = [
            PreprocessedData(
                timestamp=datetime.utcnow(),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.5,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            )
        ]
        
        result = frequency_service.analyze_frequency(data)
        
        assert result == {}
    
    def test_add_frequency_analysis_to_data(self, frequency_service, sinusoidal_data):
        """Test ajout des résultats d'analyse aux données"""
        analysis_result = {
            "method": "fft",
            "dominant_frequency": 10.0,
            "total_power": 100.0
        }
        
        result = frequency_service.add_frequency_analysis_to_data(
            sinusoidal_data,
            analysis_result
        )
        
        assert len(result) == len(sinusoidal_data)
        assert result[0].frequency_analysis is not None
        assert result[0].frequency_analysis["method"] == "fft"

