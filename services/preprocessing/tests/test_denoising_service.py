"""
Tests pour le service de débruitage
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from app.services.denoising_service import DenoisingService
from app.models.sensor_data import PreprocessedData


class TestDenoisingService:
    """Tests pour DenoisingService"""
    
    @pytest.fixture
    def denoising_service(self):
        return DenoisingService()
    
    @pytest.fixture
    def noisy_data(self):
        """Créer des données avec du bruit"""
        base_time = datetime.utcnow()
        data = []
        base_value = 25.0
        
        for i in range(50):
            # Signal sinusoïdal avec bruit
            signal = base_value + 2 * np.sin(2 * np.pi * i / 10)
            noise = np.random.randn() * 0.5
            value = signal + noise
            
            data.append(PreprocessedData(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=value,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={}
            ))
        return data
    
    def test_denoise_single_sensor_butterworth(self, denoising_service, noisy_data):
        """Test débruitage avec filtre Butterworth"""
        # Utiliser une fréquence de coupure raisonnable (moitié de la fréquence d'échantillonnage)
        # Les données ont un intervalle de 1 seconde, donc sampling_rate = 1 Hz
        # Utiliser highcut = 0.1 Hz pour un filtre passe-bas
        result = denoising_service.denoise_single_sensor(
            noisy_data,
            method="butterworth",
            lowcut=None,
            highcut=0.1  # Filtre passe-bas à 0.1 Hz (normalisé pour sampling_rate = 1 Hz)
        )
        
        assert len(result) == len(noisy_data)
        assert result[0].preprocessing_metadata.get("denoised", False) is True
        assert result[0].preprocessing_metadata.get("denoising_method") == "butterworth"
    
    def test_denoise_single_sensor_moving_average(self, denoising_service, noisy_data):
        """Test débruitage avec moyenne mobile"""
        # La méthode utilise window_size=5 par défaut
        result = denoising_service.denoise_single_sensor(
            noisy_data,
            method="moving_average"
        )
        
        assert len(result) == len(noisy_data)
        assert result[0].preprocessing_metadata.get("denoised", False) is True
        assert result[0].preprocessing_metadata.get("denoising_method") == "moving_average"
    
    def test_denoise_single_sensor_savgol(self, denoising_service, noisy_data):
        """Test débruitage avec filtre Savitzky-Golay"""
        # La méthode utilise window_length=5, polyorder=2 par défaut
        result = denoising_service.denoise_single_sensor(
            noisy_data,
            method="savgol"
        )
        
        assert len(result) == len(noisy_data)
        assert result[0].preprocessing_metadata.get("denoised", False) is True
        assert result[0].preprocessing_metadata.get("denoising_method") == "savgol"
    
    def test_denoise_disabled(self, denoising_service):
        """Test que le débruitage est désactivé si enable_denoising = False"""
        denoising_service.enable_denoising = False
        
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
        
        result = denoising_service.denoise_single_sensor(data)
        
        assert len(result) == len(data)
        assert result[0].value == data[0].value  # Pas de modification

