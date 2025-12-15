"""
Tests pour le service de calcul de features fréquentielles
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from app.services.frequency_features_service import FrequencyFeaturesService
from app.models.feature_data import PreprocessedDataReference, ExtractedFeature


class TestFrequencyFeaturesService:
    """Tests pour FrequencyFeaturesService"""

    @pytest.fixture
    def frequency_features_service(self):
        return FrequencyFeaturesService()

    @pytest.fixture
    def sample_data(self):
        """Données d'exemple pour les tests"""
        base_time = datetime.utcnow()
        data = []
        base_value = 25.0

        for i in range(100):
            # Signal sinusoïdal avec plusieurs fréquences
            value = base_value + 2 * np.sin(2 * np.pi * i / 10) + np.sin(2 * np.pi * i / 5)

            data.append(PreprocessedDataReference(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=value,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={},
                frequency_analysis=None
            ))
        return data

    def test_calculate_frequency_features(self, frequency_features_service, sample_data):
        """Test calcul des features fréquentielles"""
        features = frequency_features_service.calculate_frequency_features(
            sample_data
        )

        assert len(features) > 0
        assert all(isinstance(f, ExtractedFeature) for f in features)
        assert all(f.feature_type == "frequency" for f in features)
        assert all(f.asset_id == "ASSET001" for f in features)
        assert all(f.sensor_id == "SENSOR001" for f in features)

        # Vérifier que certaines features sont présentes
        feature_names = [f.feature_name for f in features]
        assert "spectral_centroid" in feature_names or "spectral_rolloff" in feature_names

    def test_calculate_frequency_features_empty_data(self, frequency_features_service):
        """Test avec données vides"""
        features = frequency_features_service.calculate_frequency_features([])
        assert len(features) == 0

    def test_calculate_frequency_features_insufficient_data(self, frequency_features_service):
        """Test avec données insuffisantes"""
        data = [
            PreprocessedDataReference(
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + i * 0.1,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={},
                frequency_analysis=None
            ) for i in range(5)  # Pas assez de points pour FFT
        ]

        features = frequency_features_service.calculate_frequency_features(data)
        # Le service devrait gérer ce cas gracieusement
        assert isinstance(features, list)

    def test_calculate_band_energy(self, frequency_features_service, sample_data):
        """Test calcul de l'énergie par bande"""
        features = frequency_features_service.calculate_band_energy(
            sample_data
        )

        assert len(features) > 0
        assert all(isinstance(f, ExtractedFeature) for f in features)
        assert all(f.feature_type == "frequency" for f in features)

        # Vérifier que les features d'énergie de bande sont présentes
        feature_names = [f.feature_name for f in features]
        assert any("band_energy" in name for name in feature_names)

