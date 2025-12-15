"""
Tests pour le service de calcul de features ondelettes
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from app.services.wavelet_features_service import WaveletFeaturesService
from app.models.feature_data import PreprocessedDataReference, ExtractedFeature


class TestWaveletFeaturesService:
    """Tests pour WaveletFeaturesService"""

    @pytest.fixture
    def wavelet_features_service(self):
        return WaveletFeaturesService()

    @pytest.fixture
    def sample_data(self):
        """Données d'exemple pour les tests"""
        base_time = datetime.utcnow()
        data = []
        base_value = 25.0

        for i in range(64):  # Puissance de 2 pour les ondelettes
            # Signal avec variation
            value = base_value + 2 * np.sin(2 * np.pi * i / 10) + np.random.randn() * 0.1

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

    def test_is_available(self, wavelet_features_service):
        """Test disponibilité du service"""
        # Le service devrait être disponible si PyWavelets est installé
        available = wavelet_features_service.is_available()
        assert isinstance(available, bool)

    @pytest.mark.skipif(
        not WaveletFeaturesService().is_available(),
        reason="PyWavelets n'est pas disponible"
    )
    def test_calculate_wavelet_features(self, wavelet_features_service, sample_data):
        """Test calcul des features ondelettes"""
        if not wavelet_features_service.is_available():
            pytest.skip("PyWavelets n'est pas disponible")

        features = wavelet_features_service.calculate_wavelet_features(
            sample_data
        )

        assert len(features) > 0
        assert all(isinstance(f, ExtractedFeature) for f in features)
        assert all(f.feature_type == "wavelet" for f in features)
        assert all(f.asset_id == "ASSET001" for f in features)
        assert all(f.sensor_id == "SENSOR001" for f in features)

        # Vérifier que certaines features sont présentes
        feature_names = [f.feature_name for f in features]
        assert any("wavelet" in name.lower() for name in feature_names)

    def test_calculate_wavelet_features_empty_data(self, wavelet_features_service):
        """Test avec données vides"""
        features = wavelet_features_service.calculate_wavelet_features([])
        assert len(features) == 0

    def test_calculate_wavelet_features_insufficient_data(self, wavelet_features_service):
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
            ) for i in range(5)  # Pas assez de points pour ondelettes
        ]

        if wavelet_features_service.is_available():
            features = wavelet_features_service.calculate_wavelet_features(data)
            # Le service devrait gérer ce cas gracieusement
            assert isinstance(features, list)

