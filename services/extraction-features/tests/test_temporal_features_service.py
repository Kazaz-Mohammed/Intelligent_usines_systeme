"""
Tests pour le service de calcul de features temporelles
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from app.services.temporal_features_service import TemporalFeaturesService
from app.models.feature_data import PreprocessedDataReference, ExtractedFeature


class TestTemporalFeaturesService:
    """Tests pour TemporalFeaturesService"""

    @pytest.fixture
    def temporal_features_service(self):
        return TemporalFeaturesService()

    @pytest.fixture
    def sample_data(self):
        """Données d'exemple pour les tests"""
        base_time = datetime.utcnow()
        data = []
        base_value = 25.0

        for i in range(50):
            # Signal sinusoïdal avec variation
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

    def test_calculate_temporal_features(self, temporal_features_service, sample_data):
        """Test calcul des features temporelles"""
        features = temporal_features_service.calculate_temporal_features(
            sample_data
        )

        assert len(features) > 0
        assert all(isinstance(f, ExtractedFeature) for f in features)
        assert all(f.feature_type == "temporal" for f in features)
        assert all(f.asset_id == "ASSET001" for f in features)
        assert all(f.sensor_id == "SENSOR001" for f in features)

        # Vérifier que certaines features sont présentes
        feature_names = [f.feature_name for f in features]
        assert "rms" in feature_names
        assert "mean" in feature_names
        assert "std" in feature_names

    def test_calculate_temporal_features_empty_data(self, temporal_features_service):
        """Test avec données vides"""
        features = temporal_features_service.calculate_temporal_features([])
        assert len(features) == 0

    def test_calculate_temporal_features_single_point(self, temporal_features_service):
        """Test avec un seul point de données"""
        data = [
            PreprocessedDataReference(
                timestamp=datetime.utcnow(),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={},
                frequency_analysis=None
            )
        ]

        features = temporal_features_service.calculate_temporal_features(data)
        # Avec un seul point, on peut calculer certaines features mais pas toutes
        # Le service devrait gérer ce cas gracieusement
        assert isinstance(features, list)

    def test_calculate_temporal_features_rms(self, temporal_features_service, sample_data):
        """Test calcul RMS"""
        features = temporal_features_service.calculate_temporal_features(sample_data)

        rms_features = [f for f in features if f.feature_name == "rms"]
        assert len(rms_features) > 0
        assert rms_features[0].feature_value > 0

    def test_calculate_temporal_features_kurtosis(self, temporal_features_service, sample_data):
        """Test calcul kurtosis"""
        features = temporal_features_service.calculate_temporal_features(sample_data)

        kurtosis_features = [f for f in features if f.feature_name == "kurtosis"]
        assert len(kurtosis_features) > 0
        assert not np.isnan(kurtosis_features[0].feature_value)

    def test_calculate_temporal_features_crest_factor(self, temporal_features_service, sample_data):
        """Test calcul crest factor"""
        features = temporal_features_service.calculate_temporal_features(sample_data)

        crest_factor_features = [f for f in features if f.feature_name == "crest_factor"]
        assert len(crest_factor_features) > 0
        assert crest_factor_features[0].feature_value > 0

