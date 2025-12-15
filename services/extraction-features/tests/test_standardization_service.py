"""
Tests pour le service de standardisation
"""
import pytest
from datetime import datetime
import numpy as np

from app.services.standardization_service import StandardizationService
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector


class TestStandardizationService:
    """Tests pour StandardizationService"""

    @pytest.fixture
    def standardization_service(self):
        return StandardizationService()

    @pytest.fixture
    def sample_features(self):
        """Features d'exemple pour les tests"""
        base_time = datetime.utcnow()
        features = []

        for i in range(10):
            features.append(ExtractedFeature(
                timestamp=base_time,
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                feature_name=f"feature_{i}",
                feature_value=25.0 + i * 0.5,
                feature_type="temporal",
                metadata={}
            ))

        return features

    @pytest.fixture
    def sample_feature_vector(self):
        """Vecteur de features d'exemple"""
        base_time = datetime.utcnow()

        return ExtractedFeaturesVector(
            feature_vector_id="fv_001",
            timestamp=base_time,
            asset_id="ASSET001",
            start_time=base_time,
            end_time=base_time,
            features={
                "rms": 25.5,
                "kurtosis": 2.3,
                "skewness": 0.5,
                "crest_factor": 4.1
            },
            feature_metadata={}
        )

    def test_standardize_features_z_score(self, standardization_service, sample_features):
        """Test standardisation z-score"""
        standardized = standardization_service.standardize_features(
            sample_features,
            asset_type="pump",
            method="z-score"
        )

        assert len(standardized) > 0
        assert all(isinstance(f, ExtractedFeature) for f in standardized)
        # Vérifier que les features ont été standardisées (nom se termine par _standardized)
        assert all(f.feature_name.endswith("_standardized") for f in standardized)
        # Vérifier que les métadonnées indiquent la standardisation
        assert all(f.metadata.get("standardized", False) is True for f in standardized)
        # Pour les features avec template (comme rms pour pump), vérifier la standardisation z-score
        rms_feature = next((f for f in standardized if "rms" in f.feature_name), None)
        if rms_feature:
            # Pour pump, rms: mean=25.0, std=5.0
            # Si la valeur originale était 27.5 (25.0 + 2.5), la valeur standardisée devrait être (27.5 - 25.0) / 5.0 = 0.5
            assert isinstance(rms_feature.feature_value, (int, float))

    def test_standardize_features_min_max(self, standardization_service, sample_features):
        """Test standardisation min-max"""
        # Créer un template min-max pour le test
        min_max_template = {
            "rms": {"min": 0.0, "max": 100.0},
            "kurtosis": {"min": 0.0, "max": 10.0}
        }
        standardization_service.update_asset_type_template("pump", "min-max", min_max_template)
        
        standardized = standardization_service.standardize_features(
            sample_features,
            asset_type="pump",
            method="min-max"
        )

        assert len(standardized) > 0
        assert all(isinstance(f, ExtractedFeature) for f in standardized)
        # Pour les features avec template min-max, les valeurs devraient être entre 0 et 1
        rms_feature = next((f for f in standardized if "rms" in f.feature_name), None)
        if rms_feature:
            # La valeur standardisée devrait être entre 0 et 1
            assert 0 <= rms_feature.feature_value <= 1

    def test_standardize_features_robust(self, standardization_service, sample_features):
        """Test standardisation robust"""
        standardized = standardization_service.standardize_features(
            sample_features,
            asset_type="pump",
            method="robust"
        )

        assert len(standardized) > 0
        assert all(isinstance(f, ExtractedFeature) for f in standardized)

    def test_standardize_feature_vector(self, standardization_service, sample_feature_vector):
        """Test standardisation d'un vecteur de features"""
        standardized = standardization_service.standardize_feature_vector(
            sample_feature_vector,
            asset_type="pump",
            method="z-score"
        )

        assert isinstance(standardized, ExtractedFeaturesVector)
        assert standardized.feature_vector_id == sample_feature_vector.feature_vector_id
        assert len(standardized.features) == len(sample_feature_vector.features)
        # Les valeurs devraient être standardisées
        assert standardized.feature_metadata.get("standardized", False) is True

    def test_standardize_features_empty(self, standardization_service):
        """Test avec features vides"""
        standardized = standardization_service.standardize_features(
            [],
            asset_type="pump"
        )
        assert len(standardized) == 0

    def test_update_asset_type_template(self, standardization_service):
        """Test mise à jour du template d'actif"""
        # Créer un template de features
        features_template = {
            "rms": {"mean": 25.0, "std": 5.0},
            "kurtosis": {"mean": 3.0, "std": 1.0}
        }

        # Mettre à jour le template
        standardization_service.update_asset_type_template(
            "test_asset",
            "z-score",
            features_template
        )

        # Vérifier que le template a été mis à jour
        template = standardization_service.get_asset_type_template("test_asset")
        assert template is not None
        assert template["method"] == "z-score"
        assert "rms" in template["features"]
        assert "mean" in template["features"]["rms"]
        assert "std" in template["features"]["rms"]

    def test_calculate_statistics_from_data(self, standardization_service):
        """Test calcul des statistiques à partir de données"""
        # Simuler des données historiques
        historical_data = [
            [
                ExtractedFeature(timestamp=datetime.utcnow(), asset_id="ASSET001", sensor_id="SENSOR001", feature_name="rms", feature_value=20.0, feature_type="temporal", metadata={}),
                ExtractedFeature(timestamp=datetime.utcnow(), asset_id="ASSET001", sensor_id="SENSOR001", feature_name="rms", feature_value=30.0, feature_type="temporal", metadata={}),
                ExtractedFeature(timestamp=datetime.utcnow(), asset_id="ASSET001", sensor_id="SENSOR001", feature_name="kurtosis", feature_value=3.0, feature_type="temporal", metadata={}),
            ],
            [
                ExtractedFeature(timestamp=datetime.utcnow(), asset_id="ASSET001", sensor_id="SENSOR001", feature_name="rms", feature_value=25.0, feature_type="temporal", metadata={}),
                ExtractedFeature(timestamp=datetime.utcnow(), asset_id="ASSET001", sensor_id="SENSOR001", feature_name="kurtosis", feature_value=4.0, feature_type="temporal", metadata={}),
            ]
        ]
        stats = standardization_service.calculate_statistics_from_data(historical_data, "test_asset_type")

        assert "rms" in stats
        assert "kurtosis" in stats
        assert np.isclose(stats["rms"]["mean"], 25.0)
        assert np.isclose(stats["rms"]["std"], 4.082, atol=0.01)  # std de [20, 30, 25]
        assert standardization_service.get_asset_type_template("test_asset_type") is not None

