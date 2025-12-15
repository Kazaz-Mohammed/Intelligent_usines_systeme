"""
Configuration pytest pour les tests
"""
import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import List

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.feature_data import (
    ExtractedFeature,
    ExtractedFeaturesVector,
    PreprocessedDataReference,
    WindowedDataReference
)


@pytest.fixture
def sample_preprocessed_data() -> List[PreprocessedDataReference]:
    """Données prétraitées d'exemple"""
    base_time = datetime.utcnow()
    data = []
    
    for i in range(100):
        data.append(PreprocessedDataReference(
            timestamp=base_time + timedelta(seconds=i),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            value=25.0 + (i % 10) * 0.5,
            unit="°C",
            quality=2,
            source_type="TEST",
            preprocessing_metadata={"cleaned": True},
            frequency_analysis=None
        ))
    
    return data


@pytest.fixture
def sample_windowed_data() -> WindowedDataReference:
    """Fenêtre de données d'exemple"""
    base_time = datetime.utcnow()
    sensor_data = {
        "SENSOR001": [
            PreprocessedDataReference(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="ASSET001",
                sensor_id="SENSOR001",
                value=25.0 + (i % 10) * 0.5,
                unit="°C",
                quality=2,
                source_type="TEST",
                preprocessing_metadata={"cleaned": True},
                frequency_analysis=None
            ) for i in range(100)
        ]
    }
    
    return WindowedDataReference(
        window_id="window_001",
        asset_id="ASSET001",
        start_time=base_time,
        end_time=base_time + timedelta(seconds=100),
        sensor_data=sensor_data,
        metadata={"window_size": 100, "overlap": 0.5}
    )


@pytest.fixture
def sample_extracted_features() -> List[ExtractedFeature]:
    """Features extraites d'exemple"""
    base_time = datetime.utcnow()
    features = []
    
    feature_names = ["rms", "kurtosis", "skewness", "crest_factor", "spectral_centroid"]
    
    for i, feature_name in enumerate(feature_names):
        features.append(ExtractedFeature(
            timestamp=base_time,
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            feature_name=feature_name,
            feature_value=25.0 + i * 0.5,
            feature_type="temporal" if i < 4 else "frequency",
            metadata={"source": "test"}
        ))
    
    return features


@pytest.fixture
def sample_feature_vector() -> ExtractedFeaturesVector:
    """Vecteur de features d'exemple"""
    base_time = datetime.utcnow()
    
    return ExtractedFeaturesVector(
        feature_vector_id="fv_001",
        timestamp=base_time,
        asset_id="ASSET001",
        start_time=base_time - timedelta(seconds=100),
        end_time=base_time,
        features={
            "rms": 25.5,
            "kurtosis": 2.3,
            "skewness": 0.5,
            "crest_factor": 4.1,
            "spectral_centroid": 150.2
        },
        feature_metadata={
            "window_size": 100,
            "standardized": False,
            "asset_type": "pump"
        }
    )


@pytest.fixture
def mock_timescaledb_service(monkeypatch):
    """Mock du service TimescaleDB"""
    from unittest.mock import MagicMock
    
    mock_service = MagicMock()
    mock_service.insert_extracted_feature = MagicMock()
    mock_service.insert_extracted_features_batch = MagicMock()
    mock_service.insert_feature_vector = MagicMock()
    mock_service.get_features_by_asset = MagicMock(return_value=[])
    
    return mock_service


@pytest.fixture
def mock_kafka_producer(monkeypatch):
    """Mock du service Kafka Producer"""
    from unittest.mock import MagicMock
    
    mock_producer = MagicMock()
    mock_producer.publish_extracted_feature = MagicMock()
    mock_producer.publish_extracted_features_batch = MagicMock()
    mock_producer.publish_feature_vector = MagicMock()
    
    return mock_producer


@pytest.fixture
def mock_kafka_consumer(monkeypatch):
    """Mock du service Kafka Consumer"""
    from unittest.mock import MagicMock
    
    mock_consumer = MagicMock()
    mock_consumer.consume_preprocessed_data = MagicMock()
    mock_consumer.consume_windowed_data = MagicMock()
    
    return mock_consumer


@pytest.fixture
def mock_asset_service(monkeypatch):
    """Mock du service Asset"""
    from unittest.mock import MagicMock
    
    mock_service = MagicMock()
    mock_service.get_asset_type = MagicMock(return_value="pump")
    mock_service.get_asset_info = MagicMock(return_value={
        "id": "ASSET001",
        "name": "Pompe Centrifuge #1",
        "type": "pump",
        "location": "Atelier A",
        "line_id": "LINE1",
        "criticity": "high"
    })
    
    return mock_service

