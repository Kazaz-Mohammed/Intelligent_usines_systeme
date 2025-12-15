"""
Tests d'intégration Kafka
"""
import pytest
from datetime import datetime, timedelta

from app.services.kafka_consumer import KafkaConsumerService
from app.services.kafka_producer import KafkaProducerService
from app.models.feature_data import ExtractedFeature, ExtractedFeaturesVector


@pytest.mark.integration
@pytest.mark.kafka
class TestKafkaIntegration:
    """Tests d'intégration Kafka"""

    @pytest.fixture
    def kafka_producer(self):
        return KafkaProducerService()

    @pytest.fixture
    def kafka_consumer(self):
        return KafkaConsumerService()

    @pytest.fixture
    def sample_feature(self):
        """Feature d'exemple"""
        return ExtractedFeature(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            feature_name="rms",
            feature_value=25.5,
            feature_type="temporal",
            metadata={"source": "test"}
        )

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
                "skewness": 0.5
            },
            feature_metadata={"window_size": 100}
        )

    def test_publish_extracted_feature(self, kafka_producer, sample_feature):
        """Test publication d'une feature"""
        if not kafka_producer.producer:
            pytest.skip("Kafka non disponible")

        try:
            kafka_producer.publish_extracted_feature(sample_feature)
            # Si la publication réussit, pas d'exception
            assert True
        except Exception as e:
            pytest.fail(f"Publication échouée: {e}")

    def test_publish_extracted_features_batch(self, kafka_producer, sample_feature):
        """Test publication d'un lot de features"""
        if not kafka_producer.producer:
            pytest.skip("Kafka non disponible")

        features = [sample_feature] * 10

        try:
            kafka_producer.publish_extracted_features_batch(features)
            # Si la publication réussit, pas d'exception
            assert True
        except Exception as e:
            pytest.fail(f"Publication batch échouée: {e}")

    def test_publish_feature_vector(self, kafka_producer, sample_feature_vector):
        """Test publication d'un vecteur de features"""
        if not kafka_producer.producer:
            pytest.skip("Kafka non disponible")

        try:
            kafka_producer.publish_feature_vector(sample_feature_vector)
            # Si la publication réussit, pas d'exception
            assert True
        except Exception as e:
            pytest.fail(f"Publication vecteur échouée: {e}")

    def test_consumer_creation(self, kafka_consumer):
        """Test création du consumer"""
        if not kafka_consumer.consumer:
            pytest.skip("Kafka non disponible")

        assert kafka_consumer.consumer is not None

    def test_consumer_close(self, kafka_consumer):
        """Test fermeture du consumer"""
        kafka_consumer.close()
        # Ne devrait pas lever d'exception
        assert True

