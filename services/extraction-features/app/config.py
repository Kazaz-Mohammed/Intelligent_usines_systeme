"""
Configuration du service Extraction Features
"""
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Configuration du service"""
    
    # Service
    service_name: str = "extraction-features-service"
    service_port: int = 8083
    log_level: str = "INFO"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "extraction-features-service-v2"  # Changed to force reading from beginning
    kafka_topic_input_preprocessed: str = "preprocessed-data"
    kafka_topic_input_windowed: str = "windowed-data"
    kafka_topic_output: str = "extracted-features"
    kafka_auto_offset_reset: str = "earliest"
    kafka_enable_auto_commit: bool = True
    
    # Database
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "predictive_maintenance"
    database_user: str = "pmuser"
    database_password: str = "pmpassword"
    
    # Feature Extraction
    enable_temporal_features: bool = True
    enable_frequency_features: bool = True
    enable_wavelet_features: bool = True
    temporal_features_list: List[str] = [
        "mean", "std", "var", "min", "max", "median",
        "rms", "kurtosis", "skewness", "crest_factor",
        "peak_to_peak", "form_factor"
    ]
    frequency_features_list: List[str] = [
        "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
        "zero_crossing_rate", "spectral_flatness"
    ]
    
    # Feature Store (Feast)
    feast_repo_path: Optional[str] = "./feast_repo"
    feast_enable: bool = True
    feast_online_store_type: str = "redis"  # redis, sqlite, postgres
    feast_offline_store_type: str = "file"  # file, bigquery, s3
    
    # Standardization
    enable_standardization: bool = True
    standardization_method: str = "z-score"  # z-score, min-max, robust
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

