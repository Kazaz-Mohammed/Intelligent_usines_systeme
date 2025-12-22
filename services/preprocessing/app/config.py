"""
Configuration du service Prétraitement
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration du service"""
    
    # Service
    service_name: str = "preprocessing-service"
    service_port: int = 8082
    log_level: str = "INFO"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "preprocessing-service"
    kafka_topic_input: str = "sensor-data"
    kafka_topic_output: str = "preprocessed-data"
    kafka_auto_offset_reset: str = "latest"  # Use "latest" to skip old messages, "earliest" to process all
    kafka_enable_auto_commit: bool = True
    
    # Database
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "predictive_maintenance"
    database_user: str = "pmuser"
    database_password: str = "pmpassword"
    
    # Preprocessing
    window_size: int = 100  # Taille fenêtre pour ML
    window_overlap: float = 0.5  # Chevauchement fenêtres (50%)
    resampling_rate: Optional[float] = None  # Hz, None = pas de rééchantillonnage
    outlier_threshold: float = 3.0  # Nombre d'écarts-types pour détection outliers
    enable_denoising: bool = True
    enable_frequency_analysis: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

