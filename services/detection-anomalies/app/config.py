"""
Configuration du service Detection Anomalies
"""
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Configuration du service"""
    
    # Service
    service_name: str = "detection-anomalies-service"
    service_port: int = 8084
    log_level: str = "INFO"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "detection-anomalies-service"
    kafka_topic_input_features: str = "extracted-features"
    kafka_topic_output_anomalies: str = "anomalies-detected"
    kafka_auto_offset_reset: str = "earliest"
    kafka_enable_auto_commit: bool = True
    
    # Database
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "predictive_maintenance"
    database_user: str = "pmuser"
    database_password: str = "pmpassword"
    
    # Anomaly Detection Models
    # Note: LSTM Autoencoder requires PyTorch which can be slow to initialize
    # Set to False for faster startup during testing
    enable_isolation_forest: bool = True
    enable_one_class_svm: bool = True
    enable_lstm_autoencoder: bool = False  # Disabled by default for faster startup
    
    # Isolation Forest
    isolation_forest_contamination: float = 0.1
    isolation_forest_n_estimators: int = 100
    isolation_forest_max_samples: str = "auto"
    
    # One-Class SVM
    one_class_svm_nu: float = 0.1
    one_class_svm_kernel: str = "rbf"
    one_class_svm_gamma: str = "scale"
    
    # LSTM Autoencoder
    lstm_autoencoder_encoder_layers: List[int] = [64, 32, 16]
    lstm_autoencoder_decoder_layers: List[int] = [16, 32, 64]
    lstm_autoencoder_sequence_length: int = 10
    lstm_autoencoder_batch_size: int = 32
    lstm_autoencoder_epochs: int = 50
    lstm_autoencoder_learning_rate: float = 0.001
    lstm_autoencoder_threshold_percentile: float = 95.0
    
    # Scoring
    anomaly_score_threshold: float = 0.5
    adaptive_threshold_enabled: bool = True
    criticality_levels: List[str] = ["low", "medium", "high", "critical"]
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "anomaly-detection"
    mlflow_enabled: bool = True
    
    # Model Persistence
    models_save_dir: str = "models"
    auto_load_models: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

