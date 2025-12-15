"""
Configuration du service Prediction RUL
"""
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Configuration du service"""
    
    # Service
    service_name: str = "prediction-rul-service"
    service_port: int = 8085
    log_level: str = "INFO"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "prediction-rul-service"
    kafka_topic_input_features: str = "extracted-features"
    kafka_topic_output_rul: str = "rul-predictions"
    kafka_auto_offset_reset: str = "earliest"
    kafka_enable_auto_commit: bool = True
    
    # Database
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "predictive_maintenance"
    database_user: str = "pmuser"
    database_password: str = "pmpassword"
    
    # RUL Prediction Models
    # Note: LSTM/GRU/TCN require PyTorch which can be slow to initialize
    # Set to False for faster startup during testing
    enable_lstm: bool = True  # Enabled for production
    enable_gru: bool = False  # Disabled by default for faster startup
    enable_tcn: bool = False  # Disabled by default for faster startup
    enable_xgboost: bool = True  # Faster, enabled by default
    
    # LSTM Parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 20
    lstm_batch_size: int = 32
    lstm_epochs: int = 100
    lstm_learning_rate: float = 0.001
    
    # GRU Parameters
    gru_hidden_size: int = 64
    gru_num_layers: int = 2
    gru_dropout: float = 0.2
    gru_sequence_length: int = 20
    gru_batch_size: int = 32
    gru_epochs: int = 100
    gru_learning_rate: float = 0.001
    
    # TCN Parameters
    tcn_num_channels: List[int] = [64, 128, 256]
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    tcn_sequence_length: int = 20
    tcn_batch_size: int = 32
    tcn_epochs: int = 100
    tcn_learning_rate: float = 0.001
    
    # XGBoost Parameters
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_subsample: float = 0.8
    
    # Transfer Learning
    transfer_learning_enabled: bool = True
    transfer_learning_pretrained_path: Optional[str] = None
    transfer_learning_freeze_layers: bool = False
    
    # Calibration
    calibration_enabled: bool = True
    calibration_method: str = "isotonic"  # isotonic, platt, temperature_scaling
    
    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "rul-prediction"
    mlflow_enabled: bool = True
    
    # Model Persistence
    models_save_dir: str = "models"  # Directory to save/load trained models
    auto_load_models: bool = True  # Automatically load saved models on startup
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

