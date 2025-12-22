"""
Configuration du service Dashboard Usine
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List, Union


class Settings(BaseSettings):
    """Configuration du service"""
    
    # Service
    service_name: str = "dashboard-usine-service"
    service_port: int = 8091
    log_level: str = "INFO"
    
    # Database
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "predictive_maintenance"
    database_user: str = "pmuser"
    database_password: str = "pmpassword"
    
    # External Services URLs
    extraction_features_url: str = "http://localhost:8083"
    detection_anomalies_url: str = "http://localhost:8084"
    prediction_rul_url: str = "http://localhost:8085"
    orchestrator_url: str = "http://localhost:8087"
    dashboard_monitoring_url: str = "http://localhost:8090"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "dashboard-usine-service"
    kafka_topics: str = "extracted-features,anomalies-detected,rul-predictions"
    
    # WebSocket
    websocket_path: str = "/ws/dashboard"
    websocket_ping_interval: int = 25
    websocket_ping_timeout: int = 10
    
    # Frontend
    frontend_url: str = "http://localhost:3000"
    cors_origins: Union[str, List[str]] = "http://localhost:3000,http://localhost:8091"
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    # Export
    export_max_records: int = 10000
    export_pdf_template_path: Optional[str] = None
    
    # Grafana
    grafana_url: Optional[str] = None
    grafana_api_key: Optional[str] = None
    
    # PostGIS
    enable_gis: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

