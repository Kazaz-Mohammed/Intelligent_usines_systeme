-- Initialisation PostgreSQL avec extension TimescaleDB

-- Créer extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Base de données pour séries temporelles (raw sensor data)
CREATE TABLE IF NOT EXISTS raw_sensor_data (
    time TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    sensor_id VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    quality INTEGER,
    metadata JSONB
);

-- Convertir en table hypertable TimescaleDB
SELECT create_hypertable('raw_sensor_data', 'time', if_not_exists => TRUE);

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_raw_sensor_asset_time ON raw_sensor_data (asset_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_raw_sensor_sensor_time ON raw_sensor_data (sensor_id, time DESC);

-- Table pour données prétraitées individuelles
CREATE TABLE IF NOT EXISTS preprocessed_sensor_data (
    time TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    sensor_id VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    quality INTEGER,
    source_type VARCHAR(50),
    preprocessing_metadata JSONB,
    frequency_analysis JSONB,
    PRIMARY KEY (time, asset_id, sensor_id)
);

-- Convertir en table hypertable TimescaleDB
SELECT create_hypertable('preprocessed_sensor_data', 'time', if_not_exists => TRUE);

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_preprocessed_sensor_asset_time ON preprocessed_sensor_data (asset_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_preprocessed_sensor_sensor_time ON preprocessed_sensor_data (sensor_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_preprocessed_sensor_asset_sensor_time ON preprocessed_sensor_data (asset_id, sensor_id, time DESC);

-- Table pour fenêtres (windowed data) pour ML
CREATE TABLE IF NOT EXISTS windowed_sensor_data (
    window_id VARCHAR(100) PRIMARY KEY,
    asset_id VARCHAR(100) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    sensor_data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_windowed_sensor_asset_time ON windowed_sensor_data (asset_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_windowed_sensor_time_range ON windowed_sensor_data USING GIST (tstzrange(start_time, end_time));

-- Table pour données prétraitées (ancienne table, conservée pour compatibilité)
CREATE TABLE IF NOT EXISTS processed_windows (
    time TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    window_id VARCHAR(100) NOT NULL,
    window_size INTEGER,
    data JSONB NOT NULL,
    metadata JSONB
);

SELECT create_hypertable('processed_windows', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_processed_windows_asset_time ON processed_windows (asset_id, time DESC);

-- Table pour événements d'anomalies
CREATE TABLE IF NOT EXISTS anomaly_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset_id VARCHAR(100) NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    anomaly_type VARCHAR(50),
    severity VARCHAR(20),
    details JSONB,
    resolved BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_asset_time ON anomaly_events (asset_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_events_resolved ON anomaly_events (resolved, timestamp DESC);

-- Table pour prédictions RUL
CREATE TABLE IF NOT EXISTS rul_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset_id VARCHAR(100) NOT NULL,
    rul_mean DOUBLE PRECISION NOT NULL,
    rul_lower DOUBLE PRECISION,
    rul_upper DOUBLE PRECISION,
    confidence_interval DOUBLE PRECISION,
    model_version VARCHAR(50),
    features JSONB,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_rul_predictions_asset_time ON rul_predictions (asset_id, timestamp DESC);

-- Table pour référentiel des actifs
CREATE TABLE IF NOT EXISTS assets (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    location VARCHAR(255),
    line_id VARCHAR(100),
    criticity VARCHAR(20) DEFAULT 'medium',
    manufacturer VARCHAR(100),
    installation_date DATE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table pour ordres de maintenance
CREATE TABLE IF NOT EXISTS maintenance_orders (
    id SERIAL PRIMARY KEY,
    order_number VARCHAR(100) UNIQUE NOT NULL,
    asset_id VARCHAR(100) NOT NULL REFERENCES assets(id),
    order_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    scheduled_date TIMESTAMPTZ,
    completion_date TIMESTAMPTZ,
    assigned_to VARCHAR(100),
    description TEXT,
    work_details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_maintenance_orders_asset ON maintenance_orders (asset_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_orders_status ON maintenance_orders (status, scheduled_date);

-- Fonction pour mettre à jour updated_at automatiquement
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers pour updated_at
CREATE TRIGGER update_assets_updated_at BEFORE UPDATE ON assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_maintenance_orders_updated_at BEFORE UPDATE ON maintenance_orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Vues utiles pour le dashboard
CREATE OR REPLACE VIEW v_asset_status AS
SELECT 
    a.id,
    a.name,
    a.type,
    a.criticity,
    a.location,
    a.line_id,
    COALESCE(latest_rul.rul_mean, NULL) as current_rul,
    COALESCE(latest_rul.confidence_interval, NULL) as rul_confidence,
    COALESCE(recent_anomaly.anomaly_score, NULL) as latest_anomaly_score,
    COALESCE(recent_anomaly.severity, NULL) as latest_anomaly_severity,
    COALESCE(pending_orders.count, 0) as pending_maintenance_orders,
    a.updated_at
FROM assets a
LEFT JOIN LATERAL (
    SELECT rul_mean, confidence_interval
    FROM rul_predictions
    WHERE asset_id = a.id
    ORDER BY timestamp DESC
    LIMIT 1
) latest_rul ON TRUE
LEFT JOIN LATERAL (
    SELECT anomaly_score, severity
    FROM anomaly_events
    WHERE asset_id = a.id AND resolved = FALSE
    ORDER BY timestamp DESC
    LIMIT 1
) recent_anomaly ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) as count
    FROM maintenance_orders
    WHERE asset_id = a.id AND status = 'pending'
) pending_orders ON TRUE;

-- Table pour features extraites individuelles
CREATE TABLE IF NOT EXISTS extracted_features (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    sensor_id VARCHAR(100),
    feature_name VARCHAR(100) NOT NULL,
    feature_value DOUBLE PRECISION NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    metadata JSONB,
    PRIMARY KEY (timestamp, asset_id, sensor_id, feature_name)
);

-- Convertir en table hypertable TimescaleDB
SELECT create_hypertable('extracted_features', 'timestamp', if_not_exists => TRUE);

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_extracted_features_asset_time ON extracted_features (asset_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_extracted_features_sensor_time ON extracted_features (sensor_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_extracted_features_feature_name ON extracted_features (feature_name);
CREATE INDEX IF NOT EXISTS idx_extracted_features_feature_type ON extracted_features (feature_type);
CREATE INDEX IF NOT EXISTS idx_extracted_features_asset_feature_time ON extracted_features (asset_id, feature_name, timestamp DESC);

-- Table pour vecteurs de features extraites
CREATE TABLE IF NOT EXISTS extracted_feature_vectors (
    feature_vector_id VARCHAR(100) PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    feature_metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_extracted_feature_vectors_asset_time ON extracted_feature_vectors (asset_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_extracted_feature_vectors_time_range ON extracted_feature_vectors USING GIST (tstzrange(start_time, end_time));
CREATE INDEX IF NOT EXISTS idx_extracted_feature_vectors_created_at ON extracted_feature_vectors (created_at DESC);

-- Insertion de données d'exemple (pour tests)
INSERT INTO assets (id, name, type, location, line_id, criticity) VALUES
('ASSET001', 'Pompe Centrifuge #1', 'pump', 'Atelier A', 'LINE1', 'high'),
('ASSET002', 'Moteur Électrique #1', 'motor', 'Atelier A', 'LINE1', 'medium'),
('ASSET003', 'Convoyeur #1', 'conveyor', 'Atelier B', 'LINE2', 'low')
ON CONFLICT (id) DO NOTHING;

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully with TimescaleDB extension';
    RAISE NOTICE 'Tables created: raw_sensor_data, processed_windows, anomaly_events, rul_predictions, assets, maintenance_orders';
    RAISE NOTICE 'Sample assets inserted: ASSET001, ASSET002, ASSET003';
END $$;

