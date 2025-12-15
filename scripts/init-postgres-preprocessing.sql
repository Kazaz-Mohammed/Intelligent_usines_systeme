-- Script d'initialisation pour les tables du service Prétraitement

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
    frequency_analysis JSONB
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

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'Preprocessing tables created: preprocessed_sensor_data, windowed_sensor_data';
END $$;

