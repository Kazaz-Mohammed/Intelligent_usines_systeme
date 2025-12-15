-- Initialisation des tables pour le service Extraction Features
-- À exécuter après init-postgres.sql

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

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'Tables pour Extraction Features créées avec succès';
END $$;

