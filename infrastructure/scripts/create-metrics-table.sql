-- Script pour créer la table metrics pour dashboard-monitoring
-- Exécuter avec: psql -h localhost -U pmuser -d predictive_maintenance -f create-metrics-table.sql

CREATE TABLE IF NOT EXISTS metrics (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    labels JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index pour améliorer les performances des requêtes
CREATE INDEX IF NOT EXISTS idx_metrics_service_name ON metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_metric_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_service_timestamp ON metrics(service_name, timestamp DESC);

-- Commentaires
COMMENT ON TABLE metrics IS 'Table pour stocker les métriques des microservices';
COMMENT ON COLUMN metrics.service_name IS 'Nom du service (ex: ingestion-iiot, orchestrateur-maintenance)';
COMMENT ON COLUMN metrics.metric_name IS 'Nom de la métrique (ex: response_time, error_count)';
COMMENT ON COLUMN metrics.value IS 'Valeur de la métrique';
COMMENT ON COLUMN metrics.unit IS 'Unité de la métrique (ex: ms, count, %)';
COMMENT ON COLUMN metrics.labels IS 'Labels additionnels au format JSON';
COMMENT ON COLUMN metrics.timestamp IS 'Timestamp de la métrique';
COMMENT ON COLUMN metrics.created_at IS 'Date de création de l''enregistrement';

