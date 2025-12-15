#!/bin/bash

# Script pour attendre que tous les services soient prêts

set -e

echo "⏳ Attente du démarrage des services..."

# Fonction pour vérifier si un service est prêt
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=60
    local attempt=0
    
    echo "Vérification de $service_name..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name est prêt"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo "Tentative $attempt/$max_attempts pour $service_name..."
        sleep 2
    done
    
    echo "❌ $service_name n'est pas prêt après $max_attempts tentatives"
    return 1
}

# Attendre les services infrastructure
wait_for_service "PostgreSQL" "http://localhost:5432" || true
wait_for_service "Kafka" "http://localhost:9092" || true

# Attendre les services applicatifs
wait_for_service "IngestionIIoT" "http://localhost:8081/health"
wait_for_service "Preprocessing" "http://localhost:8082/health"
wait_for_service "ExtractionFeatures" "http://localhost:8083/health"
wait_for_service "DetectionAnomalies" "http://localhost:8084/health"
wait_for_service "PredictionRUL" "http://localhost:8085/health"
wait_for_service "OrchestrateurMaintenance" "http://localhost:8087/health"
wait_for_service "DashboardMonitoring" "http://localhost:8086/health"

echo "✅ Tous les services sont prêts!"

