#!/bin/bash

# Script pour configurer l'environnement E2E

set -e

echo "🔧 Configuration de l'environnement E2E..."

# Créer les topics Kafka
echo "📦 Création des topics Kafka..."
docker exec e2e-kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1 \
    --topic raw-sensor-data

docker exec e2e-kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1 \
    --topic preprocessed-data

docker exec e2e-kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1 \
    --topic extracted-features

docker exec e2e-kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1 \
    --topic anomalies-detected

docker exec e2e-kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1 \
    --topic rul-predictions

docker exec e2e-kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1 \
    --topic work-orders

echo "✅ Topics Kafka créés"

# Initialiser la base de données
echo "📊 Initialisation de la base de données..."
# Les services créeront automatiquement les tables au démarrage

echo "✅ Configuration E2E terminée!"

