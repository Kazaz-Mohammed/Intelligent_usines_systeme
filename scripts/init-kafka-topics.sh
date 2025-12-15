#!/bin/bash

# Script d'initialisation des topics Kafka
# Ce script doit être exécuté après le démarrage de Kafka

KAFKA_CONTAINER="kafka"
KAFKA_BOOTSTRAP_SERVER="localhost:9092"

# Liste des topics à créer
TOPICS=(
    "sensor-data:3:1"           # Données brutes des capteurs (3 partitions, 1 replica)
    "preprocessed-data:3:1"       # Données prétraitées (3 partitions, 1 replica)
    "features:3:1"                # Caractéristiques extraites (3 partitions, 1 replica)
    "anomalies:3:1"               # Événements d'anomalies (3 partitions, 1 replica)
    "rul-predictions:3:1"         # Prédictions RUL (3 partitions, 1 replica)
    "maintenance-orders:3:1"      # Ordres de maintenance (3 partitions, 1 replica)
)

echo "Initialisation des topics Kafka..."

# Fonction pour créer un topic
create_topic() {
    local topic_config=$1
    IFS=':' read -r topic_name partitions replicas <<< "$topic_config"
    
    echo "Création du topic: $topic_name (partitions: $partitions, replicas: $replicas)"
    
    docker exec -it $KAFKA_CONTAINER kafka-topics \
        --create \
        --bootstrap-server $KAFKA_BOOTSTRAP_SERVER \
        --topic "$topic_name" \
        --partitions "$partitions" \
        --replication-factor "$replicas" \
        --if-not-exists \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Topic $topic_name créé avec succès"
    else
        echo "⚠️  Topic $topic_name existe déjà ou erreur lors de la création"
    fi
}

# Attendre que Kafka soit prêt
echo "Attente du démarrage de Kafka..."
sleep 10

# Créer tous les topics
for topic_config in "${TOPICS[@]}"; do
    create_topic "$topic_config"
done

# Lister tous les topics
echo ""
echo "Topics Kafka disponibles:"
docker exec -it $KAFKA_CONTAINER kafka-topics \
    --list \
    --bootstrap-server $KAFKA_BOOTSTRAP_SERVER

echo ""
echo "✅ Initialisation des topics Kafka terminée"

