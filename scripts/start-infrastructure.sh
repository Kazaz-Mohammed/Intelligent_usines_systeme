#!/bin/bash

# Script de démarrage de l'infrastructure complète

echo "🚀 Démarrage de l'infrastructure Predictive Maintenance..."
echo ""

# Vérifier que Docker est en cours d'exécution
if ! docker info > /dev/null 2>&1; then
    echo "❌ Erreur: Docker n'est pas en cours d'exécution"
    exit 1
fi

# Aller dans le répertoire du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/infrastructure" || exit 1

# Copier .env.example vers .env si .env n'existe pas
if [ ! -f .env ]; then
    if [ -f ../.env.example ]; then
        echo "📋 Copie de .env.example vers .env..."
        cp ../.env.example .env
        echo "⚠️  N'oubliez pas de modifier .env avec vos valeurs de production!"
    else
        echo "⚠️  .env.example non trouvé, création de .env vide..."
        touch .env
    fi
fi

# Démarrer les services
echo "🐳 Démarrage des conteneurs Docker..."
docker-compose up -d

# Attendre que les services soient prêts
echo ""
echo "⏳ Attente du démarrage des services..."
sleep 15

# Vérifier l'état des services
echo ""
echo "📊 État des services:"
docker-compose ps

# Initialiser les bases de données (via init script dans PostgreSQL)
echo ""
echo "📦 PostgreSQL sera initialisé automatiquement via init script..."

# Initialiser les topics Kafka
echo ""
echo "📨 Initialisation des topics Kafka..."
bash ../scripts/init-kafka-topics.sh

# Initialiser les buckets MinIO
echo ""
echo "🪣 Initialisation des buckets MinIO..."
bash ../scripts/init-minio-buckets.sh

echo ""
echo "✅ Infrastructure démarrée avec succès!"
echo ""
echo "📍 Services disponibles:"
echo "   - Kafka: localhost:9092"
echo "   - Kafka UI: http://localhost:8080 (si activé avec --profile tools)"
echo "   - PostgreSQL: localhost:5432"
echo "   - pgAdmin: http://localhost:5050 (si activé avec --profile tools)"
echo "   - InfluxDB: http://localhost:8086"
echo "   - MinIO Console: http://localhost:9001"
echo "   - Redis: localhost:6379"
echo ""
echo "📝 Pour arrêter l'infrastructure: docker-compose down"
echo "📝 Pour voir les logs: docker-compose logs -f"

