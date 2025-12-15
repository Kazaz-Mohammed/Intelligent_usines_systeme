#!/bin/bash

# Script simplifié pour builder toutes les images Docker
# À exécuter depuis la racine du projet

set -e

echo "🔨 Build de toutes les images Docker"
echo "======================================"
echo ""

# Vérifier qu'on est dans la racine du projet
if [ ! -d "services" ]; then
    echo "❌ Erreur: Ce script doit être exécuté depuis la racine du projet"
    echo "   Exemple: cd ~/Desktop/Predictive\ Maintenance\ Projet"
    exit 1
fi

SERVICES=(
    "ingestion-iiot"
    "preprocessing"
    "extraction-features"
    "detection-anomalies"
    "prediction-rul"
    "orchestrateur-maintenance"
    "dashboard-monitoring"
)

for service in "${SERVICES[@]}"; do
    echo "📦 Building $service..."
    
    if [ -d "services/$service" ]; then
        cd "services/$service"
        
        if [ -f "Dockerfile" ]; then
            docker build -t predictive-maintenance/$service:latest .
            echo "✅ $service buildé"
        else
            echo "⚠️  Dockerfile non trouvé pour $service"
        fi
        
        cd - > /dev/null
    else
        echo "⚠️  Service $service non trouvé"
    fi
    
    echo ""
done

echo "✅ Toutes les images sont buildées!"
echo ""
echo "📋 Images créées:"
docker images | grep predictive-maintenance || echo "Aucune image trouvée"

