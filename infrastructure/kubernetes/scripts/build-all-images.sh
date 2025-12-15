#!/bin/bash

# Script pour builder toutes les images Docker

set -e

# Trouver la racine du projet (dossier contenant "services" et "infrastructure")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERVICES=(
    "ingestion-iiot"
    "preprocessing"
    "extraction-features"
    "detection-anomalies"
    "prediction-rul"
    "orchestrateur-maintenance"
    "dashboard-monitoring"
)

echo "🔨 Build de toutes les images Docker"
echo "======================================"
echo ""
echo "📁 Racine du projet: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Vérifier qu'on est dans le bon répertoire
if [ ! -d "services" ]; then
    echo "❌ Erreur: Répertoire 'services' non trouvé"
    echo "   Assurez-vous d'exécuter ce script depuis la racine du projet"
    exit 1
fi

for service in "${SERVICES[@]}"; do
    echo "📦 Building $service..."
    
    SERVICE_PATH="services/$service"
    if [ -d "$SERVICE_PATH" ]; then
        cd "$SERVICE_PATH"
        
        if [ -f "Dockerfile" ]; then
            echo "   📄 Dockerfile trouvé, build en cours..."
            docker build -t predictive-maintenance/$service:latest .
            echo "✅ $service buildé"
        else
            echo "⚠️  Dockerfile non trouvé pour $service dans $SERVICE_PATH"
        fi
        
        cd "$PROJECT_ROOT"
    else
        echo "⚠️  Service $service non trouvé dans $SERVICE_PATH"
    fi
    
    echo ""
done

echo "✅ Toutes les images sont buildées!"
echo ""
echo "📋 Images créées:"
docker images | grep predictive-maintenance

