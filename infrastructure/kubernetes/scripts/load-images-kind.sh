#!/bin/bash

# Script pour charger toutes les images Docker dans kind

set -e

CLUSTER_NAME="predictive-maintenance"
SERVICES=(
    "ingestion-iiot"
    "preprocessing"
    "extraction-features"
    "detection-anomalies"
    "prediction-rul"
    "orchestrateur-maintenance"
    "dashboard-monitoring"
)

echo "📦 Chargement des images dans kind"
echo "==================================="
echo ""

# Vérifier que kind est installé
if ! command -v kind &> /dev/null; then
    echo "❌ kind n'est pas installé"
    echo "Installer: choco install kind"
    exit 1
fi

# Vérifier que le cluster existe
if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "❌ Cluster $CLUSTER_NAME n'existe pas"
    echo "Créer d'abord: ./scripts/setup-kind.sh"
    exit 1
fi

# Charger chaque image
for service in "${SERVICES[@]}"; do
    IMAGE="predictive-maintenance/$service:latest"
    
    echo "📤 Chargement de $IMAGE..."
    
    if docker images | grep -q "predictive-maintenance/$service"; then
        kind load docker-image $IMAGE --name $CLUSTER_NAME
        echo "✅ $service chargé"
    else
        echo "⚠️  Image $IMAGE non trouvée"
        echo "   Builder d'abord: docker build -t $IMAGE services/$service/"
    fi
    
    echo ""
done

echo "✅ Chargement terminé!"
echo ""
echo "📋 Images chargées dans kind:"
kind get nodes --name $CLUSTER_NAME | xargs -I {} docker exec {} crictl images | grep predictive-maintenance || echo "Aucune image trouvée"

