#!/bin/bash

# Script pour tester le déploiement Kubernetes

set -e

NAMESPACE="predictive-maintenance"

echo "🧪 Tests du déploiement Kubernetes"
echo "=================================="
echo ""

# Fonction pour vérifier le statut d'un déploiement
check_deployment() {
    local deployment=$1
    echo "📋 Vérification de $deployment..."
    
    if kubectl get deployment $deployment -n $NAMESPACE &>/dev/null; then
        local ready=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
        local desired=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.replicas}')
        
        if [ "$ready" == "$desired" ] && [ "$ready" != "0" ]; then
            echo "✅ $deployment: $ready/$desired pods prêts"
            return 0
        else
            echo "⚠️  $deployment: $ready/$desired pods prêts"
            return 1
        fi
    else
        echo "❌ $deployment: Déploiement non trouvé"
        return 1
    fi
}

# Fonction pour vérifier les pods
check_pods() {
    echo ""
    echo "📦 Statut des pods:"
    kubectl get pods -n $NAMESPACE
    
    echo ""
    echo "🔍 Pods non prêts:"
    kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running || true
}

# Fonction pour vérifier les services
check_services() {
    echo ""
    echo "🌐 Services:"
    kubectl get services -n $NAMESPACE
}

# Fonction pour tester un endpoint
test_endpoint() {
    local service=$1
    local port=$2
    local path=$3
    
    echo ""
    echo "🧪 Test de $service sur $path..."
    
    # Port-forward en arrière-plan
    kubectl port-forward -n $NAMESPACE service/$service $port:$port > /dev/null 2>&1 &
    local pf_pid=$!
    sleep 3
    
    # Tester l'endpoint
    if curl -f -s "http://localhost:$port$path" > /dev/null 2>&1; then
        echo "✅ $service accessible sur http://localhost:$port$path"
        kill $pf_pid 2>/dev/null || true
        return 0
    else
        echo "❌ $service non accessible"
        kill $pf_pid 2>/dev/null || true
        return 1
    fi
}

# Vérifier le namespace
echo "1. Vérification du namespace..."
if kubectl get namespace $NAMESPACE &>/dev/null; then
    echo "✅ Namespace $NAMESPACE existe"
else
    echo "❌ Namespace $NAMESPACE n'existe pas"
    exit 1
fi

# Vérifier les ConfigMaps
echo ""
echo "2. Vérification des ConfigMaps..."
kubectl get configmaps -n $NAMESPACE

# Vérifier les Secrets
echo ""
echo "3. Vérification des Secrets..."
kubectl get secrets -n $NAMESPACE

# Vérifier les déploiements
echo ""
echo "4. Vérification des déploiements..."
check_deployment "postgresql"
check_deployment "zookeeper"
check_deployment "kafka"
check_deployment "ingestion-iiot"
check_deployment "preprocessing"
check_deployment "extraction-features"
check_deployment "detection-anomalies"
check_deployment "prediction-rul"
check_deployment "orchestrateur-maintenance"
check_deployment "dashboard-monitoring"

# Vérifier les pods
check_pods

# Vérifier les services
check_services

# Tester les endpoints
echo ""
echo "5. Tests des endpoints..."
test_endpoint "ingestion-iiot-service" "8081" "/health" || true
test_endpoint "detection-anomalies-service" "8084" "/health" || true
test_endpoint "prediction-rul-service" "8085" "/health" || true
test_endpoint "dashboard-monitoring-service" "8086" "/health" || true

echo ""
echo "✅ Tests terminés!"

