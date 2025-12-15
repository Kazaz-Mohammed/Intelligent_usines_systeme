#!/bin/bash

# Script de health check complet après déploiement

set -e

NAMESPACE="predictive-maintenance"
TIMEOUT=300

echo "🏥 Health Check - Plateforme Predictive Maintenance"
echo "====================================================="
echo ""

# Fonction pour attendre qu'un déploiement soit prêt
wait_for_deployment() {
    local deployment=$1
    echo "⏳ Attente de $deployment..."
    
    kubectl wait --for=condition=available \
        deployment/$deployment \
        -n $NAMESPACE \
        --timeout=${TIMEOUT}s || {
        echo "❌ $deployment n'est pas prêt après ${TIMEOUT}s"
        return 1
    }
    
    echo "✅ $deployment est prêt"
    return 0
}

# Fonction pour tester un endpoint
test_endpoint() {
    local service=$1
    local port=$2
    local path=$3
    
    echo "🧪 Test de $service..."
    
    # Port-forward
    kubectl port-forward -n $NAMESPACE service/$service $port:$port > /dev/null 2>&1 &
    local pf_pid=$!
    sleep 5
    
    # Test
    if curl -f -s "http://localhost:$port$path" > /dev/null 2>&1; then
        echo "✅ $service accessible"
        kill $pf_pid 2>/dev/null || true
        return 0
    else
        echo "❌ $service non accessible"
        kill $pf_pid 2>/dev/null || true
        return 1
    fi
}

# 1. Vérifier le namespace
echo "1. Vérification du namespace..."
if kubectl get namespace $NAMESPACE &>/dev/null; then
    echo "✅ Namespace existe"
else
    echo "❌ Namespace n'existe pas"
    exit 1
fi

# 2. Attendre les déploiements infrastructure
echo ""
echo "2. Attente des services infrastructure..."
wait_for_deployment "postgresql"
wait_for_deployment "zookeeper"
wait_for_deployment "kafka"

# 3. Attendre les services applicatifs
echo ""
echo "3. Attente des services applicatifs..."
wait_for_deployment "ingestion-iiot"
wait_for_deployment "preprocessing"
wait_for_deployment "extraction-features"
wait_for_deployment "detection-anomalies"
wait_for_deployment "prediction-rul"
wait_for_deployment "orchestrateur-maintenance"
wait_for_deployment "dashboard-monitoring"

# 4. Vérifier les pods
echo ""
echo "4. Statut des pods:"
kubectl get pods -n $NAMESPACE

# Compter les pods prêts
READY_PODS=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase=Running --no-headers | wc -l)
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE --no-headers | wc -l)

echo ""
echo "📊 Pods: $READY_PODS/$TOTAL_PODS prêts"

# 5. Tester les endpoints
echo ""
echo "5. Tests des endpoints de santé..."
test_endpoint "ingestion-iiot-service" "8081" "/health" || true
test_endpoint "preprocessing-service" "8082" "/health" || true
test_endpoint "extraction-features-service" "8083" "/health" || true
test_endpoint "detection-anomalies-service" "8084" "/health" || true
test_endpoint "prediction-rul-service" "8085" "/health" || true
test_endpoint "orchestrateur-maintenance-service" "8087" "/health" || true
test_endpoint "dashboard-monitoring-service" "8086" "/health" || true

# 6. Vérifier les services
echo ""
echo "6. Services:"
kubectl get services -n $NAMESPACE

# 7. Vérifier les ressources
echo ""
echo "7. Utilisation des ressources:"
kubectl top pods -n $NAMESPACE 2>/dev/null || echo "⚠️  metrics-server non disponible"

echo ""
echo "✅ Health check terminé!"

