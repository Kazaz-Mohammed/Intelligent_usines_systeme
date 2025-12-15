#!/bin/bash

# Script pour déployer toute la plateforme sur Kubernetes

set -e

echo "🚀 Déploiement de la plateforme Predictive Maintenance sur Kubernetes..."

# 1. Créer le namespace
echo "📦 Création du namespace..."
kubectl apply -f namespace.yaml

# 2. Créer les ConfigMaps
echo "⚙️  Création des ConfigMaps..."
kubectl apply -f configmaps/

# 3. Créer les Secrets (vérifier que secrets.yaml existe)
if [ -f "secrets/secrets.yaml" ]; then
    echo "🔐 Création des Secrets..."
    kubectl apply -f secrets/secrets.yaml
else
    echo "⚠️  secrets/secrets.yaml n'existe pas. Créer depuis secrets-template.yaml"
    exit 1
fi

# 4. Déployer PostgreSQL
echo "🐘 Déploiement de PostgreSQL..."
kubectl apply -f postgresql/

# 5. Attendre que PostgreSQL soit prêt
echo "⏳ Attente de PostgreSQL..."
kubectl wait --for=condition=ready pod -l app=postgresql -n predictive-maintenance --timeout=300s

# 6. Déployer Kafka
echo "📨 Déploiement de Kafka..."
kubectl apply -f kafka/

# 7. Attendre que Kafka soit prêt
echo "⏳ Attente de Kafka..."
kubectl wait --for=condition=ready pod -l app=kafka -n predictive-maintenance --timeout=300s

# 8. Déployer les services applicatifs
echo "🔧 Déploiement des services applicatifs..."
kubectl apply -f services/

# 9. Attendre que les services soient prêts
echo "⏳ Attente des services..."
kubectl wait --for=condition=ready pod -l app=ingestion-iiot -n predictive-maintenance --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=preprocessing -n predictive-maintenance --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=extraction-features -n predictive-maintenance --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=detection-anomalies -n predictive-maintenance --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=prediction-rul -n predictive-maintenance --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=orchestrateur-maintenance -n predictive-maintenance --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=dashboard-monitoring -n predictive-maintenance --timeout=300s || true

# 10. Déployer l'Ingress
echo "🌐 Déploiement de l'Ingress..."
kubectl apply -f ingress/

echo "✅ Déploiement terminé!"
echo ""
echo "📊 Statut des pods:"
kubectl get pods -n predictive-maintenance

echo ""
echo "🌐 Services:"
kubectl get services -n predictive-maintenance

