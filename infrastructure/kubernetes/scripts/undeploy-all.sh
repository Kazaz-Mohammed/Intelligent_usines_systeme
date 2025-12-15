#!/bin/bash

# Script pour supprimer toute la plateforme de Kubernetes

set -e

echo "🗑️  Suppression de la plateforme Predictive Maintenance de Kubernetes..."

# Supprimer dans l'ordre inverse du déploiement
kubectl delete -f ingress/ --ignore-not-found=true
kubectl delete -f services/ --ignore-not-found=true
kubectl delete -f kafka/ --ignore-not-found=true
kubectl delete -f postgresql/ --ignore-not-found=true
kubectl delete -f configmaps/ --ignore-not-found=true
kubectl delete -f secrets/secrets.yaml --ignore-not-found=true
kubectl delete -f namespace.yaml --ignore-not-found=true

echo "✅ Suppression terminée!"

