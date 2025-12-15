#!/bin/bash

# Script pour nettoyer l'environnement E2E

set -e

echo "🧹 Nettoyage de l'environnement E2E..."

# Arrêter et supprimer les conteneurs
echo "🛑 Arrêt des conteneurs..."
docker-compose -f docker-compose.e2e.yml down -v

# Supprimer les volumes (optionnel)
read -p "Supprimer les volumes? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Suppression des volumes..."
    docker volume prune -f
fi

echo "✅ Nettoyage terminé!"

