#!/bin/bash

# Script pour exécuter les tests E2E

set -e

echo "🚀 Démarrage des tests E2E..."

# Vérifier que les services sont prêts
./scripts/wait-for-services.sh

# Exécuter les tests Python
if [ -d "src/python" ]; then
    echo "📝 Exécution des tests Python..."
    cd src/python
    python -m pytest -v
    cd ../..
fi

# Exécuter les tests Java
if [ -d "src/java" ]; then
    echo "☕ Exécution des tests Java..."
    cd src/java
    mvn test
    cd ../..
fi

echo "✅ Tests E2E terminés!"

