#!/bin/bash
# Script pour exécuter les tests avec Docker

set -e

echo "=== Exécution des tests avec Docker ==="
echo ""

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier que Docker est disponible
if ! command -v docker &> /dev/null; then
    error "Docker n'est pas installé ou non disponible dans le PATH"
    exit 1
fi

info "Docker est disponible"

# Vérifier que docker-compose est disponible
if ! command -v docker-compose &> /dev/null; then
    error "docker-compose n'est pas installé ou non disponible dans le PATH"
    exit 1
fi

info "docker-compose est disponible"

# Options de test
TEST_TYPE="${1:-all}"
COVERAGE="${2:-false}"

info "Type de test: $TEST_TYPE"
info "Couverture: $COVERAGE"

# Construire l'image de test
info "Construction de l'image de test..."
docker build -f Dockerfile.test -t preprocessing-test:latest .

if [ $? -ne 0 ]; then
    error "Échec de la construction de l'image"
    exit 1
fi

info "Image construite avec succès"

# Créer le réseau si nécessaire
info "Création du réseau Docker..."
docker network create predictive-maintenance 2>/dev/null || warn "Réseau déjà existant"

# Démarrer les services dépendants
info "Démarrage des services dépendants (Kafka, PostgreSQL)..."
docker-compose -f docker-compose.test.yml up -d kafka zookeeper postgresql

# Attendre que les services soient prêts
info "Attente que les services soient prêts..."
sleep 30

# Vérifier que Kafka est prêt
info "Vérification de Kafka..."
timeout=60
elapsed=0
while ! docker exec kafka-test nc -z localhost 9092 2>/dev/null; do
    if [ $elapsed -ge $timeout ]; then
        error "Kafka n'est pas prêt après $timeout secondes"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
info "Kafka est prêt"

# Vérifier que PostgreSQL est prêt
info "Vérification de PostgreSQL..."
timeout=60
elapsed=0
while ! docker exec postgresql-test pg_isready -U pmuser -d predictive_maintenance 2>/dev/null; do
    if [ $elapsed -ge $timeout ]; then
        error "PostgreSQL n'est pas prêt après $timeout secondes"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
info "PostgreSQL est prêt"

# Exécuter les tests
info "Exécution des tests..."

case $TEST_TYPE in
    unit)
        info "Tests unitaires uniquement"
        docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v --tb=short -m "not integration"
        ;;
    integration)
        info "Tests d'intégration uniquement"
        docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v --tb=short -m integration
        ;;
    all)
        info "Tous les tests"
        if [ "$COVERAGE" = "true" ]; then
            docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v --tb=short --cov=app --cov-report=html --cov-report=term-missing
        else
            docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v --tb=short
        fi
        ;;
    *)
        error "Type de test inconnu: $TEST_TYPE"
        echo "Usage: $0 [unit|integration|all] [coverage=true|false]"
        exit 1
        ;;
esac

TEST_EXIT_CODE=$?

# Afficher les résultats
if [ $TEST_EXIT_CODE -eq 0 ]; then
    info "Tests réussis!"
else
    error "Tests échoués avec le code: $TEST_EXIT_CODE"
fi

# Arrêter les services (optionnel)
read -p "Arrêter les services? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Arrêt des services..."
    docker-compose -f docker-compose.test.yml down
fi

exit $TEST_EXIT_CODE

