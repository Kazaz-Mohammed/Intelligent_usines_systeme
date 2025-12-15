# Tests d'Intégration End-to-End (E2E)

## Description

Tests d'intégration end-to-end pour valider le fonctionnement complet de la plateforme de maintenance prédictive, de l'ingestion des données jusqu'au dashboard.

## Architecture E2E

```
IngestionIIoT → Prétraitement → ExtractionFeatures → DetectionAnomalies → PredictionRUL → OrchestrateurMaintenance → DashboardMonitoring
```

## Prérequis

- Docker et Docker Compose
- Java 17+
- Python 3.11+
- Maven 3.8+
- Kafka (via Docker)
- PostgreSQL (via Docker)

## Structure

```
tests-e2e/
├── src/
│   ├── java/ (Tests Java)
│   └── python/ (Tests Python)
├── docker-compose.e2e.yml
├── scripts/
│   ├── setup-e2e.sh
│   ├── run-e2e-tests.sh
│   └── cleanup-e2e.sh
└── README.md
```

## Services testés

1. **IngestionIIoT** (Port 8081)
2. **Prétraitement** (Port 8082)
3. **ExtractionFeatures** (Port 8083)
4. **DetectionAnomalies** (Port 8084)
5. **PredictionRUL** (Port 8085)
6. **OrchestrateurMaintenance** (Port 8087)
7. **DashboardMonitoring** (Port 8086)

## Scénarios de test

### Scénario 1 : Flux complet de détection d'anomalie
1. Ingestion de données IoT
2. Prétraitement des données
3. Extraction de caractéristiques
4. Détection d'anomalie
5. Création d'intervention
6. Affichage dans le dashboard

### Scénario 2 : Flux complet de prédiction RUL
1. Ingestion de données IoT
2. Prétraitement des données
3. Extraction de caractéristiques
4. Prédiction RUL
5. Planification d'intervention
6. Affichage dans le dashboard

### Scénario 3 : Monitoring et alertes
1. Vérification de la santé des services
2. Création d'alertes
3. Notification via WebSocket
4. Affichage dans le dashboard

## Exécution

```bash
# Démarrer tous les services
docker-compose -f docker-compose.e2e.yml up -d

# Attendre que tous les services soient prêts
./scripts/wait-for-services.sh

# Exécuter les tests E2E
./scripts/run-e2e-tests.sh

# Nettoyer
./scripts/cleanup-e2e.sh
```

## État

🚧 **Phase 10 en cours - Tests E2E à créer**

