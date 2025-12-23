# Maintenance Prédictive Temps-Réel pour Usines Intelligentes

Plateforme de maintenance prédictive intégrant ML/DL, Data Mining (KNIME), et Architecture Microservices pour la détection d'anomalies et la prédiction de la durée de vie résiduelle (RUL) des équipements industriels.

## 🎯 Vue d'Ensemble

Cette plateforme combine **3 modules académiques** en une solution complète :

1. **ML & DL** : Modèles de prédiction RUL (LSTM, XGBoost) et détection d'anomalies (Isolation Forest, Autoencodeurs)
2. **Data Mining** : Analyse exploratoire avec KNIME Analytics Platform
3. **Architecture Microservices** : Système distribué avec Spring Boot, FastAPI, Docker, Kubernetes

## 🏗️ Architecture du Système

### Flux de Données

```
IngestionIIoT → Prétraitement → ExtractionFeatures
                                      ↓
                    DétectionAnomalies + PrédictionRUL
                                      ↓
                    OrchestrateurMaintenance
                                      ↓
                        DashboardUsine (React + GIS)
```

### 7 Microservices

1. **Ingestion-IIoT** (Spring Boot) : Collecte données PLC/SCADA via OPC UA, Modbus, MQTT
2. **Prétraitement** (FastAPI) : Nettoyage, normalisation et validation des données
3. **Extraction-Features** (FastAPI) : Calcul caractéristiques temporelles/fréquentielles
4. **Détection-Anomalies** (FastAPI + ML) : Détection anomalies temps-réel avec Isolation Forest et Autoencodeurs
5. **Prédiction-RUL** (FastAPI + ML) : Estimation RUL avec LSTM et XGBoost
6. **Orchestrateur-Maintenance** (Spring Boot) : Planification optimisée des interventions
7. **Dashboard-Usine** (React + FastAPI) : Interface temps-réel avec visualisations GIS

### Infrastructure

- **Messaging** : Apache Kafka (Zookeeper)
- **Databases** : PostgreSQL (TimescaleDB), InfluxDB, MinIO (S3-compatible)
- **Cache** : Redis
- **Monitoring** : Prometheus, Grafana (optionnel)
- **Tools** : Kafka UI, pgAdmin, OPC UA Simulator (optionnel)

## 📊 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
- 21 capteurs
- 3 réglages moteur
- 4 scénarios de dégradation
- Format CSV

## 🚀 Installation et Démarrage

### Prérequis

- **Docker** & **Docker Compose** (version 3.8+)
- **Git**
- **8GB RAM minimum** (recommandé: 16GB)
- **Ports disponibles** : 3000, 4840, 5050, 5432, 6379, 8080-8091, 9000-9001, 9092-9093

### Installation

1. **Cloner le dépôt**
```bash
git clone https://github.com/Kazaz-Mohammed/usines_intelligentes.git
cd usines_intelligentes
```

2. **Configurer les variables d'environnement**
```bash
# Copier le fichier d'exemple
cp env.example .env

# Éditer .env avec vos valeurs (optionnel, valeurs par défaut disponibles)
# POSTGRES_DB=predictive_maintenance
# POSTGRES_USER=pmuser
# POSTGRES_PASSWORD=pmpassword
# MINIO_ROOT_USER=minioadmin
# MINIO_ROOT_PASSWORD=minioadmin
# INFLUXDB_TOKEN=pm-token-change-in-production
```

3. **Initialiser l'infrastructure** (Kafka, PostgreSQL, MinIO, etc.)
```bash
# Windows PowerShell
.\scripts\init-kafka-topics.ps1
.\scripts\init-minio-buckets.ps1

# Linux/Mac
chmod +x scripts/*.sh
./scripts/init-kafka-topics.sh
./scripts/init-minio-buckets.sh
```

4. **Démarrer tous les services**
```bash
# Démarrer l'infrastructure et les services
docker-compose up -d

# Vérifier le statut
docker-compose ps

# Voir les logs
docker-compose logs -f
```

5. **Démarrer avec outils de développement** (Kafka UI, pgAdmin, OPC UA Simulator)
```bash
docker-compose --profile tools up -d
```

### Accès aux Services

- **Dashboard Frontend** : http://localhost:3000
- **Dashboard Backend API** : http://localhost:8091
- **Kafka UI** : http://localhost:8080 (si activé avec `--profile tools`)
- **pgAdmin** : http://localhost:5050 (si activé avec `--profile tools`)
- **MinIO Console** : http://localhost:9001 (minioadmin/minioadmin)
- **OPC UA Simulator** : opc.tcp://localhost:4840 (si activé)

### Services API

- **Ingestion-IIoT** : http://localhost:8081
- **Prétraitement** : http://localhost:8082
- **Extraction-Features** : http://localhost:8083
- **Détection-Anomalies** : http://localhost:8084
- **Prédiction-RUL** : http://localhost:8085
- **Orchestrateur-Maintenance** : http://localhost:8087

## 📁 Structure du Projet

```
usines_intelligentes/
├── services/                    # Microservices
│   ├── ingestion-iiot/          # Service Spring Boot
│   ├── preprocessing/           # Service FastAPI
│   ├── extraction-features/     # Service FastAPI
│   ├── detection-anomalies/     # Service FastAPI + ML
│   ├── prediction-rul/          # Service FastAPI + ML
│   ├── orchestrateur-maintenance/ # Service Spring Boot
│   └── dashboard-usine/         # Frontend React + Backend FastAPI
├── ml_pipeline/                 # Pipeline ML (entraînement modèles)
│   ├── ml_pipeline_tutorial.ipynb
│   └── saved_models/            # Modèles entraînés
├── data-mining/                 # Workflows KNIME
├── datasets/                    # Dataset NASA C-MAPSS
├── infrastructure/              # Configuration K8s, scripts
├── scripts/                     # Scripts utilitaires
├── docs/                        # Documentation technique
├── docker-compose.yml           # Configuration Docker Compose
└── README.md                    # Ce fichier
```

## 🔧 Utilisation

### 1. Démarrer le système complet

```bash
docker-compose up -d
```

### 2. Vérifier la santé des services

```bash
# Vérifier tous les services
docker-compose ps

# Vérifier un service spécifique
curl http://localhost:8081/health
curl http://localhost:8082/health
curl http://localhost:8083/health
curl http://localhost:8084/health
curl http://localhost:8085/health
curl http://localhost:8087/health
curl http://localhost:8091/health
```

### 3. Entraîner les modèles ML

Voir [ml_pipeline/README.md](ml_pipeline/README.md) pour les instructions d'entraînement.

### 4. Tester avec OPC UA Simulator

```bash
# Démarrer le simulateur OPC UA
docker-compose --profile tools up -d opcua-simulator

# Le service Ingestion-IIoT se connectera automatiquement
```

### 5. Arrêter le système

```bash
docker-compose down

# Supprimer aussi les volumes (⚠️ supprime les données)
docker-compose down -v
```

## 🧪 Tests

```bash
# Tests unitaires (dans chaque service)
cd services/[service-name]
# Python: pytest
# Java: ./mvnw test

# Tests d'intégration
docker-compose up -d
# Exécuter les scripts de test dans scripts/
```

## 📊 Monitoring

- **Logs** : `docker-compose logs -f [service-name]`
- **Métriques** : Prometheus (si configuré)
- **Visualisation** : Grafana (si configuré)
- **Kafka** : Kafka UI (http://localhost:8080)

## 🔒 Sécurité

- ⚠️ **Important** : Changer tous les mots de passe par défaut en production
- Utiliser des variables d'environnement pour les secrets
- Activer TLS/SSL pour les communications
- Configurer l'authentification JWT

## 🛠️ Technologies

- **Backend Java** : Spring Boot 3.x, Eclipse Milo (OPC UA)
- **Backend Python** : FastAPI, PyTorch, scikit-learn, XGBoost
- **ML/DL** : PyTorch (LSTM), XGBoost, Isolation Forest, Autoencodeurs
- **Data Mining** : KNIME Analytics Platform
- **Messaging** : Apache Kafka
- **Databases** : PostgreSQL (TimescaleDB), InfluxDB, MinIO
- **Frontend** : React.js, Next.js, WebSockets, Plotly
- **Infrastructure** : Docker, Docker Compose, Kubernetes
- **Monitoring** : Prometheus, Grafana

## 📝 Documentation

- [Documentation Architecture](docs/ARCHITECTURE_MICROSERVICES.md)
- [ML Pipeline](ml_pipeline/README.md)
- [Infrastructure](infrastructure/README.md)
- [Scripts](scripts/README.md)

## 🤝 Contribution

Ce projet est développé dans le cadre académique. Pour questions ou suggestions, créer une issue sur GitHub.

## 📄 Licence

(À définir selon besoins)

## 🔗 Liens Utiles

- Repository : https://github.com/Kazaz-Mohammed/usines_intelligentes.git
- [Documentation Spring Boot](https://spring.io/projects/spring-boot)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation PyTorch](https://pytorch.org/docs/)
- [Documentation Kafka](https://kafka.apache.org/documentation/)

---

**Note** : Ce système est en développement actif. Consulter la documentation dans `docs/` pour plus de détails.
