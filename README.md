# Maintenance Prédictive Temps-Réel pour Usines Intelligentes

Plateforme de maintenance prédictive intégrant ML/DL, Data Mining (KNIME), et Architecture Microservices.

## 📚 Documentation du Projet

Ce projet combine **3 modules académiques** en une solution complète :

1. **ML & DL (Système d'Information Géographique)** : Modèles de prédiction RUL et détection d'anomalies
2. **Data Mining** : Analyse exploratoire avec KNIME Analytics Platform
3. **Architecture Microservices** : Système distribué avec Spring Boot, Docker, Kubernetes

### 📖 Documents Essentiels

- **[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)** : Explication complète du projet et intégration des 3 modules
- **[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)** : Plan de développement détaillé phase par phase avec tests
- **[AI_PROMPT_TEMPLATE.md](AI_PROMPT_TEMPLATE.md)** : Templates de prompts pour assistance IA
- **[GITHUB_WORKFLOW.md](GITHUB_WORKFLOW.md)** : Stratégie Git et workflow de développement

## 🎯 Objectif

Développer une plateforme temps-réel capable de :
- ✅ Détecter précocement les anomalies
- ✅ Estimer la Remaining Useful Life (RUL) des équipements
- ✅ Planifier des interventions optimales
- ✅ S'intégrer aux systèmes OT/IT (SCADA/MES/CMMS/ERP)

## 🏗️ Architecture

### 7 Microservices

```
IngestionIIoT → Prétraitement → ExtractionFeatures
                                      ↓
                    DétectionAnomalies + PrédictionRUL
                                      ↓
                    OrchestrateurMaintenance
                                      ↓
                        DashboardUsine (React + GIS)
```

1. **IngestionIIoT** : Collecte données PLC/SCADA (OPC UA, Modbus, MQTT)
2. **Prétraitement** : Nettoyage et normalisation des données
3. **ExtractionFeatures** : Calcul caractéristiques temporelles/fréquentielles
4. **DétectionAnomalies** : Détection anomalies temps-réel (PyOD, Autoencodeurs)
5. **PrédictionRUL** : Estimation RUL (LSTM/GRU/TCN custom)
6. **OrchestrateurMaintenance** : Planification optimisée (Drools, OR-Tools)
7. **DashboardUsine** : Interface temps-réel avec visualisations GIS

### Technologies

- **Backend Java** : Spring Boot, Eclipse Milo (OPC UA)
- **Backend Python** : FastAPI, PyTorch, PyOD
- **ML/DL** : Modèles custom (LSTM, GRU, TCN, Autoencodeurs)
- **Data Mining** : KNIME Analytics Platform
- **Messaging** : Apache Kafka
- **Databases** : PostgreSQL, TimescaleDB, InfluxDB, Feast, MLflow
- **Frontend** : React.js, WebSockets, Plotly, Grafana
- **GIS** : PostGIS, Leaflet/Mapbox
- **Infrastructure** : Docker, Kubernetes
- **Monitoring** : Prometheus, Grafana, OpenTelemetry

## 📊 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
- 21 capteurs
- 3 réglages moteur
- 4 scénarios de dégradation
- Format CSV

## 🚀 Démarrage Rapide

### Prérequis

- Docker & Docker Compose
- Java 17+ (pour services Spring Boot)
- Python 3.9+ (pour services Python)
- Node.js 18+ (pour frontend React)
- KNIME Analytics Platform (pour data mining)
- Git

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/Kazaz-Mohammed/usines_intelligentes.git
cd usines_intelligentes

# (À venir) Démarrer l'infrastructure
docker-compose up -d

# (À venir) Démarrer les services
# ...
```

## 📋 Plan de Développement

Le projet est organisé en **12 phases** :

1. **Phase 0** : Initialisation GitHub ✅
2. **Phase 1** : Infrastructure Docker
3. **Phase 2** : Service IngestionIIoT
4. **Phase 3** : Service Prétraitement
5. **Phase 4** : Service ExtractionFeatures
6. **Phase 5** : Data Mining KNIME
7. **Phase 6** : Service DétectionAnomalies
8. **Phase 7** : Service PrédictionRUL
9. **Phase 8** : Service OrchestrateurMaintenance
10. **Phase 9** : Service DashboardUsine
11. **Phase 10** : Intégration E2E
12. **Phase 11** : Déploiement Kubernetes
13. **Phase 12** : Finalisation Documentation

Voir [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) pour les détails.

## 🧪 Tests

Chaque phase inclut :
- Tests unitaires (couverture > 70%)
- Tests d'intégration
- Tests de performance
- Validation manuelle

**Règle** : Ne pas passer à la phase suivante sans validation complète de la phase actuelle.

## 📝 Workflow Git

### Branches

- `main` : Code production-ready
- `develop` : Développement principal
- `feature/[nom]` : Nouvelles fonctionnalités

### Convention de Commits

```
[TYPE][SERVICE] Description

Exemples:
[feat][ingestion-iiot] Ajout support OPC UA
[fix][preprocessing] Correction rééchantillonnage
[test][prediction-rul] Tests unitaires modèle LSTM
```

Voir [GITHUB_WORKFLOW.md](GITHUB_WORKFLOW.md) pour plus de détails.

## 🤖 Assistance IA

Utiliser les templates de [AI_PROMPT_TEMPLATE.md](AI_PROMPT_TEMPLATE.md) pour :
- Développement de chaque service
- Résolution de problèmes
- Optimisation
- Documentation

## 📦 Structure du Projet

```
projet/
├── services/
│   ├── ingestion-iiot/          # Service Spring Boot
│   ├── pre-traitement/          # Service FastAPI
│   ├── extraction-features/     # Service FastAPI
│   ├── detection-anomalies/     # Service FastAPI + ML
│   ├── prediction-rul/          # Service FastAPI + ML
│   ├── orchestrateur-maintenance/  # Service Spring Boot
│   └── dashboard-usine/         # Frontend React + Backend FastAPI
├── ml-models/
│   ├── rul-prediction/          # Modèles LSTM/GRU/TCN
│   └── anomaly-detection/       # Modèles PyOD + Autoencodeurs
├── data-mining/
│   └── knime-workflows/         # Workflows KNIME
├── datasets/
│   └── nasa-cmapss/             # Dataset NASA C-MAPSS
├── infrastructure/
│   ├── docker/                  # Dockerfiles
│   ├── kubernetes/              # Manifests K8s
│   └── docker-compose.yml       # Compose pour dev local
├── tests/                       # Tests E2E
├── docs/                        # Documentation
└── scripts/                     # Scripts utilitaires
```

## 🔒 Sécurité

- Pas de secrets/credentials dans le code
- Utilisation de variables d'environnement
- Chiffrement des communications (TLS)
- Authentification/Authorization (JWT)

## 📊 Monitoring

- **Prometheus** : Collecte métriques
- **Grafana** : Visualisation et dashboards
- **OpenTelemetry** : Traces distribuées
- **Logging** : Centralisé (ELK/Loki)

## 🎓 Modules Académiques

### Module 1 : ML & DL (GIS)
- Création modèles custom (pas de modules pré-existants)
- Architecture PyTorch pour LSTM/GRU/TCN
- Intégration GIS avec PostGIS

### Module 2 : Data Mining
- Workflows KNIME pour exploration
- Analyse statistique et sélection features
- Préparation données pour ML

### Module 3 : Microservices
- Architecture Spring Boot
- Communication REST/gRPC
- Déploiement Docker/Kubernetes
- Observabilité complète

## 📈 Résultats Attendus

À la fin du développement :
- ✅ 7 microservices fonctionnels
- ✅ Modèles ML/DL custom entraînés
- ✅ Workflows KNIME documentés
- ✅ Pipeline temps-réel opérationnel
- ✅ Dashboard interactif
- ✅ Infrastructure Kubernetes
- ✅ Documentation complète

## 🤝 Contribution

Ce projet est développé dans le cadre académique. Pour questions ou suggestions, créer une issue.

## 📄 Licence

(À définir selon besoins)

## 🔗 Liens Utiles

- [Documentation Spring Boot](https://spring.io/projects/spring-boot)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation PyTorch](https://pytorch.org/docs/)
- [Documentation KNIME](https://docs.knime.com/)
- [Documentation Kafka](https://kafka.apache.org/documentation/)
- [Documentation Kubernetes](https://kubernetes.io/docs/)

## 📞 Contact

- Repository : https://github.com/Kazaz-Mohammed/usines_intelligentes.git

---

## ⚠️ État Actuel

**Phase actuelle** : Phase 0 - Initialisation

Le projet est en cours de développement. Suivre le plan dans [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) pour voir l'avancement.

---

**Note** : Ce README sera mis à jour au fur et à mesure du développement.

