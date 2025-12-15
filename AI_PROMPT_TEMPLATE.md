# Prompt Template pour Assistant IA - Développement Projet Maintenance Prédictive

## Prompt Principal

```
Je développe une plateforme de maintenance prédictive temps-réel pour usines intelligentes. 
Voici le contexte et mes besoins spécifiques :

## Contexte du Projet

**Objectif** : Plateforme temps-réel capable de :
- Détecter précocement les anomalies
- Estimer la Remaining Useful Life (RUL) des équipements
- Planifier des interventions optimales
- S'intégrer aux systèmes OT/IT (SCADA/MES/CMMS/ERP)

**Dataset** : NASA C-MAPSS (CSV, 21 capteurs, 3 réglages moteur, 4 scénarios)

**Architecture** : 7 microservices avec les technologies suivantes :
- IngestionIIoT : Spring Boot, OPC UA (Eclipse Milo), Kafka, TimescaleDB, InfluxDB, MinIO
- Prétraitement : Python/FastAPI, Pandas, SciPy, Kafka Streams
- ExtractionFeatures : Python, tsfresh/tsflex, PyWavelets, Feast (feature store)
- DétectionAnomalies : PyOD, PyTorch (autoencodeurs), PostgreSQL
- PrédictionRUL : PyTorch (LSTM/GRU/TCN), XGBoost, MLflow
- OrchestrateurMaintenance : Spring Boot, Drools, OR-Tools, FastAPI
- DashboardUsine : React.js, WebSockets, Grafana, PostGIS

**Contraintes** :
- Pour ML/DL : Créer les modèles de zéro (pas de modules pré-existants)
- Pour Data Mining : Utiliser KNIME Analytics Platform
- Pour Microservices : Architecture Spring Boot, Docker, Kubernetes
- Déploiement avec Docker/Kubernetes
- Observabilité avec OpenTelemetry, Prometheus, Grafana

## Ma Demande Spécifique

[INSÉRER ICI LA DEMANDE SPÉCIFIQUE, exemples ci-dessous]

## Structure du Projet Actuel

[DESCRIRE L'ÉTAT ACTUEL : quelle phase, quels services existent, ce qui est fait]

## Ce dont j'ai besoin

[PRÉCISER : code, tests, documentation, configuration, etc.]
```

---

## Exemples de Prompts par Phase

### Phase 1 : Infrastructure Docker

```
Je suis en Phase 1 - Infrastructure Docker du projet de maintenance prédictive.

**Objectif** : Créer un docker-compose.yml avec :
- Kafka + Zookeeper
- PostgreSQL + extension TimescaleDB
- InfluxDB
- MinIO
- Redis

**Requis** :
- Configuration des variables d'environnement
- Health checks pour tous les services
- Scripts d'initialisation (création databases, topics Kafka, buckets MinIO)
- Volumes persistants pour les données

**Fichiers à créer** :
- docker-compose.yml
- Scripts d'initialisation dans infrastructure/scripts/
- .env.example avec toutes les variables

**Validation** : Tous les conteneurs doivent démarrer, être accessibles, et les health checks passer.

Peux-tu créer ces fichiers avec une configuration production-ready ?
```

---

### Phase 2 : Service IngestionIIoT

```
Je développe le service IngestionIIoT (Phase 2).

**Technologies** : Spring Boot, Eclipse Milo (OPC UA), Kafka, TimescaleDB, MinIO

**Fonctionnalités requises** :
1. Connecteur OPC UA pour lire données depuis simulateur
2. Normalisation des données (horodatage unifié, conversion unités)
3. Publication sur Kafka (topic "sensor-data")
4. Stockage dans TimescaleDB (table "raw_sensor_data")
5. Archivage données brutes dans MinIO
6. Gestion QoS et buffer en cas de perte réseau

**Structure Spring Boot** :
- Configuration OPC UA
- Service de normalisation
- Kafka producer
- Repository TimescaleDB
- Client MinIO
- REST API pour contrôle/statut

**Tests** :
- Tests unitaires avec données simulées
- Tests d'intégration avec simulateur OPC UA
- Tests de résilience (perte réseau simulée)

**Dataset** : Utiliser format similaire à NASA C-MAPSS pour tests (21 capteurs)

Peux-tu créer la structure complète du service avec toutes les dépendances et configurations ?
```

---

### Phase 3 : Service Prétraitement

```
Je développe le service Prétraitement (Phase 3).

**Technologies** : Python/FastAPI, Pandas, SciPy, Kafka

**Fonctionnalités requises** :
1. Consumer Kafka (topic "sensor-data")
2. Nettoyage données (détection valeurs aberrantes, imputation)
3. Rééchantillonnage (synchronisation multi-capteurs)
4. Débruitage (filtres passe-bande)
5. Analyse fréquentielle (STFT/FFT)
6. Fenêtrage glissant (fenêtres de 50-100 timestamps)
7. Publication sur Kafka (topic "preprocessed-data")
8. Stockage dans TimescaleDB (table "processed_windows")

**API FastAPI** :
- GET /health
- GET /stats
- POST /reprocess/{sensor_id}

**Tests** :
- Tests unitaires sur fonctions de nettoyage
- Tests avec dataset NASA C-MAPSS
- Validation qualité des fenêtres générées
- Tests de performance

Peux-tu créer le service complet avec tests et documentation ?
```

---

### Phase 4 : Service ExtractionFeatures

```
Je développe le service ExtractionFeatures (Phase 4).

**Technologies** : Python, tsfresh/tsflex, PyWavelets, Feast

**Fonctionnalités requises** :
1. Consumer Kafka (topic "preprocessed-data")
2. Calcul caractéristiques temporelles : RMS, kurtosis, crest factor, variance
3. Calcul caractéristiques fréquentielles : énergie de bande, centroides spectraux, order tracking
4. Transformées ondelettes (PyWavelets)
5. Feature store Feast (online + offline)
6. Standardisation par type d'actif (pompes, moteurs, convoyeurs, CNC)
7. Publication sur Kafka (topic "features")

**Feature Store Feast** :
- Configuration Feast
- Définition schema de features
- Entités : asset_id, timestamp
- Features : rms, kurtosis, crest_factor, spectral_centroid, etc.

**API FastAPI** :
- GET /features/{asset_id} (récupération depuis Feast)
- GET /health
- POST /compute (calcul manuel si besoin)

**Tests** :
- Validation calculs (comparaison manuelle sur échantillon)
- Tests Feast (écriture/lecture)
- Tests de performance

Peux-tu créer le service avec intégration Feast complète ?
```

---

### Phase 5 : Data Mining KNIME

```
Je travaille sur le module Data Mining avec KNIME (Phase 5).

**Objectif** : Créer des workflows KNIME pour :
1. Exploration dataset NASA C-MAPSS
2. Analyse statistique (corrélations, distributions)
3. Visualisations (scatter plots, heatmaps, time series)
4. Sélection de features importantes
5. Préparation données pour entraînement ML

**Workflows à créer** :
1. `01-exploration-data.knwf` : Exploration initiale
2. `02-statistical-analysis.knwf` : Analyses statistiques
3. `03-feature-selection.knwf` : Sélection de features
4. `04-data-preparation.knwf` : Préparation pour ML

**Documentation** :
- Description de chaque workflow
- Paramètres configurables
- Résultats attendus
- Guide d'utilisation

**Intégration** :
- Scripts Python pour exécution automatisée via KNIME Server API (si possible)
- Export des résultats (CSV, JSON)

Peux-tu créer les workflows KNIME avec documentation complète ?
Note : Je n'ai pas accès à KNIME ici, mais je peux utiliser les workflows que tu crées.
```

---

### Phase 6 : Service DétectionAnomalies

```
Je développe le service DétectionAnomalies (Phase 6).

**Technologies** : Python/FastAPI, PyOD, PyTorch, MLflow

**Fonctionnalités requises** :
1. Consumer Kafka (topic "features")
2. Modèles d'anomalies :
   - IsolationForest (PyOD)
   - One-Class SVM (PyOD)
   - LSTM Autoencoder (PyTorch - création custom)
3. Scoring temps-réel
4. Seuils adaptatifs par criticité actif
5. Journalisation événements dans PostgreSQL
6. MLflow tracking (expériences, métriques, modèles)

**Modèles Custom** :
- LSTM Autoencoder architecture :
  * Encoder : LSTM layers (64, 32, 16)
  * Decoder : LSTM layers (16, 32, 64)
  * Reconstruction error comme score d'anomalie
- Entraînement sur données normales uniquement
- Seuil basé sur percentile (ex: 95ème)

**API FastAPI** :
- POST /detect (scoring)
- GET /anomalies (historique)
- POST /train (ré-entraînement)
- GET /metrics (métriques MLflow)

**Tests** :
- Entraînement sur dataset NASA
- Validation métriques (Precision, Recall, F1)
- Tests avec anomalies connues
- Tests de performance temps-réel

Peux-tu créer le service avec modèles custom complets et MLflow intégration ?
```

---

### Phase 7 : Service PrédictionRUL

```
Je développe le service PrédictionRUL (Phase 7).

**Technologies** : Python/FastAPI, PyTorch, XGBoost, MLflow

**Fonctionnalités requises** :
1. Consumer Kafka (topic "features")
2. Modèles RUL (création custom, pas de modules pré-existants) :
   - LSTM : 2-3 couches LSTM (128, 64) + Dense
   - GRU : Architecture similaire LSTM
   - TCN : Temporal Convolutional Network custom
   - XGBoost : Baseline regression
3. Transfer Learning :
   - Pré-entraînement sur NASA C-MAPSS (scénarios 1-4)
   - Fine-tuning sur données usine simulées
4. Calibration : Intervalles de confiance (quantiles)
5. MLflow : Tracking, comparaison, registry

**Architecture Modèles** :
- Input : Séquences de features (fenêtres de 50-100 timestamps)
- Output : RUL + intervalles de confiance [lower, upper]
- Loss : MAE + quantile loss pour calibration

**Entraînement** :
- Dataset : NASA C-MAPSS (train/test split)
- Validation croisée temporelle
- Early stopping
- Hyperparameter tuning (optuna ou grid search)

**API FastAPI** :
- POST /predict (prédiction temps-réel)
- GET /model/info (info modèle actuel)
- POST /train (ré-entraînement)
- GET /metrics (métriques MLflow)

**Tests** :
- Entraînement validé sur NASA C-MAPSS
- Métriques : MAE < 10 cycles, RMSE acceptable
- Validation transfer learning
- Tests de performance

Peux-tu créer les modèles custom complets (sans utiliser de modules pré-existants) avec architecture détaillée, entraînement et MLflow intégration ?
```

---

### Phase 8 : Service OrchestrateurMaintenance

```
Je développe le service OrchestrateurMaintenance (Phase 8).

**Technologies** : Spring Boot, Drools, OR-Tools, FastAPI

**Fonctionnalités requises** :
1. Consumer Kafka (topics "anomalies", "rul-predictions")
2. Moteur de règles Drools :
   - Règles métier exemples :
     * Si RUL < 120h et stock_piece = 0 → Créer commande + inspection sous 24h
     * Si anomalie_critique et fenêtre_arrêt_disponible → Planifier intervention
     * Si RUL < 50h → Intervention urgente (bypass contraintes normales)
   - Contraintes : sécurité, SLA, fenêtres d'arrêt, inventaire
3. Optimisation planning (OR-Tools) :
   - Minimiser coûts (main-d'œuvre, arrêt production)
   - Contraintes : disponibilité techniciens, pièces, fenêtres
   - Allocation ressources optimale
4. Génération ordres de travail (format JSON standard CMMS)
5. Intégration simulée CMMS/ERP

**Structure Spring Boot** :
- Configuration Drools (KIE)
- Service d'optimisation OR-Tools (wrapper Python via API ou intégration Java)
- Repository PostgreSQL (référentiel actifs, BOM, criticité)
- REST API FastAPI ou Spring

**API** :
- POST /evaluate (évaluation règles)
- POST /optimize (optimisation planning)
- GET /work-orders (liste ordres)
- POST /work-orders (création)
- GET /assets/{id}/criticity

**Tests** :
- Tests règles Drools (scénarios multiples)
- Tests optimisation (validation solutions)
- Tests d'intégration

Peux-tu créer le service complet avec exemples de règles Drools et optimisation OR-Tools ?
```

---

### Phase 9 : Service DashboardUsine

```
Je développe le service DashboardUsine (Phase 9).

**Technologies** : React.js, FastAPI, WebSockets, Grafana, Plotly, PostGIS

**Fonctionnalités requises** :
1. Frontend React.js :
   - Dashboard temps-réel avec WebSockets
   - Visualisations Plotly (graphiques temps-réel, spectres)
   - Heatmaps de criticité par ligne/atelier
   - KPI : MTBF, MTTR, OEE, disponibilité
   - Drill-down par actif (détails, historique)
   - Intégration Grafana (embeddable)
2. Backend FastAPI :
   - REST API (métadonnées, historique)
   - WebSocket server (données temps-réel)
   - Export PDF/CSV
   - Intégration PostGIS pour cartographie atelier
3. Cartographie (GIS) :
   - Carte interactive d'atelier
   - Positionnement actifs avec statut visuel
   - Clustering par zone

**Structure React** :
- Components modulaires
- State management (Context API ou Redux)
- WebSocket client (reconnection automatique)
- Responsive design

**Pages** :
- Dashboard principal
- Vue détaillée actif
- Historique/Rapports
- Configuration

**API Backend** :
- GET /dashboard/summary (KPI globaux)
- GET /dashboard/assets (liste actifs avec statut)
- GET /assets/{id}/details
- GET /assets/{id}/history
- WebSocket : /ws/dashboard (stream temps-réel)
- POST /export/pdf
- POST /export/csv

**Tests** :
- Tests E2E (Cypress ou Playwright)
- Tests de charge WebSocket
- Tests responsive

Peux-tu créer le frontend React complet avec backend FastAPI et intégration WebSocket ?
```

---

### Phase 10 : Intégration E2E

```
Je suis en Phase 10 - Intégration E2E du projet.

**Objectif** : Intégrer tous les 7 services et valider le pipeline complet.

**Requis** :
1. Communication inter-services :
   - Service discovery
   - Load balancing
   - Retry policies
   - Circuit breakers (Resilience4j ou équivalent)
2. Observabilité :
   - OpenTelemetry (traces distribuées)
   - Logging centralisé (ELK ou Loki)
   - Monitoring Prometheus
   - Dashboards Grafana
3. Sécurité :
   - Authentification (JWT)
   - Authorization (RBAC)
   - Chiffrement communications (TLS)
   - Secrets management (Vault ou env)
4. Tests E2E :
   - Scénario complet : ingestion → prédiction → recommandation
   - Validation flux end-to-end
   - Performance globale (latence < 500ms)

**Configuration** :
- docker-compose.yml complet avec tous les services
- Configuration OpenTelemetry
- Configuration Prometheus/Grafana
- Configuration sécurité

**Tests** :
- Scripts de test E2E
- Données simulées complètes
- Validation métriques

Peux-tu créer la configuration complète d'intégration avec observabilité et sécurité ?
```

---

### Phase 11 : Kubernetes

```
Je suis en Phase 11 - Déploiement Kubernetes.

**Objectif** : Déployer la plateforme sur Kubernetes.

**Requis** :
1. Manifests Kubernetes :
   - Deployments pour chaque service
   - Services (ClusterIP, LoadBalancer si besoin)
   - ConfigMaps (configurations)
   - Secrets (credentials)
   - Ingress (routing HTTP)
   - StatefulSets (Kafka, databases)
2. Helm Charts (optionnel mais recommandé) :
   - Templates pour tous les services
   - Values files (dev, staging, prod)
   - Documentation
3. CI/CD :
   - GitHub Actions
   - Build Docker images
   - Tests automatiques
   - Déploiement automatique (staging)
4. Production Ready :
   - Health checks (liveness/readiness)
   - Resource limits/requests
   - Auto-scaling (HPA)
   - Rolling updates
   - Rollback strategy

**Configuration** :
- Namespace dédié
- Service mesh (Istio ou Linkerd) optionnel
- Ingress controller (Nginx ou Traefik)

Peux-tu créer tous les manifests Kubernetes et configuration CI/CD GitHub Actions ?
```

---

## Prompt Générique pour Corrections

```
Je travaille sur [NOM DU SERVICE/MODULE] de mon projet de maintenance prédictive.

**Problème rencontré** :
[DESCRIRE LE PROBLÈME]

**Contexte** :
- Technologies : [LISTER]
- État actuel : [CE QUI FONCTIONNE]
- Erreur : [LOGS/MESSAGES D'ERREUR]

**Ce que j'ai essayé** :
[ACTIONS DÉJÀ PRISES]

**Résultat attendu** :
[CE QUI DOIT FONCTIONNER]

Peux-tu m'aider à diagnostiquer et corriger le problème ?
```

---

## Prompt pour Optimisation

```
Je cherche à optimiser [COMPOSANT] de mon projet de maintenance prédictive.

**Composant** : [NOM]
**Métrique à optimiser** : [Performance, Latence, Précision, etc.]
**Valeur actuelle** : [VALEUR]
**Cible** : [VALEUR CIBLE]

**Contexte technique** :
[DESCRIPTION TECHNIQUE]

**Contraintes** :
- [CONTRAINTE 1]
- [CONTRAINTE 2]

Peux-tu proposer des optimisations avec explications et code si applicable ?
```

---

## Notes d'Utilisation

1. **Adaptez le prompt** selon votre phase et besoin spécifique
2. **Fournissez le contexte** : état actuel du code, erreurs, logs
3. **Soyez précis** sur les technologies et versions
4. **Mentionnez les contraintes** (pas de modules pré-existants pour ML, etc.)
5. **Demandez des tests** : toujours inclure des tests dans la demande

---

## Checklist avant d'envoyer le prompt

- [ ] Phase identifiée
- [ ] Technologies précisées
- [ ] Objectifs clairs
- [ ] Contraintes mentionnées
- [ ] État actuel décrit
- [ ] Résultat attendu défini

