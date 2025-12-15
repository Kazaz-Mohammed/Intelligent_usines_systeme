# Service Dashboard Usine

## Description

Interface web temps-réel pour la visualisation de l'état des équipements, alertes, RUL et planification de maintenance avec cartographie GIS.

## Fonctionnalités

- **Dashboard React.js temps-réel** : Interface moderne avec mises à jour en temps réel via WebSocket
- **Visualisations Plotly** : Graphiques interactifs pour features, RUL, anomalies
- **Intégration Grafana** : Panneaux Grafana intégrés
- **WebSockets** : Mises à jour live des données
- **Cartographie GIS (PostGIS)** : Carte interactive de l'usine avec positionnement des actifs
- **Export PDF/CSV** : Génération de rapports
- **KPI** : Calcul et affichage de MTBF, MTTR, OEE, disponibilité

## Technologies

### Backend
- **FastAPI** : Framework web asynchrone
- **PostgreSQL + PostGIS** : Base de données avec support spatial
- **WebSockets** : Communication temps-réel
- **Kafka** : Consommation des événements en temps réel
- **asyncpg** : Client PostgreSQL asynchrone

### Frontend
- **React.js 18** : Framework UI
- **TypeScript** : Typage statique
- **Vite** : Build tool moderne
- **Plotly.js** : Visualisations interactives
- **React Leaflet** : Cartes interactives
- **Socket.io Client** : WebSocket client

## Structure

```
dashboard-usine/
├── backend/
│   ├── app/
│   │   ├── api/          # Endpoints REST
│   │   ├── models/       # Modèles Pydantic
│   │   ├── services/     # Clients pour autres services
│   │   ├── database/     # PostgreSQL, PostGIS
│   │   ├── websocket/    # Gestion WebSocket
│   │   ├── config.py
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/   # Composants React
│   │   ├── pages/        # Pages
│   │   ├── services/     # API client
│   │   ├── hooks/        # Hooks React
│   │   ├── context/      # State management
│   │   └── types/        # Types TypeScript
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Installation

### Prérequis

- Python 3.11+
- Node.js 18+
- PostgreSQL avec extension PostGIS
- Kafka (pour les mises à jour temps-réel)

### Backend

```bash
cd backend
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Configuration

Copier `.env.example` vers `.env` et configurer :

```bash
# Backend
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=predictive_maintenance
EXTRACTION_FEATURES_URL=http://localhost:8083
DETECTION_ANOMALIES_URL=http://localhost:8084
PREDICTION_RUL_URL=http://localhost:8085
ORCHESTRATOR_URL=http://localhost:8087
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

## Démarrage

### Développement

**Backend :**
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8091 --reload
```

**Frontend :**
```bash
cd frontend
npm run dev
```

### Production (Docker)

```bash
docker-compose up -d
```

## API REST

### Assets
- `GET /api/v1/assets` - Liste des actifs
- `GET /api/v1/assets/{id}` - Détails d'un actif
- `GET /api/v1/assets/{id}/features` - Features d'un actif

### Anomalies
- `GET /api/v1/anomalies` - Liste des anomalies
- `GET /api/v1/anomalies/{id}` - Détails d'une anomalie

### RUL
- `GET /api/v1/rul` - Prédictions RUL
- `GET /api/v1/rul/{asset_id}/latest` - Dernière prédiction RUL

### Interventions
- `GET /api/v1/interventions` - Interventions de maintenance
- `GET /api/v1/interventions/active` - Interventions actives

### KPIs
- `GET /api/v1/kpis/summary` - Résumé des KPIs (MTBF, MTTR, OEE)
- `GET /api/v1/kpis/trend/{metric_name}` - Tendance d'un KPI

### GIS
- `GET /api/v1/gis/assets` - Localisations des actifs
- `GET /api/v1/gis/assets/within-radius` - Actifs dans un rayon
- `GET /api/v1/gis/floor-plan` - Plan d'étage

### Export
- `POST /api/v1/export/csv` - Export CSV
- `POST /api/v1/export/pdf` - Export PDF

### Grafana
- `GET /api/v1/grafana/dashboard-url` - URL du dashboard Grafana

## WebSocket

Endpoint : `WS /ws/dashboard`

Messages reçus :
- `feature_update` : Mise à jour de features
- `anomaly_detected` : Nouvelle anomalie détectée
- `rul_prediction` : Nouvelle prédiction RUL

## Migrations Base de Données

```bash
cd backend
python -m app.database.migrations
```

## État

✅ **Phase 9 COMPLÉTÉE** - Service opérationnel

### Fonctionnalités implémentées

- ✅ Backend FastAPI avec endpoints REST complets
- ✅ WebSocket pour mises à jour temps-réel
- ✅ Intégration avec tous les services (extraction-features, detection-anomalies, prediction-rul, orchestrateur-maintenance)
- ✅ Frontend React avec TypeScript
- ✅ Visualisations Plotly (RUL gauge, time series, heatmap)
- ✅ Cartographie GIS avec Leaflet
- ✅ Calcul des KPIs (MTBF, MTTR, OEE)
- ✅ Export PDF/CSV
- ✅ Pages de détail (Asset, Anomalies, Maintenance)

## Prochaines étapes

- [ ] Tests unitaires et d'intégration
- [ ] Amélioration des visualisations
- [ ] Optimisation des performances
- [ ] Documentation API complète
- [ ] Tests E2E avec Cypress/Playwright
