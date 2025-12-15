# Explication du Projet - Maintenance Prédictive Temps-Réel

## Vue d'Ensemble

Ce projet combine **3 modules académiques** en une solution intégrée de maintenance prédictive pour usines intelligentes :

1. **ML & DL (Système d'Information Géographique)** : Modèles de machine learning et deep learning pour prédiction RUL et détection d'anomalies, avec intégration GIS pour visualisation spatiale
2. **Data Mining** : Exploration et analyse de données avec KNIME Analytics Platform
3. **Architecture Microservices** : Système distribué avec Spring Boot, Docker, Kubernetes

---

## Comment Développer ce Projet

### Approche Modulaire

Le projet est divisé en **7 microservices indépendants** qui communiquent via Kafka (messaging) et APIs REST/gRPC. Chaque service peut être développé et testé séparément.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ IngestionIIoT│────▶│Prétraitement│────▶│Extraction  │
│             │     │             │     │Features    │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                    ┌─────────────┐     ┌─────────────┐
                    │Détection    │◀────│Prédiction   │
                    │Anomalies    │     │RUL          │
                    └─────────────┘     └─────────────┘
                             │                │
                             ▼                ▼
                    ┌──────────────────────────────┐
                    │ OrchestrateurMaintenance     │
                    └──────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────────────────┐
                    │ DashboardUsine               │
                    │ (React + Visualisations)      │
                    └──────────────────────────────┘
```

### Flux de Données

1. **Collecte** : Les capteurs industriels (vibration, température, courant) envoient des données
2. **Ingestion** : Service IngestionIIoT collecte via OPC UA/Modbus/MQTT
3. **Prétraitement** : Nettoyage, normalisation, fenêtrage
4. **Extraction Features** : Calcul de caractéristiques (RMS, kurtosis, fréquences)
5. **ML/DL** : 
   - Détection d'anomalies (modèles custom)
   - Prédiction RUL (LSTM/GRU/TCN custom)
6. **Orchestration** : Planification optimisée des interventions
7. **Visualisation** : Dashboard temps-réel avec cartographie

---

## Intégration des 3 Modules

### Module 1 : ML & DL (Système d'Information Géographique)

**Services concernés** :
- `ExtractionFeatures` : Calcul de features
- `DétectionAnomalies` : Modèles custom (PyOD, Autoencodeurs)
- `PrédictionRUL` : Modèles custom (LSTM/GRU/TCN)
- `DashboardUsine` : Visualisation GIS avec PostGIS

**Contraintes importantes** :
- ✅ **Créer les modèles de zéro** (pas de modules pré-existants)
- ✅ Architecture custom avec PyTorch
- ✅ Intégration GIS pour visualisation spatiale des actifs

**Technologies** :
- PyTorch pour modèles DL
- PyOD pour détection anomalies (baseline, puis custom)
- PostGIS pour données géospatiales
- React.js + Leaflet/Mapbox pour visualisation

**Dataset** : NASA C-MAPSS pour entraînement initial

---

### Module 2 : Data Mining avec KNIME

**Services concernés** :
- Analyse exploratoire en amont du ML
- Sélection de features
- Validation des données

**Approche** :
1. Créer workflows KNIME pour :
   - Exploration dataset NASA C-MAPSS
   - Analyse statistique (corrélations, distributions)
   - Visualisations
   - Sélection de features importantes
   - Préparation données pour entraînement

2. Intégration avec pipeline :
   - Export des résultats KNIME
   - Utilisation dans service ExtractionFeatures
   - Validation des features sélectionnées

**Workflows à créer** :
- `01-exploration-data.knwf`
- `02-statistical-analysis.knwf`
- `03-feature-selection.knwf`
- `04-data-preparation.knwf`

**Intégration** :
- Les workflows KNIME produisent des insights
- Ces insights guident le développement des services ML
- Validation croisée entre KNIME et modèles ML

---

### Module 3 : Architecture Microservices

**Tous les services** utilisent cette architecture :

**Stack Technologique** :
- **Spring Boot** : Services Java (Ingestion, Orchestrateur)
- **FastAPI** : Services Python (ML/DL services)
- **Docker** : Conteneurisation
- **Kubernetes** : Orchestration
- **Kafka** : Messaging asynchrone
- **PostgreSQL/TimescaleDB** : Base de données
- **REST/gRPC** : Communication synchrone
- **Swagger/OpenAPI** : Documentation API
- **Prometheus/Grafana** : Monitoring

**Principes** :
- Services indépendants et déployables séparément
- Communication via APIs bien définies
- Découplage via Kafka (event-driven)
- Observabilité (OpenTelemetry)
- Scalabilité horizontale

**Patterns** :
- Circuit breakers pour résilience
- Retry policies
- Service discovery
- Configuration centralisée

---

## Stratégie de Développement

### 1. Développement Incrémental

Commencer par l'infrastructure, puis chaque service un par un :

```
Phase 0: Infrastructure → Docker Compose
Phase 1: Ingestion → Données dans le système
Phase 2: Prétraitement → Données propres
Phase 3: Extraction Features → Caractéristiques calculées
Phase 4: Data Mining KNIME → Analyse exploratoire
Phase 5: Détection Anomalies → Alertes précoces
Phase 6: Prédiction RUL → Estimation durée de vie
Phase 7: Orchestrateur → Planification
Phase 8: Dashboard → Visualisation
Phase 9: Intégration E2E → Pipeline complet
Phase 10: Kubernetes → Déploiement production
```

### 2. Test-First Approach

**Pour chaque phase** :
1. Écrire les tests d'abord (TDD si possible)
2. Implémenter la fonctionnalité
3. Valider tous les tests passent
4. Mesurer la performance
5. Documenter
6. Commit et push sur GitHub

### 3. Validation Continue

**À chaque étape** :
- ✅ Tests unitaires (couverture > 70%)
- ✅ Tests d'intégration
- ✅ Tests de performance
- ✅ Validation manuelle si nécessaire
- ✅ Documentation à jour

### 4. Version Control Strict

**Workflow Git** :
- Branche `feature/[nom]` pour chaque développement
- Push fréquent (minimum quotidien)
- Pull Request vers `develop`
- Code review avant merge
- Tag après chaque phase complète

---

## Points Clés par Module

### Module ML/DL

**Important** : Créer les modèles de zéro signifie :
- ❌ Pas d'utilisation de modèles pré-entraînés
- ❌ Pas de libraries "plug-and-play" pour RUL
- ✅ Implémenter architecture LSTM/GRU/TCN manuellement
- ✅ Entraîner sur dataset NASA C-MAPSS
- ✅ Fine-tuning avec transfer learning

**Architecture Recommandée LSTM RUL** :
```python
# Exemple conceptuel
class RULLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=21, hidden_size=128, num_layers=2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1)
        self.fc = nn.Linear(64, 1)  # RUL output
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Last timestep
        rul = self.fc(out)
        return rul
```

### Module Data Mining KNIME

**Objectif** : Explorer et préparer les données, pas remplacer les services ML

**Workflow** :
1. Import dataset NASA C-MAPSS
2. Analyse descriptive (statistiques, distributions)
3. Corrélations entre capteurs
4. Sélection features (importance, redondance)
5. Préparation pour ML (normalisation, split train/test)
6. Export résultats (CSV, JSON)

**Intégration** :
- Les insights KNIME guident le choix de features
- Validation croisée avec modèles ML
- Documentation des décisions

### Module Microservices

**Architecture Distribuée** :
- Services indépendants (peuvent être développés séparément)
- Communication asynchrone (Kafka) pour découplage
- APIs REST/gRPC pour communication synchrone
- Databases par service (polyglot persistence)

**Défis à Gérer** :
- Gestion des erreurs distribuées
- Latence réseau
- Cohérence des données (eventual consistency)
- Monitoring distribué (OpenTelemetry)

---

## Intégration des Technologies

### Pipeline Complet

```
Données Industrielles (PLC/SCADA)
    ↓
[IngestionIIoT] → Kafka → TimescaleDB, MinIO
    ↓
[Prétraitement] → Kafka → TimescaleDB
    ↓
[ExtractionFeatures] → Feast → Kafka
    ↓
    ├─→ [DétectionAnomalies] → PostgreSQL (événements)
    └─→ [PrédictionRUL] → MLflow (modèles)
    ↓
[OrchestrateurMaintenance] → PostgreSQL (ordres)
    ↓
[DashboardUsine] ← WebSocket ← APIs REST
    ↓
Visualisation Temps-Réel (React + GIS)
```

### Technologies Clés par Service

| Service | Tech Principal | DB | Messaging |
|---------|---------------|-----|-----------|
| IngestionIIoT | Spring Boot | TimescaleDB, MinIO | Kafka |
| Prétraitement | FastAPI | TimescaleDB | Kafka |
| ExtractionFeatures | FastAPI | Feast | Kafka |
| DétectionAnomalies | FastAPI | PostgreSQL | Kafka |
| PrédictionRUL | FastAPI | MLflow, MinIO | Kafka |
| OrchestrateurMaintenance | Spring Boot | PostgreSQL | Kafka |
| DashboardUsine | React.js | PostgreSQL, TimescaleDB | WebSocket |

---

## Conseils de Développement

### 1. Commencer Simple

- Version minimale fonctionnelle d'abord (MVP)
- Itérer et améliorer
- Ajouter complexité progressivement

### 2. Utiliser des Simulateurs

- Simulateur OPC UA pour tests
- Dataset NASA C-MAPSS pour ML
- Données synthétiques pour validation

### 3. Documentation Continue

- Documenter chaque service
- APIs avec Swagger/OpenAPI
- README par service
- Diagrammes d'architecture

### 4. Tests Automatisés

- CI/CD avec GitHub Actions
- Tests automatiques avant merge
- Validation automatique des builds

### 5. Monitoring Tôt

- Ajouter logging tôt
- Configuration Prometheus/Grafana dès le début
- Métriques clés définies

---

## Résultat Attendu

À la fin du développement, vous aurez :

1. ✅ **7 microservices** fonctionnels et déployables
2. ✅ **Modèles ML/DL custom** entraînés et validés
3. ✅ **Workflows KNIME** documentés pour data mining
4. ✅ **Pipeline temps-réel** opérationnel
5. ✅ **Dashboard interactif** avec visualisations
6. ✅ **Infrastructure Kubernetes** pour production
7. ✅ **Documentation complète** selon standards SoftwareX
8. ✅ **Code versionné** sur GitHub avec historique

---

## Prochaines Étapes

1. **Lire** le `DEVELOPMENT_PLAN.md` pour le plan détaillé
2. **Consulter** `AI_PROMPT_TEMPLATE.md` pour les prompts IA
3. **Démarrer** par Phase 0 : Initialisation GitHub
4. **Suivre** le plan phase par phase
5. **Tester** à chaque étape avant de continuer

---

## Questions Fréquentes

**Q : Puis-je utiliser des modèles pré-entraînés ?**
R : Non, les modèles ML/DL doivent être créés de zéro (contrainte module).

**Q : KNIME est-il obligatoire ?**
R : Oui pour le module Data Mining. Les workflows KNIME sont nécessaires.

**Q : Puis-je utiliser d'autres frameworks que Spring Boot ?**
R : Pour les services Java, Spring Boot est recommandé (module Microservices). Pour Python, FastAPI est approprié.

**Q : Comment tester sans vrais capteurs industriels ?**
R : Utiliser des simulateurs OPC UA et le dataset NASA C-MAPSS pour développement.

**Q : Kubernetes est-il obligatoire ?**
R : Oui pour le déploiement production (module Microservices). Docker Compose suffit pour développement local.

