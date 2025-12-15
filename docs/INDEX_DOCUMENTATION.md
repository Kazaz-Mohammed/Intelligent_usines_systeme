# Index de la Documentation - Maintenance Prédictive

## Vue d'Ensemble

Ce document sert d'index central pour toute la documentation de la plateforme de maintenance prédictive. Il organise et référence tous les documents créés pour faciliter la navigation.

---

## 📚 Documentation Disponible

### 1. Diagramme BPMN - Processus Métiers
**Fichier** : [`DIAGRAMME_BPMN.md`](./DIAGRAMME_BPMN.md)

**Contenu** :
- Processus principal : Cycle de vie de la maintenance prédictive
- Processus de gestion des anomalies critiques
- Processus de planification préventive basée sur RUL
- Processus d'optimisation de la planification
- Processus d'amélioration continue des modèles ML
- Processus d'intégration avec systèmes externes
- Rôles et responsabilités
- Métriques et KPIs
- Gestion des exceptions

**Utilisation** : Comprendre les processus métiers, les flux de travail et les règles de gestion.

---

### 2. Architecture Microservices
**Fichier** : [`ARCHITECTURE_MICROSERVICES.md`](./ARCHITECTURE_MICROSERVICES.md)

**Contenu** :
- Schéma d'architecture global
- Détail de chaque microservice :
  - Rôle et responsabilités
  - Technologies utilisées
  - Bases de données associées
  - Méthodes de communication (synchrone/asynchrone)
  - Ports et endpoints
- Matrice de communication entre services
- Infrastructure partagée (Kafka, PostgreSQL, MinIO, etc.)
- Patterns architecturaux
- Sécurité et scalabilité
- Monitoring et observabilité

**Utilisation** : Comprendre l'architecture technique, les technologies utilisées et les interactions entre services.

---

### 3. Diagrammes de Classes
**Fichier** : [`DIAGRAMMES_CLASSES.md`](./DIAGRAMMES_CLASSES.md)

**Contenu** :
- Diagramme de classes pour chaque microservice :
  1. Ingestion-IIoT
  2. Prétraitement
  3. Extraction-Features
  4. Détection-Anomalies
  5. Prédiction-RUL
  6. Orchestrateur-Maintenance
  7. Dashboard-Monitoring
- Description des classes principales
- Relations entre classes
- Relations inter-services

**Utilisation** : Comprendre la structure interne de chaque service, les responsabilités des classes et les dépendances.

---

### 4. Diagrammes de Cas d'Utilisation
**Fichier** : [`DIAGRAMMES_CAS_UTILISATION.md`](./DIAGRAMMES_CAS_UTILISATION.md)

**Contenu** :
- Diagrammes de cas d'utilisation pour chaque microservice
- Description détaillée de chaque cas d'utilisation :
  - Acteurs
  - Préconditions
  - Flux principal
  - Flux alternatifs
  - Postconditions
- Cas d'utilisation transversaux
- Relations entre cas d'utilisation
- Scénarios d'utilisation complets

**Utilisation** : Comprendre les fonctionnalités de chaque service, les interactions utilisateur et les scénarios d'usage.

---

## 🗺️ Navigation par Rôle

### Pour les Développeurs
1. Commencer par : **Architecture Microservices**
2. Ensuite : **Diagrammes de Classes**
3. Pour comprendre le contexte : **Diagramme BPMN**

### Pour les Architectes
1. Commencer par : **Architecture Microservices**
2. Ensuite : **Diagramme BPMN**
3. Complément : **Diagrammes de Classes**

### Pour les Analystes Métier
1. Commencer par : **Diagramme BPMN**
2. Ensuite : **Diagrammes de Cas d'Utilisation**
3. Complément : **Architecture Microservices** (vue d'ensemble)

### Pour les Data Scientists
1. Commencer par : **Architecture Microservices** (sections ML)
2. Ensuite : **Diagrammes de Classes** (services ML)
3. Complément : **Diagramme BPMN** (processus d'amélioration continue)

### Pour les Chefs de Projet
1. Vue d'ensemble : **Architecture Microservices**
2. Processus : **Diagramme BPMN**
3. Fonctionnalités : **Diagrammes de Cas d'Utilisation**

---

## 📊 Vue d'Ensemble des Microservices

### Liste des Services

| Service | Port | Langage | Framework | Base de Données |
|---------|------|---------|-----------|-----------------|
| Ingestion-IIoT | 8081 | Java | Spring Boot | PostgreSQL + TimescaleDB, MinIO |
| Prétraitement | 8082 | Python | FastAPI | PostgreSQL + TimescaleDB |
| Extraction-Features | 8083 | Python | FastAPI | PostgreSQL + TimescaleDB, Feast |
| Détection-Anomalies | 8084 | Python | FastAPI | PostgreSQL, MLflow |
| Prédiction-RUL | 8085 | Python | FastAPI | PostgreSQL, MLflow, MinIO |
| Dashboard-Monitoring | 8086 | Java | Spring Boot | PostgreSQL |
| Orchestrateur-Maintenance | 8087 | Java | Spring Boot | PostgreSQL |
| Dashboard-Usine | 3000 | TypeScript | React.js | - |

---

## 🔄 Flux de Données Principal

```
Données IIoT (OPC UA/Modbus/MQTT)
    ↓
[Ingestion-IIoT] → Kafka: sensor-data
    ↓
[Prétraitement] → Kafka: preprocessed-data
    ↓
[Extraction-Features] → Kafka: extracted-features
    ↓
    ├─→ [Détection-Anomalies] → Kafka: anomalies-detected
    └─→ [Prédiction-RUL] → Kafka: rul-predictions
    ↓
[Orchestrateur-Maintenance] → Kafka: work-orders
    ↓
[Dashboard-Monitoring] ← REST/WebSocket
    ↓
[Dashboard-Usine] (Visualisation)
```

---

## 🛠️ Technologies Principales

### Backend
- **Java Services** : Spring Boot 3.2.0, Java 17
- **Python Services** : FastAPI 0.104.1, Python 3.11
- **Frontend** : React.js 18+, TypeScript

### Infrastructure
- **Messaging** : Apache Kafka
- **Databases** : PostgreSQL + TimescaleDB
- **Object Storage** : MinIO
- **Cache** : Redis
- **ML Registry** : MLflow
- **Feature Store** : Feast

### ML/DL
- **Frameworks** : PyTorch, scikit-learn, XGBoost
- **Libraries** : PyOD, tsfresh, PyWavelets

### Optimisation
- **Rules Engine** : Drools
- **Optimization** : OR-Tools

---

## 📈 Métriques et KPIs

### Métriques Techniques
- Temps de réponse des APIs
- Throughput Kafka
- Taux d'erreur
- Disponibilité des services

### Métriques Métier
- Précision des prédictions RUL
- Taux de détection d'anomalies
- Temps de réaction aux anomalies critiques
- Taux d'utilisation des techniciens
- Coût de maintenance

---

## 🔐 Sécurité

### Authentification
- OAuth2 / JWT pour les APIs
- SSO pour le dashboard

### Autorisation
- RBAC (Role-Based Access Control)
- Permissions granulaires

### Chiffrement
- TLS pour toutes les communications
- Chiffrement des données sensibles

---

## 📝 Documentation Additionnelle

### Documentation par Service
Chaque service possède sa propre documentation dans son répertoire :
- `README.md` : Vue d'ensemble du service
- `ARCHITECTURE.md` : Architecture détaillée (si disponible)
- Guides spécifiques (Kafka, MLflow, etc.)

### Documentation Projet
- `README.md` : Documentation principale du projet
- `PROJECT_EXPLANATION.md` : Explication complète du projet
- `DEVELOPMENT_PLAN.md` : Plan de développement
- `CURRENT_STATUS.md` : État actuel du projet

---

## 🚀 Démarrage Rapide

### Pour Comprendre le Projet
1. Lire `README.md` (vue d'ensemble)
2. Lire `PROJECT_EXPLANATION.md` (détails)
3. Consulter `ARCHITECTURE_MICROSERVICES.md` (architecture)

### Pour Développer
1. Lire `ARCHITECTURE_MICROSERVICES.md` (technologies)
2. Consulter `DIAGRAMMES_CLASSES.md` (structure code)
3. Lire la documentation du service spécifique

### Pour Comprendre les Processus
1. Lire `DIAGRAMME_BPMN.md` (processus métiers)
2. Consulter `DIAGRAMMES_CAS_UTILISATION.md` (fonctionnalités)

---

## 📞 Support

Pour toute question ou clarification sur la documentation :
1. Consulter la documentation du service concerné
2. Vérifier les exemples de code dans les tests
3. Consulter les guides spécifiques (Kafka, MLflow, etc.)

---

## 🔄 Mise à Jour de la Documentation

Cette documentation est maintenue à jour avec le code. En cas de modification :
1. Mettre à jour les diagrammes concernés
2. Mettre à jour les descriptions
3. Vérifier la cohérence entre les documents

---

## 📅 Historique des Versions

- **v1.0.0** (Décembre 2024) : Documentation initiale complète
  - Diagramme BPMN
  - Architecture Microservices
  - Diagrammes de Classes
  - Diagrammes de Cas d'Utilisation

---

## ✅ Checklist de Lecture

Pour une compréhension complète du système :

- [ ] Vue d'ensemble : Architecture Microservices
- [ ] Processus métiers : Diagramme BPMN
- [ ] Structure technique : Diagrammes de Classes
- [ ] Fonctionnalités : Diagrammes de Cas d'Utilisation
- [ ] Documentation spécifique des services utilisés
- [ ] Guides d'intégration (Kafka, MLflow, etc.)

---

**Dernière mise à jour** : Décembre 2024

