# Phase 8 - Service OrchestrateurMaintenance - Démarrage

## Statut

🚧 **Phase 8 en cours - Structure de base créée**

## Ce qui a été fait

### 1. Structure de base ✅
- ✅ `pom.xml` avec toutes les dépendances :
  - Spring Boot 3.2.0
  - Drools 8.44.0 (moteur de règles)
  - OR-Tools 9.8.3296 (optimisation)
  - PostgreSQL, Kafka, Lombok, Jackson
- ✅ `OrchestrateurMaintenanceApplication.java` - Point d'entrée Spring Boot
- ✅ `application.yml` - Configuration complète
- ✅ `OrchestrateurConfig.java` - Configuration personnalisée

### 2. Modèles de données ✅
- ✅ `PriorityLevel` - Enum pour niveaux de priorité (CRITICAL, HIGH, MEDIUM, LOW)
- ✅ `InterventionRequest` - Requête pour créer une intervention
- ✅ `WorkOrder` - Entité JPA pour ordres de travail
- ✅ `WorkOrderStatus` - Enum pour statuts (PENDING, SCHEDULED, IN_PROGRESS, etc.)
- ✅ `MapToJsonConverter` - Convertisseur JPA pour JSONB

### 3. Contrôleurs ✅
- ✅ `HealthController` - Endpoints de santé (`/`, `/health`)

## Prochaines étapes

### 2. Moteur de règles Drools ⏳
- [ ] Configuration Drools (KieContainer, KieSession)
- [ ] Fichiers de règles (.drl) pour décisions de maintenance
- [ ] Service de règles (DroolsRuleService)
- [ ] Tests des règles

### 3. Optimisation OR-Tools ⏳
- [ ] Service d'optimisation (OptimizationService)
- [ ] Modèles d'optimisation (planning, allocation ressources)
- [ ] Résolution avec OR-Tools
- [ ] Tests d'optimisation

### 4. Service de planification ⏳
- [ ] Service de planification (PlanningService)
- [ ] Intégration Drools + OR-Tools
- [ ] Gestion des contraintes
- [ ] Tests de planification

### 5. Génération d'ordres de travail ⏳
- [ ] Service de génération (WorkOrderService)
- [ ] Repository JPA (WorkOrderRepository)
- [ ] Numérotation automatique
- [ ] Assignation aux techniciens

### 6. API REST ⏳
- [ ] `POST /api/v1/interventions` - Créer intervention
- [ ] `GET /api/v1/work-orders` - Lister ordres de travail
- [ ] `PUT /api/v1/work-orders/{id}` - Mettre à jour ordre
- [ ] `POST /api/v1/planning/optimize` - Optimiser planning

### 7. Intégration Kafka ⏳
- [ ] Consumer pour anomalies détectées
- [ ] Consumer pour prédictions RUL
- [ ] Producer pour ordres de travail
- [ ] Producer pour plans de maintenance

### 8. Tests ⏳
- [ ] Tests unitaires (services, règles)
- [ ] Tests d'intégration (API, Kafka)
- [ ] Tests d'optimisation

## Notes

- Architecture similaire à Phase 2 (IngestionIIoT) mais avec focus sur orchestration
- Drools pour règles métier complexes
- OR-Tools pour optimisation mathématique
- Intégration avec services Phase 6 (anomalies) et Phase 7 (RUL)

