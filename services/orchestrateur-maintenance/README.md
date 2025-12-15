# Service OrchestrateurMaintenance

## Description

Service Spring Boot responsable de la planification optimisée des interventions de maintenance. Il orchestre les décisions de maintenance basées sur les anomalies détectées et les prédictions RUL.

## Fonctionnalités

- **Moteur de règles Drools** : Décisions automatisées basées sur des règles métier
- **Optimisation planning avec OR-Tools** : Planification optimale des interventions
- **Génération d'ordres de travail** : Création automatique d'ordres de travail
- **Intégration CMMS/ERP** : Synchronisation avec systèmes externes
- **Gestion des contraintes** : Sécurité, SLA, inventaire, disponibilité

## Technologies

- **Spring Boot 3.2.0**
- **Drools 8.44.0** : Moteur de règles
- **OR-Tools 9.8.3296** : Optimisation
- **PostgreSQL** : Base de données
- **Kafka** : Messaging
- **Jackson** : Sérialisation JSON
- **Lombok** : Réduction du code boilerplate

## Structure

```
orchestrateur-maintenance/
├── src/
│   ├── main/
│   │   ├── java/com/predictivemaintenance/orchestrateur/
│   │   │   ├── OrchestrateurMaintenanceApplication.java
│   │   │   ├── config/
│   │   │   ├── controller/
│   │   │   ├── model/
│   │   │   ├── service/
│   │   │   ├── repository/
│   │   │   └── kafka/
│   │   └── resources/
│   │       └── application.yml
│   └── test/
└── pom.xml
```

## État

✅ **Phase 8 COMPLÉTÉE** - Service opérationnel et prêt pour la production

### Fonctionnalités implémentées

- ✅ **Moteur de règles Drools** : 12 règles de maintenance
- ✅ **Service de planification** : Orchestration Drools + Optimisation
- ✅ **Génération d'ordres de travail** : Création automatique
- ✅ **API REST** : 9 endpoints complets
- ✅ **Intégration Kafka** : Consumer/Producer temps-réel
- ✅ **Tests** : 8 classes de tests (unitaires + intégration)

## Installation

```bash
# Compiler le projet
mvn clean install

# Lancer l'application
mvn spring-boot:run
```

## Configuration

Les paramètres peuvent être configurés via variables d'environnement ou `application.yml` :

```yaml
spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/predictive_maintenance
    username: pmuser
    password: pmpassword
  
  kafka:
    bootstrap-servers: localhost:9092

orchestrateur:
  maintenance:
    kafka:
      topic-anomalies: anomalies-detected
      topic-rul-predictions: rul-predictions
    optimization:
      enabled: true
    constraints:
      max-technicians-per-shift: 10
    sla:
      critical-response-hours: 1
```

## API REST

### Health Check
- `GET /api/v1/health` - Statut du service
- `GET /api/v1/` - Informations du service

## Prochaines étapes

- [ ] Implémenter le moteur de règles Drools
- [ ] Implémenter l'optimisation avec OR-Tools
- [ ] Créer les services de planification
- [ ] Implémenter la génération d'ordres de travail
- [ ] Intégration Kafka (consumer/producer)
- [ ] Tests unitaires et intégration
- [ ] Documentation complète

## Notes

- Service similaire à IngestionIIoT (Phase 2) mais avec focus sur orchestration
- Utilise Drools pour les règles métier complexes
- OR-Tools pour l'optimisation mathématique du planning
