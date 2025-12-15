# Service Dashboard & Monitoring

## Description

Service Spring Boot pour le dashboard et le monitoring en temps réel de la plateforme de maintenance prédictive. Fournit des visualisations, métriques, alertes et statistiques pour tous les services.

## Fonctionnalités

- **Dashboard temps-réel** : Visualisation des données en temps réel
- **Métriques des services** : Monitoring de tous les microservices
- **Statistiques de maintenance** : KPIs et métriques de maintenance
- **Alertes et notifications** : Système d'alertes en temps réel
- **Graphiques et visualisations** : Charts pour données temporelles
- **Rapports** : Génération de rapports de maintenance

## Technologies

- **Spring Boot 3.2.0**
- **Spring WebFlux** : Pour le streaming temps-réel (WebSocket/SSE)
- **PostgreSQL** : Base de données pour historisation
- **Kafka** : Consommation des événements en temps réel
- **Prometheus** : Métriques
- **Grafana** : Visualisations (intégration)
- **WebSocket/SSE** : Streaming temps-réel
- **Chart.js/D3.js** : Visualisations côté client

## Structure

```
dashboard-monitoring/
├── src/
│   ├── main/
│   │   ├── java/com/predictivemaintenance/dashboard/
│   │   │   ├── DashboardApplication.java
│   │   │   ├── config/
│   │   │   ├── controller/
│   │   │   ├── service/
│   │   │   ├── model/
│   │   │   ├── websocket/
│   │   │   └── kafka/
│   │   └── resources/
│   │       ├── application.yml
│   │       └── static/ (HTML, CSS, JS)
│   └── test/
└── pom.xml
```

## État

✅ **Phase 9 COMPLÉTÉE** - Service opérationnel et prêt pour la production

### Fonctionnalités implémentées

- ✅ **Dashboard temps-réel** : WebSocket/SSE pour mises à jour automatiques
- ✅ **Monitoring des services** : Vérification automatique de la santé
- ✅ **Système d'alertes** : Gestion complète avec notifications
- ✅ **API REST** : 14 endpoints complets
- ✅ **Interface web** : Dashboard HTML/CSS/JS moderne
- ✅ **Tests** : 7 classes de tests (unitaires + intégration)

## Installation

```bash
# Compiler le projet
mvn clean install

# Lancer l'application
mvn spring-boot:run
```

## Configuration

```yaml
spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/predictive_maintenance
    username: pmuser
    password: pmpassword
  
  kafka:
    bootstrap-servers: localhost:9092

dashboard:
  monitoring:
    update-interval-seconds: 5
    metrics-retention-days: 30
    alerts:
      enabled: true
      email-enabled: false
      sms-enabled: false
```

## API REST

### Dashboard
- `GET /api/v1/dashboard/overview` - Vue d'ensemble
- `GET /api/v1/dashboard/metrics` - Métriques en temps réel
- `GET /api/v1/dashboard/statistics` - Statistiques agrégées

### Monitoring
- `GET /api/v1/monitoring/services` - Statut des services
- `GET /api/v1/monitoring/health` - Health check global
- `GET /api/v1/monitoring/metrics` - Métriques Prometheus

### Alertes
- `GET /api/v1/alerts` - Liste des alertes
- `POST /api/v1/alerts` - Créer une alerte
- `PUT /api/v1/alerts/{id}/acknowledge` - Acquitter une alerte

## WebSocket/SSE

- `WS /ws/dashboard` - WebSocket pour updates temps-réel
- `GET /sse/metrics` - Server-Sent Events pour métriques

## Prochaines étapes

- [ ] Créer la structure de base
- [ ] Implémenter les endpoints de dashboard
- [ ] Implémenter le monitoring des services
- [ ] Créer le système d'alertes
- [ ] Implémenter WebSocket/SSE pour temps-réel
- [ ] Créer l'interface web (HTML/CSS/JS)
- [ ] Intégration avec Prometheus/Grafana
- [ ] Tests et documentation

