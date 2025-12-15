# Guide Kafka - Service OrchestrateurMaintenance

## Vue d'ensemble

Le service `orchestrateur-maintenance` consomme les anomalies et prédictions RUL depuis Kafka et publie automatiquement les ordres de travail générés.

## Fonctionnalités

### ✅ Consumer Kafka
- **Topic `anomalies-detected`** : Consomme les anomalies détectées
- **Topic `rul-predictions`** : Consomme les prédictions RUL
- Traitement automatique avec création d'interventions
- Gestion des erreurs et retry

### ✅ Producer Kafka
- **Topic `work-orders`** : Publie les ordres de travail créés
- **Topic `maintenance-plans`** : Publie les plans de maintenance
- Publication batch supportée
- Callbacks de livraison

## Configuration

Dans `application.yml` :

```yaml
spring:
  kafka:
    bootstrap-servers: ${KAFKA_BOOTSTRAP_SERVERS:localhost:9092}
    consumer:
      group-id: orchestrateur-maintenance-service
      auto-offset-reset: earliest
    producer:
      acks: all
      retries: 3

orchestrateur:
  maintenance:
    kafka:
      topic-anomalies: anomalies-detected
      topic-rul-predictions: rul-predictions
      topic-work-orders: work-orders
      topic-maintenance-plans: maintenance-plans
```

## Format des Messages

### Input : Anomalie Détectée

**Topic** : `anomalies-detected`

```json
{
    "asset_id": "PUMP_001",
    "sensor_id": "SENSOR_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "final_score": 0.95,
    "is_anomaly": true,
    "criticality": "critical"
}
```

### Input : Prédiction RUL

**Topic** : `rul-predictions`

```json
{
    "asset_id": "PUMP_001",
    "sensor_id": "SENSOR_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "rul_prediction": 30.5,
    "confidence_interval_lower": 20.0,
    "confidence_interval_upper": 40.0,
    "model_used": "ensemble"
}
```

### Output : Ordre de Travail

**Topic** : `work-orders`

```json
{
    "workOrderNumber": "WO-1705312345678",
    "assetId": "PUMP_001",
    "sensorId": "SENSOR_001",
    "priority": "CRITICAL",
    "interventionType": "corrective",
    "status": "SCHEDULED",
    "scheduledStartTime": "2024-01-15T11:30:00",
    "scheduledEndTime": "2024-01-15T13:30:00",
    "estimatedDurationMinutes": 120,
    "assignedTechnicianId": null,
    "assignedTeam": null
}
```

### Output : Plan de Maintenance

**Topic** : `maintenance-plans`

```json
{
    "assetId": "PUMP_001",
    "planType": "preventive",
    "scheduledDate": "2024-01-20T08:00:00",
    "workOrderNumbers": ["WO-001", "WO-002"],
    "description": "Plan de maintenance préventive"
}
```

## Flux de Traitement

### 1. Anomalie Détectée

```
Kafka (anomalies-detected)
    ↓
KafkaConsumerService.consumeAnomaly()
    ↓
convertAnomalyToInterventionRequest()
    ↓
KafkaOrchestrationService.processAnomalyIntervention()
    ↓
PlanningService.planIntervention()
    ↓
DroolsRuleService.evaluateIntervention()
    ↓
WorkOrderService.save()
    ↓
KafkaProducerService.publishWorkOrder()
    ↓
Kafka (work-orders)
```

### 2. Prédiction RUL

```
Kafka (rul-predictions)
    ↓
KafkaConsumerService.consumeRulPrediction()
    ↓
convertRulToInterventionRequest()
    ↓
KafkaOrchestrationService.processRulIntervention()
    ↓
shouldCreateIntervention() (vérification)
    ↓
Si RUL < 200 ou priorité CRITICAL/HIGH:
    PlanningService.planIntervention()
    ↓
    WorkOrderService.save()
    ↓
    KafkaProducerService.publishWorkOrder()
    ↓
    Kafka (work-orders)
```

## Utilisation

### Démarrer le Service

Le service démarre automatiquement les consumers Kafka au démarrage :

```bash
mvn spring-boot:run
```

Les consumers sont actifs et traitent les messages en temps-réel.

### Vérifier les Consumers

```bash
# Voir les groupes de consommateurs
kafka-consumer-groups --bootstrap-server localhost:9092 --list

# Voir les lag des consumers
kafka-consumer-groups --bootstrap-server localhost:9092 \
    --group orchestrateur-maintenance-anomalies-group --describe
```

### Publier un Message de Test

```bash
# Publier une anomalie
kafka-console-producer --bootstrap-server localhost:9092 \
    --topic anomalies-detected <<EOF
{"asset_id":"TEST_001","sensor_id":"SENSOR_001","timestamp":"2024-01-15T10:30:00Z","final_score":0.95,"is_anomaly":true,"criticality":"critical"}
EOF

# Publier une prédiction RUL
kafka-console-producer --bootstrap-server localhost:9092 \
    --topic rul-predictions <<EOF
{"asset_id":"TEST_001","sensor_id":"SENSOR_001","timestamp":"2024-01-15T10:30:00Z","rul_prediction":30.0,"confidence_interval_lower":20.0,"confidence_interval_upper":40.0,"model_used":"ensemble"}
EOF
```

## Gestion des Erreurs

### Erreurs de Désérialisation
- Les messages invalides sont loggés et ignorés
- Le message n'est pas acknowledgé pour permettre le retry

### Erreurs de Traitement
- Les erreurs sont loggées avec stack trace
- Le message n'est pas acknowledgé pour permettre le retry
- Kafka retry automatique selon la configuration

### Dead Letter Queue (Futur)
- Implémenter un DLQ pour les messages en échec répétés
- Monitoring des messages en échec

## Performance

### Configuration Optimisée
- **Concurrency** : 3 threads pour traitement parallèle
- **Batch Size** : 10 messages par poll
- **Ack Mode** : MANUAL_IMMEDIATE (acknowledgement manuel)
- **Idempotence** : Activée pour le producer

### Recommandations
- Ajuster `concurrency` selon le nombre de partitions
- Monitorer le lag des consumers
- Ajuster `max.poll.records` selon la charge

## Monitoring

### Métriques Kafka
- Lag des consumers
- Taux de traitement
- Erreurs de désérialisation
- Temps de traitement

### Logs
```bash
# Voir les logs de consommation
grep "Message reçu" logs/application.log

# Voir les erreurs
grep "Erreur lors du traitement" logs/application.log
```

## Troubleshooting

### Consumer ne consomme pas
- Vérifier que Kafka est démarré
- Vérifier les topics existent
- Vérifier le group-id
- Vérifier les logs pour erreurs

### Messages non traités
- Vérifier le format JSON
- Vérifier les logs pour erreurs de désérialisation
- Vérifier que les services (PlanningService, etc.) fonctionnent

### Producer ne publie pas
- Vérifier la connexion Kafka
- Vérifier les logs pour erreurs
- Vérifier que les topics existent

## Exemples

### Exemple 1 : Traitement d'une Anomalie

1. **Message reçu** :
```json
{
    "asset_id": "PUMP_001",
    "is_anomaly": true,
    "criticality": "critical",
    "final_score": 0.95
}
```

2. **Intervention créée** :
- Priorité : CRITICAL (déterminée par Drools)
- Type : corrective
- Temps de réponse : 1 heure (SLA)

3. **Ordre de travail publié** :
```json
{
    "workOrderNumber": "WO-1705312345678",
    "assetId": "PUMP_001",
    "priority": "CRITICAL",
    "status": "SCHEDULED"
}
```

### Exemple 2 : Traitement d'une RUL

1. **Message reçu** :
```json
{
    "asset_id": "PUMP_002",
    "rul_prediction": 25.0
}
```

2. **Vérification** : RUL < 200 → Intervention nécessaire

3. **Intervention créée** :
- Priorité : CRITICAL (RUL < 50)
- Type : predictive
- Temps de réponse : 1 heure (SLA)

4. **Ordre de travail publié**

## Notes

- Les consumers sont automatiquement démarrés au démarrage du service
- Le traitement est asynchrone et non-bloquant
- Les messages sont acknowledgés seulement après traitement réussi
- Support batch pour traitement de plusieurs messages

