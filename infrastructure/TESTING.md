# Guide de Test - Infrastructure Phase 1

## Tests à Effectuer

### 1. Validation de la Configuration

```bash
cd infrastructure
docker-compose config
```

**Attendu** : Aucune erreur de configuration

### 2. Démarrage des Services

```bash
docker-compose up -d
```

**Attendu** : Tous les services démarrent sans erreur

### 3. Vérification de l'État des Services

```bash
docker-compose ps
```

**Attendu** : 
- Tous les services en état "Up"
- Health checks "healthy" pour tous les services

### 4. Tests de Connectivité

#### PostgreSQL

```bash
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "SELECT version();"
```

**Attendu** : Version de PostgreSQL affichée

#### Vérifier TimescaleDB

```bash
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "SELECT * FROM pg_extension WHERE extname = 'timescaledb';"
```

**Attendu** : Extension timescaledb listée

#### Vérifier les Tables

```bash
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "\dt"
```

**Attendu** : Tables créées (raw_sensor_data, processed_windows, anomaly_events, etc.)

#### Kafka

```bash
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Attendu** : Liste des topics (peut être vide avant initialisation)

#### InfluxDB

Accéder à http://localhost:8086 dans un navigateur

**Attendu** : Interface web InfluxDB accessible

#### MinIO

Accéder à http://localhost:9001 dans un navigateur

**Attendu** : Interface console MinIO accessible (se connecter avec minioadmin/minioadmin par défaut)

#### Redis

```bash
docker exec -it redis redis-cli -a redispassword PING
```

**Attendu** : Réponse "PONG"

### 5. Initialisation des Topics Kafka

```powershell
# Windows
..\scripts\init-kafka-topics.ps1

# Linux/Mac
../scripts/init-kafka-topics.sh
```

**Attendu** : 6 topics créés avec succès

Vérifier :
```bash
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Attendu** : Topics listés : sensor-data, preprocessed-data, features, anomalies, rul-predictions, maintenance-orders

### 6. Initialisation des Buckets MinIO

```powershell
# Windows
..\scripts\init-minio-buckets.ps1

# Linux/Mac
../scripts/init-minio-buckets.sh
```

**Attendu** : 5 buckets créés avec succès

Vérifier via l'interface web MinIO (http://localhost:9001)

**Attendu** : Buckets visibles : raw-sensor-data, processed-data, model-artifacts, mlflow-artifacts, backups

### 7. Test d'Insertion dans PostgreSQL

```bash
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "INSERT INTO raw_sensor_data (time, asset_id, sensor_id, value, unit) VALUES (NOW(), 'ASSET001', 'SENSOR001', 25.5, '°C'); SELECT * FROM raw_sensor_data LIMIT 1;"
```

**Attendu** : Insertion réussie et données retournées

### 8. Test de Kafka (Publication/Consommation)

#### Publier un message

```bash
docker exec -it kafka kafka-console-producer --bootstrap-server localhost:9092 --topic sensor-data
# Taper un message de test, puis Ctrl+C
```

#### Consommer un message (dans un autre terminal)

```bash
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic sensor-data --from-beginning
```

**Attendu** : Message reçu

### 9. Test Redis

```bash
docker exec -it redis redis-cli -a redispassword SET test_key "test_value"
docker exec -it redis redis-cli -a redispassword GET test_key
```

**Attendu** : "test_value"

### 10. Vérification des Logs

```bash
docker-compose logs --tail=50
```

**Attendu** : Pas d'erreurs critiques

## Checklist de Validation

- [ ] Tous les conteneurs démarrent sans erreur
- [ ] Health checks passent pour tous les services
- [ ] PostgreSQL accessible et TimescaleDB fonctionnel
- [ ] Tables créées dans PostgreSQL
- [ ] Kafka accessible et topics initialisés
- [ ] InfluxDB accessible via interface web
- [ ] MinIO accessible et buckets créés
- [ ] Redis accessible et fonctionnel
- [ ] Test d'insertion PostgreSQL réussi
- [ ] Test Kafka (pub/sub) réussi
- [ ] Test Redis (set/get) réussi

## En Cas de Problème

### Erreur "port already in use"

Modifier les ports dans `docker-compose.yml` ou arrêter le service qui utilise le port

### Erreur de permission

Sur Linux : `sudo chown -R $USER:$USER volumes/`

### Service ne démarre pas

Vérifier les logs : `docker-compose logs [service-name]`

### Health check échoue

Attendre quelques secondes puis vérifier à nouveau : `docker-compose ps`

## Arrêt des Services

```bash
docker-compose down
```

Pour supprimer aussi les volumes (⚠️ supprime les données) :

```bash
docker-compose down -v
```

