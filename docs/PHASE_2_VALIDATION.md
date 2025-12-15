# Phase 2 : Validation du Service IngestionIIoT

## Date : 3 novembre 2025

## Tests Effectués

### ✅ Compilation Maven
- **Statut** : ✅ RÉUSSI
- **Commande** : `mvn clean compile -DskipTests`
- **Résultat** : BUILD SUCCESS (5:52 min)
- **Fichiers compilés** : 14 source files

### ✅ Infrastructure Docker
- **PostgreSQL/TimescaleDB** : ✅ En cours d'exécution
- **Kafka** : ✅ En cours d'exécution
- **MinIO** : ✅ En cours d'exécution
- **Redis** : ✅ En cours d'exécution

### ⏳ Démarrage du Service
- **Commande** : `mvn spring-boot:run -Dspring-boot.run.profiles=local`
- **Profil** : `local` (OPC UA désactivé pour éviter erreurs)
- **Port** : 8081
- **Variables d'environnement configurées** :
  - `OPCUA_ENABLED=false`
  - `DATABASE_HOST=localhost`
  - `KAFKA_BOOTSTRAP_SERVERS=localhost:9092`
  - `MINIO_ENDPOINT=http://localhost:9000`

### 📋 Endpoints à Tester

1. **Health Check**
   - URL: `http://localhost:8081/api/v1/ingestion/health`
   - Méthode: GET
   - Réponse attendue: `{"status":"UP","service":"ingestion-iiot"}`

2. **Status**
   - URL: `http://localhost:8081/api/v1/ingestion/status`
   - Méthode: GET
   - Réponse attendue: `{"service":"ingestion-iiot","status":"running"}`

3. **Ingestion de données**
   - URL: `http://localhost:8081/api/v1/ingestion/data`
   - Méthode: POST
   - Body: JSON avec SensorData
   - Réponse attendue: `{"status":"success"}`

## Configuration Créée

### Fichiers de Configuration
- ✅ `application.yml` : Configuration principale
- ✅ `application-dev.yml` : Configuration développement
- ✅ `application-local.yml` : Configuration locale (OPC UA désactivé)
- ✅ `application-test.yml` : Configuration tests

### Scripts de Test
- ✅ `scripts/test-service-startup.ps1` : Script complet de test
- ✅ `scripts/validate-service.ps1` : Script de validation
- ✅ `scripts/quick-test-service.ps1` : Test rapide

## Prochaines Étapes

1. **Démarrer le service manuellement** :
   ```powershell
   cd services\ingestion-iiot
   $env:OPCUA_ENABLED="false"
   $env:DATABASE_HOST="localhost"
   $env:KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
   $env:MINIO_ENDPOINT="http://localhost:9000"
   mvn spring-boot:run -Dspring-boot.run.profiles=local
   ```

2. **Tester les endpoints** :
   ```powershell
   # Health
   Invoke-RestMethod -Uri "http://localhost:8081/api/v1/ingestion/health" -Method GET
   
   # Status
   Invoke-RestMethod -Uri "http://localhost:8081/api/v1/ingestion/status" -Method GET
   
   # Ingestion
   $data = @{
       timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
       assetId = "ASSET001"
       sensorId = "SENSOR001"
       value = 25.5
       unit = "°C"
       quality = 2
       sourceType = "TEST"
   } | ConvertTo-Json
   
   Invoke-RestMethod -Uri "http://localhost:8081/api/v1/ingestion/data" `
       -Method POST -Body $data -ContentType "application/json"
   ```

3. **Vérifier les logs** pour s'assurer que :
   - Les connexions à PostgreSQL fonctionnent
   - Les connexions à Kafka fonctionnent
   - Les connexions à MinIO fonctionnent
   - Les données sont bien traitées

## Résultats Attendus

### ✅ Succès si :
- Service démarre sans erreur
- Health endpoint retourne 200 OK
- Status endpoint retourne 200 OK
- Ingestion endpoint accepte les données
- Pas d'erreurs dans les logs

### ⚠️ À vérifier si erreurs :
- Connexion PostgreSQL : Vérifier credentials et port
- Connexion Kafka : Vérifier que Kafka est démarré
- Connexion MinIO : Vérifier endpoint et credentials
- OPC UA : Désactivé par défaut dans profil local

## Notes

- Le service prend généralement 30-60 secondes pour démarrer
- OPC UA est désactivé dans le profil `local` pour éviter les erreurs de connexion
- Les variables d'environnement peuvent être définies dans un fichier `.env` ou via PowerShell

