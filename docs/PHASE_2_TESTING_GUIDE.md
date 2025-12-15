# Guide de Test - Phase 2 : Service IngestionIIoT

## 📋 Prérequis

1. **Docker Desktop** doit être démarré
2. **Java 17** installé
3. **Maven** installé
4. **Infrastructure Docker** démarrée

## 🚀 Étape 1 : Démarrer l'Infrastructure

```powershell
# Aller dans le répertoire du projet
cd "C:\Users\DELL\Desktop\Predictive Maintenance Projet"

# Démarrer l'infrastructure Docker
docker-compose -f infrastructure/docker-compose.yml up -d

# Attendre 30-60 secondes que tous les services soient prêts
Start-Sleep -Seconds 30

# Vérifier que les services sont en cours d'exécution
docker ps
```

**Services attendus** :
- ✅ postgresql (PostgreSQL/TimescaleDB)
- ✅ kafka (Apache Kafka)
- ✅ minio (MinIO)
- ✅ redis (Redis)
- ✅ zookeeper (Zookeeper pour Kafka)

## 🔧 Étape 2 : Configurer les Variables d'Environnement

```powershell
# Définir les variables d'environnement pour le service
$env:OPCUA_ENABLED = "false"  # Désactiver OPC UA pour éviter erreurs
$env:DATABASE_HOST = "localhost"
$env:DATABASE_PORT = "5432"
$env:DATABASE_NAME = "predictive_maintenance"
$env:DATABASE_USER = "pmuser"
$env:DATABASE_PASSWORD = "pmpassword"
$env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
$env:MINIO_ENDPOINT = "http://localhost:9000"
$env:MINIO_ACCESS_KEY = "minioadmin"
$env:MINIO_SECRET_KEY = "minioadmin"
$env:SERVER_PORT = "8081"
```

## 🏃 Étape 3 : Démarrer le Service

```powershell
# Aller dans le répertoire du service
cd "C:\Users\DELL\Desktop\Predictive Maintenance Projet\services\ingestion-iiot"

# Démarrer le service avec le profil local
mvn spring-boot:run -Dspring-boot.run.profiles=local
```

**Le service va** :
1. Compiler le projet (si nécessaire)
2. Démarrer Spring Boot
3. Se connecter à PostgreSQL, Kafka, MinIO
4. Écouter sur le port 8081

**Temps de démarrage** : 30-60 secondes

## ✅ Étape 4 : Tester les Endpoints

### Test 1 : Health Check

```powershell
# Dans un nouveau terminal PowerShell
Invoke-RestMethod -Uri "http://localhost:8081/api/v1/ingestion/health" -Method GET
```

**Résultat attendu** :
```json
{
  "status": "UP",
  "service": "ingestion-iiot"
}
```

### Test 2 : Status

```powershell
Invoke-RestMethod -Uri "http://localhost:8081/api/v1/ingestion/status" -Method GET
```

**Résultat attendu** :
```json
{
  "service": "ingestion-iiot",
  "status": "running"
}
```

### Test 3 : Ingestion de Données

```powershell
# Créer les données de test
$testData = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    assetId = "ASSET001"
    sensorId = "SENSOR001"
    value = 25.5
    unit = "°C"
    quality = 2
    sourceType = "TEST"
} | ConvertTo-Json

# Envoyer les données
$headers = @{ "Content-Type" = "application/json" }
Invoke-RestMethod -Uri "http://localhost:8081/api/v1/ingestion/data" `
    -Method POST -Body $testData -Headers $headers
```

**Résultat attendu** :
```json
{
  "status": "success"
}
```

## 🔍 Étape 5 : Vérifier les Logs

Dans le terminal où le service est démarré, vous devriez voir :
- ✅ Connexion à PostgreSQL réussie
- ✅ Connexion à Kafka réussie
- ✅ Connexion à MinIO réussie
- ✅ Service démarré sur le port 8081

## 🐛 Dépannage

### Erreur : Docker Desktop non démarré
```
error during connect: open //./pipe/dockerDesktopLinuxEngine
```
**Solution** : Démarrer Docker Desktop

### Erreur : Port 8081 déjà utilisé
```
Port 8081 is already in use
```
**Solution** : 
- Arrêter le service existant
- Ou changer le port : `$env:SERVER_PORT = "8082"`

### Erreur : Connexion PostgreSQL échouée
```
Connection refused to localhost:5432
```
**Solution** :
- Vérifier que PostgreSQL est démarré : `docker ps | Select-String postgres`
- Vérifier les credentials dans `application.yml`

### Erreur : Connexion Kafka échouée
```
Connection refused to localhost:9092
```
**Solution** :
- Vérifier que Kafka est démarré : `docker ps | Select-String kafka`
- Attendre 1-2 minutes après le démarrage de Kafka

### Erreur : Connexion MinIO échouée
```
Connection refused to localhost:9000
```
**Solution** :
- Vérifier que MinIO est démarré : `docker ps | Select-String minio`
- Vérifier l'endpoint dans `application.yml`

## 📊 Validation Complète

Une fois tous les tests réussis, vous pouvez :

1. **Vérifier que les données sont dans PostgreSQL** :
```powershell
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "SELECT COUNT(*) FROM raw_sensor_data;"
```

2. **Vérifier que les messages sont dans Kafka** :
```powershell
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic sensor-data --from-beginning --max-messages 1
```

3. **Vérifier que les fichiers sont dans MinIO** :
```powershell
docker exec -it minio mc ls local/raw-sensor-data
```

## ✅ Checklist de Validation

- [ ] Infrastructure Docker démarrée
- [ ] Service IngestionIIoT démarré
- [ ] Health endpoint répond 200 OK
- [ ] Status endpoint répond 200 OK
- [ ] Ingestion endpoint accepte les données
- [ ] Pas d'erreurs dans les logs
- [ ] Données visibles dans PostgreSQL
- [ ] Messages visibles dans Kafka
- [ ] Fichiers visibles dans MinIO

## 🎯 Prochaines Étapes

Une fois la validation réussie :
1. ✅ Merger la branche `feature/service-ingestion-iiot` dans `develop`
2. ✅ Créer un tag `v0.2.0`
3. ✅ Passer à la Phase 3 : Service Prétraitement

