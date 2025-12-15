# Guide Docker - Service Extraction Features

## Vue d'ensemble

Ce guide explique comment construire et exécuter le service Extraction Features avec Docker.

## Prérequis

- Docker Desktop installé et en cours d'exécution
- Docker Compose v2.0+
- Réseau Docker `predictive-maintenance-network` créé

## Construction de l'image

### Construction manuelle

```bash
cd services/extraction-features
docker build -t extraction-features-service:0.1.0 .
```

### Construction avec docker-compose

```bash
cd services/extraction-features
docker-compose build
```

## Exécution du service

### Avec docker-compose (recommandé)

```bash
cd services/extraction-features
docker-compose up -d
```

### Exécution manuelle

```bash
docker run -d \
  --name extraction-features-service \
  --network predictive-maintenance-network \
  -p 8083:8083 \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  -e DATABASE_HOST=postgresql \
  -e DATABASE_PORT=5432 \
  -e DATABASE_NAME=predictive_maintenance \
  -e DATABASE_USER=pmuser \
  -e DATABASE_PASSWORD=pmpassword \
  extraction-features-service:0.1.0
```

## Variables d'environnement

### Variables requises

- `KAFKA_BOOTSTRAP_SERVERS`: Serveurs Kafka (ex: `kafka:9092`)
- `DATABASE_HOST`: Hôte TimescaleDB (ex: `postgresql`)
- `DATABASE_PORT`: Port TimescaleDB (ex: `5432`)
- `DATABASE_NAME`: Nom de la base de données (ex: `predictive_maintenance`)
- `DATABASE_USER`: Utilisateur de la base de données (ex: `pmuser`)
- `DATABASE_PASSWORD`: Mot de passe de la base de données (ex: `pmpassword`)

### Variables optionnelles

- `SERVICE_NAME`: Nom du service (défaut: `extraction-features-service`)
- `SERVICE_PORT`: Port du service (défaut: `8083`)
- `LOG_LEVEL`: Niveau de log (défaut: `INFO`)
- `KAFKA_CONSUMER_GROUP`: Groupe de consommateurs Kafka (défaut: `extraction-features-service`)
- `KAFKA_TOPIC_INPUT_PREPROCESSED`: Topic Kafka pour données prétraitées (défaut: `preprocessed-data`)
- `KAFKA_TOPIC_INPUT_WINDOWED`: Topic Kafka pour fenêtres de données (défaut: `windowed-data`)
- `KAFKA_TOPIC_OUTPUT`: Topic Kafka pour features extraites (défaut: `extracted-features`)
- `ENABLE_TEMPORAL_FEATURES`: Activer les features temporelles (défaut: `true`)
- `ENABLE_FREQUENCY_FEATURES`: Activer les features fréquentielles (défaut: `true`)
- `ENABLE_WAVELET_FEATURES`: Activer les features ondelettes (défaut: `true`)
- `ENABLE_STANDARDIZATION`: Activer la standardisation (défaut: `true`)
- `FEAST_ENABLE`: Activer Feast Feature Store (défaut: `true`)
- `FEAST_REPO_PATH`: Chemin vers le repository Feast (défaut: `./feast_repo`)
- `FEAST_ONLINE_STORE_TYPE`: Type de store en ligne Feast (défaut: `redis`)
- `FEAST_OFFLINE_STORE_TYPE`: Type de store hors ligne Feast (défaut: `file`)
- `STANDARDIZATION_METHOD`: Méthode de standardisation (défaut: `z-score`)

## Vérification du service

### Health check

```bash
curl http://localhost:8083/health
```

Réponse attendue :
```json
{
  "status": "healthy",
  "service": "extraction-features-service",
  "version": "0.1.0"
}
```

### Status du service

```bash
curl http://localhost:8083/api/v1/features/status
```

### Logs

```bash
# Avec docker-compose
docker-compose logs -f extraction-features-service

# Avec docker run
docker logs -f extraction-features-service
```

## Arrêt du service

### Avec docker-compose

```bash
docker-compose down
```

### Arrêt manuel

```bash
docker stop extraction-features-service
docker rm extraction-features-service
```

## Intégration avec l'infrastructure

Le service Extraction Features doit être ajouté au `docker-compose.yml` principal de l'infrastructure pour être exécuté avec les autres services.

### Ajout au docker-compose principal

Ajouter le service suivant au fichier `infrastructure/docker-compose.yml` :

```yaml
extraction-features-service:
  build:
    context: ../services/extraction-features
    dockerfile: Dockerfile
  container_name: extraction-features-service
  ports:
    - "8083:8083"
  environment:
    - SERVICE_NAME=extraction-features-service
    - SERVICE_PORT=8083
    - LOG_LEVEL=INFO
    - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
    - KAFKA_CONSUMER_GROUP=extraction-features-service
    - KAFKA_TOPIC_INPUT_PREPROCESSED=preprocessed-data
    - KAFKA_TOPIC_INPUT_WINDOWED=windowed-data
    - KAFKA_TOPIC_OUTPUT=extracted-features
    - DATABASE_HOST=postgresql
    - DATABASE_PORT=5432
    - DATABASE_NAME=predictive_maintenance
    - DATABASE_USER=pmuser
    - DATABASE_PASSWORD=pmpassword
    - ENABLE_TEMPORAL_FEATURES=true
    - ENABLE_FREQUENCY_FEATURES=true
    - ENABLE_WAVELET_FEATURES=true
    - ENABLE_STANDARDIZATION=true
    - FEAST_ENABLE=true
    - FEAST_REPO_PATH=/app/feast_repo
    - FEAST_ONLINE_STORE_TYPE=redis
    - FEAST_OFFLINE_STORE_TYPE=file
    - STANDARDIZATION_METHOD=z-score
  depends_on:
    kafka:
      condition: service_healthy
    postgresql:
      condition: service_healthy
    redis:
      condition: service_healthy
  networks:
    - predictive-maintenance-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8083/health')"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 40s
```

## Dépannage

### Le service ne démarre pas

1. Vérifier que les services dépendants sont en cours d'exécution :
   ```bash
   docker ps
   ```

2. Vérifier les logs :
   ```bash
   docker logs extraction-features-service
   ```

3. Vérifier la connectivité réseau :
   ```bash
   docker network inspect predictive-maintenance-network
   ```

### Erreurs de connexion Kafka

- Vérifier que Kafka est accessible : `kafka:29092` (interne) ou `localhost:9092` (externe)
- Vérifier que les topics existent :
  ```bash
  docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
  ```

### Erreurs de connexion TimescaleDB

- Vérifier que PostgreSQL/TimescaleDB est en cours d'exécution
- Vérifier les credentials de la base de données
- Vérifier que les tables nécessaires existent

### Erreurs Feast

- Si Feast n'est pas configuré, désactiver-le avec `FEAST_ENABLE=false`
- Vérifier que Redis est accessible si `FEAST_ONLINE_STORE_TYPE=redis`

## Développement

### Mode développement avec volumes

Pour développer avec rechargement automatique, monter le code source :

```yaml
volumes:
  - ./app:/app/app
  - ./requirements.txt:/app/requirements.txt
```

### Tests dans Docker

Pour exécuter les tests dans un conteneur :

```bash
docker build -t extraction-features-test -f Dockerfile.test .
docker run --rm extraction-features-test pytest tests/ -v
```

## Production

### Optimisations pour la production

1. Utiliser une image multi-stage pour réduire la taille
2. Utiliser des secrets Docker pour les mots de passe
3. Configurer des limites de ressources (CPU, mémoire)
4. Utiliser un orchestrateur (Kubernetes) pour la haute disponibilité

### Exemple de configuration production

```yaml
deploy:
  resources:
    limits:
      cpus: '1'
      memory: 2G
    reservations:
      cpus: '0.5'
      memory: 1G
```

