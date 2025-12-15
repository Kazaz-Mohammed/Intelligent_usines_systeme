# Infrastructure Docker

## Description

Configuration Docker Compose pour le développement local de la plateforme de maintenance prédictive.

## Services

### Services Principaux

- **Zookeeper** : Coordination pour Kafka (port 2181)
- **Kafka** : Messaging asynchrone (ports 9092, 9093)
- **PostgreSQL + TimescaleDB** : Base de données avec extension pour séries temporelles (port 5432)
- **InfluxDB** : Base de données séries temporelles spécialisée (port 8086)
- **MinIO** : Stockage objet S3-compatible (ports 9000, 9001)
- **Redis** : Cache et stockage clé-valeur (port 6379)

### Services Optionnels (profile: tools)

- **Kafka UI** : Interface web pour gérer Kafka (port 8080)
- **pgAdmin** : Interface web pour gérer PostgreSQL (port 5050)

## Démarrage Rapide

### Prérequis

- Docker et Docker Compose installés
- Au moins 4GB de RAM disponible
- Ports libres : 2181, 5432, 6379, 8086, 9000, 9001, 9092, 9093

### Démarrage (Linux/Mac)

```bash
cd infrastructure
./../scripts/start-infrastructure.sh
```

### Démarrage (Windows PowerShell)

```powershell
cd infrastructure
..\scripts\start-infrastructure.ps1
```

### Démarrage Manuel

```bash
cd infrastructure
docker-compose up -d
```

### Avec Outils de Gestion (Kafka UI, pgAdmin)

```bash
docker-compose --profile tools up -d
```

## Initialisation

Les scripts d'initialisation créent automatiquement :

1. **PostgreSQL** : Tables et extensions TimescaleDB (via `init-postgres.sql`)
2. **Kafka** : Topics nécessaires (via `init-kafka-topics.sh` ou `.ps1`)
3. **MinIO** : Buckets pour stockage (via `init-minio-buckets.sh` ou `.ps1`)

### Topics Kafka Créés

- `sensor-data` : Données brutes des capteurs
- `preprocessed-data` : Données prétraitées
- `features` : Caractéristiques extraites
- `anomalies` : Événements d'anomalies
- `rul-predictions` : Prédictions RUL
- `maintenance-orders` : Ordres de maintenance

### Buckets MinIO Créés

- `raw-sensor-data` : Données brutes
- `processed-data` : Données prétraitées
- `model-artifacts` : Artefacts de modèles ML
- `mlflow-artifacts` : Artefacts MLflow
- `backups` : Backups

## Configuration

### Variables d'Environnement

Copier `.env.example` vers `.env` et modifier les valeurs :

```bash
cp .env.example .env
# Éditer .env avec vos valeurs
```

**⚠️ Important** : Changer les mots de passe par défaut en production !

## Accès aux Services

| Service | URL/Host | Port | Credentials |
|---------|----------|------|-------------|
| Kafka | localhost | 9092 | - |
| Kafka UI | http://localhost:8080 | 8080 | - |
| PostgreSQL | localhost | 5432 | Voir .env |
| pgAdmin | http://localhost:5050 | 5050 | Voir .env |
| InfluxDB | http://localhost:8086 | 8086 | Voir .env |
| MinIO | http://localhost:9000 | 9000 | Voir .env |
| MinIO Console | http://localhost:9001 | 9001 | Voir .env |
| Redis | localhost | 6379 | Voir .env |

## Commandes Utiles

### Voir les logs

```bash
docker-compose logs -f [service-name]
```

### Arrêter les services

```bash
docker-compose down
```

### Arrêter et supprimer les volumes (⚠️ supprime les données)

```bash
docker-compose down -v
```

### Redémarrer un service

```bash
docker-compose restart [service-name]
```

### Vérifier l'état des services

```bash
docker-compose ps
```

### Health Checks

Tous les services ont des health checks configurés. Vérifier :

```bash
docker-compose ps
```

## Tests de Connectivité

### PostgreSQL

```bash
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "SELECT version();"
```

### Kafka

```bash
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

### Redis

```bash
docker exec -it redis redis-cli -a redispassword PING
```

### MinIO

Accéder à http://localhost:9001 et se connecter avec les credentials de `.env`

### InfluxDB

Accéder à http://localhost:8086 et se connecter avec les credentials de `.env`

## Dépannage

### Port déjà utilisé

Si un port est déjà utilisé, modifier les mappings dans `docker-compose.yml`

### Erreur de permissions (Linux)

```bash
sudo chown -R $USER:$USER volumes/
```

### Redémarrer depuis zéro

```bash
docker-compose down -v
docker-compose up -d
```

### Vérifier les logs d'erreur

```bash
docker-compose logs | grep -i error
```

## Prochaines Étapes

Une fois l'infrastructure démarrée, vous pouvez :
- Développer les services d'application (Phase 2+)
- Connecter les services aux bases de données
- Tester l'envoi/réception de messages Kafka

## Notes

- Les données sont persistées dans des volumes Docker
- Les health checks permettent de s'assurer que les services sont prêts
- Le réseau `predictive-maintenance-network` permet la communication entre services
