# Service IngestionIIoT

## Description

Service Spring Boot responsable de la collecte de données depuis les systèmes industriels (PLC/SCADA) via différents protocoles.

## Fonctionnalités

- ✅ Connecteurs OPC UA (Eclipse Milo)
- ⏳ Support Modbus (à implémenter)
- ⏳ Support MQTT (à implémenter)
- ✅ Normalisation des données
- ✅ Publication sur Kafka
- ✅ Stockage dans TimescaleDB
- ✅ Archivage dans MinIO

## Technologies

- **Spring Boot 3.2.0**
- **Eclipse Milo** (OPC UA)
- **Apache Kafka** (Spring Kafka)
- **PostgreSQL/TimescaleDB** (Spring Data JPA)
- **MinIO** (Client Java)
- **Java 17**

## Structure du Projet

```
ingestion-iiot/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/predictivemaintenance/ingestion/
│   │   │       ├── config/          # Configurations
│   │   │       ├── controller/      # REST API
│   │   │       ├── model/           # Modèles de données
│   │   │       └── service/         # Services métier
│   │   └── resources/
│   │       └── application.yml      # Configuration
│   └── test/                        # Tests
├── pom.xml                          # Maven dependencies
├── Dockerfile                        # Image Docker
└── README.md
```

## Configuration

### Variables d'Environnement

```bash
# Database
DATABASE_HOST=postgresql
DATABASE_PORT=5432
DATABASE_NAME=predictive_maintenance
DATABASE_USER=pmuser
DATABASE_PASSWORD=pmpassword

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# MinIO
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=raw-sensor-data

# OPC UA
OPCUA_ENABLED=true
OPCUA_ENDPOINT_URL=opc.tcp://localhost:4840
```

## Démarrage

### Local (avec Maven)

```bash
mvn spring-boot:run
```

### Docker

```bash
docker build -t ingestion-iiot:latest .
docker run -p 8081:8081 ingestion-iiot:latest
```

### Avec Docker Compose

Le service sera ajouté au `docker-compose.yml` dans la Phase 10 (Intégration E2E).

## API Endpoints

- `GET /actuator/health` - Health check
- `GET /api/v1/ingestion/health` - Service health
- `GET /api/v1/ingestion/status` - Service status
- `POST /api/v1/ingestion/data` - Ingest data manually

## État

🚧 **En développement** (Phase 2)

### Complété
- ✅ Structure Spring Boot
- ✅ Configuration Kafka
- ✅ Service de normalisation
- ✅ Service Kafka Producer
- ✅ Service TimescaleDB
- ✅ Service MinIO
- ✅ Service OPC UA (basique)
- ✅ REST API

### À compléter
- ⏳ Tests unitaires
- ⏳ Tests d'intégration
- ⏳ Support Modbus
- ⏳ Support MQTT
- ⏳ Buffer edge pour résilience
- ⏳ Gestion d'erreurs avancée
