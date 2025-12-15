# Docker Guide for Predictive Maintenance System

This guide explains how to run the entire predictive maintenance system using Docker.

## Prerequisites

- Docker Desktop installed (or Docker Engine + Docker Compose)
- At least 8GB of RAM available
- At least 10GB of free disk space

## Quick Start

### 1. Start All Services

```bash
# Build and start all services
docker-compose up --build

# Or start in detached mode (background)
docker-compose up -d --build
```

### 2. Access the Services

Once all services are running, you can access:

- **Dashboard Frontend**: http://localhost:3000
- **Dashboard Backend API**: http://localhost:8091
- **Preprocessing Service**: http://localhost:8082
- **Extraction Features Service**: http://localhost:8083
- **Detection Anomalies Service**: http://localhost:8084
- **Prediction RUL Service**: http://localhost:8085
- **Orchestrator Maintenance Service**: http://localhost:8087
- **Kafka**: localhost:9092
- **PostgreSQL**: localhost:5432

### 3. Optional Tools

To start optional management tools (Kafka UI, pgAdmin):

```bash
docker-compose --profile tools up -d
```

- **Kafka UI**: http://localhost:8080
- **pgAdmin**: http://localhost:5050
  - Email: admin@pm.local
  - Password: admin (default)

## Service Details

### Infrastructure Services

1. **Zookeeper** - Required for Kafka coordination
2. **Kafka** - Message broker for real-time data streaming
3. **PostgreSQL** - Main database (TimescaleDB for time-series data)

### Application Services

1. **Preprocessing** - Cleans and normalizes sensor data
2. **Extraction Features** - Extracts temporal, frequency, and wavelet features
3. **Detection Anomalies** - ML models for anomaly detection
4. **Prediction RUL** - Remaining Useful Life predictions
5. **Orchestrator Maintenance** - Work order management
6. **Dashboard Backend** - API gateway and data aggregation
7. **Dashboard Frontend** - Next.js web interface

## Common Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f dashboard-frontend
docker-compose logs -f prediction-rul
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Restart a Service

```bash
# Restart specific service
docker-compose restart dashboard-backend

# Rebuild and restart
docker-compose up -d --build dashboard-backend
```

### Check Service Status

```bash
# List all running containers
docker-compose ps

# Check health status
docker-compose ps --format json | jq '.[] | {name: .Name, status: .State}'
```

## Environment Variables

Create a `.env` file in the project root to customize settings:

```env
# Database
POSTGRES_DB=predictive_maintenance
POSTGRES_USER=pmuser
POSTGRES_PASSWORD=pmpassword

# pgAdmin
PGADMIN_EMAIL=admin@pm.local
PGADMIN_PASSWORD=admin
```

## Volumes

Docker volumes persist data across container restarts:

- `postgresql-data` - Database data
- `kafka-data` - Kafka message logs
- `zookeeper-data` - Zookeeper state
- `detection-anomalies-models` - Trained anomaly detection models
- `prediction-rul-models` - Trained RUL prediction models

## Troubleshooting

### Services Won't Start

1. Check if ports are already in use:
   ```bash
   # Windows PowerShell
   netstat -ano | findstr :3000
   netstat -ano | findstr :8091
   
   # Linux/Mac
   lsof -i :3000
   lsof -i :8091
   ```

2. Check Docker logs:
   ```bash
   docker-compose logs [service-name]
   ```

3. Verify Docker has enough resources:
   - Docker Desktop: Settings → Resources → Memory (at least 8GB)

### Database Connection Issues

1. Wait for PostgreSQL to be fully ready (healthcheck passes)
2. Check database credentials in `.env` file
3. Verify network connectivity:
   ```bash
   docker-compose exec dashboard-backend ping postgresql
   ```

### Kafka Connection Issues

1. Wait for Kafka to be fully ready (can take 2-3 minutes)
2. Check Kafka is accessible:
   ```bash
   docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092
   ```

### Model Files Not Found

Model files are stored in Docker volumes. If models are missing:

1. Train models first (they will be saved to volumes)
2. Check volume exists:
   ```bash
   docker volume ls | grep models
   ```

### Frontend Build Issues

If the Next.js frontend fails to build:

1. Check Node.js version (should be 18+)
2. Clear build cache:
   ```bash
   docker-compose build --no-cache dashboard-frontend
   ```

## Development Mode

For development, you can mount source code as volumes:

```yaml
# In docker-compose.yml, add volumes to services:
volumes:
  - ./services/dashboard-backend:/app
  - /app/venv  # Exclude venv from mount
```

Then restart:
```bash
docker-compose up -d
```

## Production Deployment

For production:

1. Use environment-specific `.env` files
2. Set strong passwords for databases
3. Enable SSL/TLS for services
4. Use Docker secrets for sensitive data
5. Set up proper backup strategies for volumes
6. Configure resource limits:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

## Building Individual Services

To build a specific service:

```bash
# Build single service
docker-compose build dashboard-backend

# Build without cache
docker-compose build --no-cache dashboard-backend
```

## Network

All services communicate via the `predictive-maintenance-network` bridge network. Services can reach each other using their container names as hostnames (e.g., `kafka:29092`, `postgresql:5432`).

## Health Checks

All services have health checks configured. Check health status:

```bash
docker-compose ps
```

Services marked as "healthy" are ready to accept requests.

## Next Steps

1. Start all services: `docker-compose up -d --build`
2. Wait for all services to be healthy (check with `docker-compose ps`)
3. Access the dashboard at http://localhost:3000
4. Train models using the training scripts
5. Send test data through the pipeline

For more information, see the individual service README files.

