# Docker Setup Complete ✅

All Docker configuration files have been created for the Predictive Maintenance System.

## Files Created

### 1. **docker-compose.yml** (Root)
Main orchestration file that defines all services:
- Infrastructure: Zookeeper, Kafka, PostgreSQL
- Application Services: All 7 microservices
- Optional Tools: Kafka UI, pgAdmin (with `--profile tools`)

### 2. **Dockerfile** for Next.js Dashboard
Location: `services/dashboard-usine/predictive-maintenance-dashboard/Dockerfile`
- Multi-stage build (builder + runner)
- Uses Next.js standalone output
- Optimized for production

### 3. **.dockerignore** (Root)
Excludes unnecessary files from Docker builds

### 4. **.dockerignore** (Dashboard)
Excludes Next.js build artifacts and dependencies

### 5. **docker-start.sh** (Linux/Mac)
Interactive script to manage Docker services

### 6. **docker-start.ps1** (Windows)
PowerShell script to manage Docker services

### 7. **DOCKER_GUIDE.md**
Comprehensive guide on using Docker with the system

## How Docker Works

### Image Building
When you run `docker-compose up --build`:
1. Docker reads each service's `Dockerfile`
2. Builds a Docker image for each service
3. Images are cached for faster subsequent builds

### Container Running
1. Docker creates containers from the images
2. Containers run in isolated environments
3. Services communicate via Docker network

### Data Persistence
- Volumes store data (databases, models) outside containers
- Data persists even when containers are stopped/removed

## Quick Start Commands

### Start Everything
```bash
# Build and start all services
docker-compose up --build

# Or in background
docker-compose up -d --build
```

### Using Helper Scripts
```bash
# Windows PowerShell
.\docker-start.ps1

# Linux/Mac
./docker-start.sh
```

### Manual Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Stop services
docker-compose down

# Stop and remove data
docker-compose down -v

# Check status
docker-compose ps
```

## Service URLs

Once started, access services at:

- **Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8091
- **Preprocessing**: http://localhost:8082
- **Extraction Features**: http://localhost:8083
- **Detection Anomalies**: http://localhost:8084
- **Prediction RUL**: http://localhost:8085
- **Orchestrator**: http://localhost:8087
- **Kafka**: localhost:9092
- **PostgreSQL**: localhost:5432

## Important Notes

1. **First Build**: The first build will take 10-20 minutes as it downloads all dependencies and builds images

2. **Model Files**: Trained models are stored in Docker volumes:
   - `detection-anomalies-models`
   - `prediction-rul-models`

3. **Database**: PostgreSQL data persists in `postgresql-data` volume

4. **Network**: All services communicate via `predictive-maintenance-network`

5. **Health Checks**: Services have health checks - wait for them to be "healthy" before using

6. **Port Conflicts**: Make sure ports 3000, 5432, 8082-8087, 8091, 9092 are not in use

## Next Steps

1. **Start the system**:
   ```bash
   docker-compose up -d --build
   ```

2. **Wait for services to be healthy**:
   ```bash
   docker-compose ps
   ```
   Wait until all services show "healthy" status

3. **Access the dashboard**:
   Open http://localhost:3000 in your browser

4. **Train models** (if needed):
   Models will be saved to Docker volumes automatically

5. **Send test data**:
   Use the data ingestion scripts to send sensor data

## Troubleshooting

See `DOCKER_GUIDE.md` for detailed troubleshooting steps.

## Environment Variables

Create a `.env` file in the project root to customize:

```env
POSTGRES_DB=predictive_maintenance
POSTGRES_USER=pmuser
POSTGRES_PASSWORD=your_secure_password
PGADMIN_EMAIL=admin@pm.local
PGADMIN_PASSWORD=admin
```

## Production Considerations

For production deployment:
- Use environment-specific `.env` files
- Set strong passwords
- Enable SSL/TLS
- Configure resource limits
- Set up backup strategies
- Use Docker secrets for sensitive data

---

**All Docker files are ready!** 🎉

Run `docker-compose up -d --build` to start the entire system.

