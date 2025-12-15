# Preprocessing Worker

## Important: Two Processes Needed

The preprocessing service requires **TWO separate processes**:

1. **FastAPI Server** (REST API) - Port 8082
   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8082
   ```

2. **Kafka Worker** (Background consumer) - Consumes from Kafka
   ```powershell
   python -m app.worker --mode streaming
   ```

## Why Two Processes?

- **FastAPI Server**: Provides REST API endpoints for manual data processing
- **Kafka Worker**: Automatically consumes messages from `sensor-data` topic and processes them

## Quick Start

### Terminal 1: Start FastAPI Server
```powershell
cd services/preprocessing
python -m uvicorn app.main:app --host 0.0.0.0 --port 8082
```

### Terminal 2: Start Kafka Worker
```powershell
cd services/preprocessing
python -m app.worker --mode streaming
```

### Terminal 3: Send Test Data
```powershell
cd services/preprocessing
python test_send_data.py
```

## Verify It's Working

After starting the worker, you should see logs like:
```
INFO: Démarrage du pipeline de prétraitement en mode: streaming
INFO: Kafka consumer créé pour topic: sensor-data
INFO: Démarrage de la consommation Kafka...
```

When you send data, you should see:
```
INFO: Message traité: asset=ASSET001, sensor=SENSOR001
INFO: Donnée prétraitée publiée: asset=ASSET001, sensor=SENSOR001
```

## Check Processed Data

To see processed data in Kafka:
```powershell
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic preprocessed-data --from-beginning --max-messages 5 --timeout-ms 5000
```

