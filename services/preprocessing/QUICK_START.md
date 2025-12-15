# Quick Start Guide - Preprocessing Service

## ⚠️ IMPORTANT: You Need TWO Terminals

The preprocessing service requires **TWO separate processes**:

### Terminal 1: FastAPI Server (REST API)
```powershell
cd services/preprocessing
python -m uvicorn app.main:app --host 0.0.0.0 --port 8082
```

**What you'll see:**
```
INFO: Démarrage du service preprocessing-service
INFO: Configuration Kafka: localhost:9092
INFO: Topic input: sensor-data
INFO: Topic output: preprocessed-data
INFO: Application startup complete.
```

### Terminal 2: Kafka Worker (Background Consumer) ⭐ THIS IS WHAT'S MISSING
```powershell
cd services/preprocessing
python -m app.worker --mode streaming
```

**What you'll see:**
```
INFO: Démarrage du worker en mode: streaming
INFO: Démarrage du pipeline de prétraitement en mode: streaming
INFO: Kafka consumer créé pour topic: sensor-data
INFO: Démarrage de la consommation Kafka...
```

## How to Test

### Step 1: Start Both Processes
- Terminal 1: FastAPI server (already running ✅)
- Terminal 2: **Start the worker** (run the command above)

### Step 2: Send Test Data
In a **third terminal**:
```powershell
cd services/preprocessing
python test_send_data.py
```

### Step 3: Watch the Worker Terminal
You should see in Terminal 2 (worker):
```
INFO: Message traité: asset=ASSET001, sensor=SENSOR001
INFO: Donnée prétraitée publiée: asset=ASSET001, sensor=SENSOR001
```

## Troubleshooting

### Worker not processing messages?
1. ✅ Check worker is running (Terminal 2)
2. ✅ Check Kafka is running: `docker ps | findstr kafka`
3. ✅ Check topic exists: `docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092`
4. ✅ Check messages were sent: Look for "✓ Sent" in test script output

### No logs in worker?
- Make sure you're looking at Terminal 2 (worker), not Terminal 1 (FastAPI server)
- Worker logs will show "Message traité" when processing

## Summary

- **FastAPI Server** = REST API endpoints (manual processing)
- **Kafka Worker** = Automatic background processing from Kafka ⭐ **YOU NEED THIS**

