# Quick Test Guide - NASA CMAPSS Data

## ✅ Script is Ready!

The script `load_and_send_cmapss.py` is working and can send NASA CMAPSS data through your microservices.

## Quick Test Commands

### Small Test (Recommended First)
```powershell
cd datasets/nasa-cmapss
python load_and_send_cmapss.py --file train_FD001.txt --max-units 3 --max-cycles 20
```
- Sends data from 3 engines, 20 cycles each
- ~1,320 messages total (3 units × 20 cycles × 22 sensors/ops)

### Medium Test
```powershell
python load_and_send_cmapss.py --file train_FD001.txt --max-units 10 --max-cycles 50
```
- Sends data from 10 engines, 50 cycles each
- ~11,000 messages total

### Larger Test
```powershell
python load_and_send_cmapss.py --file train_FD001.txt --max-units 20 --max-cycles 100
```
- Sends data from 20 engines, 100 cycles each
- ~44,000 messages total

## What to Watch

### 1. Preprocessing Worker Terminal
You should see:
```
INFO: Message traité: asset=ENGINE_FD001_001, sensor=T2
INFO: Donnée prétraitée publiée: asset=ENGINE_FD001_001, sensor=T2
```

### 2. Extraction Features Worker (if running)
Should consume from `preprocessed-data` and extract features

### 3. Detection Anomalies Worker (if running)
Should consume from `extracted-features` and detect anomalies

### 4. Prediction RUL Worker (if running)
Should consume from `extracted-features` and predict RUL

## Troubleshooting

### No messages in preprocessing?
- ✅ Check preprocessing worker is running
- ✅ Check Kafka is running: `docker ps | findstr kafka`
- ✅ Check topic exists: `docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092`

### Too slow?
- Reduce `--delay` (default: 10ms)
- Or remove delay: `--delay 0`

### Too fast / overwhelming?
- Increase `--delay` to 100ms or more
- Reduce `--max-units` and `--max-cycles`

## Data Flow

```
NASA CMAPSS Data
    ↓
load_and_send_cmapss.py
    ↓
Kafka: sensor-data
    ↓
Preprocessing Worker
    ↓
Kafka: preprocessed-data
    ↓
Extraction Features Worker
    ↓
Kafka: extracted-features
    ↓
Detection Anomalies Worker → Kafka: anomalies-detected
Prediction RUL Worker → Kafka: rul-predictions
```

## Next Steps

1. Start all workers
2. Run the script with small test
3. Watch worker logs for processing
4. Check for any errors
5. Gradually increase data size

