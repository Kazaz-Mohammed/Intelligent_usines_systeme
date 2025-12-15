# Using NASA CMAPSS Dataset with Microservices

## Quick Start

### 1. Make sure all services are running:

- **Preprocessing Worker**: `python -m app.worker --mode streaming` (in services/preprocessing)
- **Extraction Features Worker**: (if you have one)
- **Detection Anomalies Worker**: `python -m app.worker` (in services/detection-anomalies)
- **Prediction RUL Worker**: (if you have one)

### 2. Send NASA CMAPSS data:

```powershell
cd datasets/nasa-cmapss

# Small test (5 units, 50 cycles each)
python load_and_send_cmapss.py --file train_FD001.txt --max-units 5 --max-cycles 50

# Medium test (10 units, 100 cycles each)
python load_and_send_cmapss.py --file train_FD001.txt --max-units 10 --max-cycles 100

# Full dataset (all units, all cycles) - WARNING: This will send A LOT of data
python load_and_send_cmapss.py --file train_FD001.txt --max-units 0 --max-cycles 0
```

## Options

- `--file`: CMAPSS file to use (train_FD001.txt, train_FD002.txt, etc.)
- `--max-units`: Limit number of engine units (0 = all)
- `--max-cycles`: Limit cycles per unit (0 = all)
- `--delay`: Delay between batches in milliseconds (default: 10ms)

## What Happens

1. **Script loads** NASA CMAPSS data file
2. **Converts** to sensor data format (one message per sensor per cycle)
3. **Sends to Kafka** topic `sensor-data`
4. **Preprocessing service** consumes, processes, publishes to `preprocessed-data`
5. **Extraction features** consumes, extracts features, publishes to `extracted-features`
6. **Detection anomalies** and **Prediction RUL** consume features and process

## Monitoring

Watch the worker terminals to see:
- Messages being received
- Processing logs
- Any errors

## Troubleshooting

### Script fails to load file
- Check file path is correct
- Ensure pandas is installed: `python -m pip install pandas`

### No messages processed
- Check Kafka is running: `docker ps | findstr kafka`
- Check workers are running
- Check topic exists: `docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092`

### Too many messages
- Use `--max-units` and `--max-cycles` to limit data
- Increase `--delay` to slow down sending

