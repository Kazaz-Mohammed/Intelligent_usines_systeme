# OPC UA Setup Guide

This guide explains how to set up the OPC UA server simulator and connect it to the ingestion service.

## Prerequisites

1. **Python 3.8+** installed
2. **OPC UA Python library** installed:
   ```bash
   pip install opcua
   ```

## Quick Start

### Step 1: Install OPC UA Library

```bash
pip install opcua
```

### Step 2: Start OPC UA Simulator

```bash
python scripts/opcua_server_simulator.py
```

The server will start on `opc.tcp://localhost:4840` and expose:
- **3 Assets**: ENGINE_FD001_000, ENGINE_FD002_019, ENGINE_FD004_000
- **21 Sensors per asset**: T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32
- **Total: 63 OPC UA nodes**

### Step 3: Start Ingestion Service

The ingestion service is already configured to connect to the OPC UA simulator. Just start it:

```bash
cd services/ingestion-iiot
mvn spring-boot:run
```

Or if using Docker:
```bash
docker-compose up ingestion-iiot
```

## How It Works

1. **OPC UA Simulator** (`scripts/opcua_server_simulator.py`):
   - Creates an OPC UA server on port 4840
   - Exposes sensor nodes for all 3 engines
   - Updates sensor values every second with realistic degradation
   - Node ID format: `ns=2;s=<ASSET_ID>_<SENSOR_ID>`

2. **Ingestion Service** (`services/ingestion-iiot`):
   - Connects to OPC UA server on startup
   - Reads all configured nodes every 5 seconds (configurable)
   - Normalizes data (timestamps, units, quality)
   - Publishes to Kafka topic `sensor-data`
   - Stores in TimescaleDB `raw_sensor_data` table
   - Archives in MinIO (if enabled)

## Configuration

### OPC UA Simulator

Edit `scripts/opcua_server_simulator.py` to:
- Change server endpoint (default: `opc.tcp://0.0.0.0:4840`)
- Add/remove assets
- Modify sensor ranges
- Adjust update frequency

### Ingestion Service

Edit `services/ingestion-iiot/src/main/resources/application.yml`:
- `opcua.endpoint-url`: OPC UA server URL
- `opcua.enabled`: Enable/disable OPC UA (default: true)
- `opcua.nodes`: List of nodes to monitor
- `app.ingestion.flush-interval`: Data collection interval in milliseconds (default: 5000ms)

## Verification

### Check OPC UA Server

1. The simulator should show:
   ```
   ✓ OPC UA Server started successfully!
   ✓ Listening for connections...
   ```

2. You can use an OPC UA client (like UaExpert) to browse:
   - Endpoint: `opc.tcp://localhost:4840`
   - Browse to: `Objects > IndustrialAssets > ENGINE_FD001_000 > Sensors`

### Check Ingestion Service

1. Check logs for connection:
   ```
   Connected to OPC UA server: opc.tcp://localhost:4840
   ```

2. Check for data collection:
   ```
   Processed X sensor data records
   ```

3. Verify Kafka messages:
   ```bash
   kafka-console-consumer --bootstrap-server localhost:9092 --topic sensor-data --from-beginning
   ```

4. Check TimescaleDB:
   ```sql
   SELECT COUNT(*) FROM raw_sensor_data;
   SELECT * FROM raw_sensor_data ORDER BY time DESC LIMIT 10;
   ```

## Troubleshooting

### OPC UA Connection Failed

**Error**: `Failed to connect to OPC UA server`

**Solutions**:
1. Ensure OPC UA simulator is running
2. Check endpoint URL in `application.yml`
3. Verify firewall allows port 4840
4. Check OPC UA simulator logs

### No Data Collected

**Symptoms**: Ingestion service connects but no data is collected

**Solutions**:
1. Verify node IDs in `application.yml` match simulator format
2. Check OPC UA simulator is updating values (should see cycle logs)
3. Increase logging: Set `logging.level.com.predictivemaintenance.ingestion: DEBUG`

### Kafka Not Receiving Data

**Solutions**:
1. Verify Kafka is running: `docker ps | grep kafka`
2. Check Kafka topic exists: `kafka-topics --list --bootstrap-server localhost:9092`
3. Check ingestion service logs for Kafka errors

## Advanced: Connect to Real OPC UA Server

To connect to a real PLC/SCADA OPC UA server:

1. **Get the endpoint URL** from your OPC UA server (e.g., `opc.tcp://192.168.1.100:4840`)

2. **Browse the server** to find node IDs using an OPC UA client

3. **Update `application.yml`**:
   ```yaml
   opcua:
     endpoint-url: opc.tcp://192.168.1.100:4840  # Your server IP
     nodes:
       - node-id: "ns=2;s=PLC1.TemperatureSensor1"  # Actual node from your PLC
         asset-id: "ENGINE_FD001_000"
         sensor-id: "T2"
         unit: "°C"
   ```

4. **Restart ingestion service**

## Data Flow

```
OPC UA Simulator (Port 4840)
    ↓ (every 1 second: updates values)
OPC UA Nodes (63 nodes)
    ↓ (every 5 seconds: reads values)
Ingestion Service (Port 8081)
    ↓ (normalizes & publishes)
Kafka Topic: sensor-data
    ↓
Preprocessing Service
    ↓
Feature Extraction
    ↓
RUL Prediction
    ↓
Anomaly Detection
    ↓
Dashboard
```

## Next Steps

Once OPC UA is working:
1. Monitor data flow through the pipeline
2. Check dashboard for real-time updates
3. Verify RUL predictions and anomaly detections
4. Adjust sensor ranges or degradation rates as needed

