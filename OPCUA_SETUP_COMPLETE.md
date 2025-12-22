# OPC UA Setup - Complete ✅

## What Was Created

### 1. OPC UA Server Simulator
**File**: `scripts/opcua_server_simulator.py`

- Simulates an OPC UA server on port 4840
- Exposes 63 sensor nodes (3 assets × 21 sensors)
- Updates values every second with realistic degradation
- Node format: `ns=2;s=<ASSET_ID>_<SENSOR_ID>`

### 2. Ingestion Service Configuration
**File**: `services/ingestion-iiot/src/main/resources/application.yml`

- Updated with all 63 OPC UA node mappings
- Configured for 3 engines: ENGINE_FD001_000, ENGINE_FD002_019, ENGINE_FD004_000
- All 21 sensors per engine mapped correctly

### 3. Setup Documentation
**File**: `scripts/README_OPCUA_SETUP.md`

- Complete setup guide
- Troubleshooting section
- Verification steps

### 4. Quick Start Scripts
- `scripts/start_opcua_simulator.bat` (Windows)
- `scripts/start_opcua_simulator.sh` (Linux/Mac)

### 5. Requirements File
**File**: `scripts/requirements_opcua.txt`

- Lists required Python library: `opcua==0.98.13`

## Quick Start

### Step 1: Install OPC UA Library

```bash
pip install opcua
```

Or use the requirements file:
```bash
pip install -r scripts/requirements_opcua.txt
```

### Step 2: Start OPC UA Simulator

**Windows:**
```bash
scripts\start_opcua_simulator.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/start_opcua_simulator.sh
./scripts/start_opcua_simulator.sh
```

**Or directly:**
```bash
python scripts/opcua_server_simulator.py
```

You should see:
```
============================================================
OPC UA Server Simulator for Predictive Maintenance
============================================================

Server Endpoint: opc.tcp://0.0.0.0:4840
Assets: ENGINE_FD001_000, ENGINE_FD002_019, ENGINE_FD004_000
Sensors per asset: 21
Total nodes: 63

✓ OPC UA Server started successfully!
✓ Listening for connections...
```

### Step 3: Start Ingestion Service

The ingestion service is already configured. Just start it:

```bash
cd services/ingestion-iiot
mvn spring-boot:run
```

You should see in the logs:
```
Connected to OPC UA server: opc.tcp://localhost:4840
Processed X sensor data records
```

## Data Flow

```
OPC UA Simulator (Port 4840)
    ↓ Updates every 1 second
63 OPC UA Nodes (3 assets × 21 sensors)
    ↓ Reads every 5 seconds
Ingestion Service (Port 8081)
    ↓ Normalizes & publishes
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

## Verification

### 1. Check OPC UA Server
- Server should be running and showing cycle updates
- Endpoint: `opc.tcp://localhost:4840`

### 2. Check Ingestion Service Logs
Look for:
- `Connected to OPC UA server`
- `Processed X sensor data records`

### 3. Check Kafka
```bash
kafka-console-consumer --bootstrap-server localhost:9092 --topic sensor-data --from-beginning
```

### 4. Check Database
```sql
SELECT COUNT(*) FROM raw_sensor_data;
SELECT * FROM raw_sensor_data ORDER BY time DESC LIMIT 10;
```

## Configuration

### Change Update Frequency

**OPC UA Simulator** (`scripts/opcua_server_simulator.py`):
- Line 95: `await asyncio.sleep(1)` - Change to adjust update interval

**Ingestion Service** (`application.yml`):
- `app.ingestion.flush-interval: 5000` - Change to adjust collection interval (milliseconds)

### Add More Assets/Sensors

1. Edit `scripts/opcua_server_simulator.py`:
   - Add to `ASSETS` list
   - Add to `SENSORS` dictionary if needed

2. Edit `services/ingestion-iiot/src/main/resources/application.yml`:
   - Add node mappings under `opcua.nodes`

## Troubleshooting

### "ModuleNotFoundError: No module named 'opcua'"
**Solution**: Install the library
```bash
pip install opcua
```

### "Failed to connect to OPC UA server"
**Solution**: 
1. Ensure OPC UA simulator is running
2. Check endpoint URL in `application.yml` matches simulator
3. Verify port 4840 is not blocked by firewall

### "No data collected"
**Solution**:
1. Verify node IDs in `application.yml` match simulator format: `ns=2;s=<ASSET_ID>_<SENSOR_ID>`
2. Check OPC UA simulator logs for updates
3. Increase logging: Set `logging.level.com.predictivemaintenance.ingestion: DEBUG`

## Next Steps

1. ✅ OPC UA simulator created
2. ✅ Ingestion service configured
3. ⏳ Start OPC UA simulator
4. ⏳ Start ingestion service
5. ⏳ Verify data flow through pipeline
6. ⏳ Check dashboard for real-time updates

## Benefits

- ✅ **Automatic data collection** - No manual commands needed
- ✅ **Real-time pipeline** - Data flows automatically through all services
- ✅ **Realistic simulation** - Degradation over time mimics real equipment
- ✅ **Scalable** - Easy to add more assets/sensors
- ✅ **Production-ready** - Can connect to real OPC UA servers

---

**Status**: ✅ Setup Complete - Ready to Use!

