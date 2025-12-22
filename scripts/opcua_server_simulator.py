"""
OPC UA Server Simulator for Predictive Maintenance
Simulates sensor data from CMAPSS engines as OPC UA nodes
"""
import asyncio
import random
import time
from datetime import datetime
from opcua import Server, ua

# Configuration
SERVER_ENDPOINT = "opc.tcp://0.0.0.0:4840"
ASSETS = ["ENGINE_FD001_000", "ENGINE_FD002_019", "ENGINE_FD004_000"]

# Sensor configurations based on CMAPSS dataset
SENSORS = {
    "T2": {"range": (500, 650), "unit": "R"},
    "T24": {"range": (1000, 1200), "unit": "R"},
    "T30": {"range": (1500, 2000), "unit": "R"},
    "T50": {"range": (800, 1000), "unit": "R"},
    "P2": {"range": (1.0, 2.0), "unit": "psia"},
    "P15": {"range": (5.0, 15.0), "unit": "psia"},
    "P30": {"range": (10.0, 30.0), "unit": "psia"},
    "Nf": {"range": (1000, 5000), "unit": "rpm"},
    "Nc": {"range": (8000, 12000), "unit": "rpm"},
    "epr": {"range": (1.0, 2.5), "unit": "--"},
    "Ps30": {"range": (5.0, 20.0), "unit": "psia"},
    "phi": {"range": (0.0, 1.0), "unit": "--"},
    "NRf": {"range": (0.9, 1.0), "unit": "--"},
    "NRc": {"range": (0.9, 1.0), "unit": "--"},
    "BPR": {"range": (1.0, 5.0), "unit": "--"},
    "farB": {"range": (0.0, 0.1), "unit": "--"},
    "htBleed": {"range": (0.0, 1.0), "unit": "--"},
    "Nf_dmd": {"range": (2000, 4000), "unit": "rpm"},
    "PCNfR_dmd": {"range": (5000, 10000), "unit": "rpm"},
    "W31": {"range": (0.0, 100.0), "unit": "--"},
    "W32": {"range": (0.0, 100.0), "unit": "--"},
}

class OPCUAServerSimulator:
    def __init__(self):
        # Create server with default configuration (allows anonymous connections)
        self.server = Server()
        self.server.set_endpoint(SERVER_ENDPOINT)
        self.server.set_server_name("Predictive Maintenance OPC UA Simulator")
        # Allow anonymous connections (no certificate required for simulator)
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        
        # Setup namespace
        uri = "http://predictivemaintenance.iiot"
        idx = self.server.register_namespace(uri)
        self.namespace_idx = idx  # Store for use in start() method
        
        # Create objects for each asset
        self.objects = {}
        self.nodes = {}
        
        # Create root object
        root = self.server.get_objects_node()
        
        # Create variables directly under root with flat node IDs
        # This ensures the node IDs match what the ingestion service expects: ns=2;s=ASSET_SENSOR
        for asset_id in ASSETS:
            for sensor_id, sensor_config in SENSORS.items():
                node_id_string = f"{asset_id}_{sensor_id}"
                initial_value = random.uniform(*sensor_config["range"])
                
                # Create variable with explicit flat node ID
                # Format: ns=<idx>;s=ASSET_SENSOR (matches ingestion service config)
                var_node_id = ua.NodeId(node_id_string, idx)
                var = root.add_variable(
                    var_node_id,
                    node_id_string,  # Browse name
                    initial_value
                )
                var.set_writable()
                # Data type is automatically inferred from initial_value (float = Double)
                
                # Store node reference
                self.nodes[node_id_string] = {
                    "variable": var,
                    "range": sensor_config["range"],
                    "unit": sensor_config["unit"],
                    "base_value": initial_value,
                    "cycle": 0
                }
    
    async def update_values(self):
        """Continuously update sensor values with realistic degradation"""
        cycle = 0
        while True:
            cycle += 1
            for node_id, node_data in self.nodes.items():
                var = node_data["variable"]
                min_val, max_val = node_data["range"]
                base_value = node_data["base_value"]
                
                # Simulate degradation over time (gradual drift)
                degradation_factor = 1 + (cycle * 0.0001)
                noise = random.uniform(-0.02, 0.02)
                
                new_value = base_value * degradation_factor * (1 + noise)
                new_value = max(min_val, min(max_val, new_value))
                
                var.set_value(new_value)
            
            # Log every 100 cycles
            if cycle % 100 == 0:
                print(f"[Cycle {cycle}] Updated {len(self.nodes)} sensor values")
            
            await asyncio.sleep(1)  # Update every second
    
    def start(self):
        """Start the OPC UA server"""
        print("=" * 60)
        print("OPC UA Server Simulator for Predictive Maintenance")
        print("=" * 60)
        print(f"\nServer Endpoint: {SERVER_ENDPOINT}")
        print(f"Assets: {', '.join(ASSETS)}")
        print(f"Sensors per asset: {len(SENSORS)}")
        print(f"Total nodes: {len(ASSETS) * len(SENSORS)}")
        
        # Show namespace index and node format
        print(f"\nNamespace Index: {self.namespace_idx}")
        print(f"Node ID Format: ns={self.namespace_idx};s=<ASSET_ID>_<SENSOR_ID>")
        print("Examples:")
        print(f"  - ns={self.namespace_idx};s=ENGINE_FD001_000_T2")
        print(f"  - ns={self.namespace_idx};s=ENGINE_FD001_000_epr")
        print(f"  - ns={self.namespace_idx};s=ENGINE_FD002_019_P30")
        print("\n" + "=" * 60)
        
        try:
            self.server.start()
            print("✓ OPC UA Server started successfully!")
            print("✓ Listening for connections...")
            print("\nPress Ctrl+C to stop\n")
            
            # Run value updates in background
            asyncio.run(self.update_values())
        except KeyboardInterrupt:
            print("\n\nStopping OPC UA Server...")
            self.server.stop()
            print("✓ Server stopped.")
        except Exception as e:
            print(f"\n✗ Error starting server: {e}")
            raise

if __name__ == "__main__":
    simulator = OPCUAServerSimulator()
    simulator.start()

