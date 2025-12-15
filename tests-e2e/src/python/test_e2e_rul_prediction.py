"""
Test E2E : Flux complet de prédiction RUL
"""
import pytest
import requests
import time

# URLs des services
EXTRACTION_URL = "http://localhost:8083"
PREDICTION_URL = "http://localhost:8085"
ORCHESTRATEUR_URL = "http://localhost:8087"
DASHBOARD_URL = "http://localhost:8086"


class TestE2ERulPrediction:
    """Tests E2E pour le flux de prédiction RUL"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup avant chaque test"""
        self.wait_for_services()
        yield
    
    def wait_for_services(self):
        """Attendre que les services soient prêts"""
        services = [
            (EXTRACTION_URL, "ExtractionFeatures"),
            (PREDICTION_URL, "PredictionRUL"),
            (ORCHESTRATEUR_URL, "OrchestrateurMaintenance"),
            (DASHBOARD_URL, "DashboardMonitoring")
        ]
        
        for url, name in services:
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ {name} est prêt")
                        break
                except:
                    if attempt == max_attempts - 1:
                        pytest.fail(f"{name} n'est pas prêt")
                    time.sleep(2)
    
    def test_complete_rul_prediction_flow(self):
        """Test du flux complet de prédiction RUL"""
        
        # 1. Créer une séquence de features
        print("\n1. Création d'une séquence de features...")
        features_sequence = [
            {"rms": 8.5, "kurtosis": 2.1, "skewness": 0.7, "peak": 12.0, "crest_factor": 3.2},
            {"rms": 9.0, "kurtosis": 2.3, "skewness": 0.8, "peak": 13.0, "crest_factor": 3.3},
            {"rms": 9.5, "kurtosis": 2.5, "skewness": 0.9, "peak": 14.0, "crest_factor": 3.4},
            {"rms": 10.0, "kurtosis": 2.7, "skewness": 1.0, "peak": 15.0, "crest_factor": 3.5},
            {"rms": 10.5, "kurtosis": 2.9, "skewness": 1.1, "peak": 16.0, "crest_factor": 3.6}
        ]
        
        rul_request = {
            "asset_id": "PUMP_003",
            "sensor_id": "SENSOR_003",
            "features_sequence": features_sequence,
            "timestamp": "2024-01-15T10:00:00Z"
        }
        
        # 2. Prédire la RUL
        print("\n2. Prédiction de la RUL...")
        response = requests.post(
            f"{PREDICTION_URL}/api/v1/rul/predict",
            json=rul_request
        )
        
        assert response.status_code == 200, f"Prédiction RUL échouée: {response.text}"
        result = response.json()
        assert "predicted_rul" in result
        print(f"✅ RUL prédite: {result.get('predicted_rul')} cycles")
        
        # 3. Vérifier la création d'intervention si RUL faible
        print("\n3. Vérification de la création d'intervention...")
        if result.get("predicted_rul", 0) < 200:
            time.sleep(5)
            
            response = requests.get(
                f"{ORCHESTRATEUR_URL}/api/v1/work-orders/asset/PUMP_003"
            )
            assert response.status_code == 200
            work_orders = response.json()
            print(f"✅ Work orders trouvés: {len(work_orders)}")
        
        # 4. Vérifier l'affichage dans le dashboard
        print("\n4. Vérification du dashboard...")
        response = requests.get(f"{DASHBOARD_URL}/api/v1/dashboard/overview")
        assert response.status_code == 200
        overview = response.json()
        assert "totalAssets" in overview
        print("✅ Dashboard accessible")
        
        print("\n✅ Flux complet de prédiction RUL réussi!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

