"""
Test E2E : Monitoring et alertes
"""
import pytest
import requests
import time

# URLs des services
DASHBOARD_URL = "http://localhost:8086"
INGESTION_URL = "http://localhost:8081"
DETECTION_URL = "http://localhost:8084"


class TestE2EMonitoring:
    """Tests E2E pour le monitoring et les alertes"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup avant chaque test"""
        self.wait_for_services()
        yield
    
    def wait_for_services(self):
        """Attendre que les services soient prêts"""
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{DASHBOARD_URL}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ DashboardMonitoring est prêt")
                    break
            except:
                if attempt == max_attempts - 1:
                    pytest.fail("DashboardMonitoring n'est pas prêt")
                time.sleep(2)
    
    def test_service_monitoring(self):
        """Test du monitoring des services"""
        
        # 1. Vérifier le statut des services
        print("\n1. Vérification du statut des services...")
        response = requests.get(f"{DASHBOARD_URL}/api/v1/monitoring/services")
        assert response.status_code == 200, f"Monitoring échoué: {response.text}"
        
        services = response.json()
        assert isinstance(services, dict)
        print(f"✅ Services monitorés: {len(services)}")
        
        # 2. Vérifier le health check global
        print("\n2. Vérification du health check global...")
        response = requests.get(f"{DASHBOARD_URL}/api/v1/monitoring/health")
        assert response.status_code == 200
        health = response.json()
        assert "status" in health
        assert "upServices" in health
        print(f"✅ Statut global: {health.get('status')}")
        print(f"✅ Services UP: {health.get('upServices')}/{health.get('totalServices')}")
    
    def test_alert_creation_and_notification(self):
        """Test de création d'alerte et notification"""
        
        # 1. Créer une alerte
        print("\n1. Création d'une alerte...")
        alert = {
            "type": "SERVICE_DOWN",
            "severity": "CRITICAL",
            "title": "Service Down - Test E2E",
            "description": "Service de test est down",
            "sourceService": "test-service"
        }
        
        response = requests.post(
            f"{DASHBOARD_URL}/api/v1/alerts",
            json=alert
        )
        assert response.status_code == 201, f"Création d'alerte échouée: {response.text}"
        created_alert = response.json()
        alert_id = created_alert.get("id")
        print(f"✅ Alerte créée: ID {alert_id}")
        
        # 2. Vérifier que l'alerte est dans la liste
        print("\n2. Vérification de la liste des alertes...")
        response = requests.get(f"{DASHBOARD_URL}/api/v1/alerts/active")
        assert response.status_code == 200
        alerts = response.json()
        assert len(alerts) > 0
        print(f"✅ Alertes actives: {len(alerts)}")
        
        # 3. Acquitter l'alerte
        print("\n3. Acquittement de l'alerte...")
        response = requests.put(
            f"{DASHBOARD_URL}/api/v1/alerts/{alert_id}/acknowledge",
            json={"acknowledgedBy": "test-user"}
        )
        assert response.status_code == 200
        acknowledged_alert = response.json()
        assert acknowledged_alert.get("status") == "ACKNOWLEDGED"
        print("✅ Alerte acquittée")
    
    def test_dashboard_overview(self):
        """Test de la vue d'ensemble du dashboard"""
        
        print("\n1. Récupération de la vue d'ensemble...")
        response = requests.get(f"{DASHBOARD_URL}/api/v1/dashboard/overview")
        assert response.status_code == 200, f"Dashboard échoué: {response.text}"
        
        overview = response.json()
        assert "totalAssets" in overview
        assert "systemStatus" in overview
        assert "serviceMetrics" in overview
        
        print(f"✅ Vue d'ensemble récupérée")
        print(f"   - Actifs surveillés: {overview.get('totalAssets')}")
        print(f"   - Statut système: {overview.get('systemStatus')}")
        print(f"   - Services: {len(overview.get('serviceMetrics', {}))}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

