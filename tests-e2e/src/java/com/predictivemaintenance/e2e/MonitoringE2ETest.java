package com.predictivemaintenance.e2e;

import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Test E2E : Monitoring et alertes
 */
public class MonitoringE2ETest extends E2ETestBase {
    
    @Test
    void testServiceMonitoring() {
        System.out.println("\n=== Test E2E : Monitoring des services ===\n");
        
        // 1. Vérifier le statut des services
        System.out.println("1. Vérification du statut des services...");
        @SuppressWarnings("unchecked")
        Map<String, Object> services = webClient.get()
                .uri(DASHBOARD_URL + "/api/v1/monitoring/services")
                .retrieve()
                .bodyToMono(Map.class)
                .block(Duration.ofSeconds(10));
        
        assertThat(services).isNotNull();
        System.out.println("✅ Services monitorés: " + services.size());
        
        // 2. Vérifier le health check global
        System.out.println("\n2. Vérification du health check global...");
        Map<String, Object> health = webClient.get()
                .uri(DASHBOARD_URL + "/api/v1/monitoring/health")
                .retrieve()
                .bodyToMono(Map.class)
                .block(Duration.ofSeconds(10));
        
        assertThat(health).isNotNull();
        assertThat(health).containsKey("status");
        assertThat(health).containsKey("upServices");
        
        System.out.println("✅ Statut global: " + health.get("status"));
        System.out.println("✅ Services UP: " + health.get("upServices") + "/" + health.get("totalServices"));
    }
    
    @Test
    void testAlertCreation() {
        System.out.println("\n=== Test E2E : Création d'alerte ===\n");
        
        // 1. Créer une alerte
        System.out.println("1. Création d'une alerte...");
        Map<String, Object> alert = Map.of(
                "type", "SERVICE_DOWN",
                "severity", "CRITICAL",
                "title", "Service Down - Test E2E",
                "description", "Service de test est down",
                "sourceService", "test-service"
        );
        
        Map<String, Object> createdAlert = webClient.post()
                .uri(DASHBOARD_URL + "/api/v1/alerts")
                .bodyValue(alert)
                .retrieve()
                .bodyToMono(Map.class)
                .block(Duration.ofSeconds(10));
        
        assertThat(createdAlert).isNotNull();
        assertThat(createdAlert).containsKey("id");
        
        Object alertId = createdAlert.get("id");
        System.out.println("✅ Alerte créée: ID " + alertId);
        
        // 2. Vérifier que l'alerte est dans la liste
        System.out.println("\n2. Vérification de la liste des alertes...");
        @SuppressWarnings("unchecked")
        java.util.List<Map<String, Object>> alerts = webClient.get()
                .uri(DASHBOARD_URL + "/api/v1/alerts/active")
                .retrieve()
                .bodyToMono(java.util.List.class)
                .block(Duration.ofSeconds(10));
        
        assertThat(alerts).isNotNull();
        assertThat(alerts.size()).isGreaterThan(0);
        System.out.println("✅ Alertes actives: " + alerts.size());
    }
}

