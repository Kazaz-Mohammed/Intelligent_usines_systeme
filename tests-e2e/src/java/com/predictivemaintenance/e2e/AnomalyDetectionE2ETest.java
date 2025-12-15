package com.predictivemaintenance.e2e;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;
import org.awaitility.Awaitility;

import java.time.Duration;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Test E2E : Flux complet de détection d'anomalie
 */
public class AnomalyDetectionE2ETest extends E2ETestBase {
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Test
    void testCompleteAnomalyDetectionFlow() {
        System.out.println("\n=== Test E2E : Flux complet de détection d'anomalie ===\n");
        
        // 1. Ingestion de données IoT
        System.out.println("1. Ingestion de données IoT...");
        Map<String, Object> sensorData = Map.of(
                "assetId", "PUMP_E2E_001",
                "sensorId", "SENSOR_E2E_001",
                "timestamp", "2024-01-15T10:00:00Z",
                "values", Map.of(
                        "vibration", 5.2,
                        "temperature", 75.5,
                        "pressure", 1.2,
                        "current", 4.8,
                        "rpm", 1500
                )
        );
        
        String ingestionResponse = webClient.post()
                .uri(INGESTION_URL + "/api/v1/sensors/data")
                .bodyValue(sensorData)
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.ofSeconds(10));
        
        assertThat(ingestionResponse).isNotNull();
        System.out.println("✅ Données ingérées");
        
        // 2. Attendre le traitement (via Kafka)
        System.out.println("\n2. Attente du traitement via Kafka...");
        Awaitility.await()
                .atMost(Duration.ofMinutes(2))
                .pollInterval(Duration.ofSeconds(5))
                .until(() -> checkAnomalyDetected("PUMP_E2E_001"));
        
        System.out.println("✅ Anomalie détectée");
        
        // 3. Vérifier la création d'intervention
        System.out.println("\n3. Vérification de la création d'intervention...");
        Awaitility.await()
                .atMost(Duration.ofMinutes(2))
                .pollInterval(Duration.ofSeconds(5))
                .until(() -> checkWorkOrderCreated("PUMP_E2E_001"));
        
        System.out.println("✅ Intervention créée");
        
        // 4. Vérifier l'affichage dans le dashboard
        System.out.println("\n4. Vérification du dashboard...");
        Map<String, Object> overview = webClient.get()
                .uri(DASHBOARD_URL + "/api/v1/dashboard/overview")
                .retrieve()
                .bodyToMono(Map.class)
                .block(Duration.ofSeconds(10));
        
        assertThat(overview).isNotNull();
        assertThat(overview).containsKey("totalAssets");
        System.out.println("✅ Dashboard accessible");
        
        System.out.println("\n✅ Flux complet de détection d'anomalie réussi!");
    }
    
    /**
     * Vérifie si une anomalie a été détectée
     */
    private boolean checkAnomalyDetected(String assetId) {
        try {
            Map<String, Object> response = webClient.get()
                    .uri(uriBuilder -> uriBuilder
                            .path(DETECTION_URL + "/api/v1/anomalies/")
                            .queryParam("asset_id", assetId)
                            .queryParam("limit", 10)
                            .build())
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block(Duration.ofSeconds(5));
            
            if (response != null && response.containsKey("anomalies")) {
                @SuppressWarnings("unchecked")
                java.util.List<Map<String, Object>> anomalies = 
                        (java.util.List<Map<String, Object>>) response.get("anomalies");
                return anomalies != null && !anomalies.isEmpty();
            }
            return false;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Vérifie si un work order a été créé
     */
    private boolean checkWorkOrderCreated(String assetId) {
        try {
            @SuppressWarnings("unchecked")
            java.util.List<Map<String, Object>> workOrders = webClient.get()
                    .uri(ORCHESTRATEUR_URL + "/api/v1/work-orders/asset/" + assetId)
                    .retrieve()
                    .bodyToMono(java.util.List.class)
                    .block(Duration.ofSeconds(5));
            
            return workOrders != null && !workOrders.isEmpty();
        } catch (Exception e) {
            return false;
        }
    }
}

