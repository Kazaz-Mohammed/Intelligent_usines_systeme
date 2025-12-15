package com.predictivemaintenance.e2e;

import org.junit.jupiter.api.BeforeAll;
import org.springframework.web.reactive.function.client.WebClient;
import org.awaitility.Awaitility;

import java.time.Duration;
import java.util.Map;

/**
 * Classe de base pour les tests E2E
 */
public abstract class E2ETestBase {
    
    protected static final String INGESTION_URL = "http://localhost:8081";
    protected static final String PREPROCESSING_URL = "http://localhost:8082";
    protected static final String EXTRACTION_URL = "http://localhost:8083";
    protected static final String DETECTION_URL = "http://localhost:8084";
    protected static final String PREDICTION_URL = "http://localhost:8085";
    protected static final String ORCHESTRATEUR_URL = "http://localhost:8087";
    protected static final String DASHBOARD_URL = "http://localhost:8086";
    
    protected static WebClient webClient;
    
    @BeforeAll
    static void setUp() {
        webClient = WebClient.builder().build();
        waitForServices();
    }
    
    /**
     * Attend que tous les services soient prêts
     */
    protected static void waitForServices() {
        Map<String, String> services = Map.of(
                "IngestionIIoT", INGESTION_URL,
                "Preprocessing", PREPROCESSING_URL,
                "ExtractionFeatures", EXTRACTION_URL,
                "DetectionAnomalies", DETECTION_URL,
                "PredictionRUL", PREDICTION_URL,
                "OrchestrateurMaintenance", ORCHESTRATEUR_URL,
                "DashboardMonitoring", DASHBOARD_URL
        );
        
        for (Map.Entry<String, String> entry : services.entrySet()) {
            String serviceName = entry.getKey();
            String serviceUrl = entry.getValue();
            
            Awaitility.await()
                    .atMost(Duration.ofMinutes(5))
                    .pollInterval(Duration.ofSeconds(2))
                    .until(() -> isServiceReady(serviceUrl));
            
            System.out.println("✅ " + serviceName + " est prêt");
        }
    }
    
    /**
     * Vérifie si un service est prêt
     */
    private static boolean isServiceReady(String url) {
        try {
            String response = webClient.get()
                    .uri(url + "/health")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(5));
            return response != null;
        } catch (Exception e) {
            return false;
        }
    }
}

