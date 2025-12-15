package com.predictivemaintenance.dashboard.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

/**
 * Configuration du dashboard
 */
@Configuration
@ConfigurationProperties(prefix = "dashboard")
@Data
public class DashboardConfig {
    
    private Monitoring monitoring = new Monitoring();
    
    @Data
    public static class Monitoring {
        private String updateIntervalSeconds = "5";
        private Integer metricsRetentionDays = 30;
        private Map<String, String> services = Map.of(
                "ingestion-iiot", "http://localhost:8081",
                "extraction-features", "http://localhost:8083",
                "detection-anomalies", "http://localhost:8084",
                "prediction-rul", "http://localhost:8085",
                "orchestrateur-maintenance", "http://localhost:8087"
        );
        private Alerts alerts = new Alerts();
    }
    
    @Data
    public static class Alerts {
        private Boolean enabled = true;
        private Boolean emailEnabled = false;
        private Boolean smsEnabled = false;
        private Integer thresholdCritical = 10;
        private Integer thresholdHigh = 5;
    }
    
    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }
    
    public String getUpdateIntervalSeconds() {
        return monitoring.getUpdateIntervalSeconds();
    }
    
    public Map<String, String> getServices() {
        return monitoring.getServices();
    }
}

