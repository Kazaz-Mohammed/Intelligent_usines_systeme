package com.predictivemaintenance.orchestrateur.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration du service OrchestrateurMaintenance
 */
@Configuration
@ConfigurationProperties(prefix = "orchestrateur.maintenance")
@Data
public class OrchestrateurConfig {
    
    private Kafka kafka = new Kafka();
    private Optimization optimization = new Optimization();
    private Constraints constraints = new Constraints();
    private Sla sla = new Sla();
    
    @Data
    public static class Kafka {
        private String topicAnomalies = "anomalies-detected";
        private String topicRulPredictions = "rul-predictions";
        private String topicWorkOrders = "work-orders";
        private String topicMaintenancePlans = "maintenance-plans";
    }
    
    @Data
    public static class Optimization {
        private boolean enabled = true;
        private int maxIterations = 1000;
        private int timeLimitSeconds = 60;
    }
    
    @Data
    public static class Constraints {
        private int maxTechniciansPerShift = 10;
        private int minTimeBetweenInterventionsHours = 2;
        private int maxInterventionDurationHours = 8;
        private boolean safetyConstraintsEnabled = true;
    }
    
    @Data
    public static class Sla {
        private int criticalResponseHours = 1;
        private int highPriorityResponseHours = 4;
        private int mediumPriorityResponseHours = 24;
        private int lowPriorityResponseHours = 72;
    }
}

