package com.predictivemaintenance.orchestrateur.kafka;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;

import java.util.function.Consumer;

/**
 * Service Kafka Consumer pour consommer les anomalies et prédictions RUL
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class KafkaConsumerService {
    
    private final OrchestrateurConfig config;
    private final ObjectMapper objectMapper;
    private final KafkaOrchestrationService orchestrationService;
    
    /**
     * Consomme les anomalies détectées depuis le topic anomalies-detected
     */
    @KafkaListener(
            topics = "#{@orchestrateurConfig.kafka.topicAnomalies}",
            groupId = "#{@orchestrateurConfig.kafka.topicAnomalies}-group",
            containerFactory = "kafkaListenerContainerFactory"
    )
    public void consumeAnomaly(
            @Payload String message,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
            Acknowledgment acknowledgment
    ) {
        try {
            log.debug("Message reçu du topic {}: {}", topic, message);
            
            // Désérialiser le message JSON
            AnomalyMessage anomalyMessage = objectMapper.readValue(message, AnomalyMessage.class);
            
            // Convertir en InterventionRequest
            InterventionRequest request = convertAnomalyToInterventionRequest(anomalyMessage);
            
            // Traiter l'intervention
            orchestrationService.processAnomalyIntervention(request);
            
            // Acknowledger le message
            if (acknowledgment != null) {
                acknowledgment.acknowledge();
            }
            
            log.info("Anomalie traitée pour l'actif: {}", anomalyMessage.getAssetId());
            
        } catch (Exception e) {
            log.error("Erreur lors du traitement de l'anomalie: {}", e.getMessage(), e);
            // Ne pas acknowledger en cas d'erreur pour permettre le retry
        }
    }
    
    /**
     * Consomme les prédictions RUL depuis le topic rul-predictions
     */
    @KafkaListener(
            topics = "${orchestrateur.maintenance.kafka.topic-rul-predictions:rul-predictions}",
            groupId = "orchestrateur-maintenance-rul-group",
            containerFactory = "kafkaListenerContainerFactory"
    )
    public void consumeRulPrediction(
            @Payload String message,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
            Acknowledgment acknowledgment
    ) {
        try {
            log.debug("Message RUL reçu du topic {}: {}", topic, message);
            
            // Désérialiser le message JSON
            RulMessage rulMessage = objectMapper.readValue(message, RulMessage.class);
            
            // Convertir en InterventionRequest
            InterventionRequest request = convertRulToInterventionRequest(rulMessage);
            
            // Traiter l'intervention
            orchestrationService.processRulIntervention(request);
            
            // Acknowledger le message
            if (acknowledgment != null) {
                acknowledgment.acknowledge();
            }
            
            log.info("Prédiction RUL traitée pour l'actif: {}", rulMessage.getAssetId());
            
        } catch (Exception e) {
            log.error("Erreur lors du traitement de la prédiction RUL: {}", e.getMessage(), e);
            // Ne pas acknowledger en cas d'erreur pour permettre le retry
        }
    }
    
    /**
     * Convertit un message d'anomalie en InterventionRequest
     * Package-private for testing
     */
    InterventionRequest convertAnomalyToInterventionRequest(AnomalyMessage anomalyMessage) {
        // Déterminer la priorité basée sur la criticité
        PriorityLevel priority = switch (anomalyMessage.getCriticality().toLowerCase()) {
            case "critical" -> PriorityLevel.CRITICAL;
            case "high" -> PriorityLevel.HIGH;
            case "medium" -> PriorityLevel.MEDIUM;
            case "low" -> PriorityLevel.LOW;
            default -> PriorityLevel.MEDIUM;
        };
        
        return InterventionRequest.builder()
                .assetId(anomalyMessage.getAssetId())
                .sensorId(anomalyMessage.getSensorId())
                .priority(priority)
                .interventionType("corrective")
                .description("Intervention basée sur anomalie détectée: " + 
                        anomalyMessage.getCriticality())
                .anomalyContext(InterventionRequest.AnomalyContext.builder()
                        .isAnomaly(anomalyMessage.getIsAnomaly())
                        .criticality(anomalyMessage.getCriticality())
                        .anomalyScore(anomalyMessage.getFinalScore())
                        .build())
                .build();
    }
    
    /**
     * Convertit un message RUL en InterventionRequest
     * Package-private for testing
     */
    InterventionRequest convertRulToInterventionRequest(RulMessage rulMessage) {
        // Déterminer la priorité basée sur la RUL prédite
        PriorityLevel priority;
        if (rulMessage.getRulPrediction() < 50) {
            priority = PriorityLevel.CRITICAL;
        } else if (rulMessage.getRulPrediction() < 150) {
            priority = PriorityLevel.HIGH;
        } else if (rulMessage.getRulPrediction() < 300) {
            priority = PriorityLevel.MEDIUM;
        } else {
            priority = PriorityLevel.LOW;
        }
        
        return InterventionRequest.builder()
                .assetId(rulMessage.getAssetId())
                .sensorId(rulMessage.getSensorId())
                .priority(priority)
                .interventionType("predictive")
                .description("Intervention basée sur prédiction RUL: " + 
                        rulMessage.getRulPrediction() + " cycles restants")
                .rulContext(InterventionRequest.RulContext.builder()
                        .predictedRul(rulMessage.getRulPrediction())
                        .confidenceIntervalLower(rulMessage.getConfidenceIntervalLower())
                        .confidenceIntervalUpper(rulMessage.getConfidenceIntervalUpper())
                        .build())
                .build();
    }
    
    /**
     * Classe pour représenter un message d'anomalie depuis Kafka
     * Uses @JsonProperty to map Python snake_case to Java camelCase
     */
    @lombok.Data
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class AnomalyMessage {
        @JsonProperty("asset_id")
        private String assetId;
        
        @JsonProperty("sensor_id")
        private String sensorId;
        
        @JsonProperty("timestamp")
        private String timestamp;
        
        @JsonProperty("final_score")
        private Double finalScore;
        
        @JsonProperty("is_anomaly")
        private Boolean isAnomaly;
        
        @JsonProperty("criticality")
        private String criticality;
    }
    
    /**
     * Classe pour représenter un message RUL depuis Kafka
     * Uses @JsonProperty to map Python snake_case to Java camelCase
     */
    @lombok.Data
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class RulMessage {
        @JsonProperty("asset_id")
        private String assetId;
        
        @JsonProperty("sensor_id")
        private String sensorId;
        
        @JsonProperty("timestamp")
        private String timestamp;
        
        @JsonProperty("rul_prediction")
        private Double rulPrediction;
        
        @JsonProperty("confidence_interval_lower")
        private Double confidenceIntervalLower;
        
        @JsonProperty("confidence_interval_upper")
        private Double confidenceIntervalUpper;
        
        @JsonProperty("model_used")
        private String modelUsed;
    }
}

