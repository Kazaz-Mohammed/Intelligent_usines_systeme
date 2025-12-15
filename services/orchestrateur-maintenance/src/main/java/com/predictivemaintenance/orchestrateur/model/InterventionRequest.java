package com.predictivemaintenance.orchestrateur.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * Requête pour créer une intervention de maintenance
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InterventionRequest {
    
    @NotBlank(message = "L'ID de l'actif est requis")
    private String assetId;
    
    private String sensorId;
    
    @NotNull(message = "Le niveau de priorité est requis")
    private PriorityLevel priority;
    
    @NotBlank(message = "Le type d'intervention est requis")
    private String interventionType; // "preventive", "corrective", "predictive"
    
    private String description;
    
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime requestedStartTime;
    
    private Integer estimatedDurationMinutes;
    
    private Map<String, Object> metadata;
    
    // Données de contexte (anomalies, RUL, etc.)
    private AnomalyContext anomalyContext;
    private RulContext rulContext;
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AnomalyContext {
        private Double anomalyScore;
        private String criticality;
        private Boolean isAnomaly;
    }
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RulContext {
        private Double predictedRul;
        private Double confidenceIntervalLower;
        private Double confidenceIntervalUpper;
    }
}

