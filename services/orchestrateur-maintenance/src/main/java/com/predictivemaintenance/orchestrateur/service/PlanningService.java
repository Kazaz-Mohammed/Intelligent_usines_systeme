package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Service de planification des interventions de maintenance
 * Orchestre Drools (règles) et OR-Tools (optimisation)
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class PlanningService {
    
    private final DroolsRuleService droolsRuleService;
    private final OptimizationService optimizationService;
    private final OrchestrateurConfig config;
    
    /**
     * Planifie une intervention en utilisant les règles Drools
     * 
     * @param request Requête d'intervention
     * @return Ordre de travail planifié
     */
    public WorkOrder planIntervention(InterventionRequest request) {
        log.info("Planification de l'intervention pour l'actif: {}", request.getAssetId());
        
        // 1. Évaluer avec Drools pour déterminer priorité et type
        DroolsRuleService.MaintenanceDecision decision = 
                droolsRuleService.evaluateIntervention(request);
        
        // 2. Calculer le temps de réponse basé sur la priorité et les SLA
        LocalDateTime scheduledStartTime = calculateScheduledStartTime(
                decision.getPriority(),
                request.getRequestedStartTime()
        );
        
        // 3. Estimer la durée si non fournie
        int estimatedDuration = request.getEstimatedDurationMinutes() != null ?
                request.getEstimatedDurationMinutes() :
                estimateDuration(decision.getInterventionType(), decision.getPriority());
        
        LocalDateTime scheduledEndTime = scheduledStartTime.plusMinutes(estimatedDuration);
        
        // 4. Créer l'ordre de travail
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber(generateWorkOrderNumber())
                .assetId(request.getAssetId())
                .sensorId(request.getSensorId())
                .priority(decision.getPriority())
                .interventionType(decision.getInterventionType())
                .description(request.getDescription() != null ? 
                        request.getDescription() : decision.getReason())
                .status(WorkOrderStatus.SCHEDULED)
                .scheduledStartTime(scheduledStartTime)
                .scheduledEndTime(scheduledEndTime)
                .estimatedDurationMinutes(estimatedDuration)
                .metadata(request.getMetadata())
                .build();
        
        log.info("Intervention planifiée: {} pour l'actif {} avec priorité {}", 
                workOrder.getWorkOrderNumber(), request.getAssetId(), decision.getPriority());
        
        return workOrder;
    }
    
    /**
     * Planifie plusieurs interventions avec optimisation
     * 
     * @param requests Liste de requêtes d'intervention
     * @return Liste d'ordres de travail optimisés
     */
    public List<WorkOrder> planInterventionsOptimized(List<InterventionRequest> requests) {
        log.info("Planification optimisée de {} interventions", requests.size());
        
        if (!config.getOptimization().isEnabled()) {
            // Si l'optimisation est désactivée, planifier séquentiellement
            return requests.stream()
                    .map(this::planIntervention)
                    .toList();
        }
        
        // 1. Évaluer toutes les requêtes avec Drools
        List<DroolsRuleService.MaintenanceDecision> decisions = requests.stream()
                .map(droolsRuleService::evaluateIntervention)
                .toList();
        
        // 2. Optimiser avec OR-Tools
        List<WorkOrder> workOrders = optimizationService.optimizeSchedule(decisions, requests);
        
        log.info("{} interventions planifiées et optimisées", workOrders.size());
        
        return workOrders;
    }
    
    /**
     * Calcule le temps de début planifié basé sur la priorité et les SLA
     */
    private LocalDateTime calculateScheduledStartTime(
            PriorityLevel priority, 
            LocalDateTime requestedStartTime
    ) {
        LocalDateTime now = LocalDateTime.now();
        
        // Si un temps de début est demandé et qu'il est dans le futur, l'utiliser
        if (requestedStartTime != null && requestedStartTime.isAfter(now)) {
            return requestedStartTime;
        }
        
        // Sinon, calculer basé sur les SLA
        int responseHours = switch (priority) {
            case CRITICAL -> config.getSla().getCriticalResponseHours();
            case HIGH -> config.getSla().getHighPriorityResponseHours();
            case MEDIUM -> config.getSla().getMediumPriorityResponseHours();
            case LOW -> config.getSla().getLowPriorityResponseHours();
        };
        
        return now.plusHours(responseHours);
    }
    
    /**
     * Estime la durée d'une intervention basée sur le type et la priorité
     */
    private int estimateDuration(String interventionType, PriorityLevel priority) {
        // Durées estimées en minutes
        return switch (interventionType) {
            case "corrective" -> switch (priority) {
                case CRITICAL -> 120; // 2 heures
                case HIGH -> 180; // 3 heures
                case MEDIUM -> 240; // 4 heures
                case LOW -> 360; // 6 heures
            };
            case "predictive" -> 90; // 1.5 heures
            case "preventive" -> 60; // 1 heure
            default -> 120; // 2 heures par défaut
        };
    }
    
    /**
     * Génère un numéro d'ordre de travail unique
     */
    private String generateWorkOrderNumber() {
        return "WO-" + System.currentTimeMillis();
    }
}

