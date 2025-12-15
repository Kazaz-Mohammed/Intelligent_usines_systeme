package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import com.predictivemaintenance.orchestrateur.service.DroolsRuleService.MaintenanceDecision;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Service d'optimisation du planning avec OR-Tools
 * Pour l'instant, implémentation simplifiée (optimisation basique)
 * OR-Tools sera intégré dans une version future
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class OptimizationService {
    
    private final OrchestrateurConfig config;
    
    /**
     * Optimise le planning des interventions
     * 
     * @param decisions Décisions Drools pour chaque intervention
     * @param requests Requêtes d'intervention originales
     * @return Liste d'ordres de travail optimisés
     */
    public List<WorkOrder> optimizeSchedule(
            List<MaintenanceDecision> decisions,
            List<InterventionRequest> requests
    ) {
        log.info("Optimisation du planning pour {} interventions", decisions.size());
        
        if (decisions.size() != requests.size()) {
            throw new IllegalArgumentException("Le nombre de décisions doit correspondre au nombre de requêtes");
        }
        
        // Pour l'instant, optimisation simple basée sur la priorité
        // OR-Tools sera intégré pour une optimisation plus avancée
        
        List<WorkOrder> workOrders = new ArrayList<>();
        LocalDateTime currentTime = LocalDateTime.now();
        
        // Trier par priorité (CRITICAL d'abord, puis HIGH, MEDIUM, LOW)
        List<Integer> sortedIndices = new ArrayList<>();
        for (int i = 0; i < decisions.size(); i++) {
            sortedIndices.add(i);
        }
        
        sortedIndices.sort(Comparator.comparingInt(i -> 
                decisions.get(i).getPriority().getLevel()));
        
        // Planifier dans l'ordre de priorité
        for (int idx : sortedIndices) {
            MaintenanceDecision decision = decisions.get(idx);
            InterventionRequest request = requests.get(idx);
            
            // Calculer le temps de début
            LocalDateTime startTime = calculateOptimalStartTime(
                    currentTime, 
                    decision.getPriority(),
                    request.getRequestedStartTime()
            );
            
            // Estimer la durée
            int duration = request.getEstimatedDurationMinutes() != null ?
                    request.getEstimatedDurationMinutes() :
                    estimateDuration(decision.getInterventionType(), decision.getPriority());
            
            LocalDateTime endTime = startTime.plusMinutes(duration);
            
            // Vérifier les contraintes
            if (config.getConstraints().isSafetyConstraintsEnabled()) {
                startTime = applySafetyConstraints(startTime, workOrders);
            }
            
            // Créer l'ordre de travail
            WorkOrder workOrder = WorkOrder.builder()
                    .workOrderNumber(generateWorkOrderNumber())
                    .assetId(request.getAssetId())
                    .sensorId(request.getSensorId())
                    .priority(decision.getPriority())
                    .interventionType(decision.getInterventionType())
                    .description(request.getDescription() != null ? 
                            request.getDescription() : decision.getReason())
                    .status(WorkOrderStatus.SCHEDULED)
                    .scheduledStartTime(startTime)
                    .scheduledEndTime(endTime)
                    .estimatedDurationMinutes(duration)
                    .metadata(request.getMetadata())
                    .build();
            
            workOrders.add(workOrder);
            
            // Mettre à jour le temps courant (avec délai minimum entre interventions)
            currentTime = endTime.plusHours(
                    config.getConstraints().getMinTimeBetweenInterventionsHours()
            );
        }
        
        log.info("Planning optimisé: {} ordres de travail créés", workOrders.size());
        
        return workOrders;
    }
    
    /**
     * Calcule le temps de début optimal
     */
    private LocalDateTime calculateOptimalStartTime(
            LocalDateTime currentTime,
            PriorityLevel priority,
            LocalDateTime requestedStartTime
    ) {
        if (requestedStartTime != null && requestedStartTime.isAfter(currentTime)) {
            return requestedStartTime;
        }
        
        // Temps de réponse basé sur la priorité
        int responseHours = switch (priority) {
            case CRITICAL -> config.getSla().getCriticalResponseHours();
            case HIGH -> config.getSla().getHighPriorityResponseHours();
            case MEDIUM -> config.getSla().getMediumPriorityResponseHours();
            case LOW -> config.getSla().getLowPriorityResponseHours();
        };
        
        return currentTime.plusHours(responseHours);
    }
    
    /**
     * Applique les contraintes de sécurité
     */
    private LocalDateTime applySafetyConstraints(
            LocalDateTime proposedTime,
            List<WorkOrder> existingOrders
    ) {
        // Vérifier qu'il n'y a pas de chevauchement avec des interventions critiques
        for (WorkOrder order : existingOrders) {
            if (order.getPriority() == PriorityLevel.CRITICAL) {
                LocalDateTime orderEnd = order.getScheduledEndTime();
                if (proposedTime.isBefore(orderEnd)) {
                    // Décaler après la fin de l'intervention critique
                    proposedTime = orderEnd.plusHours(
                            config.getConstraints().getMinTimeBetweenInterventionsHours()
                    );
                }
            }
        }
        
        return proposedTime;
    }
    
    /**
     * Estime la durée d'une intervention
     */
    private int estimateDuration(String interventionType, PriorityLevel priority) {
        return switch (interventionType) {
            case "corrective" -> switch (priority) {
                case CRITICAL -> 120;
                case HIGH -> 180;
                case MEDIUM -> 240;
                case LOW -> 360;
            };
            case "predictive" -> 90;
            case "preventive" -> 60;
            default -> 120;
        };
    }
    
    /**
     * Génère un numéro d'ordre de travail
     */
    private String generateWorkOrderNumber() {
        return "WO-" + System.currentTimeMillis();
    }
}

