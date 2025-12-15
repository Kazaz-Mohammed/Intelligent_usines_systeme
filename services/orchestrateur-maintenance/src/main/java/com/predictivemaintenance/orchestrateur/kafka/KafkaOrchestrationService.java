package com.predictivemaintenance.orchestrateur.kafka;

import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.service.PlanningService;
import com.predictivemaintenance.orchestrateur.service.WorkOrderService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * Service d'orchestration pour traiter les messages Kafka
 * et créer automatiquement des interventions de maintenance
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class KafkaOrchestrationService {
    
    private final PlanningService planningService;
    private final WorkOrderService workOrderService;
    private final KafkaProducerService kafkaProducerService;
    
    /**
     * Traite une anomalie détectée et crée une intervention
     * 
     * @param request Requête d'intervention basée sur l'anomalie
     */
    public void processAnomalyIntervention(InterventionRequest request) {
        log.info("Traitement d'une intervention basée sur anomalie pour l'actif: {}", 
                request.getAssetId());
        
        try {
            // Planifier l'intervention
            WorkOrder workOrder = planningService.planIntervention(request);
            
            // Sauvegarder l'ordre de travail
            WorkOrder saved = workOrderService.save(workOrder);
            
            // Publier sur Kafka
            kafkaProducerService.publishWorkOrder(saved);
            
            log.info("Intervention créée et publiée: {} pour l'actif {}", 
                    saved.getWorkOrderNumber(), request.getAssetId());
            
        } catch (Exception e) {
            log.error("Erreur lors du traitement de l'intervention basée sur anomalie: {}", 
                    e.getMessage(), e);
            throw e;
        }
    }
    
    /**
     * Traite une prédiction RUL et crée une intervention si nécessaire
     * 
     * @param request Requête d'intervention basée sur la RUL
     */
    public void processRulIntervention(InterventionRequest request) {
        log.info("Traitement d'une intervention basée sur RUL pour l'actif: {}", 
                request.getAssetId());
        
        try {
            // Vérifier si une intervention est nécessaire
            // (seulement si RUL est faible ou si priorité élevée)
            if (shouldCreateIntervention(request)) {
                // Planifier l'intervention
                WorkOrder workOrder = planningService.planIntervention(request);
                
                // Sauvegarder l'ordre de travail
                WorkOrder saved = workOrderService.save(workOrder);
                
                // Publier sur Kafka
                kafkaProducerService.publishWorkOrder(saved);
                
                log.info("Intervention RUL créée et publiée: {} pour l'actif {}", 
                        saved.getWorkOrderNumber(), request.getAssetId());
            } else {
                log.debug("Aucune intervention nécessaire pour l'actif {} (RUL suffisante)", 
                        request.getAssetId());
            }
            
        } catch (Exception e) {
            log.error("Erreur lors du traitement de l'intervention basée sur RUL: {}", 
                    e.getMessage(), e);
            throw e;
        }
    }
    
    /**
     * Détermine si une intervention doit être créée basée sur la RUL
     */
    private boolean shouldCreateIntervention(InterventionRequest request) {
        if (request.getRulContext() == null) {
            return false;
        }
        
        Double predictedRul = request.getRulContext().getPredictedRul();
        
        // Créer une intervention si RUL < 200 cycles ou si priorité CRITICAL/HIGH
        return predictedRul != null && (
                predictedRul < 200 ||
                request.getPriority() == PriorityLevel.CRITICAL ||
                request.getPriority() == PriorityLevel.HIGH
        );
    }
}

