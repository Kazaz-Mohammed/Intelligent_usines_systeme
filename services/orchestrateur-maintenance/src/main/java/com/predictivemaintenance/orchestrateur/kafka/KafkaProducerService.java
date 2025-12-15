package com.predictivemaintenance.orchestrateur.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Service Kafka Producer pour publier les ordres de travail et plans de maintenance
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class KafkaProducerService {
    
    private final KafkaTemplate<String, String> kafkaTemplate;
    private final OrchestrateurConfig config;
    private final ObjectMapper objectMapper;
    
    /**
     * Publie un ordre de travail sur le topic work-orders
     * 
     * @param workOrder Ordre de travail à publier
     */
    public void publishWorkOrder(WorkOrder workOrder) {
        try {
            String message = objectMapper.writeValueAsString(convertWorkOrderToMessage(workOrder));
            
            CompletableFuture<SendResult<String, String>> future = kafkaTemplate.send(
                    config.getKafka().getTopicWorkOrders(),
                    workOrder.getAssetId(),
                    message
            );
            
            future.whenComplete((result, exception) -> {
                if (exception == null) {
                    log.info("Ordre de travail publié: {} sur topic {}", 
                            workOrder.getWorkOrderNumber(), 
                            config.getKafka().getTopicWorkOrders());
                } else {
                    log.error("Erreur lors de la publication de l'ordre de travail: {}", 
                            exception.getMessage(), exception);
                }
            });
            
        } catch (Exception e) {
            log.error("Erreur lors de la sérialisation de l'ordre de travail: {}", 
                    e.getMessage(), e);
        }
    }
    
    /**
     * Publie plusieurs ordres de travail (batch)
     * 
     * @param workOrders Liste d'ordres de travail
     */
    public void publishWorkOrdersBatch(List<WorkOrder> workOrders) {
        log.info("Publication de {} ordres de travail en batch", workOrders.size());
        
        for (WorkOrder workOrder : workOrders) {
            publishWorkOrder(workOrder);
        }
    }
    
    /**
     * Publie un plan de maintenance sur le topic maintenance-plans
     * 
     * @param maintenancePlan Plan de maintenance à publier
     */
    public void publishMaintenancePlan(MaintenancePlanMessage maintenancePlan) {
        try {
            String message = objectMapper.writeValueAsString(maintenancePlan);
            
            CompletableFuture<SendResult<String, String>> future = kafkaTemplate.send(
                    config.getKafka().getTopicMaintenancePlans(),
                    maintenancePlan.getAssetId(),
                    message
            );
            
            future.whenComplete((result, exception) -> {
                if (exception == null) {
                    log.info("Plan de maintenance publié pour l'actif: {} sur topic {}", 
                            maintenancePlan.getAssetId(),
                            config.getKafka().getTopicMaintenancePlans());
                } else {
                    log.error("Erreur lors de la publication du plan de maintenance: {}", 
                            exception.getMessage(), exception);
                }
            });
            
        } catch (Exception e) {
            log.error("Erreur lors de la sérialisation du plan de maintenance: {}", 
                    e.getMessage(), e);
        }
    }
    
    /**
     * Convertit un WorkOrder en message Kafka
     */
    private WorkOrderMessage convertWorkOrderToMessage(WorkOrder workOrder) {
        return WorkOrderMessage.builder()
                .workOrderNumber(workOrder.getWorkOrderNumber())
                .assetId(workOrder.getAssetId())
                .sensorId(workOrder.getSensorId())
                .priority(workOrder.getPriority().name())
                .interventionType(workOrder.getInterventionType())
                .status(workOrder.getStatus().name())
                .scheduledStartTime(workOrder.getScheduledStartTime() != null ? 
                        workOrder.getScheduledStartTime().toString() : null)
                .scheduledEndTime(workOrder.getScheduledEndTime() != null ? 
                        workOrder.getScheduledEndTime().toString() : null)
                .estimatedDurationMinutes(workOrder.getEstimatedDurationMinutes())
                .assignedTechnicianId(workOrder.getAssignedTechnicianId())
                .assignedTeam(workOrder.getAssignedTeam())
                .build();
    }
    
    /**
     * Classe pour représenter un message d'ordre de travail pour Kafka
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class WorkOrderMessage {
        private String workOrderNumber;
        private String assetId;
        private String sensorId;
        private String priority;
        private String interventionType;
        private String status;
        private String scheduledStartTime;
        private String scheduledEndTime;
        private Integer estimatedDurationMinutes;
        private String assignedTechnicianId;
        private String assignedTeam;
    }
    
    /**
     * Classe pour représenter un plan de maintenance pour Kafka
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class MaintenancePlanMessage {
        private String assetId;
        private String planType; // "preventive", "predictive", "corrective"
        private String scheduledDate;
        private List<String> workOrderNumbers;
        private String description;
    }
}

