package com.predictivemaintenance.orchestrateur.kafka;

import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import com.predictivemaintenance.orchestrateur.service.PlanningService;
import com.predictivemaintenance.orchestrateur.service.WorkOrderService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Tests pour KafkaOrchestrationService
 */
class KafkaOrchestrationServiceTest {
    
    @Mock
    private PlanningService planningService;
    
    @Mock
    private WorkOrderService workOrderService;
    
    @Mock
    private KafkaProducerService kafkaProducerService;
    
    private KafkaOrchestrationService orchestrationService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        orchestrationService = new KafkaOrchestrationService(
                planningService, workOrderService, kafkaProducerService
        );
    }
    
    @Test
    void testProcessAnomalyIntervention() {
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET001")
                .priority(PriorityLevel.HIGH)
                .interventionType("corrective")
                .build();
        
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-001")
                .assetId("ASSET001")
                .status(WorkOrderStatus.SCHEDULED)
                .build();
        
        when(planningService.planIntervention(request)).thenReturn(workOrder);
        when(workOrderService.save(workOrder)).thenReturn(workOrder);
        
        orchestrationService.processAnomalyIntervention(request);
        
        verify(planningService, times(1)).planIntervention(request);
        verify(workOrderService, times(1)).save(workOrder);
        verify(kafkaProducerService, times(1)).publishWorkOrder(workOrder);
    }
    
    @Test
    void testProcessRulInterventionWithLowRul() {
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET002")
                .priority(PriorityLevel.CRITICAL)
                .interventionType("predictive")
                .rulContext(InterventionRequest.RulContext.builder()
                        .predictedRul(30.0)
                        .build())
                .build();
        
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-002")
                .assetId("ASSET002")
                .status(WorkOrderStatus.SCHEDULED)
                .build();
        
        when(planningService.planIntervention(request)).thenReturn(workOrder);
        when(workOrderService.save(workOrder)).thenReturn(workOrder);
        
        orchestrationService.processRulIntervention(request);
        
        verify(planningService, times(1)).planIntervention(request);
        verify(workOrderService, times(1)).save(workOrder);
        verify(kafkaProducerService, times(1)).publishWorkOrder(workOrder);
    }
    
    @Test
    void testProcessRulInterventionWithHighRul() {
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET003")
                .priority(PriorityLevel.LOW)
                .interventionType("predictive")
                .rulContext(InterventionRequest.RulContext.builder()
                        .predictedRul(500.0)
                        .build())
                .build();
        
        orchestrationService.processRulIntervention(request);
        
        // Ne devrait pas créer d'intervention si RUL est élevée
        verify(planningService, never()).planIntervention(any());
        verify(workOrderService, never()).save(any());
        verify(kafkaProducerService, never()).publishWorkOrder(any());
    }
    
    @Test
    void testProcessRulInterventionWithCriticalPriority() {
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET004")
                .priority(PriorityLevel.CRITICAL)
                .interventionType("predictive")
                .rulContext(InterventionRequest.RulContext.builder()
                        .predictedRul(400.0) // RUL élevée mais priorité CRITICAL
                        .build())
                .build();
        
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-004")
                .assetId("ASSET004")
                .status(WorkOrderStatus.SCHEDULED)
                .build();
        
        when(planningService.planIntervention(request)).thenReturn(workOrder);
        when(workOrderService.save(workOrder)).thenReturn(workOrder);
        
        orchestrationService.processRulIntervention(request);
        
        // Devrait créer une intervention même avec RUL élevée si priorité CRITICAL
        verify(planningService, times(1)).planIntervention(request);
        verify(workOrderService, times(1)).save(workOrder);
        verify(kafkaProducerService, times(1)).publishWorkOrder(workOrder);
    }
}

