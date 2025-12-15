package com.predictivemaintenance.orchestrateur.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.kafka.core.KafkaTemplate;

import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Tests pour KafkaProducerService
 */
class KafkaProducerServiceTest {
    
    @Mock
    private KafkaTemplate<String, String> kafkaTemplate;
    
    @Mock
    private OrchestrateurConfig config;
    
    private ObjectMapper objectMapper;
    private KafkaProducerService kafkaProducerService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        objectMapper = new ObjectMapper();
        
        OrchestrateurConfig.Kafka kafkaConfig = new OrchestrateurConfig.Kafka();
        kafkaConfig.setTopicWorkOrders("work-orders");
        kafkaConfig.setTopicMaintenancePlans("maintenance-plans");
        when(config.getKafka()).thenReturn(kafkaConfig);
        
        kafkaProducerService = new KafkaProducerService(kafkaTemplate, config, objectMapper);
    }
    
    @Test
    void testPublishWorkOrder() {
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-123")
                .assetId("ASSET001")
                .priority(PriorityLevel.HIGH)
                .interventionType("corrective")
                .status(WorkOrderStatus.SCHEDULED)
                .scheduledStartTime(LocalDateTime.now())
                .scheduledEndTime(LocalDateTime.now().plusHours(2))
                .estimatedDurationMinutes(120)
                .build();
        
        kafkaProducerService.publishWorkOrder(workOrder);
        
        // Vérifier que send a été appelé
        verify(kafkaTemplate, times(1)).send(
                eq("work-orders"),
                eq("ASSET001"),
                anyString()
        );
    }
    
    @Test
    void testPublishWorkOrdersBatch() {
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder()
                        .workOrderNumber("WO-001")
                        .assetId("ASSET001")
                        .priority(PriorityLevel.HIGH)
                        .status(WorkOrderStatus.SCHEDULED)
                        .build(),
                WorkOrder.builder()
                        .workOrderNumber("WO-002")
                        .assetId("ASSET002")
                        .priority(PriorityLevel.MEDIUM)
                        .status(WorkOrderStatus.SCHEDULED)
                        .build()
        );
        
        kafkaProducerService.publishWorkOrdersBatch(workOrders);
        
        // Vérifier que send a été appelé 2 fois
        verify(kafkaTemplate, times(2)).send(
                eq("work-orders"),
                anyString(),
                anyString()
        );
    }
    
    @Test
    void testPublishMaintenancePlan() {
        KafkaProducerService.MaintenancePlanMessage plan = 
                KafkaProducerService.MaintenancePlanMessage.builder()
                        .assetId("ASSET001")
                        .planType("preventive")
                        .scheduledDate(LocalDateTime.now().toString())
                        .description("Plan de maintenance préventive")
                        .build();
        
        kafkaProducerService.publishMaintenancePlan(plan);
        
        // Vérifier que send a été appelé
        verify(kafkaTemplate, times(1)).send(
                eq("maintenance-plans"),
                eq("ASSET001"),
                anyString()
        );
    }
}

