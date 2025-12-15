package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.service.DroolsRuleService.MaintenanceDecision;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests pour OptimizationService
 */
class OptimizationServiceTest {
    
    private OptimizationService optimizationService;
    private OrchestrateurConfig config;
    
    @BeforeEach
    void setUp() {
        config = new OrchestrateurConfig();
        
        OrchestrateurConfig.Optimization optimization = new OrchestrateurConfig.Optimization();
        optimization.setEnabled(true);
        config.setOptimization(optimization);
        
        OrchestrateurConfig.Constraints constraints = new OrchestrateurConfig.Constraints();
        constraints.setMinTimeBetweenInterventionsHours(2);
        constraints.setSafetyConstraintsEnabled(true);
        config.setConstraints(constraints);
        
        OrchestrateurConfig.Sla sla = new OrchestrateurConfig.Sla();
        sla.setCriticalResponseHours(1);
        sla.setHighPriorityResponseHours(4);
        sla.setMediumPriorityResponseHours(24);
        sla.setLowPriorityResponseHours(72);
        config.setSla(sla);
        
        optimizationService = new OptimizationService(config);
    }
    
    @Test
    void testOptimizeSchedule() {
        List<MaintenanceDecision> decisions = List.of(
                createDecision("ASSET001", PriorityLevel.CRITICAL),
                createDecision("ASSET002", PriorityLevel.MEDIUM),
                createDecision("ASSET003", PriorityLevel.HIGH)
        );
        
        List<InterventionRequest> requests = List.of(
                createRequest("ASSET001", PriorityLevel.CRITICAL),
                createRequest("ASSET002", PriorityLevel.MEDIUM),
                createRequest("ASSET003", PriorityLevel.HIGH)
        );
        
        List<WorkOrder> workOrders = optimizationService.optimizeSchedule(decisions, requests);
        
        assertNotNull(workOrders);
        assertEquals(3, workOrders.size());
        
        // Vérifier que CRITICAL est planifié en premier
        assertEquals(PriorityLevel.CRITICAL, workOrders.get(0).getPriority());
    }
    
    @Test
    void testOptimizeScheduleEmpty() {
        List<WorkOrder> workOrders = optimizationService.optimizeSchedule(
                List.of(), List.of()
        );
        
        assertNotNull(workOrders);
        assertTrue(workOrders.isEmpty());
    }
    
    @Test
    void testOptimizeScheduleSizeMismatch() {
        List<MaintenanceDecision> decisions = List.of(createDecision("ASSET001", PriorityLevel.HIGH));
        List<InterventionRequest> requests = List.of(
                createRequest("ASSET001", PriorityLevel.HIGH),
                createRequest("ASSET002", PriorityLevel.MEDIUM)
        );
        
        assertThrows(IllegalArgumentException.class, () -> {
            optimizationService.optimizeSchedule(decisions, requests);
        });
    }
    
    private MaintenanceDecision createDecision(String assetId, PriorityLevel priority) {
        return MaintenanceDecision.builder()
                .assetId(assetId)
                .priority(priority)
                .interventionType("corrective")
                .build();
    }
    
    private InterventionRequest createRequest(String assetId, PriorityLevel priority) {
        return InterventionRequest.builder()
                .assetId(assetId)
                .priority(priority)
                .interventionType("corrective")
                .build();
    }
}

