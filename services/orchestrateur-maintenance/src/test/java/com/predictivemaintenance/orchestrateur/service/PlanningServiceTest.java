package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.kie.api.KieServices;
import org.kie.api.builder.KieFileSystem;
import org.kie.api.builder.KieModule;
import org.kie.api.builder.KieBuilder;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests pour le service PlanningService
 */
class PlanningServiceTest {
    
    private PlanningService planningService;
    private DroolsRuleService droolsRuleService;
    private OptimizationService optimizationService;
    private OrchestrateurConfig config;
    
    @BeforeEach
    void setUp() {
        // Configuration
        config = new OrchestrateurConfig();
        OrchestrateurConfig.Optimization optimization = new OrchestrateurConfig.Optimization();
        optimization.setEnabled(true);
        config.setOptimization(optimization);
        
        OrchestrateurConfig.Constraints constraints = new OrchestrateurConfig.Constraints();
        constraints.setMinTimeBetweenInterventionsHours(2);
        config.setConstraints(constraints);
        
        OrchestrateurConfig.Sla sla = new OrchestrateurConfig.Sla();
        sla.setCriticalResponseHours(1);
        sla.setHighPriorityResponseHours(4);
        sla.setMediumPriorityResponseHours(24);
        sla.setLowPriorityResponseHours(72);
        config.setSla(sla);
        
        // Drools
        KieServices kieServices = KieServices.Factory.get();
        KieFileSystem kieFileSystem = kieServices.newKieFileSystem();
        kieFileSystem.write("src/main/resources/rules/maintenance-rules.drl", 
                loadRulesFromClasspath());
        KieBuilder kieBuilder = kieServices.newKieBuilder(kieFileSystem);
        kieBuilder.buildAll();
        KieModule kieModule = kieBuilder.getKieModule();
        KieContainer kieContainer = kieServices.newKieContainer(kieModule.getReleaseId());
        KieSession kieSession = kieContainer.newKieSession();
        
        droolsRuleService = new DroolsRuleService(kieSession);
        optimizationService = new OptimizationService(config);
        planningService = new PlanningService(droolsRuleService, optimizationService, config);
    }
    
    @Test
    void testPlanIntervention() {
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET001")
                .priority(PriorityLevel.HIGH)
                .interventionType("corrective")
                .anomalyContext(InterventionRequest.AnomalyContext.builder()
                        .isAnomaly(true)
                        .criticality("high")
                        .anomalyScore(0.8)
                        .build())
                .build();
        
        WorkOrder workOrder = planningService.planIntervention(request);
        
        assertNotNull(workOrder);
        assertNotNull(workOrder.getWorkOrderNumber());
        assertEquals("ASSET001", workOrder.getAssetId());
        assertNotNull(workOrder.getScheduledStartTime());
        assertNotNull(workOrder.getScheduledEndTime());
    }
    
    @Test
    void testPlanInterventionsOptimized() {
        List<InterventionRequest> requests = List.of(
                InterventionRequest.builder()
                        .assetId("ASSET001")
                        .priority(PriorityLevel.CRITICAL)
                        .interventionType("corrective")
                        .anomalyContext(InterventionRequest.AnomalyContext.builder()
                                .isAnomaly(true)
                                .criticality("critical")
                                .anomalyScore(0.95)
                                .build())
                        .build(),
                InterventionRequest.builder()
                        .assetId("ASSET002")
                        .priority(PriorityLevel.MEDIUM)
                        .interventionType("preventive")
                        .build()
        );
        
        List<WorkOrder> workOrders = planningService.planInterventionsOptimized(requests);
        
        assertEquals(2, workOrders.size());
        // Vérifier que CRITICAL est planifié avant MEDIUM
        assertTrue(workOrders.get(0).getPriority().getLevel() <= 
                   workOrders.get(1).getPriority().getLevel());
    }
    
    private String loadRulesFromClasspath() {
        return """
            package com.predictivemaintenance.orchestrateur.rules
            
            import com.predictivemaintenance.orchestrateur.model.*
            import com.predictivemaintenance.orchestrateur.service.DroolsRuleService.MaintenanceDecision
            
            rule "Default Priority"
                when
                    decision : MaintenanceDecision(priority == null)
                then
                    decision.setPriority(PriorityLevel.MEDIUM);
            end
            """;
    }
}

