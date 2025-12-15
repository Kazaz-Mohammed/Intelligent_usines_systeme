package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.kie.api.KieServices;
import org.kie.api.builder.KieFileSystem;
import org.kie.api.builder.KieModule;
import org.kie.api.builder.KieBuilder;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests pour le service DroolsRuleService
 */
class DroolsRuleServiceTest {
    
    private DroolsRuleService droolsRuleService;
    private KieSession kieSession;
    
    @BeforeEach
    void setUp() {
        // Créer une session Drools pour les tests
        KieServices kieServices = KieServices.Factory.get();
        KieFileSystem kieFileSystem = kieServices.newKieFileSystem();
        
        // Charger les règles depuis le classpath
        String rules = loadRulesFromClasspath();
        kieFileSystem.write("src/main/resources/rules/maintenance-rules.drl", rules);
        
        KieBuilder kieBuilder = kieServices.newKieBuilder(kieFileSystem);
        kieBuilder.buildAll();
        KieModule kieModule = kieBuilder.getKieModule();
        KieContainer kieContainer = kieServices.newKieContainer(kieModule.getReleaseId());
        
        kieSession = kieContainer.newKieSession();
        droolsRuleService = new DroolsRuleService(kieSession);
    }
    
    @Test
    void testAnomalieCritique() {
        // Créer une requête avec anomalie critique
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET001")
                .priority(PriorityLevel.MEDIUM) // Priorité initiale
                .interventionType("corrective")
                .anomalyContext(InterventionRequest.AnomalyContext.builder()
                        .isAnomaly(true)
                        .criticality("critical")
                        .anomalyScore(0.95)
                        .build())
                .build();
        
        // Évaluer
        DroolsRuleService.MaintenanceDecision decision = 
                droolsRuleService.evaluateIntervention(request);
        
        // Vérifier que la priorité a été mise à jour à CRITICAL
        assertEquals(PriorityLevel.CRITICAL, decision.getPriority());
        assertEquals("corrective", decision.getInterventionType());
        assertNotNull(decision.getReason());
    }
    
    @Test
    void testRulTresFaible() {
        // Créer une requête avec RUL très faible
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET002")
                .priority(PriorityLevel.MEDIUM)
                .interventionType("preventive")
                .rulContext(InterventionRequest.RulContext.builder()
                        .predictedRul(30.0)
                        .confidenceIntervalLower(20.0)
                        .confidenceIntervalUpper(40.0)
                        .build())
                .build();
        
        // Évaluer
        DroolsRuleService.MaintenanceDecision decision = 
                droolsRuleService.evaluateIntervention(request);
        
        // Vérifier que la priorité a été mise à jour à CRITICAL
        assertEquals(PriorityLevel.CRITICAL, decision.getPriority());
        assertEquals("predictive", decision.getInterventionType());
    }
    
    @Test
    void testAnomalieEtRul() {
        // Créer une requête avec anomalie critique ET RUL faible
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET003")
                .priority(PriorityLevel.MEDIUM)
                .interventionType("preventive")
                .anomalyContext(InterventionRequest.AnomalyContext.builder()
                        .isAnomaly(true)
                        .criticality("critical")
                        .anomalyScore(0.9)
                        .build())
                .rulContext(InterventionRequest.RulContext.builder()
                        .predictedRul(50.0)
                        .confidenceIntervalLower(40.0)
                        .confidenceIntervalUpper(60.0)
                        .build())
                .build();
        
        // Évaluer
        DroolsRuleService.MaintenanceDecision decision = 
                droolsRuleService.evaluateIntervention(request);
        
        // Vérifier que la priorité est CRITICAL (règle combinée)
        assertEquals(PriorityLevel.CRITICAL, decision.getPriority());
    }
    
    @Test
    void testPrioriteParDefaut() {
        // Créer une requête sans contexte
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET004")
                .priority(null)
                .interventionType("preventive")
                .build();
        
        // Évaluer
        DroolsRuleService.MaintenanceDecision decision = 
                droolsRuleService.evaluateIntervention(request);
        
        // Vérifier que la priorité par défaut est MEDIUM
        assertEquals(PriorityLevel.MEDIUM, decision.getPriority());
    }
    
    private String loadRulesFromClasspath() {
        try {
            java.io.InputStream inputStream = getClass()
                    .getClassLoader()
                    .getResourceAsStream("rules/maintenance-rules.drl");
            if (inputStream != null) {
                return new String(inputStream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
            }
        } catch (Exception e) {
            // Ignorer
        }
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

