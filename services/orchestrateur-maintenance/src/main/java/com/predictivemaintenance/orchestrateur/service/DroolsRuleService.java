package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.kie.api.runtime.KieSession;
import org.springframework.stereotype.Service;

/**
 * Service pour exécuter les règles Drools et prendre des décisions de maintenance
 */
@Service
@Slf4j
public class DroolsRuleService {
    
    private final KieSession kieSession;
    
    public DroolsRuleService(KieSession kieSession) {
        this.kieSession = kieSession;
        log.info("DroolsRuleService initialisé");
    }
    
    /**
     * Évalue une requête d'intervention et détermine la priorité et le type
     * basé sur les règles Drools
     * 
     * @param request Requête d'intervention
     * @return Décision de maintenance avec priorité et type déterminés
     */
    public MaintenanceDecision evaluateIntervention(InterventionRequest request) {
        log.debug("Évaluation de l'intervention pour l'actif: {}", request.getAssetId());
        
        // Créer un objet de décision pour Drools
        MaintenanceDecision decision = MaintenanceDecision.builder()
                .assetId(request.getAssetId())
                .sensorId(request.getSensorId())
                .priority(request.getPriority()) // Priorité initiale (peut être modifiée par les règles)
                .interventionType(request.getInterventionType())
                .anomalyContext(request.getAnomalyContext())
                .rulContext(request.getRulContext())
                .build();
        
        // Insérer le fait dans la session Drools
        kieSession.insert(decision);
        
        // Exécuter les règles
        int rulesFired = kieSession.fireAllRules();
        
        log.info("Règles Drools exécutées: {} règles déclenchées pour l'actif {}", 
                rulesFired, request.getAssetId());
        
        // Retirer le fait de la session
        kieSession.delete(kieSession.getFactHandle(decision));
        
        return decision;
    }
    
    /**
     * Classe interne pour représenter une décision de maintenance
     * utilisée par Drools
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MaintenanceDecision {
        private String assetId;
        private String sensorId;
        private PriorityLevel priority;
        private String interventionType;
        private String reason;
        
        // Contexte d'anomalie
        private InterventionRequest.AnomalyContext anomalyContext;
        
        // Contexte RUL
        private InterventionRequest.RulContext rulContext;
    }
}

