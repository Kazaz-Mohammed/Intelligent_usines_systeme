package com.predictivemaintenance.orchestrateur.config;

import org.kie.api.KieServices;
import org.kie.api.builder.KieBuilder;
import org.kie.api.builder.KieFileSystem;
import org.kie.api.builder.KieModule;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration Drools pour le moteur de règles
 */
@Configuration
public class DroolsConfig {
    
    private static final String RULES_PATH = "rules/maintenance-rules.drl";
    
    @Bean
    public KieServices kieServices() {
        return KieServices.Factory.get();
    }
    
    @Bean
    public KieFileSystem kieFileSystem(KieServices kieServices) {
        KieFileSystem kieFileSystem = kieServices.newKieFileSystem();
        // Charger les règles depuis le classpath
        String rules = loadRulesFromClasspath();
        kieFileSystem.write("src/main/resources/" + RULES_PATH, rules);
        return kieFileSystem;
    }
    
    @Bean
    public KieContainer kieContainer(KieServices kieServices, KieFileSystem kieFileSystem) {
        KieBuilder kieBuilder = kieServices.newKieBuilder(kieFileSystem);
        kieBuilder.buildAll();
        KieModule kieModule = kieBuilder.getKieModule();
        return kieServices.newKieContainer(kieModule.getReleaseId());
    }
    
    @Bean
    public KieSession kieSession(KieContainer kieContainer) {
        return kieContainer.newKieSession();
    }
    
    /**
     * Charge les règles depuis le classpath
     * Pour l'instant, retourne un template de règles
     */
    private String loadRulesFromClasspath() {
        // Les règles seront dans src/main/resources/rules/maintenance-rules.drl
        // Pour l'instant, on retourne un template
        return """
            package com.predictivemaintenance.orchestrateur.rules
            
            import com.predictivemaintenance.orchestrateur.model.*
            import java.time.LocalDateTime
            
            // Règles de maintenance seront ajoutées ici
            """;
    }
}

