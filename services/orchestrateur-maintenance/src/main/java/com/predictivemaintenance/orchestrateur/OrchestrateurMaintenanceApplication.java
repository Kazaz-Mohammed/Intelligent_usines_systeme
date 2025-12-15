package com.predictivemaintenance.orchestrateur;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.kafka.annotation.EnableKafka;

/**
 * Application principale du service OrchestrateurMaintenance
 * 
 * Responsable de :
 * - Planification optimisée des interventions de maintenance
 * - Moteur de règles Drools pour décisions automatisées
 * - Optimisation avec OR-Tools
 * - Génération d'ordres de travail
 * - Intégration CMMS/ERP
 */
@SpringBootApplication
@EnableKafka
public class OrchestrateurMaintenanceApplication {

    public static void main(String[] args) {
        SpringApplication.run(OrchestrateurMaintenanceApplication.class, args);
    }
}

