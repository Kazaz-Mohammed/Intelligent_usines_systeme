package com.predictivemaintenance.dashboard.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;

/**
 * Initialise la base de données si nécessaire
 */
@Component
@Slf4j
@Order(1)
public class DatabaseInitializer implements CommandLineRunner {
    
    @Autowired(required = false)
    private JdbcTemplate jdbcTemplate;
    
    @Override
    public void run(String... args) {
        if (jdbcTemplate != null) {
            try {
                // Vérifier si les tables existent
                jdbcTemplate.execute("SELECT 1 FROM metrics LIMIT 1");
                log.info("Tables de base de données déjà créées");
            } catch (Exception e) {
                log.warn("Les tables n'existent pas encore - Hibernate devrait les créer automatiquement");
            }
        }
    }
}

