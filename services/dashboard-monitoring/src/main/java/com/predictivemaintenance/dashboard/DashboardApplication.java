package com.predictivemaintenance.dashboard;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Application principale pour le service Dashboard & Monitoring
 */
@SpringBootApplication
@EnableScheduling
public class DashboardApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(DashboardApplication.class, args);
    }
}

