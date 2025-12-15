package com.predictivemaintenance.orchestrateur.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Contrôleur pour les endpoints de santé et d'information
 */
@RestController
@RequestMapping("/api/v1")
public class HealthController {
    
    @GetMapping("/")
    public ResponseEntity<Map<String, Object>> root() {
        Map<String, Object> response = new HashMap<>();
        response.put("service", "orchestrateur-maintenance-service");
        response.put("version", "0.8.0");
        response.put("status", "running");
        response.put("timestamp", LocalDateTime.now());
        return ResponseEntity.ok(response);
    }
    
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "healthy");
        response.put("service", "orchestrateur-maintenance-service");
        response.put("timestamp", LocalDateTime.now());
        return ResponseEntity.ok(response);
    }
}

