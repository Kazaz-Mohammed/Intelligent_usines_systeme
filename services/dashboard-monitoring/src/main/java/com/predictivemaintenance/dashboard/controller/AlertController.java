package com.predictivemaintenance.dashboard.controller;

import com.predictivemaintenance.dashboard.model.Alert;
import com.predictivemaintenance.dashboard.service.AlertService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Contrôleur pour les alertes
 */
@RestController
@RequestMapping("/api/v1/alerts")
@RequiredArgsConstructor
public class AlertController {
    
    private final AlertService alertService;
    
    /**
     * Récupère toutes les alertes
     * 
     * GET /api/v1/alerts
     */
    @GetMapping
    public ResponseEntity<List<Alert>> getAllAlerts() {
        List<Alert> alerts = alertService.getAllAlerts();
        return ResponseEntity.ok(alerts);
    }
    
    /**
     * Récupère les alertes actives
     * 
     * GET /api/v1/alerts/active
     */
    @GetMapping("/active")
    public ResponseEntity<List<Alert>> getActiveAlerts() {
        List<Alert> alerts = alertService.getActiveAlerts();
        return ResponseEntity.ok(alerts);
    }
    
    /**
     * Récupère les alertes critiques
     * 
     * GET /api/v1/alerts/critical
     */
    @GetMapping("/critical")
    public ResponseEntity<List<Alert>> getCriticalAlerts() {
        List<Alert> alerts = alertService.getCriticalAlerts();
        return ResponseEntity.ok(alerts);
    }
    
    /**
     * Récupère une alerte par ID
     * 
     * GET /api/v1/alerts/{id}
     */
    @GetMapping("/{id}")
    public ResponseEntity<Alert> getAlertById(@PathVariable Long id) {
        return alertService.getAlertById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    /**
     * Crée une nouvelle alerte
     * 
     * POST /api/v1/alerts
     */
    @PostMapping
    public ResponseEntity<Alert> createAlert(@RequestBody Alert alert) {
        Alert created = alertService.createAlert(alert);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }
    
    /**
     * Acquitte une alerte
     * 
     * PUT /api/v1/alerts/{id}/acknowledge
     */
    @PutMapping("/{id}/acknowledge")
    public ResponseEntity<Alert> acknowledgeAlert(
            @PathVariable Long id,
            @RequestBody Map<String, String> request
    ) {
        String acknowledgedBy = request.getOrDefault("acknowledgedBy", "system");
        Alert alert = alertService.acknowledgeAlert(id, acknowledgedBy);
        return ResponseEntity.ok(alert);
    }
    
    /**
     * Résout une alerte
     * 
     * PUT /api/v1/alerts/{id}/resolve
     */
    @PutMapping("/{id}/resolve")
    public ResponseEntity<Alert> resolveAlert(@PathVariable Long id) {
        Alert alert = alertService.resolveAlert(id);
        return ResponseEntity.ok(alert);
    }
    
    /**
     * Ignore une alerte
     * 
     * PUT /api/v1/alerts/{id}/dismiss
     */
    @PutMapping("/{id}/dismiss")
    public ResponseEntity<Alert> dismissAlert(@PathVariable Long id) {
        Alert alert = alertService.dismissAlert(id);
        return ResponseEntity.ok(alert);
    }
}

