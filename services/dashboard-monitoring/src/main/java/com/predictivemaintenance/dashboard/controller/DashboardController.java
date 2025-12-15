package com.predictivemaintenance.dashboard.controller;

import com.predictivemaintenance.dashboard.model.DashboardOverview;
import com.predictivemaintenance.dashboard.service.DashboardService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Contrôleur pour le dashboard
 */
@RestController
@RequestMapping("/api/v1/dashboard")
@RequiredArgsConstructor
public class DashboardController {
    
    private final DashboardService dashboardService;
    
    /**
     * Récupère la vue d'ensemble du dashboard
     * 
     * GET /api/v1/dashboard/overview
     */
    @GetMapping("/overview")
    public ResponseEntity<DashboardOverview> getOverview() {
        DashboardOverview overview = dashboardService.getOverview();
        return ResponseEntity.ok(overview);
    }
    
    /**
     * Récupère les métriques en temps réel
     * 
     * GET /api/v1/dashboard/metrics
     */
    @GetMapping("/metrics")
    public ResponseEntity<Map<String, Object>> getRealtimeMetrics() {
        Map<String, Object> metrics = dashboardService.getRealtimeMetrics();
        return ResponseEntity.ok(metrics);
    }
    
    /**
     * Récupère les statistiques agrégées
     * 
     * GET /api/v1/dashboard/statistics
     */
    @GetMapping("/statistics")
    public ResponseEntity<Map<String, Object>> getStatistics() {
        // TODO: Implémenter avec vraies statistiques
        return ResponseEntity.ok(Map.of("message", "Statistics endpoint - to be implemented"));
    }
}

