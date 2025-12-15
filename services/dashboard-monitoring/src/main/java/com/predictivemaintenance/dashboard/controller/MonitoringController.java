package com.predictivemaintenance.dashboard.controller;

import com.predictivemaintenance.dashboard.model.DashboardOverview;
import com.predictivemaintenance.dashboard.service.MonitoringService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Contrôleur pour le monitoring
 */
@RestController
@RequestMapping("/api/v1/monitoring")
@RequiredArgsConstructor
public class MonitoringController {
    
    private final MonitoringService monitoringService;
    
    /**
     * Récupère le statut de tous les services
     * 
     * GET /api/v1/monitoring/services
     */
    @GetMapping("/services")
    public ResponseEntity<Map<String, DashboardOverview.ServiceMetrics>> getServicesStatus() {
        Map<String, DashboardOverview.ServiceMetrics> services = 
                monitoringService.getAllServiceMetrics();
        return ResponseEntity.ok(services);
    }
    
    /**
     * Récupère le statut d'un service spécifique
     * 
     * GET /api/v1/monitoring/services/{serviceName}
     */
    @GetMapping("/services/{serviceName}")
    public ResponseEntity<DashboardOverview.ServiceMetrics> getServiceStatus(
            @PathVariable String serviceName
    ) {
        DashboardOverview.ServiceMetrics metrics = 
                monitoringService.getServiceMetrics(serviceName);
        return ResponseEntity.ok(metrics);
    }
    
    /**
     * Health check global
     * 
     * GET /api/v1/monitoring/health
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> getGlobalHealth() {
        Map<String, DashboardOverview.ServiceMetrics> services = 
                monitoringService.getAllServiceMetrics();
        
        long upServices = services.values().stream()
                .filter(m -> m.getStatus() == DashboardOverview.ServiceMetrics.ServiceStatus.UP)
                .count();
        
        long totalServices = services.size();
        
        return ResponseEntity.ok(Map.of(
                "status", upServices == totalServices ? "HEALTHY" : "DEGRADED",
                "upServices", upServices,
                "totalServices", totalServices,
                "services", services
        ));
    }
}

