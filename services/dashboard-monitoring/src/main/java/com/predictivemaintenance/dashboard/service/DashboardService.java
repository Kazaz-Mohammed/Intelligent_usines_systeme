package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.model.DashboardOverview;
import com.predictivemaintenance.dashboard.model.Metric;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Service principal pour le dashboard
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class DashboardService {
    
    private final MonitoringService monitoringService;
    private final MetricService metricService;
    private final AlertService alertService;
    private final StatisticsService statisticsService;
    
    /**
     * Récupère la vue d'ensemble du dashboard
     */
    public DashboardOverview getOverview() {
        log.debug("Génération de la vue d'ensemble du dashboard");
        
        // Récupérer les métriques des services
        Map<String, DashboardOverview.ServiceMetrics> serviceMetrics = 
                monitoringService.getAllServiceMetrics();
        
        // Récupérer les statistiques
        Map<String, Object> stats = statisticsService.getMainStatistics();
        
        // Compter les alertes
        long activeAlerts = alertService.countActiveAlerts();
        long criticalAlerts = alertService.countCriticalAlerts();
        
        // Déterminer le statut global du système
        DashboardOverview.SystemStatus systemStatus = determineSystemStatus(
                serviceMetrics, activeAlerts, criticalAlerts
        );
        
        // Construire la vue d'ensemble
        return DashboardOverview.builder()
                .lastUpdate(LocalDateTime.now())
                .systemStatus(systemStatus)
                .totalAssets(getLongValue(stats, "totalAssets", 0L))
                .assetsWithAnomalies(getLongValue(stats, "assetsWithAnomalies", 0L))
                .criticalAssets(getLongValue(stats, "criticalAssets", 0L))
                .activeInterventions(getLongValue(stats, "activeInterventions", 0L))
                .scheduledInterventions(getLongValue(stats, "scheduledInterventions", 0L))
                .serviceMetrics(serviceMetrics)
                .activeAlerts(activeAlerts)
                .criticalAlerts(criticalAlerts)
                .kpis(calculateKPIs(stats))
                .build();
    }
    
    /**
     * Récupère les métriques en temps réel
     */
    public Map<String, Object> getRealtimeMetrics() {
        log.debug("Récupération des métriques temps-réel");
        
        Map<String, Object> metrics = new HashMap<>();
        
        // Métriques des services
        metrics.put("services", monitoringService.getAllServiceMetrics());
        
        // Métriques système
        metrics.put("system", Map.of(
                "cpuUsage", getSystemCpuUsage(),
                "memoryUsage", getSystemMemoryUsage(),
                "diskUsage", getSystemDiskUsage()
        ));
        
        // Métriques de maintenance
        metrics.put("maintenance", statisticsService.getMaintenanceMetrics());
        
        // Dernières métriques collectées
        List<Metric> recentMetrics = metricService.getRecentMetrics(100);
        metrics.put("recentMetrics", recentMetrics);
        
        return metrics;
    }
    
    /**
     * Détermine le statut global du système
     */
    private DashboardOverview.SystemStatus determineSystemStatus(
            Map<String, DashboardOverview.ServiceMetrics> serviceMetrics,
            long activeAlerts,
            long criticalAlerts
    ) {
        // Si des alertes critiques, système critique
        if (criticalAlerts > 0) {
            return DashboardOverview.SystemStatus.CRITICAL;
        }
        
        // Vérifier si des services sont down
        boolean hasDownService = serviceMetrics.values().stream()
                .anyMatch(m -> m.getStatus() == DashboardOverview.ServiceMetrics.ServiceStatus.DOWN);
        
        if (hasDownService) {
            return DashboardOverview.SystemStatus.CRITICAL;
        }
        
        // Vérifier si des services sont dégradés
        boolean hasDegradedService = serviceMetrics.values().stream()
                .anyMatch(m -> m.getStatus() == DashboardOverview.ServiceMetrics.ServiceStatus.DEGRADED);
        
        if (hasDegradedService || activeAlerts > 5) {
            return DashboardOverview.SystemStatus.WARNING;
        }
        
        return DashboardOverview.SystemStatus.HEALTHY;
    }
    
    /**
     * Calcule les KPIs principaux
     */
    private Map<String, Double> calculateKPIs(Map<String, Object> stats) {
        Map<String, Double> kpis = new HashMap<>();
        
        long totalAssets = getLongValue(stats, "totalAssets", 0L);
        long assetsWithAnomalies = getLongValue(stats, "assetsWithAnomalies", 0L);
        long activeInterventions = getLongValue(stats, "activeInterventions", 0L);
        long scheduledInterventions = getLongValue(stats, "scheduledInterventions", 0L);
        
        // Taux d'anomalies
        if (totalAssets > 0) {
            kpis.put("anomalyRate", (double) assetsWithAnomalies / totalAssets * 100);
        } else {
            kpis.put("anomalyRate", 0.0);
        }
        
        // Taux d'interventions actives
        long totalInterventions = activeInterventions + scheduledInterventions;
        if (totalInterventions > 0) {
            kpis.put("activeInterventionRate", (double) activeInterventions / totalInterventions * 100);
        } else {
            kpis.put("activeInterventionRate", 0.0);
        }
        
        // MTBF (Mean Time Between Failures) - exemple
        kpis.put("mtbf", getDoubleValue(stats, "mtbf", 0.0));
        
        // MTTR (Mean Time To Repair) - exemple
        kpis.put("mttr", getDoubleValue(stats, "mttr", 0.0));
        
        return kpis;
    }
    
    /**
     * Récupère l'utilisation CPU du système
     */
    private double getSystemCpuUsage() {
        // TODO: Implémenter avec système de monitoring réel
        return Math.random() * 100;
    }
    
    /**
     * Récupère l'utilisation mémoire du système
     */
    private double getSystemMemoryUsage() {
        // TODO: Implémenter avec système de monitoring réel
        return Math.random() * 100;
    }
    
    /**
     * Récupère l'utilisation disque du système
     */
    private double getSystemDiskUsage() {
        // TODO: Implémenter avec système de monitoring réel
        return Math.random() * 100;
    }
    
    /**
     * Helper pour récupérer une valeur Long depuis un Map
     */
    private Long getLongValue(Map<String, Object> map, String key, Long defaultValue) {
        Object value = map.get(key);
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        return defaultValue;
    }
    
    /**
     * Helper pour récupérer une valeur Double depuis un Map
     */
    private Double getDoubleValue(Map<String, Object> map, String key, Double defaultValue) {
        Object value = map.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return defaultValue;
    }
}

