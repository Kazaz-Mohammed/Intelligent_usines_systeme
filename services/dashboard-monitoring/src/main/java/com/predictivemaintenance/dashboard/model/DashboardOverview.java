package com.predictivemaintenance.dashboard.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * Vue d'ensemble du dashboard
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DashboardOverview {
    
    /**
     * Timestamp de la dernière mise à jour
     */
    private LocalDateTime lastUpdate;
    
    /**
     * Statut global du système
     */
    private SystemStatus systemStatus;
    
    /**
     * Nombre total d'actifs surveillés
     */
    private Long totalAssets;
    
    /**
     * Nombre d'actifs avec anomalies
     */
    private Long assetsWithAnomalies;
    
    /**
     * Nombre d'actifs critiques
     */
    private Long criticalAssets;
    
    /**
     * Nombre d'interventions en cours
     */
    private Long activeInterventions;
    
    /**
     * Nombre d'interventions planifiées
     */
    private Long scheduledInterventions;
    
    /**
     * Métriques par service
     */
    private Map<String, ServiceMetrics> serviceMetrics;
    
    /**
     * Alertes actives
     */
    private Long activeAlerts;
    
    /**
     * Alertes critiques
     */
    private Long criticalAlerts;
    
    /**
     * KPIs principaux
     */
    private Map<String, Double> kpis;
    
    /**
     * Statut du système
     */
    public enum SystemStatus {
        HEALTHY,
        WARNING,
        CRITICAL,
        UNKNOWN
    }
    
    /**
     * Métriques d'un service
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ServiceMetrics {
        private String serviceName;
        private ServiceStatus status;
        private Long uptimeSeconds;
        private Long totalRequests;
        private Long errorCount;
        private Double averageResponseTime;
        private LocalDateTime lastHealthCheck;
        
        public enum ServiceStatus {
            UP,
            DOWN,
            DEGRADED,
            UNKNOWN
        }
    }
}

