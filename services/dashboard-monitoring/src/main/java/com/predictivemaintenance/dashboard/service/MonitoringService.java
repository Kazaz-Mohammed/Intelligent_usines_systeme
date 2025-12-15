package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.config.DashboardConfig;
import com.predictivemaintenance.dashboard.model.DashboardOverview;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service de monitoring des microservices
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class MonitoringService {
    
    private final DashboardConfig config;
    private final WebClient.Builder webClientBuilder;
    
    // Cache des métriques des services
    private final Map<String, DashboardOverview.ServiceMetrics> serviceMetricsCache = 
            new ConcurrentHashMap<>();
    
    /**
     * Récupère les métriques de tous les services
     */
    public Map<String, DashboardOverview.ServiceMetrics> getAllServiceMetrics() {
        return new HashMap<>(serviceMetricsCache);
    }
    
    /**
     * Récupère les métriques d'un service spécifique
     */
    public DashboardOverview.ServiceMetrics getServiceMetrics(String serviceName) {
        return serviceMetricsCache.getOrDefault(serviceName, 
                createUnknownMetrics(serviceName));
    }
    
    /**
     * Vérifie le health d'un service
     */
    @Scheduled(fixedRateString = "${dashboard.monitoring.update-interval-seconds:5}000")
    public void checkServicesHealth() {
        log.debug("Vérification de la santé des services");
        
        Map<String, String> services = config.getServices();
        
        for (Map.Entry<String, String> entry : services.entrySet()) {
            String serviceName = entry.getKey();
            String serviceUrl = entry.getValue();
            
            checkServiceHealth(serviceName, serviceUrl);
        }
    }
    
    /**
     * Vérifie le health d'un service spécifique
     */
    private void checkServiceHealth(String serviceName, String serviceUrl) {
        WebClient webClient = webClientBuilder
                .baseUrl(serviceUrl)
                .build();
        
        // Liste des endpoints à essayer (dans l'ordre de priorité)
        String[] healthEndpoints = getHealthEndpoints(serviceName);
        
        Exception lastException = null;
        
        for (String healthEndpoint : healthEndpoints) {
            try {
                long startTime = System.currentTimeMillis();
                
                // Appel au health endpoint
                String healthResponse = webClient.get()
                        .uri(healthEndpoint)
                        .retrieve()
                        .bodyToMono(String.class)
                        .timeout(Duration.ofSeconds(5))
                        .block();
                
                long responseTime = System.currentTimeMillis() - startTime;
                
                // Mettre à jour les métriques
                DashboardOverview.ServiceMetrics metrics = DashboardOverview.ServiceMetrics.builder()
                        .serviceName(serviceName)
                        .status(DashboardOverview.ServiceMetrics.ServiceStatus.UP)
                        .lastHealthCheck(LocalDateTime.now())
                        .averageResponseTime((double) responseTime)
                        .uptimeSeconds(calculateUptime(serviceName))
                        .build();
                
                serviceMetricsCache.put(serviceName, metrics);
                
                log.debug("Service {} est UP (response time: {}ms, endpoint: {})", 
                        serviceName, responseTime, healthEndpoint);
                return; // Succès, sortir de la boucle
                
            } catch (Exception e) {
                lastException = e;
                // Continuer avec le prochain endpoint
                log.debug("Service {} - endpoint {} a échoué: {}", 
                        serviceName, healthEndpoint, e.getMessage());
            }
        }
        
        // Tous les endpoints ont échoué
        log.warn("Service {} est DOWN: {}", serviceName, 
                lastException != null ? lastException.getMessage() : "Tous les endpoints ont échoué");
        
        // Mettre à jour avec statut DOWN
        DashboardOverview.ServiceMetrics metrics = DashboardOverview.ServiceMetrics.builder()
                .serviceName(serviceName)
                .status(DashboardOverview.ServiceMetrics.ServiceStatus.DOWN)
                .lastHealthCheck(LocalDateTime.now())
                .errorCount(getErrorCount(serviceName) + 1)
                .uptimeSeconds(0L)
                .build();
        
        serviceMetricsCache.put(serviceName, metrics);
    }
    
    /**
     * Retourne la liste des endpoints de health à essayer pour un service
     */
    private String[] getHealthEndpoints(String serviceName) {
        // Services Spring Boot avec endpoints personnalisés
        switch (serviceName) {
            case "ingestion-iiot":
                // ingestion-iiot a un endpoint personnalisé /api/v1/ingestion/health
                return new String[]{"/api/v1/ingestion/health", "/actuator/health", "/health"};
            case "orchestrateur-maintenance":
                // orchestrateur-maintenance a un endpoint personnalisé /api/v1/health
                return new String[]{"/api/v1/health", "/actuator/health", "/health"};
            case "dashboard-monitoring":
                return new String[]{"/actuator/health", "/health"};
            case "extraction-features":
                return new String[]{"/api/v1/features/health", "/health", "/actuator/health"};
            case "detection-anomalies":
                return new String[]{"/api/v1/anomalies/health", "/health", "/actuator/health"};
            case "prediction-rul":
                return new String[]{"/api/v1/rul/health", "/health", "/actuator/health"};
            case "preprocessing":
            case "pre-traitement":
                return new String[]{"/api/v1/preprocessing/health", "/health", "/actuator/health"};
            default:
                // Essayer plusieurs endpoints par défaut
                return new String[]{"/actuator/health", "/health", "/api/health"};
        }
    }
    
    /**
     * Calcule l'uptime d'un service (simplifié)
     */
    private Long calculateUptime(String serviceName) {
        DashboardOverview.ServiceMetrics existing = serviceMetricsCache.get(serviceName);
        if (existing != null && existing.getUptimeSeconds() != null) {
            // Si le service est UP, incrémenter l'uptime
            if (existing.getStatus() == DashboardOverview.ServiceMetrics.ServiceStatus.UP) {
                return existing.getUptimeSeconds() + 
                        Long.parseLong(config.getUpdateIntervalSeconds());
            }
        }
        return 0L;
    }
    
    /**
     * Récupère le nombre d'erreurs d'un service
     */
    private Long getErrorCount(String serviceName) {
        DashboardOverview.ServiceMetrics existing = serviceMetricsCache.get(serviceName);
        return existing != null && existing.getErrorCount() != null ? 
                existing.getErrorCount() : 0L;
    }
    
    /**
     * Crée des métriques inconnues pour un service
     */
    private DashboardOverview.ServiceMetrics createUnknownMetrics(String serviceName) {
        return DashboardOverview.ServiceMetrics.builder()
                .serviceName(serviceName)
                .status(DashboardOverview.ServiceMetrics.ServiceStatus.UNKNOWN)
                .lastHealthCheck(LocalDateTime.now())
                .uptimeSeconds(0L)
                .totalRequests(0L)
                .errorCount(0L)
                .averageResponseTime(0.0)
                .build();
    }
}

