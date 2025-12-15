package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.model.Metric;
import com.predictivemaintenance.dashboard.repository.MetricRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Service de gestion des métriques
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class MetricService {
    
    private final MetricRepository metricRepository;
    
    /**
     * Sauvegarde une métrique
     */
    @Transactional
    public Metric saveMetric(Metric metric) {
        return metricRepository.save(metric);
    }
    
    /**
     * Sauvegarde plusieurs métriques
     */
    @Transactional
    public List<Metric> saveMetrics(List<Metric> metrics) {
        return metricRepository.saveAll(metrics);
    }
    
    /**
     * Récupère les métriques récentes
     */
    public List<Metric> getRecentMetrics(int limit) {
        try {
            return metricRepository.findTopNByOrderByTimestampDesc(limit);
        } catch (Exception e) {
            log.warn("Erreur lors de la récupération des métriques récentes: {}", e.getMessage());
            return List.of();
        }
    }
    
    /**
     * Récupère les métriques par nom
     */
    public List<Metric> getMetricsByName(String metricName, LocalDateTime start, LocalDateTime end) {
        return metricRepository.findByMetricNameAndTimestampBetweenOrderByTimestampAsc(
                metricName, start, end
        );
    }
    
    /**
     * Récupère les métriques par service
     */
    public List<Metric> getMetricsByService(String serviceName, LocalDateTime start, LocalDateTime end) {
        return metricRepository.findByServiceNameAndTimestampBetweenOrderByTimestampAsc(
                serviceName, start, end
        );
    }
    
    /**
     * Récupère la dernière valeur d'une métrique
     */
    public Metric getLatestMetric(String metricName) {
        return metricRepository.findTopByMetricNameOrderByTimestampDesc(metricName)
                .orElse(null);
    }
}

