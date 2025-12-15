package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.model.Alert;
import com.predictivemaintenance.dashboard.repository.AlertRepository;
import com.predictivemaintenance.dashboard.websocket.DashboardWebSocketHandler;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Service de gestion des alertes
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class AlertService {
    
    private final AlertRepository alertRepository;
    private final ApplicationEventPublisher eventPublisher;
    
    /**
     * Crée une nouvelle alerte
     */
    @Transactional
    public Alert createAlert(Alert alert) {
        log.info("Création d'une alerte: {} - {}", alert.getType(), alert.getTitle());
        Alert saved = alertRepository.save(alert);
        
        // Publier un événement pour notifier via WebSocket
        eventPublisher.publishEvent(new AlertCreatedEvent(saved));
        
        return saved;
    }
    
    /**
     * Événement pour notifier la création d'une alerte
     */
    public static class AlertCreatedEvent {
        private final Alert alert;
        
        public AlertCreatedEvent(Alert alert) {
            this.alert = alert;
        }
        
        public Alert getAlert() {
            return alert;
        }
    }
    
    /**
     * Récupère toutes les alertes actives
     */
    public List<Alert> getActiveAlerts() {
        return alertRepository.findByStatusOrderByCreatedAtDesc(Alert.AlertStatus.ACTIVE);
    }
    
    /**
     * Récupère les alertes par sévérité
     */
    public List<Alert> getAlertsBySeverity(Alert.Severity severity) {
        return alertRepository.findBySeverityOrderByCreatedAtDesc(severity);
    }
    
    /**
     * Récupère les alertes critiques
     */
    public List<Alert> getCriticalAlerts() {
        return alertRepository.findBySeverityOrderByCreatedAtDesc(Alert.Severity.CRITICAL);
    }
    
    /**
     * Récupère une alerte par ID
     */
    public Optional<Alert> getAlertById(Long id) {
        return alertRepository.findById(id);
    }
    
    /**
     * Acquitte une alerte
     */
    @Transactional
    public Alert acknowledgeAlert(Long id, String acknowledgedBy) {
        Alert alert = alertRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Alerte non trouvée: " + id));
        
        alert.setStatus(Alert.AlertStatus.ACKNOWLEDGED);
        alert.setAcknowledgedAt(LocalDateTime.now());
        alert.setAcknowledgedBy(acknowledgedBy);
        
        log.info("Alerte {} acquittée par {}", id, acknowledgedBy);
        return alertRepository.save(alert);
    }
    
    /**
     * Résout une alerte
     */
    @Transactional
    public Alert resolveAlert(Long id) {
        Alert alert = alertRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Alerte non trouvée: " + id));
        
        alert.setStatus(Alert.AlertStatus.RESOLVED);
        alert.setUpdatedAt(LocalDateTime.now());
        
        log.info("Alerte {} résolue", id);
        return alertRepository.save(alert);
    }
    
    /**
     * Ignore une alerte
     */
    @Transactional
    public Alert dismissAlert(Long id) {
        Alert alert = alertRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Alerte non trouvée: " + id));
        
        alert.setStatus(Alert.AlertStatus.DISMISSED);
        alert.setUpdatedAt(LocalDateTime.now());
        
        log.info("Alerte {} ignorée", id);
        return alertRepository.save(alert);
    }
    
    /**
     * Compte les alertes actives
     */
    public long countActiveAlerts() {
        try {
            return alertRepository.countByStatus(Alert.AlertStatus.ACTIVE);
        } catch (Exception e) {
            log.warn("Erreur lors du comptage des alertes actives: {}", e.getMessage());
            return 0L;
        }
    }
    
    /**
     * Compte les alertes critiques
     */
    public long countCriticalAlerts() {
        try {
            return alertRepository.countBySeverity(Alert.Severity.CRITICAL);
        } catch (Exception e) {
            log.warn("Erreur lors du comptage des alertes critiques: {}", e.getMessage());
            return 0L;
        }
    }
    
    /**
     * Récupère toutes les alertes
     */
    public List<Alert> getAllAlerts() {
        return alertRepository.findAll();
    }
}

