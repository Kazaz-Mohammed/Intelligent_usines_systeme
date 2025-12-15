package com.predictivemaintenance.dashboard.websocket;

import com.predictivemaintenance.dashboard.model.DashboardOverview;
import com.predictivemaintenance.dashboard.service.DashboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Controller;

import java.util.Map;

/**
 * Handler WebSocket pour les mises à jour du dashboard en temps réel
 */
@Controller
@Slf4j
@RequiredArgsConstructor
public class DashboardWebSocketHandler {
    
    private final SimpMessagingTemplate messagingTemplate;
    private final DashboardService dashboardService;
    
    /**
     * Gère les messages entrants depuis les clients
     * Les clients peuvent envoyer des messages à /app/dashboard/subscribe
     */
    @MessageMapping("/dashboard/subscribe")
    @SendTo("/topic/dashboard/updates")
    public DashboardUpdateMessage handleSubscription(SubscriptionMessage message) {
        log.debug("Client {} s'est abonné au dashboard", message.getClientId());
        
        // Envoyer immédiatement la vue d'ensemble actuelle
        DashboardOverview overview = dashboardService.getOverview();
        
        return DashboardUpdateMessage.builder()
                .type("overview")
                .data(overview)
                .timestamp(java.time.LocalDateTime.now())
                .build();
    }
    
    /**
     * Envoie périodiquement les mises à jour du dashboard
     * Toutes les 5 secondes par défaut
     */
    @Scheduled(fixedRateString = "${dashboard.monitoring.update-interval-seconds:5}000")
    public void sendDashboardUpdates() {
        try {
            DashboardOverview overview = dashboardService.getOverview();
            
            DashboardUpdateMessage message = DashboardUpdateMessage.builder()
                    .type("overview")
                    .data(overview)
                    .timestamp(java.time.LocalDateTime.now())
                    .build();
            
            // Envoyer à tous les clients abonnés
            messagingTemplate.convertAndSend("/topic/dashboard/updates", message);
            
            log.debug("Mise à jour du dashboard envoyée via WebSocket");
            
        } catch (Exception e) {
            log.error("Erreur lors de l'envoi de la mise à jour du dashboard: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Envoie les métriques en temps réel
     */
    @Scheduled(fixedRateString = "${dashboard.monitoring.update-interval-seconds:5}000")
    public void sendMetricsUpdates() {
        try {
            Map<String, Object> metrics = dashboardService.getRealtimeMetrics();
            
            DashboardUpdateMessage message = DashboardUpdateMessage.builder()
                    .type("metrics")
                    .data(metrics)
                    .timestamp(java.time.LocalDateTime.now())
                    .build();
            
            // Envoyer à tous les clients abonnés
            messagingTemplate.convertAndSend("/topic/dashboard/metrics", message);
            
        } catch (Exception e) {
            log.error("Erreur lors de l'envoi des métriques: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Envoie les alertes en temps réel
     */
    public void sendAlertUpdate(com.predictivemaintenance.dashboard.model.Alert alert) {
        try {
            DashboardUpdateMessage message = DashboardUpdateMessage.builder()
                    .type("alert")
                    .data(alert)
                    .timestamp(java.time.LocalDateTime.now())
                    .build();
            
            // Envoyer à tous les clients abonnés
            messagingTemplate.convertAndSend("/topic/dashboard/alerts", message);
            
            log.info("Alerte envoyée via WebSocket: {}", alert.getTitle());
            
        } catch (Exception e) {
            log.error("Erreur lors de l'envoi de l'alerte: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Message de souscription
     */
    @lombok.Data
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class SubscriptionMessage {
        private String clientId;
        private String subscriptionType; // "overview", "metrics", "alerts", "all"
    }
    
    /**
     * Message de mise à jour du dashboard
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class DashboardUpdateMessage {
        private String type; // "overview", "metrics", "alert", "service-status"
        private Object data;
        private java.time.LocalDateTime timestamp;
    }
}

