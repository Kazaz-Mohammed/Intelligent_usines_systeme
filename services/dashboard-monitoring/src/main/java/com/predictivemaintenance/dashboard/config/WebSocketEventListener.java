package com.predictivemaintenance.dashboard.config;

import com.predictivemaintenance.dashboard.service.AlertService;
import com.predictivemaintenance.dashboard.websocket.DashboardWebSocketHandler;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

/**
 * Écouteur d'événements pour notifier les clients WebSocket
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class WebSocketEventListener {
    
    private final DashboardWebSocketHandler webSocketHandler;
    
    /**
     * Écoute les événements de création d'alertes
     */
    @EventListener
    public void handleAlertCreated(AlertService.AlertCreatedEvent event) {
        log.debug("Alerte créée, notification via WebSocket");
        webSocketHandler.sendAlertUpdate(event.getAlert());
    }
}

