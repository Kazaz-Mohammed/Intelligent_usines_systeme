package com.predictivemaintenance.dashboard.controller;

import com.predictivemaintenance.dashboard.service.DashboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Contrôleur pour Server-Sent Events (SSE) - Alternative à WebSocket
 */
@RestController
@RequestMapping("/sse")
@RequiredArgsConstructor
@Slf4j
public class SSEController {
    
    private final DashboardService dashboardService;
    private final ScheduledExecutorService executor = Executors.newScheduledThreadPool(2);
    
    /**
     * Endpoint SSE pour les métriques en temps réel
     * 
     * GET /sse/metrics
     */
    @GetMapping(value = "/metrics", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamMetrics() {
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);
        
        executor.scheduleAtFixedRate(() -> {
            try {
                Map<String, Object> metrics = dashboardService.getRealtimeMetrics();
                emitter.send(SseEmitter.event()
                        .name("metrics")
                        .data(metrics));
            } catch (IOException e) {
                log.error("Erreur lors de l'envoi des métriques SSE: {}", e.getMessage());
                emitter.completeWithError(e);
            }
        }, 0, 5, TimeUnit.SECONDS);
        
        emitter.onCompletion(() -> log.debug("SSE connection completed"));
        emitter.onTimeout(() -> log.debug("SSE connection timeout"));
        emitter.onError((ex) -> log.error("SSE connection error: {}", ex.getMessage()));
        
        return emitter;
    }
    
    /**
     * Endpoint SSE pour la vue d'ensemble du dashboard
     * 
     * GET /sse/dashboard
     */
    @GetMapping(value = "/dashboard", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamDashboard() {
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);
        
        executor.scheduleAtFixedRate(() -> {
            try {
                com.predictivemaintenance.dashboard.model.DashboardOverview overview = 
                        dashboardService.getOverview();
                emitter.send(SseEmitter.event()
                        .name("overview")
                        .data(overview));
            } catch (IOException e) {
                log.error("Erreur lors de l'envoi du dashboard SSE: {}", e.getMessage());
                emitter.completeWithError(e);
            }
        }, 0, 5, TimeUnit.SECONDS);
        
        emitter.onCompletion(() -> log.debug("SSE dashboard connection completed"));
        emitter.onTimeout(() -> log.debug("SSE dashboard connection timeout"));
        emitter.onError((ex) -> log.error("SSE dashboard connection error: {}", ex.getMessage()));
        
        return emitter;
    }
    
    /**
     * Endpoint SSE pour les alertes en temps réel
     * 
     * GET /sse/alerts
     */
    @GetMapping(value = "/alerts", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamAlerts() {
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);
        
        // Pour les alertes, on peut utiliser un listener Kafka ou un service dédié
        // Pour l'instant, on envoie périodiquement les alertes actives
        executor.scheduleAtFixedRate(() -> {
            try {
                // TODO: Récupérer les alertes depuis AlertService
                emitter.send(SseEmitter.event()
                        .name("alerts")
                        .data(Map.of("message", "Alerts endpoint - to be implemented")));
            } catch (IOException e) {
                log.error("Erreur lors de l'envoi des alertes SSE: {}", e.getMessage());
                emitter.completeWithError(e);
            }
        }, 0, 10, TimeUnit.SECONDS);
        
        emitter.onCompletion(() -> log.debug("SSE alerts connection completed"));
        emitter.onTimeout(() -> log.debug("SSE alerts connection timeout"));
        emitter.onError((ex) -> log.error("SSE alerts connection error: {}", ex.getMessage()));
        
        return emitter;
    }
}

