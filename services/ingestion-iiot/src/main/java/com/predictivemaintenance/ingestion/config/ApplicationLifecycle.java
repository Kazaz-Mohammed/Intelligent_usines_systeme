package com.predictivemaintenance.ingestion.config;

import com.predictivemaintenance.ingestion.service.OPCUAService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

/**
 * Gestion du cycle de vie de l'application
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ApplicationLifecycle {

    private final OPCUAService opcuaService;

    @EventListener(ApplicationReadyEvent.class)
    public void onApplicationReady() {
        log.info("Application started, initializing OPC UA connection...");
        try {
            opcuaService.connect();
        } catch (Exception e) {
            log.warn("Failed to connect to OPC UA on startup, will retry", e);
        }
    }

    @EventListener(org.springframework.context.event.ContextClosedEvent.class)
    public void onApplicationShutdown() {
        log.info("Application shutting down, disconnecting OPC UA...");
        opcuaService.disconnect();
    }
}

