package com.predictivemaintenance.ingestion.controller;

import com.predictivemaintenance.ingestion.model.SensorData;
import com.predictivemaintenance.ingestion.service.IngestionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * REST API pour contrôler et monitorer le service d'ingestion
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/ingestion")
@RequiredArgsConstructor
public class IngestionController {

    private final IngestionService ingestionService;

    /**
     * Endpoint pour recevoir des données manuelles
     */
    @PostMapping("/data")
    public ResponseEntity<Map<String, String>> ingestData(@RequestBody SensorData sensorData) {
        try {
            ingestionService.processSensorData(sensorData);
            return ResponseEntity.ok(Map.of("status", "success", "message", "Data ingested successfully"));
        } catch (Exception e) {
            log.error("Error ingesting data", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("status", "error", "message", e.getMessage()));
        }
    }

    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "UP", "service", "ingestion-iiot"));
    }

    /**
     * Status endpoint
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> status() {
        return ResponseEntity.ok(Map.of(
            "service", "ingestion-iiot",
            "status", "running",
            "version", "0.2.0-SNAPSHOT"
        ));
    }
}

