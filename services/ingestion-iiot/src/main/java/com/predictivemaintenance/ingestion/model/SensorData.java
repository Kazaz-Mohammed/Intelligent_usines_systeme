package com.predictivemaintenance.ingestion.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SensorData {
    
    @JsonFormat(shape = JsonFormat.Shape.STRING, pattern = "yyyy-MM-dd'T'HH:mm:ss.SSSZ", timezone = "UTC")
    private Instant timestamp;
    
    private String assetId;
    private String sensorId;
    private Double value;
    private String unit;
    private Integer quality; // QoS: 0=bad, 1=uncertain, 2=good
    private Map<String, Object> metadata;
    
    // Source information
    private String sourceType; // OPC_UA, MODBUS, MQTT
    private String sourceEndpoint;
}

