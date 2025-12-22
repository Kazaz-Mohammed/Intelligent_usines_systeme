package com.predictivemaintenance.ingestion.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;
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
    
    @JsonProperty("asset_id")
    private String assetId;
    
    @JsonProperty("sensor_id")
    private String sensorId;
    
    private Double value;
    private String unit;
    private Integer quality; // QoS: 0=bad, 1=uncertain, 2=good
    private Map<String, Object> metadata;
    
    // Source information
    @JsonProperty("source_type")
    private String sourceType; // OPC_UA, MODBUS, MQTT
    
    @JsonProperty("source_endpoint")
    private String sourceEndpoint;
}

