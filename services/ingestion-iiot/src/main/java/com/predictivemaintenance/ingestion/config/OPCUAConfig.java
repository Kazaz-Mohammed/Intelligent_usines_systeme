package com.predictivemaintenance.ingestion.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * Configuration pour OPC UA
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "opcua")
public class OPCUAConfig {

    private boolean enabled = true;
    private String endpointUrl;
    private int connectionTimeout = 5000;
    private int requestTimeout = 10000;
    private long subscriptionInterval = 1000;

    private List<NodeConfig> nodes;

    @Data
    public static class NodeConfig {
        private String nodeId;
        private String assetId;
        private String sensorId;
        private String unit;
    }
}

