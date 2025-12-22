package com.predictivemaintenance.ingestion.config;

import com.influxdb.client.InfluxDBClient;
import com.influxdb.client.InfluxDBClientFactory;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
@ConditionalOnProperty(name = "influxdb.enabled", havingValue = "true", matchIfMissing = false)
public class InfluxDBConfig {

    @Value("${influxdb.url}")
    private String url;

    @Value("${influxdb.token}")
    private String token;

    @Value("${influxdb.org}")
    private String organization;

    @Value("${influxdb.bucket}")
    private String bucket;

    @Bean
    public InfluxDBClient influxDBClient() {
        try {
            log.info("Initializing InfluxDB client - URL: {}, Org: {}, Bucket: {}", url, organization, bucket);
            InfluxDBClient client = InfluxDBClientFactory.create(url, token.toCharArray(), organization, bucket);
            
            // Test connection
            if (client.ping()) {
                log.info("Successfully connected to InfluxDB");
            } else {
                log.warn("InfluxDB ping failed - connection may not be available");
            }
            
            return client;
        } catch (Exception e) {
            log.error("Failed to create InfluxDB client", e);
            throw new RuntimeException("Failed to initialize InfluxDB client", e);
        }
    }
}

