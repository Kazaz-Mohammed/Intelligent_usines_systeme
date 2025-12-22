package com.predictivemaintenance.ingestion.service;

import com.influxdb.client.InfluxDBClient;
import com.influxdb.client.WriteApiBlocking;
import com.influxdb.client.domain.WritePrecision;
import com.influxdb.client.write.Point;
import com.predictivemaintenance.ingestion.model.SensorData;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Service pour stocker les données capteurs dans InfluxDB
 */
@Slf4j
@Service
@ConditionalOnProperty(name = "influxdb.enabled", havingValue = "true", matchIfMissing = false)
public class InfluxDBService {

    private final InfluxDBClient influxDBClient;
    private final WriteApiBlocking writeApi;

    @Value("${influxdb.measurement:sensor_data}")
    private String measurement;

    @Autowired
    public InfluxDBService(InfluxDBClient influxDBClient) {
        this.influxDBClient = influxDBClient;
        this.writeApi = influxDBClient.getWriteApiBlocking();
        log.info("InfluxDB Service initialized");
    }

    /**
     * Écrit une donnée capteur dans InfluxDB
     */
    public void writeSensorData(SensorData sensorData) {
        try {
            Point point = createPoint(sensorData);
            writeApi.writePoint(point);
            log.debug("Written sensor data to InfluxDB: asset={}, sensor={}", 
                sensorData.getAssetId(), sensorData.getSensorId());
        } catch (Exception e) {
            log.error("Error writing sensor data to InfluxDB", e);
            throw new RuntimeException("Failed to write to InfluxDB", e);
        }
    }

    /**
     * Écrit un batch de données capteurs dans InfluxDB
     */
    public void writeBatch(List<SensorData> sensorDataList) {
        try {
            List<Point> points = sensorDataList.stream()
                .map(this::createPoint)
                .collect(Collectors.toList());
            
            writeApi.writePoints(points);
            log.info("Written batch of {} records to InfluxDB", sensorDataList.size());
        } catch (Exception e) {
            log.error("Error writing batch to InfluxDB", e);
            throw new RuntimeException("Failed to write batch to InfluxDB", e);
        }
    }

    /**
     * Crée un Point InfluxDB à partir des données capteur
     */
    private Point createPoint(SensorData sensorData) {
        Point point = Point.measurement(measurement)
            .time(sensorData.getTimestamp(), WritePrecision.MS)
            .addTag("asset_id", sensorData.getAssetId())
            .addTag("sensor_id", sensorData.getSensorId())
            .addTag("unit", sensorData.getUnit() != null ? sensorData.getUnit() : "unknown")
            .addField("value", sensorData.getValue());

        // Add optional fields
        if (sensorData.getQuality() != null) {
            point.addField("quality", sensorData.getQuality());
        }

        if (sensorData.getSourceType() != null) {
            point.addTag("source_type", sensorData.getSourceType());
        }

        return point;
    }

    /**
     * Vérifie la connexion à InfluxDB
     */
    public boolean isHealthy() {
        try {
            return influxDBClient.ping();
        } catch (Exception e) {
            log.warn("InfluxDB health check failed", e);
            return false;
        }
    }
}

