package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.model.SensorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

/**
 * Service pour publier les données sur Kafka
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class KafkaProducerService {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Value("${spring.kafka.producer.topic:sensor-data}")
    private String topicName;

    /**
     * Publie une donnée capteur sur Kafka
     */
    public void publishSensorData(SensorData sensorData) {
        try {
            String key = generateKey(sensorData);
            
            CompletableFuture<SendResult<String, Object>> future = 
                kafkaTemplate.send(topicName, key, sensorData);

            future.whenComplete((result, exception) -> {
                if (exception == null) {
                    log.debug("Message sent successfully: topic={}, key={}, offset={}", 
                        topicName, key, result.getRecordMetadata().offset());
                } else {
                    log.error("Failed to send message: topic={}, key={}", 
                        topicName, key, exception);
                }
            });
        } catch (Exception e) {
            log.error("Error publishing sensor data to Kafka", e);
            throw new RuntimeException("Failed to publish to Kafka", e);
        }
    }

    /**
     * Publie un batch de données
     */
    public void publishBatch(java.util.List<SensorData> sensorDataList) {
        sensorDataList.forEach(this::publishSensorData);
        log.info("Published batch of {} messages to topic {}", sensorDataList.size(), topicName);
    }

    private String generateKey(SensorData sensorData) {
        // Utiliser asset_id + sensor_id comme clé pour garantir l'ordre
        return String.format("%s:%s", sensorData.getAssetId(), sensorData.getSensorId());
    }
}

