package com.predictivemaintenance.orchestrateur.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.orchestrateur.config.OrchestrateurConfig;
import com.predictivemaintenance.orchestrateur.kafka.KafkaConsumerService.AnomalyMessage;
import com.predictivemaintenance.orchestrateur.kafka.KafkaConsumerService.RulMessage;
import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Tests pour KafkaConsumerService
 */
class KafkaConsumerServiceTest {
    
    @Mock
    private OrchestrateurConfig config;
    
    @Mock
    private KafkaOrchestrationService orchestrationService;
    
    private ObjectMapper objectMapper;
    private KafkaConsumerService kafkaConsumerService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        objectMapper = new ObjectMapper();
        
        OrchestrateurConfig.Kafka kafkaConfig = new OrchestrateurConfig.Kafka();
        when(config.getKafka()).thenReturn(kafkaConfig);
        
        kafkaConsumerService = new KafkaConsumerService(config, objectMapper, orchestrationService);
    }
    
    @Test
    void testConvertAnomalyToInterventionRequest() throws Exception {
        AnomalyMessage anomalyMessage = new AnomalyMessage();
        anomalyMessage.setAssetId("ASSET001");
        anomalyMessage.setSensorId("SENSOR001");
        anomalyMessage.setIsAnomaly(true);
        anomalyMessage.setCriticality("critical");
        anomalyMessage.setFinalScore(0.95);
        
        InterventionRequest request = kafkaConsumerService.convertAnomalyToInterventionRequest(anomalyMessage);
        
        assertNotNull(request);
        assertEquals("ASSET001", request.getAssetId());
        assertEquals("SENSOR001", request.getSensorId());
        assertNotNull(request.getAnomalyContext());
        assertTrue(request.getAnomalyContext().getIsAnomaly());
        assertEquals("critical", request.getAnomalyContext().getCriticality());
    }
    
    @Test
    void testConvertRulToInterventionRequest() throws Exception {
        RulMessage rulMessage = new RulMessage();
        rulMessage.setAssetId("ASSET002");
        rulMessage.setSensorId("SENSOR002");
        rulMessage.setRulPrediction(30.0);
        rulMessage.setConfidenceIntervalLower(20.0);
        rulMessage.setConfidenceIntervalUpper(40.0);
        
        InterventionRequest request = kafkaConsumerService.convertRulToInterventionRequest(rulMessage);
        
        assertNotNull(request);
        assertEquals("ASSET002", request.getAssetId());
        assertNotNull(request.getRulContext());
        assertEquals(30.0, request.getRulContext().getPredictedRul());
    }
}

