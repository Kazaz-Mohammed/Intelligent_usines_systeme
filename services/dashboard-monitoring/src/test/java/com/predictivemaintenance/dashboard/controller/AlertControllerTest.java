package com.predictivemaintenance.dashboard.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.dashboard.model.Alert;
import com.predictivemaintenance.dashboard.service.AlertService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Tests pour AlertController
 */
@WebMvcTest(AlertController.class)
class AlertControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private AlertService alertService;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Test
    void testGetAllAlerts() throws Exception {
        List<Alert> alerts = List.of(
                Alert.builder()
                        .id(1L)
                        .title("Test Alert")
                        .severity(Alert.Severity.HIGH)
                        .build()
        );
        
        when(alertService.getAllAlerts()).thenReturn(alerts);
        
        mockMvc.perform(get("/api/v1/alerts"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$[0].title").value("Test Alert"));
    }
    
    @Test
    void testGetActiveAlerts() throws Exception {
        List<Alert> alerts = List.of(
                Alert.builder()
                        .id(1L)
                        .status(Alert.AlertStatus.ACTIVE)
                        .build()
        );
        
        when(alertService.getActiveAlerts()).thenReturn(alerts);
        
        mockMvc.perform(get("/api/v1/alerts/active"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }
    
    @Test
    void testGetAlertById() throws Exception {
        Alert alert = Alert.builder()
                .id(1L)
                .title("Test Alert")
                .build();
        
        when(alertService.getAlertById(1L)).thenReturn(Optional.of(alert));
        
        mockMvc.perform(get("/api/v1/alerts/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.title").value("Test Alert"));
    }
    
    @Test
    void testGetAlertByIdNotFound() throws Exception {
        when(alertService.getAlertById(1L)).thenReturn(Optional.empty());
        
        mockMvc.perform(get("/api/v1/alerts/1"))
                .andExpect(status().isNotFound());
    }
    
    @Test
    void testCreateAlert() throws Exception {
        Alert alert = Alert.builder()
                .type(Alert.AlertType.SERVICE_DOWN)
                .severity(Alert.Severity.CRITICAL)
                .title("Service Down")
                .build();
        
        when(alertService.createAlert(any(Alert.class))).thenReturn(alert);
        
        mockMvc.perform(post("/api/v1/alerts")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(alert)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.title").value("Service Down"));
    }
    
    @Test
    void testAcknowledgeAlert() throws Exception {
        Alert alert = Alert.builder()
                .id(1L)
                .status(Alert.AlertStatus.ACKNOWLEDGED)
                .build();
        
        when(alertService.acknowledgeAlert(1L, "user123")).thenReturn(alert);
        
        mockMvc.perform(put("/api/v1/alerts/1/acknowledge")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"acknowledgedBy\":\"user123\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ACKNOWLEDGED"));
    }
    
    @Test
    void testResolveAlert() throws Exception {
        Alert alert = Alert.builder()
                .id(1L)
                .status(Alert.AlertStatus.RESOLVED)
                .build();
        
        when(alertService.resolveAlert(1L)).thenReturn(alert);
        
        mockMvc.perform(put("/api/v1/alerts/1/resolve"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("RESOLVED"));
    }
}

