package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.model.Alert;
import com.predictivemaintenance.dashboard.repository.AlertRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.context.ApplicationEventPublisher;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Tests pour AlertService
 */
class AlertServiceTest {
    
    @Mock
    private AlertRepository alertRepository;
    
    @Mock
    private ApplicationEventPublisher eventPublisher;
    
    private AlertService alertService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        alertService = new AlertService(alertRepository, eventPublisher);
    }
    
    @Test
    void testCreateAlert() {
        Alert alert = Alert.builder()
                .type(Alert.AlertType.SERVICE_DOWN)
                .severity(Alert.Severity.CRITICAL)
                .title("Service Down")
                .description("Service is down")
                .build();
        
        when(alertRepository.save(any(Alert.class))).thenReturn(alert);
        
        Alert created = alertService.createAlert(alert);
        
        assertNotNull(created);
        assertEquals("Service Down", created.getTitle());
        verify(alertRepository, times(1)).save(alert);
        verify(eventPublisher, times(1)).publishEvent(any(AlertService.AlertCreatedEvent.class));
    }
    
    @Test
    void testGetActiveAlerts() {
        List<Alert> alerts = List.of(
                Alert.builder().status(Alert.AlertStatus.ACTIVE).build()
        );
        
        when(alertRepository.findByStatusOrderByCreatedAtDesc(Alert.AlertStatus.ACTIVE))
                .thenReturn(alerts);
        
        List<Alert> activeAlerts = alertService.getActiveAlerts();
        
        assertEquals(1, activeAlerts.size());
    }
    
    @Test
    void testAcknowledgeAlert() {
        Alert alert = Alert.builder()
                .id(1L)
                .status(Alert.AlertStatus.ACTIVE)
                .build();
        
        when(alertRepository.findById(1L)).thenReturn(Optional.of(alert));
        when(alertRepository.save(any(Alert.class))).thenReturn(alert);
        
        Alert acknowledged = alertService.acknowledgeAlert(1L, "user123");
        
        assertEquals(Alert.AlertStatus.ACKNOWLEDGED, acknowledged.getStatus());
        assertNotNull(acknowledged.getAcknowledgedAt());
        assertEquals("user123", acknowledged.getAcknowledgedBy());
    }
    
    @Test
    void testAcknowledgeAlertNotFound() {
        when(alertRepository.findById(1L)).thenReturn(Optional.empty());
        
        assertThrows(IllegalArgumentException.class, () -> {
            alertService.acknowledgeAlert(1L, "user123");
        });
    }
    
    @Test
    void testResolveAlert() {
        Alert alert = Alert.builder()
                .id(1L)
                .status(Alert.AlertStatus.ACTIVE)
                .build();
        
        when(alertRepository.findById(1L)).thenReturn(Optional.of(alert));
        when(alertRepository.save(any(Alert.class))).thenReturn(alert);
        
        Alert resolved = alertService.resolveAlert(1L);
        
        assertEquals(Alert.AlertStatus.RESOLVED, resolved.getStatus());
    }
    
    @Test
    void testCountActiveAlerts() {
        when(alertRepository.countByStatus(Alert.AlertStatus.ACTIVE)).thenReturn(5L);
        
        long count = alertService.countActiveAlerts();
        
        assertEquals(5L, count);
    }
    
    @Test
    void testCountCriticalAlerts() {
        when(alertRepository.countBySeverity(Alert.Severity.CRITICAL)).thenReturn(2L);
        
        long count = alertService.countCriticalAlerts();
        
        assertEquals(2L, count);
    }
}

