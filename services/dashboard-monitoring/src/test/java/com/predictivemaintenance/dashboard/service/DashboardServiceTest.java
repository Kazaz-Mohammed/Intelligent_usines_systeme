package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.model.DashboardOverview;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Tests pour DashboardService
 */
class DashboardServiceTest {
    
    @Mock
    private MonitoringService monitoringService;
    
    @Mock
    private MetricService metricService;
    
    @Mock
    private AlertService alertService;
    
    @Mock
    private StatisticsService statisticsService;
    
    private DashboardService dashboardService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        dashboardService = new DashboardService(
                monitoringService, metricService, alertService, statisticsService
        );
    }
    
    @Test
    void testGetOverview() {
        // Mock des services
        when(monitoringService.getAllServiceMetrics())
                .thenReturn(Map.of());
        when(statisticsService.getMainStatistics())
                .thenReturn(Map.of(
                        "totalAssets", 100L,
                        "assetsWithAnomalies", 15L,
                        "criticalAssets", 5L,
                        "activeInterventions", 8L,
                        "scheduledInterventions", 12L
                ));
        when(alertService.countActiveAlerts()).thenReturn(3L);
        when(alertService.countCriticalAlerts()).thenReturn(1L);
        
        DashboardOverview overview = dashboardService.getOverview();
        
        assertNotNull(overview);
        assertEquals(100L, overview.getTotalAssets());
        assertEquals(15L, overview.getAssetsWithAnomalies());
        assertEquals(8L, overview.getActiveInterventions());
        assertNotNull(overview.getLastUpdate());
        assertNotNull(overview.getSystemStatus());
    }
    
    @Test
    void testGetRealtimeMetrics() {
        // Mock des services
        when(monitoringService.getAllServiceMetrics())
                .thenReturn(Map.of());
        when(statisticsService.getMaintenanceMetrics())
                .thenReturn(Map.of("totalWorkOrders", 50L));
        when(metricService.getRecentMetrics(100))
                .thenReturn(java.util.Collections.emptyList());
        
        Map<String, Object> metrics = dashboardService.getRealtimeMetrics();
        
        assertNotNull(metrics);
        assertTrue(metrics.containsKey("services"));
        assertTrue(metrics.containsKey("system"));
        assertTrue(metrics.containsKey("maintenance"));
    }
}

