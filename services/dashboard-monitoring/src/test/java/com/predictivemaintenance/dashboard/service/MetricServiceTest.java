package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.model.Metric;
import com.predictivemaintenance.dashboard.repository.MetricRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Tests pour MetricService
 */
class MetricServiceTest {
    
    @Mock
    private MetricRepository metricRepository;
    
    private MetricService metricService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        metricService = new MetricService(metricRepository);
    }
    
    @Test
    void testSaveMetric() {
        Metric metric = Metric.builder()
                .metricName("cpu.usage")
                .value(75.5)
                .timestamp(LocalDateTime.now())
                .build();
        
        when(metricRepository.save(any(Metric.class))).thenReturn(metric);
        
        Metric saved = metricService.saveMetric(metric);
        
        assertNotNull(saved);
        assertEquals("cpu.usage", saved.getMetricName());
        assertEquals(75.5, saved.getValue());
        verify(metricRepository, times(1)).save(metric);
    }
    
    @Test
    void testSaveMetrics() {
        List<Metric> metrics = List.of(
                Metric.builder().metricName("metric1").value(10.0).build(),
                Metric.builder().metricName("metric2").value(20.0).build()
        );
        
        when(metricRepository.saveAll(anyList())).thenReturn(metrics);
        
        List<Metric> saved = metricService.saveMetrics(metrics);
        
        assertEquals(2, saved.size());
        verify(metricRepository, times(1)).saveAll(metrics);
    }
    
    @Test
    void testGetRecentMetrics() {
        List<Metric> metrics = List.of(
                Metric.builder().metricName("metric1").build()
        );
        
        when(metricRepository.findTopNByOrderByTimestampDesc(100))
                .thenReturn(metrics);
        
        List<Metric> recent = metricService.getRecentMetrics(100);
        
        assertEquals(1, recent.size());
    }
    
    @Test
    void testGetMetricsByName() {
        LocalDateTime start = LocalDateTime.now().minusDays(1);
        LocalDateTime end = LocalDateTime.now();
        
        List<Metric> metrics = List.of(
                Metric.builder().metricName("cpu.usage").value(75.0).build()
        );
        
        when(metricRepository.findByMetricNameAndTimestampBetweenOrderByTimestampAsc(
                "cpu.usage", start, end
        )).thenReturn(metrics);
        
        List<Metric> result = metricService.getMetricsByName("cpu.usage", start, end);
        
        assertEquals(1, result.size());
        assertEquals("cpu.usage", result.get(0).getMetricName());
    }
    
    @Test
    void testGetLatestMetric() {
        Metric metric = Metric.builder()
                .metricName("cpu.usage")
                .value(80.0)
                .build();
        
        when(metricRepository.findTopByMetricNameOrderByTimestampDesc("cpu.usage"))
                .thenReturn(Optional.of(metric));
        
        Metric latest = metricService.getLatestMetric("cpu.usage");
        
        assertNotNull(latest);
        assertEquals("cpu.usage", latest.getMetricName());
        assertEquals(80.0, latest.getValue());
    }
}

