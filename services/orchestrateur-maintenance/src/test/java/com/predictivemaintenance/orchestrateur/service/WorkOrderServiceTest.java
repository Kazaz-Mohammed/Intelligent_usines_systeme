package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import com.predictivemaintenance.orchestrateur.repository.WorkOrderRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Tests pour WorkOrderService
 */
class WorkOrderServiceTest {
    
    @Mock
    private WorkOrderRepository workOrderRepository;
    
    private WorkOrderService workOrderService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        workOrderService = new WorkOrderService(workOrderRepository);
    }
    
    @Test
    void testSave() {
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-001")
                .assetId("ASSET001")
                .priority(PriorityLevel.HIGH)
                .status(WorkOrderStatus.SCHEDULED)
                .build();
        
        when(workOrderRepository.save(workOrder)).thenReturn(workOrder);
        
        WorkOrder saved = workOrderService.save(workOrder);
        
        assertNotNull(saved);
        assertEquals("WO-001", saved.getWorkOrderNumber());
        verify(workOrderRepository, times(1)).save(workOrder);
    }
    
    @Test
    void testSaveAll() {
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder().workOrderNumber("WO-001").build(),
                WorkOrder.builder().workOrderNumber("WO-002").build()
        );
        
        when(workOrderRepository.saveAll(workOrders)).thenReturn(workOrders);
        
        List<WorkOrder> saved = workOrderService.saveAll(workOrders);
        
        assertEquals(2, saved.size());
        verify(workOrderRepository, times(1)).saveAll(workOrders);
    }
    
    @Test
    void testFindById() {
        WorkOrder workOrder = WorkOrder.builder()
                .id(1L)
                .workOrderNumber("WO-001")
                .build();
        
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));
        
        Optional<WorkOrder> found = workOrderService.findById(1L);
        
        assertTrue(found.isPresent());
        assertEquals("WO-001", found.get().getWorkOrderNumber());
    }
    
    @Test
    void testFindByWorkOrderNumber() {
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-001")
                .build();
        
        when(workOrderRepository.findByWorkOrderNumber("WO-001"))
                .thenReturn(Optional.of(workOrder));
        
        Optional<WorkOrder> found = workOrderService.findByWorkOrderNumber("WO-001");
        
        assertTrue(found.isPresent());
        assertEquals("WO-001", found.get().getWorkOrderNumber());
    }
    
    @Test
    void testFindByAssetId() {
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder().assetId("ASSET001").build(),
                WorkOrder.builder().assetId("ASSET001").build()
        );
        
        when(workOrderRepository.findByAssetIdOrderByScheduledStartTimeDesc("ASSET001"))
                .thenReturn(workOrders);
        
        List<WorkOrder> found = workOrderService.findByAssetId("ASSET001");
        
        assertEquals(2, found.size());
    }
    
    @Test
    void testFindByStatus() {
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder().status(WorkOrderStatus.SCHEDULED).build()
        );
        
        when(workOrderRepository.findByStatusOrderByScheduledStartTimeAsc(WorkOrderStatus.SCHEDULED))
                .thenReturn(workOrders);
        
        List<WorkOrder> found = workOrderService.findByStatus(WorkOrderStatus.SCHEDULED);
        
        assertEquals(1, found.size());
    }
    
    @Test
    void testUpdateStatusToInProgress() {
        WorkOrder workOrder = WorkOrder.builder()
                .id(1L)
                .status(WorkOrderStatus.SCHEDULED)
                .build();
        
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));
        when(workOrderRepository.save(any(WorkOrder.class))).thenReturn(workOrder);
        
        WorkOrder updated = workOrderService.updateStatus(1L, WorkOrderStatus.IN_PROGRESS);
        
        assertNotNull(updated);
        assertNotNull(updated.getActualStartTime());
        verify(workOrderRepository, times(1)).save(any(WorkOrder.class));
    }
    
    @Test
    void testUpdateStatusToCompleted() {
        LocalDateTime startTime = LocalDateTime.now().minusHours(2);
        WorkOrder workOrder = WorkOrder.builder()
                .id(1L)
                .status(WorkOrderStatus.IN_PROGRESS)
                .actualStartTime(startTime)
                .build();
        
        when(workOrderRepository.findById(1L)).thenReturn(Optional.of(workOrder));
        when(workOrderRepository.save(any(WorkOrder.class))).thenReturn(workOrder);
        
        WorkOrder updated = workOrderService.updateStatus(1L, WorkOrderStatus.COMPLETED);
        
        assertNotNull(updated);
        assertNotNull(updated.getActualEndTime());
        assertNotNull(updated.getActualDurationMinutes());
        verify(workOrderRepository, times(1)).save(any(WorkOrder.class));
    }
    
    @Test
    void testUpdateStatusNotFound() {
        when(workOrderRepository.findById(1L)).thenReturn(Optional.empty());
        
        assertThrows(IllegalArgumentException.class, () -> {
            workOrderService.updateStatus(1L, WorkOrderStatus.COMPLETED);
        });
    }
    
    @Test
    void testCountByStatus() {
        when(workOrderRepository.countByStatus(WorkOrderStatus.SCHEDULED)).thenReturn(5L);
        
        long count = workOrderService.countByStatus(WorkOrderStatus.SCHEDULED);
        
        assertEquals(5L, count);
    }
}

