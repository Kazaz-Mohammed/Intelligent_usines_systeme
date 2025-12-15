package com.predictivemaintenance.orchestrateur.controller;

import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import com.predictivemaintenance.orchestrateur.service.WorkOrderService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Contrôleur pour la gestion des ordres de travail
 */
@RestController
@RequestMapping("/api/v1/work-orders")
@RequiredArgsConstructor
public class WorkOrderController {
    
    private final WorkOrderService workOrderService;
    
    /**
     * Récupère tous les ordres de travail
     * 
     * GET /api/v1/work-orders
     */
    @GetMapping
    public ResponseEntity<List<WorkOrder>> getAllWorkOrders() {
        List<WorkOrder> workOrders = workOrderService.findAll();
        return ResponseEntity.ok(workOrders);
    }
    
    /**
     * Récupère un ordre de travail par ID
     * 
     * GET /api/v1/work-orders/{id}
     */
    @GetMapping("/{id}")
    public ResponseEntity<WorkOrder> getWorkOrderById(@PathVariable Long id) {
        return workOrderService.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    /**
     * Récupère un ordre de travail par numéro
     * 
     * GET /api/v1/work-orders/number/{workOrderNumber}
     */
    @GetMapping("/number/{workOrderNumber}")
    public ResponseEntity<WorkOrder> getWorkOrderByNumber(@PathVariable String workOrderNumber) {
        return workOrderService.findByWorkOrderNumber(workOrderNumber)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    /**
     * Récupère tous les ordres de travail pour un actif
     * 
     * GET /api/v1/work-orders/asset/{assetId}
     */
    @GetMapping("/asset/{assetId}")
    public ResponseEntity<List<WorkOrder>> getWorkOrdersByAsset(@PathVariable String assetId) {
        List<WorkOrder> workOrders = workOrderService.findByAssetId(assetId);
        return ResponseEntity.ok(workOrders);
    }
    
    /**
     * Récupère tous les ordres de travail par statut
     * 
     * GET /api/v1/work-orders/status/{status}
     */
    @GetMapping("/status/{status}")
    public ResponseEntity<List<WorkOrder>> getWorkOrdersByStatus(@PathVariable WorkOrderStatus status) {
        List<WorkOrder> workOrders = workOrderService.findByStatus(status);
        return ResponseEntity.ok(workOrders);
    }
    
    /**
     * Met à jour le statut d'un ordre de travail
     * 
     * PUT /api/v1/work-orders/{id}/status
     */
    @PutMapping("/{id}/status")
    public ResponseEntity<WorkOrder> updateWorkOrderStatus(
            @PathVariable Long id,
            @RequestBody Map<String, String> request
    ) {
        WorkOrderStatus newStatus = WorkOrderStatus.valueOf(request.get("status").toUpperCase());
        WorkOrder updated = workOrderService.updateStatus(id, newStatus);
        return ResponseEntity.ok(updated);
    }
    
    /**
     * Récupère les statistiques des ordres de travail
     * 
     * GET /api/v1/work-orders/stats
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getWorkOrderStats() {
        Map<String, Object> stats = Map.of(
                "total", workOrderService.findAll().size(),
                "pending", workOrderService.countByStatus(WorkOrderStatus.PENDING),
                "scheduled", workOrderService.countByStatus(WorkOrderStatus.SCHEDULED),
                "inProgress", workOrderService.countByStatus(WorkOrderStatus.IN_PROGRESS),
                "completed", workOrderService.countByStatus(WorkOrderStatus.COMPLETED),
                "cancelled", workOrderService.countByStatus(WorkOrderStatus.CANCELLED)
        );
        return ResponseEntity.ok(stats);
    }
}

