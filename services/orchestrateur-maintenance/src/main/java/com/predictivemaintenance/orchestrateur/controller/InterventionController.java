package com.predictivemaintenance.orchestrateur.controller;

import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.service.PlanningService;
import com.predictivemaintenance.orchestrateur.service.WorkOrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Contrôleur pour la gestion des interventions de maintenance
 */
@RestController
@RequestMapping("/api/v1/interventions")
@RequiredArgsConstructor
public class InterventionController {
    
    private final PlanningService planningService;
    private final WorkOrderService workOrderService;
    
    /**
     * Crée une intervention de maintenance
     * 
     * POST /api/v1/interventions
     */
    @PostMapping
    public ResponseEntity<WorkOrder> createIntervention(
            @Valid @RequestBody InterventionRequest request
    ) {
        WorkOrder workOrder = planningService.planIntervention(request);
        WorkOrder saved = workOrderService.save(workOrder);
        
        return ResponseEntity.status(HttpStatus.CREATED).body(saved);
    }
    
    /**
     * Crée plusieurs interventions avec optimisation
     * 
     * POST /api/v1/interventions/batch
     */
    @PostMapping("/batch")
    public ResponseEntity<List<WorkOrder>> createInterventionsBatch(
            @Valid @RequestBody List<InterventionRequest> requests
    ) {
        List<WorkOrder> workOrders = planningService.planInterventionsOptimized(requests);
        List<WorkOrder> saved = workOrderService.saveAll(workOrders);
        
        return ResponseEntity.status(HttpStatus.CREATED).body(saved);
    }
}

