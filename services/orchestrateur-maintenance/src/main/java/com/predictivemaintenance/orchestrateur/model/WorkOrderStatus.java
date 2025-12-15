package com.predictivemaintenance.orchestrateur.model;

/**
 * Statut d'un ordre de travail
 */
public enum WorkOrderStatus {
    PENDING("En attente"),
    SCHEDULED("Planifié"),
    IN_PROGRESS("En cours"),
    COMPLETED("Terminé"),
    CANCELLED("Annulé"),
    DEFERRED("Reporté");
    
    private final String description;
    
    WorkOrderStatus(String description) {
        this.description = description;
    }
    
    public String getDescription() {
        return description;
    }
}

