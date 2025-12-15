package com.predictivemaintenance.dashboard.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.persistence.*;
import java.time.LocalDateTime;

/**
 * Modèle d'alerte
 */
@Entity
@Table(name = "alerts")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Alert {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    /**
     * Type d'alerte
     */
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private AlertType type;
    
    /**
     * Niveau de sévérité
     */
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Severity severity;
    
    /**
     * Titre de l'alerte
     */
    @Column(nullable = false)
    private String title;
    
    /**
     * Description de l'alerte
     */
    @Column(columnDefinition = "TEXT")
    private String description;
    
    /**
     * ID de l'actif concerné (si applicable)
     */
    private String assetId;
    
    /**
     * Service source de l'alerte
     */
    private String sourceService;
    
    /**
     * Statut de l'alerte
     */
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private AlertStatus status;
    
    /**
     * Timestamp de création
     */
    @Column(name = "`created_at`", nullable = false)
    private LocalDateTime createdAt;
    
    /**
     * Timestamp de dernière mise à jour
     */
    @Column(name = "`updated_at`")
    private LocalDateTime updatedAt;
    
    /**
     * Timestamp d'acquittement (si acquittée)
     */
    @Column(name = "`acknowledged_at`")
    private LocalDateTime acknowledgedAt;
    
    /**
     * Utilisateur qui a acquitté l'alerte
     */
    private String acknowledgedBy;
    
    /**
     * Métadonnées additionnelles (JSON)
     */
    @Column(columnDefinition = "CLOB")
    private String metadata;
    
    /**
     * Type d'alerte
     */
    public enum AlertType {
        SERVICE_DOWN,
        HIGH_ANOMALY_RATE,
        CRITICAL_ASSET,
        LOW_RUL,
        INTERVENTION_OVERDUE,
        SYSTEM_ERROR,
        PERFORMANCE_DEGRADATION,
        OTHER
    }
    
    /**
     * Niveau de sévérité
     */
    public enum Severity {
        INFO,
        WARNING,
        HIGH,
        CRITICAL
    }
    
    /**
     * Statut de l'alerte
     */
    public enum AlertStatus {
        ACTIVE,
        ACKNOWLEDGED,
        RESOLVED,
        DISMISSED
    }
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
        if (status == null) {
            status = AlertStatus.ACTIVE;
        }
    }
    
    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}

