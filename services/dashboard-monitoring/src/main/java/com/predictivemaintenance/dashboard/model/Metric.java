package com.predictivemaintenance.dashboard.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.persistence.*;
import java.time.LocalDateTime;

/**
 * Modèle de métrique pour historisation
 */
@Entity
@Table(name = "metrics")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Metric {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    /**
     * Nom de la métrique
     */
    @Column(nullable = false)
    private String metricName;
    
    /**
     * Nom du service
     */
    private String serviceName;
    
    /**
     * Valeur de la métrique
     */
    @Column(name = "`value`", nullable = false)
    private Double value;
    
    /**
     * Unité de la métrique
     */
    private String unit;
    
    /**
     * Labels additionnels (JSON)
     */
    @Column(columnDefinition = "CLOB")
    private String labels;
    
    /**
     * Timestamp de la métrique
     */
    @Column(name = "`timestamp`", nullable = false)
    private LocalDateTime timestamp;
    
    /**
     * Timestamp de création
     */
    @Column(name = "`created_at`", nullable = false)
    private LocalDateTime createdAt;
    
    @PrePersist
    protected void onCreate() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
        if (timestamp == null) {
            timestamp = LocalDateTime.now();
        }
    }
}

