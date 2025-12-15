package com.predictivemaintenance.orchestrateur.model;

/**
 * Niveaux de priorité pour les interventions de maintenance
 */
public enum PriorityLevel {
    CRITICAL(1, "Critique - Intervention immédiate requise"),
    HIGH(2, "Haute - Intervention dans les 4 heures"),
    MEDIUM(3, "Moyenne - Intervention dans les 24 heures"),
    LOW(4, "Basse - Intervention planifiée");
    
    private final int level;
    private final String description;
    
    PriorityLevel(int level, String description) {
        this.level = level;
        this.description = description;
    }
    
    public int getLevel() {
        return level;
    }
    
    public String getDescription() {
        return description;
    }
    
    public static PriorityLevel fromLevel(int level) {
        for (PriorityLevel p : values()) {
            if (p.level == level) {
                return p;
            }
        }
        throw new IllegalArgumentException("Niveau de priorité invalide: " + level);
    }
}

