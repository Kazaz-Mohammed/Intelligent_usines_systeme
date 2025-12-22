// CI-Only Pipeline for Microservices (No Deployment Stage)
// This pipeline builds and analyzes code without deploying containers

pipeline {
    agent any

    tools {
        maven 'maven'
    }

    stages {
        
        stage('Cloner le dépôt') {
            steps {
                echo 'Clonage du dépôt GitHub...'
                git branch: 'main', url: 'https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme.git'
            }
        }

        stage('Build et Analyse SonarQube') {
            parallel {
                
                // ============================================
                // Services Java/Spring Boot (Maven)
                // ============================================
                
                stage('IngestionIIoT Service') {
                    stages {
                        stage('Build IngestionIIoT') {
                            steps {
                                dir('services/ingestion-iiot') {
                                    echo 'Compilation et génération du service IngestionIIoT...'
                                    script {
                                        bat 'mvn clean install -DskipTests'
                                    }
                                }
                            }
                        }
                        
                        stage('SonarQube Analysis IngestionIIoT') {
                            steps {
                                dir('services/ingestion-iiot') {
                                    script {
                                        def mvn = tool 'maven'
                                        withSonarQubeEnv('SonarQube') {
                                            bat "\"${mvn}\\bin\\mvn\" clean verify sonar:sonar ^ " +
                                                "-Dsonar.projectKey=ingestion-iiot ^ " +
                                                "-Dsonar.projectName=IngestionIIoT ^ " +
                                                "-Dsonar.sources=src/main ^ " +
                                                "-Dsonar.tests=src/test ^ " +
                                                "-Dsonar.java.binaries=target/classes ^ " +
                                                "-Dsonar.junit.reportPaths=target/surefire-reports ^ " +
                                                "-DskipTests"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                stage('OrchestrateurMaintenance Service') {
                    stages {
                        stage('Build OrchestrateurMaintenance') {
                            steps {
                                dir('services/orchestrateur-maintenance') {
                                    echo 'Compilation et génération du service OrchestrateurMaintenance...'
                                    script {
                                        bat 'mvn clean install -DskipTests'
                                    }
                                }
                            }
                        }
                        
                        stage('SonarQube Analysis OrchestrateurMaintenance') {
                            steps {
                                dir('services/orchestrateur-maintenance') {
                                    script {
                                        def mvn = tool 'maven'
                                        withSonarQubeEnv('SonarQube') {
                                            bat "\"${mvn}\\bin\\mvn\" clean verify sonar:sonar ^ " +
                                                "-Dsonar.projectKey=orchestrateur-maintenance ^ " +
                                                "-Dsonar.projectName=OrchestrateurMaintenance ^ " +
                                                "-Dsonar.sources=src/main ^ " +
                                                "-Dsonar.tests=src/test ^ " +
                                                "-Dsonar.java.binaries=target/classes ^ " +
                                                "-DskipTests"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ============================================
                // Services Python/FastAPI
                // ============================================
                
                stage('Preprocessing Service') {
                    stages {
                        stage('Build Preprocessing') {
                            steps {
                                dir('services/preprocessing') {
                                    echo 'Installation des dépendances Python pour Preprocessing...'
                                    script {
                                        bat 'python -m pip install --upgrade pip'
                                        bat 'pip install -r requirements.txt'
                                    }
                                }
                            }
                        }
                        
                        stage('SonarQube Analysis Preprocessing') {
                            steps {
                                dir('services/preprocessing') {
                                    script {
                                        def scannerHome = tool 'SonarQubeScanner'
                                        withSonarQubeEnv('SonarQube') {
                                            bat """
                                                if exist sonar-project.properties (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\"
                                                ) else (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\" ^
                                                        -Dsonar.projectKey=preprocessing ^
                                                        -Dsonar.projectName=Preprocessing ^
                                                        -Dsonar.sources=app ^
                                                        -Dsonar.python.version=3.9
                                                )
                                            """
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                stage('ExtractionFeatures Service') {
                    stages {
                        stage('Build ExtractionFeatures') {
                            steps {
                                dir('services/extraction-features') {
                                    echo 'Installation des dépendances Python pour ExtractionFeatures...'
                                    script {
                                        bat 'python -m pip install --upgrade pip'
                                        bat 'pip install -r requirements.txt'
                                    }
                                }
                            }
                        }
                        
                        stage('SonarQube Analysis ExtractionFeatures') {
                            steps {
                                dir('services/extraction-features') {
                                    script {
                                        def scannerHome = tool 'SonarQubeScanner'
                                        withSonarQubeEnv('SonarQube') {
                                            bat """
                                                if exist sonar-project.properties (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\"
                                                ) else (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\" ^
                                                        -Dsonar.projectKey=extraction-features ^
                                                        -Dsonar.projectName=ExtractionFeatures ^
                                                        -Dsonar.sources=app ^
                                                        -Dsonar.python.version=3.9
                                                )
                                            """
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                stage('DetectionAnomalies Service') {
                    stages {
                        stage('Build DetectionAnomalies') {
                            steps {
                                dir('services/detection-anomalies') {
                                    echo 'Installation des dépendances Python pour DetectionAnomalies...'
                                    script {
                                        bat 'python -m pip install --upgrade pip'
                                        bat 'pip install -r requirements.txt'
                                    }
                                }
                            }
                        }
                        
                        stage('SonarQube Analysis DetectionAnomalies') {
                            steps {
                                dir('services/detection-anomalies') {
                                    script {
                                        def scannerHome = tool 'SonarQubeScanner'
                                        withSonarQubeEnv('SonarQube') {
                                            bat """
                                                if exist sonar-project.properties (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\"
                                                ) else (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\" ^
                                                        -Dsonar.projectKey=detection-anomalies ^
                                                        -Dsonar.projectName=DetectionAnomalies ^
                                                        -Dsonar.sources=app ^
                                                        -Dsonar.python.version=3.9
                                                )
                                            """
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                stage('PredictionRUL Service') {
                    stages {
                        stage('Build PredictionRUL') {
                            steps {
                                dir('services/prediction-rul') {
                                    echo 'Installation des dépendances Python pour PredictionRUL...'
                                    script {
                                        bat 'python -m pip install --upgrade pip'
                                        bat 'pip install -r requirements.txt'
                                    }
                                }
                            }
                        }
                        
                        stage('SonarQube Analysis PredictionRUL') {
                            steps {
                                dir('services/prediction-rul') {
                                    script {
                                        def scannerHome = tool 'SonarQubeScanner'
                                        withSonarQubeEnv('SonarQube') {
                                            bat """
                                                if exist sonar-project.properties (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\"
                                                ) else (
                                                    \"${scannerHome}\\bin\\sonar-scanner.bat\" ^
                                                        -Dsonar.projectKey=prediction-rul ^
                                                        -Dsonar.projectName=PredictionRUL ^
                                                        -Dsonar.sources=app ^
                                                        -Dsonar.python.version=3.9
                                                )
                                            """
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // NOTE: Deployment stage removed - this is CI-only pipeline
        // To add CD later, uncomment the stage below:
        /*
        stage('Docker Compose - Déploiement') {
            steps {
                dir('deploy') {
                    echo 'Création et déploiement des conteneurs Docker...'
                    script {
                        bat 'docker-compose up -d --build'
                    }
                }
            }
        }
        */
    }

    post {
        always {
            echo 'Pipeline CI terminé. Consulter les résultats dans SonarQube (http://localhost:9999)'
        }
        success {
            echo '✅ Pipeline CI réussi! Tous les builds et analyses SonarQube sont OK.'
        }
        failure {
            echo '❌ Pipeline CI échoué. Vérifier les logs pour plus de détails.'
        }
    }
}

