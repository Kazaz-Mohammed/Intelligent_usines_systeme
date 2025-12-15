"""
Service MLflow pour le tracking et le registry des modèles
"""
import logging
import os
from typing import Optional, Dict, Any, List

# MLflow imports are optional - service can work without it
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Create dummy classes to avoid import errors
    class MlflowClient:
        pass
    mlflow = None

from app.config import settings

logger = logging.getLogger(__name__)


class MLflowService:
    """Service pour gérer MLflow (tracking et registry)"""
    
    def __init__(self):
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow n'est pas installé - le service fonctionnera sans tracking MLflow")
            self.enabled = False
            self.client = None
            return
            
        self.enabled = settings.mlflow_enabled
        if not self.enabled:
            logger.info("MLflow désactivé dans la configuration")
            self.client = None
            return
        
        try:
            # Configurer MLflow - try file backend first to avoid connection delays
            import os
            file_backend_path = "./mlruns"
            os.makedirs(file_backend_path, exist_ok=True)
            file_uri = f"file://{os.path.abspath(file_backend_path)}"
            
            # Try file backend first (faster, no network)
            try:
                mlflow.set_tracking_uri(file_uri)
                mlflow.set_experiment(settings.mlflow_experiment_name)
                self.client = None  # File backend doesn't need client
                logger.info(f"MLflow initialisé avec backend fichier: {file_backend_path}")
                logger.info(f"Experiment: {settings.mlflow_experiment_name}")
            except Exception as file_error:
                # If file backend fails, try server (but don't wait long)
                logger.debug(f"Backend fichier échoué, tentative serveur: {file_error}")
                try:
                    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                    mlflow.set_experiment(settings.mlflow_experiment_name)
                    self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
                    logger.info(f"MLflow initialisé: {settings.mlflow_tracking_uri}")
                    logger.info(f"Experiment: {settings.mlflow_experiment_name}")
                except Exception as server_error:
                    # Both failed, disable MLflow
                    error_msg = str(server_error)
                    if "Connection refused" in error_msg or "Failed to establish" in error_msg:
                        logger.debug(f"MLflow server non disponible, utilisation backend fichier uniquement")
                        # Fall back to file backend silently
                        mlflow.set_tracking_uri(file_uri)
                        mlflow.set_experiment(settings.mlflow_experiment_name)
                        self.client = None
                    else:
                        logger.warning(f"Impossible d'initialiser MLflow: {error_msg}")
                        logger.warning("MLflow sera désactivé - le service fonctionnera sans tracking")
                        self.enabled = False
                        self.client = None
        except Exception as e:
            # Log only a warning, not the full stack trace, since this is expected when MLflow server is not running
            error_msg = str(e)
            if "Connection refused" in error_msg or "Failed to establish" in error_msg:
                logger.warning(f"MLflow server non disponible à {settings.mlflow_tracking_uri}")
                logger.info("MLflow sera désactivé - le service fonctionnera sans tracking MLflow")
            else:
                logger.warning(f"Erreur lors de l'initialisation MLflow: {error_msg}")
                logger.info("MLflow sera désactivé - le service fonctionnera sans tracking MLflow")
            self.enabled = False
            self.client = None
    
    def start_run(self, run_name: Optional[str] = None):
        """
        Démarre un run MLflow
        
        Args:
            run_name: Nom du run (optionnel)
        
        Returns:
            ActiveRun ou None si MLflow est désactivé
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
        
        try:
            return mlflow.start_run(run_name=run_name)
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du run MLflow: {e}", exc_info=True)
            return None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log les paramètres d'un modèle
        
        Args:
            params: Dictionnaire de paramètres
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.log_params(params)
            logger.debug(f"Paramètres loggés: {list(params.keys())}")
        except Exception as e:
            logger.error(f"Erreur lors du logging des paramètres: {e}", exc_info=True)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log les métriques d'un modèle
        
        Args:
            metrics: Dictionnaire de métriques
            step: Step/epoch (optionnel)
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Métriques loggées: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Erreur lors du logging des métriques: {e}", exc_info=True)
    
    def log_model_sklearn(self, model: Any, artifact_path: str, registered_model_name: Optional[str] = None):
        """
        Log un modèle scikit-learn/PyOD
        
        Args:
            model: Modèle à logger
            artifact_path: Chemin de l'artifact
            registered_model_name: Nom du modèle dans le registry (optionnel)
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            if registered_model_name:
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
                    registered_model_name
                )
                logger.info(f"Modèle enregistré dans le registry: {registered_model_name}")
        except Exception as e:
            logger.error(f"Erreur lors du logging du modèle sklearn: {e}", exc_info=True)
    
    def log_model_pytorch(self, model: Any, artifact_path: str, registered_model_name: Optional[str] = None):
        """
        Log un modèle PyTorch
        
        Args:
            model: Modèle PyTorch à logger
            artifact_path: Chemin de l'artifact
            registered_model_name: Nom du modèle dans le registry (optionnel)
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            if registered_model_name:
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
                    registered_model_name
                )
                logger.info(f"Modèle enregistré dans le registry: {registered_model_name}")
        except Exception as e:
            logger.error(f"Erreur lors du logging du modèle PyTorch: {e}", exc_info=True)
    
    def load_model(self, model_name: str, version: Optional[int] = None, stage: Optional[str] = None):
        """
        Charge un modèle depuis le registry
        
        Args:
            model_name: Nom du modèle
            version: Version du modèle (optionnel)
            stage: Stage du modèle (None, "Staging", "Production") (optionnel)
        
        Returns:
            Modèle chargé ou None
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
        
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            # Essayer de charger comme modèle sklearn
            try:
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Modèle sklearn chargé: {model_uri}")
                return model
            except:
                pass
            
            # Essayer de charger comme modèle PyTorch
            try:
                model = mlflow.pytorch.load_model(model_uri)
                logger.info(f"Modèle PyTorch chargé: {model_uri}")
                return model
            except:
                pass
            
            logger.warning(f"Impossible de charger le modèle: {model_uri}")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}", exc_info=True)
            return None
    
    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """
        Transitionne un modèle vers un stage (Staging, Production, Archived)
        
        Args:
            model_name: Nom du modèle
            version: Version du modèle
            stage: Nouveau stage
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Modèle {model_name} v{version} transitionné vers {stage}")
        except Exception as e:
            logger.error(f"Erreur lors de la transition du modèle: {e}", exc_info=True)
    
    def get_latest_versions(self, model_name: str, stages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Récupère les dernières versions d'un modèle
        
        Args:
            model_name: Nom du modèle
            stages: Liste de stages à filtrer (optionnel)
        
        Returns:
            Liste des versions
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return []
        
        try:
            versions = self.client.get_latest_versions(model_name, stages=stages)
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status
                }
                for v in versions
            ]
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des versions: {e}", exc_info=True)
            return []
    
    def end_run(self):
        """Termine le run MLflow actif"""
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.end_run()
        except Exception as e:
            logger.error(f"Erreur lors de la fin du run: {e}", exc_info=True)

