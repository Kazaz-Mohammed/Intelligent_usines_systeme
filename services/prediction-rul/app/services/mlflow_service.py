"""
Service MLflow pour le tracking des expériences et le registre des modèles RUL
"""
import logging
import os
from typing import Dict, Any, Optional, List

# MLflow imports are optional - service can work without it
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Create dummy classes to avoid import errors
    class ActiveRun:
        pass
    mlflow = None

from app.config import settings

logger = logging.getLogger(__name__)


class MLflowService:
    """Service pour gérer MLflow (tracking et registry)"""
    
    def __init__(self):
        """Initialise le service MLflow"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow n'est pas installé - le service fonctionnera sans tracking MLflow")
            self.enabled = False
            self.tracking_uri = None
            self.experiment_name = None
            return
            
        self.enabled = settings.mlflow_enabled
        self.tracking_uri = settings.mlflow_tracking_uri
        self.experiment_name = settings.mlflow_experiment_name
        
        if self.enabled:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"MLflow initialisé: {self.tracking_uri}, experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"Erreur lors de l'initialisation MLflow: {e}. MLflow sera désactivé.")
                self.enabled = False
        else:
            logger.info("MLflow désactivé dans la configuration")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Démarre une nouvelle run MLflow
        
        Args:
            run_name: Nom de la run (optionnel)
            tags: Tags à ajouter à la run (optionnel)
        
        Returns:
            ActiveRun MLflow ou None si désactivé
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
        
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags or {})
            logger.debug(f"Run MLflow démarrée: {run.info.run_id}")
            return run
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de la run MLflow: {e}", exc_info=True)
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """
        Termine la run MLflow actuelle
        
        Args:
            status: Statut de la run ('FINISHED', 'FAILED', 'KILLED')
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.end_run(status=status)
            logger.debug("Run MLflow terminée")
        except Exception as e:
            logger.error(f"Erreur lors de la fin de la run MLflow: {e}", exc_info=True)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log des paramètres dans la run actuelle
        
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
        Log des métriques dans la run actuelle
        
        Args:
            metrics: Dictionnaire de métriques
            step: Step/epoch (optionnel)
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return
        
        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
            logger.debug(f"Métriques loggées: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Erreur lors du logging des métriques: {e}", exc_info=True)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Log un modèle dans MLflow
        
        Args:
            model: Modèle PyTorch ou scikit-learn
            artifact_path: Chemin de l'artifact
            registered_model_name: Nom du modèle dans le registry (optionnel)
        
        Returns:
            URI du modèle loggé ou None
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
        
        try:
            # Détecter le type de modèle
            if hasattr(model, 'state_dict') or hasattr(model, 'parameters'):
                # Modèle PyTorch
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )
            else:
                # Modèle scikit-learn
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )
            
            model_uri = mlflow.get_artifact_uri(artifact_path)
            logger.info(f"Modèle loggé: {model_uri}")
            return model_uri
            
        except Exception as e:
            logger.error(f"Erreur lors du logging du modèle: {e}", exc_info=True)
            return None
    
    def load_model(self, model_uri: str, model_type: str = "pytorch"):
        """
        Charge un modèle depuis MLflow
        
        Args:
            model_uri: URI du modèle (run_id/artifact_path ou registered_model_name:version)
            model_type: Type de modèle ('pytorch' ou 'sklearn')
        
        Returns:
            Modèle chargé ou None
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
        
        try:
            if model_type == "pytorch":
                model = mlflow.pytorch.load_model(model_uri)
            else:
                model = mlflow.sklearn.load_model(model_uri)
            
            logger.info(f"Modèle chargé depuis: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}", exc_info=True)
            return None
    
    def get_run_metrics(self, run_id: str) -> Dict[str, List[float]]:
        """
        Récupère les métriques d'une run
        
        Args:
            run_id: ID de la run
        
        Returns:
            Dictionnaire de métriques (nom -> liste de valeurs)
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return {}
        
        try:
            run = mlflow.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques: {e}", exc_info=True)
            return {}
    
    def get_run_params(self, run_id: str) -> Dict[str, str]:
        """
        Récupère les paramètres d'une run
        
        Args:
            run_id: ID de la run
        
        Returns:
            Dictionnaire de paramètres
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return {}
        
        try:
            run = mlflow.get_run(run_id)
            return run.data.params
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des paramètres: {e}", exc_info=True)
            return {}
    
    def get_latest_model_version(self, model_name: str) -> Optional[int]:
        """
        Récupère la dernière version d'un modèle enregistré
        
        Args:
            model_name: Nom du modèle
        
        Returns:
            Numéro de version ou None
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            latest_version = client.get_latest_versions(model_name, stages=["None"])
            if latest_version:
                return int(latest_version[0].version)
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la version: {e}", exc_info=True)
            return None
    
    def get_model_uri(self, model_name: str, version: Optional[int] = None, stage: Optional[str] = None) -> Optional[str]:
        """
        Récupère l'URI d'un modèle enregistré
        
        Args:
            model_name: Nom du modèle
            version: Version du modèle (optionnel)
            stage: Stage du modèle ('Staging', 'Production', etc.) (optionnel)
        
        Returns:
            URI du modèle ou None
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
        
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Dernière version
                latest_version = self.get_latest_model_version(model_name)
                if latest_version:
                    model_uri = f"models:/{model_name}/{latest_version}"
                else:
                    return None
            
            return model_uri
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'URI: {e}", exc_info=True)
            return None
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Recherche des runs MLflow
        
        Args:
            filter_string: Filtre MLflow (ex: "metrics.mae < 5.0")
            max_results: Nombre maximum de résultats
        
        Returns:
            Liste de runs (dict avec run_id, metrics, params, etc.)
        """
        if not self.enabled or not MLFLOW_AVAILABLE:
            return []
        
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=filter_string,
                max_results=max_results
            )
            
            results = []
            for _, run in runs.iterrows():
                results.append({
                    "run_id": run["run_id"],
                    "run_name": run.get("tags.mlflow.runName", ""),
                    "status": run["status"],
                    "metrics": {k: v for k, v in run.items() if k.startswith("metrics.")},
                    "params": {k: v for k, v in run.items() if k.startswith("params.")},
                    "start_time": run.get("start_time", None),
                    "end_time": run.get("end_time", None)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de runs: {e}", exc_info=True)
            return []
    
    def get_service_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le service MLflow"""
        return {
            "enabled": self.enabled,
            "tracking_uri": self.tracking_uri if self.enabled else None,
            "experiment_name": self.experiment_name if self.enabled else None
        }

