"""
Tests pour le service de nettoyage
"""
import pytest
from datetime import datetime
import pandas as pd
import numpy as np

from app.services.cleaning_service import CleaningService
from app.models.sensor_data import SensorData, PreprocessedData


class TestCleaningService:
    """Tests pour CleaningService"""
    
    @pytest.fixture
    def cleaning_service(self):
        return CleaningService()
    
    @pytest.fixture
    def sample_sensor_data(self):
        return SensorData(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            value=25.5,
            unit="°C",
            quality=2,
            source_type="TEST"
        )
    
    def test_clean_single_value_good_quality(self, cleaning_service, sample_sensor_data):
        """Test nettoyage d'une valeur de bonne qualité"""
        result = cleaning_service.clean_single_value(sample_sensor_data)
        
        assert result is not None
        assert result.value == 25.5
        assert result.quality == 2
        assert result.asset_id == "ASSET001"
        assert result.sensor_id == "SENSOR001"
        assert "outlier_removed" in result.preprocessing_metadata
    
    def test_clean_single_value_bad_quality(self, cleaning_service):
        """Test nettoyage d'une valeur de mauvaise qualité"""
        bad_data = SensorData(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            value=25.5,
            unit="°C",
            quality=0,  # Bad quality
            source_type="TEST"
        )
        
        result = cleaning_service.clean_single_value(bad_data)
        
        assert result is not None
        assert result.quality == 0
    
    def test_clean_single_value_with_outlier(self, cleaning_service, sample_sensor_data):
        """Test détection et correction d'outlier"""
        historical_values = [25.0, 25.1, 25.2, 25.3, 25.4] * 10  # 50 valeurs normales
        
        # Valeur outlier (très différente)
        outlier_data = SensorData(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            value=100.0,  # Outlier évident
            unit="°C",
            quality=2,
            source_type="TEST"
        )
        
        result = cleaning_service.clean_single_value(outlier_data, historical_values)
        
        assert result is not None
        assert result.preprocessing_metadata.get("outlier_removed", False) is True
        # La valeur devrait être imputée avec la médiane
        assert result.value != 100.0
    
    def test_clean_single_value_infinite(self, cleaning_service):
        """Test gestion des valeurs infinies"""
        inf_data = SensorData(
            timestamp=datetime.utcnow(),
            asset_id="ASSET001",
            sensor_id="SENSOR001",
            value=float('inf'),
            unit="°C",
            quality=2,
            source_type="TEST"
        )
        
        result = cleaning_service.clean_single_value(inf_data)
        
        assert result is not None
        assert np.isfinite(result.value)
        assert result.value == 0.0  # Valeur par défaut pour infini
    
    def test_clean_dataframe(self, cleaning_service):
        """Test nettoyage d'un DataFrame"""
        # Créer un DataFrame avec outliers et valeurs manquantes
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=100, freq='1min'),
            'asset_id': ['ASSET001'] * 100,
            'sensor_id': ['SENSOR001'] * 100,
            'value': [25.0 + np.random.randn() * 0.5 for _ in range(100)],
            'quality': [2] * 100
        })
        
        # Ajouter quelques outliers
        df.loc[10, 'value'] = 100.0  # Outlier
        df.loc[20, 'value'] = np.nan  # Valeur manquante
        df.loc[30, 'value'] = float('inf')  # Infini
        
        result = cleaning_service.clean_dataframe(df)
        
        assert len(result) > 0
        assert result['value'].isna().sum() == 0  # Pas de NaN
        assert np.isfinite(result['value']).all()  # Pas d'infini
        assert 'preprocessing_metadata' in result.columns
    
    def test_detect_outliers_iqr(self, cleaning_service):
        """Test détection d'outliers avec IQR"""
        values = pd.Series([25.0, 25.1, 25.2, 25.3, 25.4, 100.0, 25.5])
        
        outliers = cleaning_service.detect_outliers_iqr(values)
        
        assert outliers.sum() > 0  # Au moins un outlier détecté
        assert outliers.iloc[5] == True  # La valeur 100.0 devrait être un outlier

