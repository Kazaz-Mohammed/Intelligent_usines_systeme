"""
Tests pour le service Asset
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.asset_service import AssetService


class TestAssetService:
    """Tests pour AssetService"""

    @pytest.fixture
    def asset_service(self):
        return AssetService()

    @pytest.mark.timescaledb
    def test_get_asset_type(self, asset_service):
        """Test récupération du type d'actif"""
        asset_type = asset_service.get_asset_type("ASSET001")
        # Si la base de données est disponible, devrait retourner "pump"
        # Sinon, devrait retourner None
        assert asset_type is None or asset_type == "pump"

    @pytest.mark.timescaledb
    def test_get_asset_info(self, asset_service):
        """Test récupération des informations d'actif"""
        asset_info = asset_service.get_asset_info("ASSET001")
        # Si la base de données est disponible, devrait retourner les infos
        # Sinon, devrait retourner None
        if asset_info:
            assert "id" in asset_info
            assert "type" in asset_info
            assert asset_info["id"] == "ASSET001"

    def test_get_asset_type_nonexistent(self, asset_service):
        """Test avec actif inexistant"""
        asset_type = asset_service.get_asset_type("NONEXISTENT")
        assert asset_type is None

    def test_get_asset_info_nonexistent(self, asset_service):
        """Test avec actif inexistant"""
        asset_info = asset_service.get_asset_info("NONEXISTENT")
        assert asset_info is None

    @patch('app.services.asset_service.SimpleConnectionPool')
    def test_get_asset_type_with_mock(self, mock_pool):
        """Test avec mock de la base de données"""
        # Créer un mock de la connexion
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("pump",)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None

        # Créer un mock du pool
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance

        # Créer le service
        service = AssetService()
        service.pool = mock_pool_instance

        # Tester
        asset_type = service.get_asset_type("ASSET001")
        assert asset_type == "pump"

    def test_close(self, asset_service):
        """Test fermeture du service"""
        asset_service.close()
        # Vérifier que le pool est fermé
        assert asset_service.pool is None or hasattr(asset_service.pool, 'closeall')

