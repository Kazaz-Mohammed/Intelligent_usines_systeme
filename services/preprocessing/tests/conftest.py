"""
Configuration pytest pour les tests
"""
import pytest
import sys
from pathlib import Path

# Ajouter le répertoire racine au path Python
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

@pytest.fixture(scope="session")
def test_config():
    """Configuration de test"""
    return {
        "kafka_bootstrap_servers": "localhost:9092",
        "database_host": "localhost",
        "database_port": 5432,
    }

