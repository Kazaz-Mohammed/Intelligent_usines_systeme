"""
Script de test pour vérifier la connexion TimescaleDB
"""
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import sys
import os

# Forcer l'encodage UTF-8 pour éviter les problèmes sur Windows
if sys.platform == 'win32':
    os.environ['PGCLIENTENCODING'] = 'UTF8'

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'predictive_maintenance',
    'user': 'pmuser',
    'password': 'pmpassword'
}

def test_connection():
    """Test de connexion directe"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connexion directe réussie")
        
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()
            print(f"✅ Version PostgreSQL: {version[0][:50]}...")
            
            # Vérifier TimescaleDB
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            ext = cur.fetchone()
            if ext:
                print("✅ Extension TimescaleDB installée")
            else:
                print("⚠️ Extension TimescaleDB non installée")
            
            # Vérifier les tables
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name IN ('preprocessed_sensor_data', 'windowed_sensor_data')
            """)
            tables = cur.fetchall()
            print(f"✅ Tables trouvées: {[t[0] for t in tables]}")
            
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False

def test_pool():
    """Test du pool de connexions"""
    try:
        pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG
        )
        print("✅ Pool de connexions créé")
        
        conn = pool.getconn()
        print("✅ Connexion obtenue du pool")
        
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            assert result[0] == 1
            print("✅ Requête exécutée avec succès")
        
        pool.putconn(conn)
        print("✅ Connexion retournée au pool")
        
        pool.closeall()
        print("✅ Pool fermé")
        return True
    except Exception as e:
        print(f"❌ Erreur avec le pool: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Test de connexion TimescaleDB ===\n")
    
    print("1. Test de connexion directe:")
    test_connection()
    
    print("\n2. Test du pool de connexions:")
    test_pool()
    
    print("\n=== Tests terminés ===")

