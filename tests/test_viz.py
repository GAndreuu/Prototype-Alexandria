from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)

def test_endpoints():
    print("--- Testing Visualization Endpoints (TestClient) ---")
    
    # 1. Test Manifold Data
    print("\n1. Testing /visualization/manifold_data...")
    try:
        res = client.get("/visualization/manifold_data?limit=10")
        if res.status_code == 200:
            data = res.json()
            print(f"✅ Status 200 OK")
            print(f"Count: {data.get('count')}")
            points = data.get('points', [])
            if points:
                p = points[0]
                print(f"Sample Point: x={p.get('x')}, y={p.get('y')}, z={p.get('z')}")
                if 'x' in p and 'y' in p and 'z' in p:
                    print("✅ 3D Coordinates present")
                else:
                    print("❌ Missing coordinates")
            else:
                print("⚠️ No points returned (System might be empty, but endpoint works)")
        else:
            print(f"❌ Error: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 2. Test Evolution Stats
    print("\n2. Testing /visualization/evolution_stats...")
    try:
        res = client.get("/visualization/evolution_stats")
        if res.status_code == 200:
            data = res.json()
            print(f"✅ Status 200 OK")
            history = data.get('history', [])
            print(f"History Points: {len(history)}")
            if history:
                print(f"Latest Metric: {history[-1]}")
        else:
            print(f"❌ Error: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_endpoints()
