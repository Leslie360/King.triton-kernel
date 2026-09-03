"""Smoke test: check eval server /health endpoint."""
import requests, sys

def main(server_url="http://localhost:10907"):
    r = requests.get(f"{server_url}/health", timeout=5)
    assert r.status_code == 200, f"server unhealthy: {r.status_code}"
    print("✅ eval server healthy")
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
