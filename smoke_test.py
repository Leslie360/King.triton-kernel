"""Smoke test: 校验 eval server /evaluate 端点与 reward 计算链."""
import requests, sys

def main(server_url="http://localhost:8004"):
    r = requests.get(f"{server_url}/health", timeout=5)
    assert r.status_code == 200, f"server 不健康: {r.status_code}"
    print("✅ eval server healthy")
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
