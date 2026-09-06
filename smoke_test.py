"""Smoke test for the King.triton-kernel eval server.

Two levels of checks:

1. Health check (always): GET /health must return 200.
2. Grading-chain check (POST /evaluate, only when a GPU is available): submit a
   minimal kernel task and assert the response carries the grading verdict
   fields ``compiled`` / ``correctness`` / ``speedup`` / ``error_code``. This
   exercises the real "execute on GPU, then grade" path — not just process
   liveness.

If no CUDA-visible GPU is present, the grading-chain check is skipped with a
clear message (the server can still serve /health, but it has no worker to run
grading tasks).

Usage:
    python3 smoke_test.py [server_url] [--evaluate|--no-evaluate]

    server_url   default http://localhost:10907
    --evaluate   (default) run the POST /evaluate grading-chain check
    --no-evaluate  skip the grading-chain check (health only)
"""

import argparse
import sys

import requests

DEFAULT_SERVER = "http://localhost:10907"

# A minimal kernelbench task. workflow="kernelbench" requires reference_code;
# kernel_code is the candidate Triton/CUDA implementation to grade. The values
# here are deliberately trivial (identity/relu-style) so the grading chain runs
# end to end; the exact correctness verdict is not what the smoke test asserts.
MINIMAL_TASK = {
    "task_id": "smoke_local_evaluate",
    "workflow": "kernelbench",
    "toolkit": "kernelbench",
    "backend_adapter": "kernelbench",
    "backend": "cuda",
    "reference_code": (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return torch.relu(x)\n"
    ),
    "kernel_code": (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class ModelNew(nn.Module):\n"
        "    def forward(self, x):\n"
        "        # Minimal candidate kernel: compiled and run on a real GPU.\n"
        "        return torch.relu(x)\n"
    ),
    "entry_point": "Model",
    "num_correct_trials": 1,
    "num_perf_trials": 1,
    "num_warmup": 0,
    "timeout": 120,
    "priority": "high",
    "force_refresh": True,
}

# Grading verdict fields that must be present in a real /evaluate response.
VERDICT_FIELDS = ("compiled", "correctness", "speedup", "error_code")


def _gpu_available() -> bool:
    """Return True if a CUDA-visible GPU is present."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        # torch import failure (e.g. CPU-only build) counts as "no GPU".
        return False


def check_health(server_url: str) -> None:
    r = requests.get(f"{server_url}/health", timeout=5)
    assert r.status_code == 200, f"server unhealthy: HTTP {r.status_code}"
    body = r.json()
    assert "status" in body or isinstance(body, dict), f"unexpected health body: {body}"
    print("ok: GET /health returned 200")


def check_evaluate(server_url: str) -> None:
    """Submit a minimal kernel and assert the grading verdict fields appear."""
    task = dict(MINIMAL_TASK)
    task["task_id"] = "smoke_local_evaluate"
    r = requests.post(f"{server_url}/evaluate", json=task, timeout=180)
    assert r.status_code == 200, f"/evaluate failed: HTTP {r.status_code} {r.text}"
    body = r.json()

    missing = [f for f in VERDICT_FIELDS if f not in body]
    assert not missing, f"/evaluate response missing grading fields {missing}; got keys={sorted(body.keys())}"

    # Print the verdict values for visibility, but do not assert on them — a
    # correct or incorrect kernel both prove the grading chain ran end to end.
    print(
        f"ok: POST /evaluate returned grading verdict "
        f"(compiled={body.get('compiled')}, "
        f"correctness={body.get('correctness')}, "
        f"speedup={body.get('speedup')}, "
        f"error_code={body.get('error_code')})"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_url", nargs="?", default=DEFAULT_SERVER)
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=True,
        help="run the POST /evaluate grading-chain check (default)",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="skip the grading-chain check (health only)",
    )
    args = parser.parse_args(argv)

    server_url = args.server_url.rstrip("/")

    print(f"smoke_test: server_url={server_url}")
    check_health(server_url)

    if not args.evaluate:
        print("smoke_test: --no-evaluate given, skipping grading-chain check.")
        return 0

    if not _gpu_available():
        print(
            "smoke_test: NO CUDA-VISIBLE GPU DETECTED. "
            "Skipping the POST /evaluate grading-chain check — it requires a "
            "real GPU and a registered worker. Health check passed. "
            "To run the full grading chain, start with scripts/launch_local.sh "
            "on a machine with an NVIDIA GPU."
        )
        return 0

    check_evaluate(server_url)
    print("smoke_test: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
