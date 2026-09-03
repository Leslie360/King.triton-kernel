#!/bin/bash
# launch_local.sh
#
# One-click local smoke environment for King.triton-kernel.
#
# Starts the full grading chain on a single machine with a single GPU:
#   1. a local Redis instance (used by the eval server for task/result state)
#   2. the KernelGym grading API server (kernelgym-server)
#   3. one GPU worker (kernelgym-single-worker on cuda:0)
#
# It then prints the server URL you can hand to smoke_test.py.
#
# Usage:
#   bash scripts/launch_local.sh
#
# Environment overrides (all optional):
#   API_PORT   eval server port          (default 10907)
#   REDIS_PORT redis port                (default 6379)
#   NO_GPU     set to 1 to skip the worker and print a clear message
#   N          0 disables nohup backgrounding (foreground, for debugging)
#
# On exit it kills the processes it started (redis, server, worker) unless you
# pass NO_CLEANUP=1 to keep them running.
set -euo pipefail

# --- Resolve repo root so this script works from any cwd ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Config ----------------------------------------------------------------
API_PORT="${API_PORT:-10907}"
REDIS_PORT="${REDIS_PORT:-6379}"
NO_GPU="${NO_GPU:-0}"
NO_CLEANUP="${NO_CLEANUP:-0}"
FOREGROUND="${FOREGROUND:-0}"   # set to 1 to run in foreground (no nohup)

SERVER_URL="http://localhost:${API_PORT}"

# Track the PIDs we start so we can clean them up on exit.
STARTED_PIDS=""
# Set to 1 if we reused an already-running redis (so cleanup must NOT kill it).
OWN_REDIS=0

cleanup() {
    if [ "${NO_CLEANUP}" = "1" ]; then
        echo
        echo "[launch_local] NO_CLEANUP=1, leaving processes running."
        echo "[launch_local] Server:      ${SERVER_URL}"
        echo "[launch_local] To stop:     pkill -f 'kernelgym' ; pkill redis-server"
        return
    fi
    echo
    echo "[launch_local] Cleaning up (server/worker/redis)..."
    # Kill in reverse start order: worker, server.
    for pid in $(echo "${STARTED_PIDS}" | tr ' ' '\n' | tac); do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Safety net for anything that did not die.
    pkill -f "kernelgym.server.api.server:app" 2>/dev/null || true
    pkill -f "kernelgym.worker.single_worker" 2>/dev/null || true
    # Redis: only shut down the instance we started ourselves (never a shared
    # redis that was already running on REDIS_PORT before we launched).
    if [ "${OWN_REDIS}" = "1" ]; then
        redis-cli -p "${REDIS_PORT}" shutdown nosave >/dev/null 2>&1 || true
    fi
    echo "[launch_local] Cleanup complete."
}
trap cleanup EXIT INT TERM

echo "=================================================="
echo "[launch_local] King.triton-kernel one-click local smoke"
echo "=================================================="

# --- Preconditions ----------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "[launch_local] ERROR: python3 not found"; exit 1; }
python3 -c "import kernelgym, uvicorn, fastapi" 2>/dev/null \
    || { echo "[launch_local] ERROR: dependencies not installed. Run: bash setup.sh && pip install -e ."; exit 1; }
command -v redis-server >/dev/null 2>&1 \
    || { echo "[launch_local] ERROR: redis-server not found. Install Redis (apt-get install redis-server or equivalent)."; exit 1; }
command -v redis-cli >/dev/null 2>&1 \
    || { echo "[launch_local] ERROR: redis-cli not found (ships with redis-server)."; exit 1; }

# --- 1. Redis ---------------------------------------------------------------
# Redis is required by the eval server. We are collision-safe:
#   * If a healthy redis already answers PONG on REDIS_PORT, reuse it (and do
#     NOT kill it on cleanup — it may be a shared/global instance).
#   * Otherwise start our own redis, first on REDIS_PORT and, if that port is
#     taken by a non-redis process, auto-fall back to a free port.

redis_pong() {
    # Returns 0 iff redis-cli answers a clean "PONG" on the given port.
    local out
    out="$(redis-cli -p "$1" ping 2>/dev/null)"
    [ "$out" = "PONG" ]
}

if redis_pong "${REDIS_PORT}"; then
    echo "[launch_local] Reusing existing healthy redis on port ${REDIS_PORT}."
    OWN_REDIS=0
else
    # Try REDIS_PORT first; if it fails to come up, fall back to a free port.
    REDIS_DIR="$(mktemp -d)"
    REDIS_TRY_PORT="${REDIS_PORT}"
    STARTED=0
    for attempt in $(seq 1 10); do
        echo "[launch_local] Starting redis on port ${REDIS_TRY_PORT}..."
        if [ "${FOREGROUND}" = "1" ]; then
            redis-server --port "${REDIS_TRY_PORT}" --save "" --appendonly no \
                --dir "${REDIS_DIR}" &
        else
            nohup redis-server --port "${REDIS_TRY_PORT}" --save "" --appendonly no \
                --dir "${REDIS_DIR}" >"${REDIS_DIR}/redis.log" 2>&1 &
        fi
        REDIS_PID=$!

        # Bounded wait for this redis to answer a clean PONG.
        for i in $(seq 1 30); do
            if redis_pong "${REDIS_TRY_PORT}"; then
                STARTED=1
                break
            fi
            sleep 0.5
        done

        if [ "${STARTED}" = "1" ]; then
            break
        fi

        # Failed to come up on this port — kill the attempt and try the next.
        kill "${REDIS_PID}" 2>/dev/null || true
        REDIS_TRY_PORT=$((REDIS_TRY_PORT + 1))
    done

    if [ "${STARTED}" != "1" ]; then
        echo "[launch_local] ERROR: could not start redis on any port near ${REDIS_PORT}."
        exit 1
    fi

    REDIS_PORT="${REDIS_TRY_PORT}"
    OWN_REDIS=1
    STARTED_PIDS="${STARTED_PIDS} ${REDIS_PID}"
    echo "[launch_local] redis up on port ${REDIS_PORT} (pid ${REDIS_PID})."
fi

# --- 2. Eval server ---------------------------------------------------------
# Prefer the console-script entry point (installed by `pip install -e .`), but
# fall back to `python3 -m ...` so the script works even when the editable
# install's bin directory is not on PATH.
SERVER_CMD=(kernelgym-server)
if ! command -v kernelgym-server >/dev/null 2>&1; then
    SERVER_CMD=(python3 -m kernelgym.server.api.server)
fi
echo "[launch_local] Starting eval server on ${SERVER_URL} (${SERVER_CMD[*]})..."
export API_HOST="0.0.0.0"
export API_PORT
export REDIS_HOST="localhost"
export REDIS_PORT
if [ "${FOREGROUND}" = "1" ]; then
    "${SERVER_CMD[@]}" &
else
    nohup "${SERVER_CMD[@]}" >"${REPO_ROOT}/logs/launch_local_server.log" 2>&1 &
fi
SERVER_PID=$!
STARTED_PIDS="${STARTED_PIDS} ${SERVER_PID}"

# --- 3. GPU worker ----------------------------------------------------------
WORKER_CMD=(kernelgym-single-worker)
if ! command -v kernelgym-single-worker >/dev/null 2>&1; then
    WORKER_CMD=(python3 -m kernelgym.worker.single_worker)
fi
if [ "${NO_GPU}" = "1" ]; then
    echo "[launch_local] NO_GPU=1, skipping GPU worker."
    echo "[launch_local] WARNING: the server is up but no worker is registered,"
    echo "[launch_local] so POST /evaluate will block/queue forever. Run with a"
    echo "[launch_local] GPU (or set NO_GPU=0) to exercise the full grading chain."
else
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[launch_local] WARNING: no CUDA-visible GPU detected."
        echo "[launch_local] The server will still run for /health, but the"
        echo "[launch_local] grading chain (POST /evaluate) needs a real GPU."
        echo "[launch_local] Re-run with NO_GPU=1 to acknowledge and skip the worker."
    else
        echo "[launch_local] Starting GPU worker on cuda:0 (${WORKER_CMD[*]})..."
        if [ "${FOREGROUND}" = "1" ]; then
            "${WORKER_CMD[@]}" --worker-id worker-smoke-1 --device cuda:0 &
        else
            nohup "${WORKER_CMD[@]}" --worker-id worker-smoke-1 --device cuda:0 \
                >"${REPO_ROOT}/logs/launch_local_worker.log" 2>&1 &
        fi
        WORKER_PID=$!
        STARTED_PIDS="${STARTED_PIDS} ${WORKER_PID}"
        echo "[launch_local] GPU worker started (pid ${WORKER_PID})."
    fi
fi

# --- 4. Wait for the server to accept /health and report ---------------------
echo "[launch_local] Waiting for server to become healthy..."
for i in $(seq 1 60); do
    if python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('${SERVER_URL}/health', timeout=1).status==200 else 1)" 2>/dev/null; then
        echo "[launch_local] Server healthy at ${SERVER_URL}/health"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[launch_local] ERROR: server process exited. See logs/launch_local_server.log"
        exit 1
    fi
    sleep 0.5
done

echo
echo "=================================================="
echo "[launch_local] READY"
echo "  Eval server : ${SERVER_URL}"
echo "  Redis       : localhost:${REDIS_PORT}"
echo "  Next        : python3 smoke_test.py ${SERVER_URL}"
echo "  Stop        : Ctrl-C, or re-run with NO_CLEANUP=1 then pkill kernelgym"
echo "=================================================="

# Keep the script alive in foreground mode so traps fire on Ctrl-C. In nohup
# mode we sleep-wait on the server; if it dies unexpectedly we exit and clean up.
if [ "${FOREGROUND}" = "1" ]; then
    wait "${SERVER_PID}"
else
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
        sleep 5
    done
    echo "[launch_local] Server exited unexpectedly; cleaning up."
fi
