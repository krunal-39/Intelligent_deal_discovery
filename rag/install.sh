#!/usr/bin/env bash
set -e

# NOTE: faiss-gpu wheels are sensitive to CUDA / driver versions.
# If faiss-gpu install fails, the script will fall back to faiss-cpu.
# You may need to install faiss from conda/pip matching your CUDA version.

echo "Installing Python packages..."

# Core (CPU) packages
pip install -U pip
pip install -U sentence-transformers transformers tqdm numpy pandas joblib

# Light on installs for torch: user already has torch in env; otherwise uncomment:
# pip install torch --extra-index-url https://download.pytorch.org/whl/cu121

# Try faiss-gpu first (may fail depending on CUDA / wheel availability)
python - <<'PY'
import subprocess, sys
try:
    print("Trying to install faiss-gpu (fast, uses your GPU)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    print("Installed faiss-gpu")
except Exception as e:
    print("faiss-gpu install failed — falling back to faiss-cpu. Error:", e)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    print("Installed faiss-cpu")
PY

echo "Done. If you want faiss-gpu but install failed, install a wheel matching your CUDA version (or use conda)."
