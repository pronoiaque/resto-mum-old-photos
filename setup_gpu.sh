#!/usr/bin/env bash
# Installation GPU testée le 2026-11-15 sur pcob (GTX 960 2 Go, pilote 570, CUDA 12.8).
# Principe : RÉUTILISER le torch CUDA déjà installé pour python3.12 (ne pas le réinstaller),
# via un venv --system-site-packages ; n'ajouter que ce qui manque.
# Usage : bash setup_gpu.sh            (puis : bash setup_gpu.sh --test)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3.12}

if [[ "${1:-}" != "--test" ]]; then
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch CUDA absent pour ce python"; print("torch système :", torch.__version__)'
  rm -rf venv
  "$PY" -m venv --system-site-packages venv
  venv/bin/pip -q install --upgrade pip
  grep -vE '^(torch|ultralytics)' requirements.txt > /tmp/req_sans_torch.txt
  venv/bin/pip -q install -r /tmp/req_sans_torch.txt "opencv-python<5" "opencv-python-headless<5"
  # torchvision assorti au torch système, SANS dépendances (sinon pip remplace torch)
  TV=$("$PY" - <<'PY'
import torch
maj, mineur = map(int, torch.__version__.split("+")[0].split(".")[:2])
print(f"0.{mineur + 15}")          # règle torch 2.N <-> torchvision 0.(N+15)
PY
)
  CU=$("$PY" -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))')
  venv/bin/pip -q install --no-deps "torchvision==${TV}.*" --index-url "https://download.pytorch.org/whl/${CU}"
  venv/bin/pip -q install --no-deps ultralytics ultralytics-thop
  venv/bin/pip -q install polars py-cpuinfo psutil matplotlib pandas tqdm requests scipy
  # Python compilé sans liblzma ? -> module de remplacement (torchvision l'importe au démarrage)
  if ! "$PY" -c 'import lzma' 2>/dev/null; then
    SP=$(venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    cp outils/_lzma_stub.py "$SP/_lzma.py"
    echo "lzma absent : remplaçant installé (correctif propre : sudo apt install liblzma-dev + recompiler Python)"
  fi
fi

echo "--- test"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venv/bin/python - <<'PY'
import torch, torchvision, cv2, ultralytics
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("opencv", cv2.__version__, "| CascadeClassifier", hasattr(cv2, "CascadeClassifier"), "| ultralytics", ultralytics.__version__)
x = torch.randn(1000, 1000, device="cuda"); print("calcul GPU OK", float((x @ x).abs().sum()) > 0)
PY
echo "Lancer le découpage : PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venv/bin/python scripts/decoupe_local.py"
