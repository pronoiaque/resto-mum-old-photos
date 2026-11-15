#!/usr/bin/env bash
# Installation : bash setup.sh [--gpu]
set -euo pipefail
cd "$(dirname "$0")"
# Python >= 3.10 requis (client Gradio récent pour les Spaces). uv l'installe si besoin.
if command -v uv >/dev/null; then
  uv venv -q --python 3.12 venv
  VIRTUAL_ENV=venv uv pip install -q -r requirements.txt
  if [[ "${1:-}" == "--gpu" ]]; then VIRTUAL_ENV=venv uv pip install -q -r requirements-gpu.txt; fi
else
  python3 -c 'import sys; assert sys.version_info >= (3,10), "Python >= 3.10 requis (ou installer uv)"'
  python3 -m venv venv
  venv/bin/pip install -q --upgrade pip -r requirements.txt
  if [[ "${1:-}" == "--gpu" ]]; then venv/bin/pip install -q -r requirements-gpu.txt; fi
fi
echo "--- vérifications"
venv/bin/python -c "import cv2, pillow_heif, imagehash, yaml, gradio_client; print('python OK, OpenCV', cv2.__version__)"
for t in exiftool heif-convert nvidia-smi realesrgan-ncnn-vulkan; do
  if command -v "$t" >/dev/null; then echo "$t : présent"; else echo "$t : ABSENT"; fi
done
echo "exiftool absent ? -> sudo apt install libimage-exiftool-perl (avec accord)"
echo "Ensuite : éditer config.yaml (source_dir, out_dir, backends ai.*), puis lire PROMPT.md"
