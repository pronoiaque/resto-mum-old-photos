"""Teste chaque backend IA configuré (ai.*) sur une petite image de test.
Usage : python tester_ai.py [upscale visages colorisation ...]"""
import sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from common import C, run_ai

tmp = Path(tempfile.mkdtemp())
im = Image.new("L", (256, 256), 128); dr = ImageDraw.Draw(im)
dr.ellipse((70, 50, 186, 190), fill=200); dr.ellipse((100, 100, 112, 112), fill=30)
dr.ellipse((144, 100, 156, 112), fill=30); dr.arc((105, 130, 150, 160), 20, 160, fill=40, width=3)
im.convert("RGB").save(tmp / "test.png")
kinds = sys.argv[1:] or [k for k in C["ai"] if k != "fusion_generative"]
for k in kinds:
    ok, voie, rem = run_ai(k, tmp / "test.png", tmp / f"{k}.png")
    print(f"{k:14s} {'OK ' if ok else 'NON'} {voie:9s} {rem[:150]}")
print("sorties :", tmp)
