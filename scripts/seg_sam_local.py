"""Segmentation automatique des tirages avec SAM (transformers), GPU recommandé.
Contrat : python seg_sam_local.py <image.png> <sortie.json>
Écrit une liste de quadrilatères [[x,y]x4] en pixels de l'image d'entrée.

Config :  ai.segmentation: {backend: cmd,
            cmd: "python scripts/seg_sam_local.py {in} {out}"}
Variables : SAM_MODELE (défaut facebook/sam-vit-huge), SAM_PX (défaut 1024)."""
import json, os, sys
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

src, dst = sys.argv[1], sys.argv[2]
modele = os.environ.get("SAM_MODELE", "facebook/sam-vit-huge")
px = int(os.environ.get("SAM_PX", 1024))
dev = 0 if torch.cuda.is_available() else -1

im = Image.open(src).convert("RGB")
W, H = im.size
k = px / max(W, H)
sm = im.resize((int(W * k), int(H * k)))
gen = pipeline("mask-generation", model=modele, device=dev)
res = gen(sm, points_per_batch=64, pred_iou_thresh=0.9, stability_score_thresh=0.92)

h, w = sm.size[1], sm.size[0]
cands = []
for m in res["masks"]:
    m = np.asarray(m, np.uint8)
    a = m.sum()
    if not (0.004 * h * w < a < 0.9 * h * w):
        continue
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    ra = rect[1][0] * rect[1][1]
    if ra <= 0 or cv2.contourArea(c) / ra < 0.8:      # un tirage est (presque) rectangulaire
        continue
    cands.append((a, cv2.boxPoints(rect) / k, m))

# suppression des masques contenus dans un plus grand (détails intérieurs d'une photo)
cands.sort(key=lambda x: -x[0])
garde = []
for a, q, m in cands:
    if any((m & g[2]).sum() / a > 0.8 for g in garde):
        continue
    garde.append((a, q, m))
json.dump([q.round(1).tolist() for _, q, _ in garde], open(dst, "w"))
print(f"{len(garde)} tirages")
