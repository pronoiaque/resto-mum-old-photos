"""Aide à la saisie visuelle des boîtes par l'agent : aperçu quadrillé (repère "apercu").
Usage : python outil_grille.py <page> [pas=100]
Produit page_XX/apercu_grille.jpg : aperçu (grand côté = apercu_px) avec graduations
tous les `pas` px et, en surimpression, les boîtes de boites_detectees.json ou
boites_manuelles.json. L'agent lit les coins sur cette image et écrit
boites_manuelles.json avec "repere": "apercu" (voir PROMPT.md)."""
import sys
import cv2
import numpy as np
from common import C, page_dir, read_json

page = int(sys.argv[1]); pas = int(sys.argv[2]) if len(sys.argv) > 2 else 100
d = page_dir(page)
im = cv2.imread(str(d / "source_preview.jpg"))
h, w = im.shape[:2]
img = cv2.imread(str(d / "source.png"), cv2.IMREAD_REDUCED_COLOR_2)
k = C["apercu_px"] / (max(img.shape[:2]) * 2)
for x in range(0, w, pas):
    cv2.line(im, (x, 0), (x, h), (255, 0, 255), 1)
    cv2.putText(im, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
for y in range(0, h, pas):
    cv2.line(im, (0, y), (w, y), (255, 0, 255), 1)
    cv2.putText(im, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
manu = read_json(d / "boites_manuelles.json")
boites = manu or read_json(d / "boites_detectees.json", [])
for b in boites:
    q = np.array(b["quad"], np.float32)
    if not (manu and b.get("repere") == "apercu"):
        q = q * k
    cv2.polylines(im, [q.astype(np.int32)], True, (0, 200, 0), 2)
    c = q.mean(0).astype(int)
    cv2.putText(im, str(b.get("id", "?")), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite(str(d / "apercu_grille.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(d / "apercu_grille.jpg")
