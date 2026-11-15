"""Colorisation guidée par Gemini SANS génération d'image (contourne le blocage
« Image generation is not available in your country ») :
  1. Gemini (sortie TEXTE) segmente la photo et propose une couleur réaliste par zone
     -> JSON [{label, box_2d, mask (PNG base64), color "#RRGGBB"}]
  2. En local : carte de chrominance (a,b) peinte depuis les masques, affinée par un
     filtre guidé sur la luminance (colle aux contours), fusion Lab avec la luminance
     d'origine intacte -> aucun détail inventé, seule la teinte est ajoutée.
Sortie : photo_YY_couleur_guidee.png (+ _couleur_zones.json, _couleur_zones.jpg contrôle)
Usage : python colorisation_guidee.py --pages 2 [--photos 1] [--force]"""
import argparse, base64, io, time
import cv2
import numpy as np
from PIL import Image

import gemini
from common import C, journal, page_dir, photos_of, read_json, select_pages, write_json

CONSIGNE = """This is an old black-and-white photograph. Plan a realistic, period-accurate
colorization WITHOUT producing an image. Split the picture into its meaningful color regions
(each person's skin, hair, each garment, car body, windows, tyres, building, sky, foliage,
tree trunks, ground...; at most 20 regions; skip regions smaller than 0.5% of the picture).
Return ONLY a JSON list; each entry: "label" (short), "polygon" = outline of the region as
8 to 20 points [x, y] normalized 0-1000 (x right, y down) over the whole picture, and
"color" = the most plausible original color as "#RRGGBB" (natural, slightly muted,
era-consistent; realistic skin tones; "#808080" for regions that should stay neutral).
List large background regions first and small details (faces, hands) last."""


def guided(I, p, r, eps):
    """Filtre guidé (He et al.) : I guide (0-1), p à filtrer."""
    f = lambda x: cv2.boxFilter(x, -1, (r, r))
    mI, mp = f(I), f(p)
    a = (f(I * p) - mI * mp) / (f(I * I) - mI * mI + eps)
    b = mp - a * mI
    return f(a) * I + f(b)


def hex_ab(h):
    h = h.lstrip("#")
    rgb = np.uint8([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]])  # BGR
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return lab[1] - 128, lab[2] - 128


def decoder_masque(z, w, h):
    if "polygon" in z:
        m = np.zeros((h, w), np.float32)
        pts = np.array([[x * w / 1000, y * h / 1000] for x, y in z["polygon"]], np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1.0)
        return m
    y0, x0, y1, x1 = [int(v) for v in z["box_2d"]]
    X0, Y0, X1, Y1 = x0 * w // 1000, y0 * h // 1000, max(x1 * w // 1000, x0 * w // 1000 + 1), max(y1 * h // 1000, y0 * h // 1000 + 1)
    m = np.zeros((h, w), np.float32)
    s = z.get("mask", "")
    try:
        s = s.split(",", 1)[1] if s.startswith("data:") else s
        mk = np.array(Image.open(io.BytesIO(base64.b64decode(s))).convert("L"), np.float32) / 255
        m[Y0:Y1, X0:X1] = cv2.resize(mk, (X1 - X0, Y1 - Y0))
    except Exception:
        m[Y0:Y1, X0:X1] = 1.0                       # repli : boîte entière
    return m


def coloriser(src, zones):
    img = cv2.imread(str(src))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    A = np.zeros((h, w), np.float32); B = np.zeros((h, w), np.float32); Wt = np.zeros((h, w), np.float32)
    # grandes zones d'abord, les petites (visages, détails) écrasent
    for z in zones:                    # ordre de Gemini : grandes zones d'abord, détails ensuite
        m = decoder_masque(z, w, h)
        a, b = hex_ab(z.get("color", "#808080"))
        sel = m > 0.5
        A[sel], B[sel], Wt[sel] = a, b, 1.0
    I = g.astype(np.float32) / 255
    r = max(8, int(max(h, w) * 0.01)) | 1
    A = guided(I, cv2.GaussianBlur(A, (0, 0), r / 2), r, 1e-3)
    B = guided(I, cv2.GaussianBlur(B, (0, 0), r / 2), r, 1e-3)
    # moins de chroma dans les noirs profonds et les blancs purs
    att = np.clip(np.minimum(I / 0.15, (1 - I) / 0.08), 0, 1)
    lab = cv2.merge([g, np.clip(A * att + 128, 0, 255).astype(np.uint8),
                     np.clip(B * att + 128, 0, 255).astype(np.uint8)])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def controle(src, zones, out):
    img = cv2.imread(str(src)); h, w = img.shape[:2]
    for z in zones:
        if "polygon" in z:
            pts = np.array([[x * w / 1000, y * h / 1000] for x, y in z["polygon"]], np.int32)
            x0, y0 = (pts.min(0) * 1000 / [w, h]).astype(int); x1, y1 = (pts.max(0) * 1000 / [w, h]).astype(int)
        else:
            y0, x0, y1, x1 = z["box_2d"]
        hx = z.get("color", "#808080").lstrip("#")
        c = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
        p0, p1 = (x0 * w // 1000, y0 * h // 1000), (x1 * w // 1000, y1 * h // 1000)
        if "polygon" in z:
            cv2.polylines(img, [pts], True, c, max(2, w // 400))
        else:
            cv2.rectangle(img, p0, p1, c, max(2, w // 400))
        cv2.putText(img, z.get("label", "")[:20], (p0[0] + 4, p0[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, w / 1800, c, 2)
    k = 1400 / max(h, w)
    cv2.imwrite(str(out), cv2.resize(img, None, fx=k, fy=k), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="all"); ap.add_argument("--photos", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sel = {int(x) for x in a.photos.split(",") if x}
    for page in select_pages(a.pages):
        d = page_dir(page)
        for ph in photos_of(page):
            n = ph["photo"]; base = f"photo_{n:02d}"
            if sel and n not in sel:
                continue
            meta = read_json(d / f"{base}.json", {})
            src, out = d / f"{base}_restaure.png", d / f"{base}_couleur_guidee.png"
            if meta.get("classe") != "nb" or not src.exists() or (out.exists() and not a.force):
                continue
            zf = d / f"{base}_couleur_zones.json"
            t = time.time()
            zones = read_json(zf)
            if zones is None:
                zones = gemini.appeler(C["gemini"]["modele_vision"], CONSIGNE, [src], json_schema=True,
                                       objet=f"zones couleur p{page} {base}", max_px=1024, max_tokens=12000)
                write_json(zf, zones)
            res = coloriser(src, zones)
            cv2.imwrite(str(out), res)
            controle(src, zones, d / f"{base}_couleur_zones.jpg")
            meta.update(colorisation_ia=True, colorisation_outil=f"guidée {C['gemini']['modele_vision']} + local")
            write_json(d / f"{base}.json", meta)
            journal(page, n, "colorisation_guidee", C["gemini"]["modele_vision"], "en_ligne",
                    duree=time.time() - t, remarque=f"{len(zones)} zones")
            print(f"page {page:02d} {base} : {len(zones)} zones -> {out.name}")
