"""Étape 0 : inventaire des sources, dédoublonnage par empreinte, conversion, diagnostic.
Usage : python s00_inventaire.py [--force]"""
import argparse, csv, fnmatch, time, zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif

from common import C, OUT, journal, page_dir, sha256

pillow_heif.register_heif_opener()
EXT = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def collecter():
    src = C["source_dir"]
    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in EXT)
    extrait = OUT / "_zip_extrait"
    for z in sorted(src.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if Path(n).suffix.lower() in EXT and not n.endswith("/"):
                    dst = extrait / z.stem / Path(n).name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(zf.read(n))
                    files.append(dst)
    return files


def diagnostic(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (1000, int(1000 * g.shape[0] / g.shape[1])))
    nette = cv2.Laplacian(s, cv2.CV_64F).var()
    lum = float(s.mean())
    sature = float((s > 250).mean())
    fond = cv2.GaussianBlur(s, (0, 0), 60)
    ombre = float(np.percentile(fond, 95) - np.percentile(fond, 5))
    return dict(nettete=round(nette, 1), luminosite=round(lum, 1),
                reflets_pct=round(100 * sature, 2), ecart_eclairage=round(ombre, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vus, lignes, page = {}, [], 0
    for f in collecter():
        h = sha256(f)
        exclu = any(fnmatch.fnmatch(f.name, m) for m in C.get("exclure", []))
        if h in vus:
            lignes.append(dict(page="", fichier=str(f), sha256=h, retenu="n",
                               motif=f"identique à {vus[h]}"))
            continue
        vus[h] = f.name
        if exclu:
            lignes.append(dict(page="", fichier=str(f), sha256=h, retenu="n", motif="exclu (config)"))
            continue
        page += 1
        d = page_dir(page)
        t = time.time()
        src_png = d / "source.png"
        if a.force or not src_png.exists():
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            im.save(src_png, compress_level=3)
            pv = im.copy(); pv.thumbnail((C["apercu_px"],) * 2)
            pv.save(d / "source_preview.jpg", quality=85)
        img = cv2.imread(str(src_png))
        dg = diagnostic(img)
        lignes.append(dict(page=page, fichier=str(f), sha256=h, retenu="o", motif="",
                           largeur=img.shape[1], hauteur=img.shape[0], **dg))
        journal(page, "", "s00_conversion", "pillow-heif", parametres=dg, duree=time.time() - t)
        print(f"page {page:02d} <- {f.name} {img.shape[1]}x{img.shape[0]} {dg}")
    cols = ["page", "fichier", "sha256", "retenu", "motif", "largeur", "hauteur",
            "nettete", "luminosite", "reflets_pct", "ecart_eclairage", "type_prise"]
    with open(OUT / "index_pages.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, cols, extrasaction="ignore")
        w.writeheader(); w.writerows(lignes)
    n_ex = sum(1 for l in lignes if l["retenu"] == "n")
    print(f"\n{page} pages retenues, {n_ex} fichiers écartés -> {OUT/'index_pages.csv'}")
    print("À FAIRE (agent) : regarder les aperçus, compléter 'type_prise', écarter les hors-sujet.")


if __name__ == "__main__":
    main()
