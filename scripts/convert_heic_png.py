"""Conversion par lot HEIC -> PNG en résolution native.
- Originaux en lecture seule ; orientation EXIF appliquée ; profil ICC et EXIF conservés.
- Idempotent (saute les PNG déjà présents, --force pour refaire), parallèle.
Usage : python convert_heic_png.py [--src ~/albums] [--dst ~/albums/pngnatives] [--jobs 4] [--force]"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()


def convertir(src, dst, force):
    out = dst / (src.stem + ".png")
    if out.exists() and not force:
        return src.name, "déjà fait", 0.0
    t = time.time()
    with Image.open(src) as im:
        exif = im.getexif()
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
        exif[0x0112] = 1                           # orientation déjà appliquée
        tmp = out.with_suffix(".png.part")
        im.save(tmp, "PNG", compress_level=6, exif=exif.tobytes(), icc_profile=icc)
        tmp.replace(out)                           # écriture atomique
        taille = f"{im.width}x{im.height}"
    return src.name, taille, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/albums")
    ap.add_argument("--dst", default="~/albums/pngnatives")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    fichiers = sorted(p for p in src.iterdir() if p.suffix.lower() in (".heic", ".heif"))
    print(f"{len(fichiers)} HEIC -> {dst} ({a.jobs} processus)")
    t0, erreurs = time.time(), 0
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(convertir, f, dst, a.force): f for f in fichiers}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                nom, info, d = fu.result()
                print(f"[{i:3d}/{len(fichiers)}] {nom} : {info} ({d:.1f} s)", flush=True)
            except Exception as e:
                erreurs += 1
                print(f"[{i:3d}/{len(fichiers)}] {futs[fu].name} : ERREUR {e}", flush=True)
    print(f"Terminé en {time.time() - t0:.0f} s, {erreurs} erreur(s).")
    sys.exit(1 if erreurs else 0)


if __name__ == "__main__":
    main()
