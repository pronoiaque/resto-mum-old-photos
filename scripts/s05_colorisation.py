"""Étape 5 : colorisation IA des tirages N&B (classe 'nb'), depuis photo_YY_restaure.png.
Backend : ai.colorisation (Space HF ou commande locale). Résultat toujours marqué IA.
Usage : python s05_colorisation.py --pages all [--force]"""
import argparse, time
from common import journal, page_dir, photos_of, read_json, run_ai, select_pages, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--pages", default="all"); ap.add_argument("--force", action="store_true")
a = ap.parse_args()
for page in select_pages(a.pages):
    d = page_dir(page)
    for ph in photos_of(page):
        base = f"photo_{ph['photo']:02d}"
        meta = read_json(d / f"{base}.json", {})
        src, out = d / f"{base}_restaure.png", d / f"{base}_couleur.png"
        if meta.get("classe") != "nb" or not src.exists() or (out.exists() and not a.force):
            continue
        t = time.time()
        prompt = (meta.get("description_ia") or "Old family photograph") + \
            f" Natural, realistic {meta.get('periode_estimee', 'period')} color photograph."
        ok, voie, rem = run_ai("colorisation", src, out, extra={"prompt": prompt})
        journal(page, ph["photo"], "s05_colorisation", "colorisation", voie,
                duree=time.time() - t, statut="ok" if ok else "echec", remarque=rem)
        if ok:
            meta.update(colorisation_ia=True, colorisation_outil=rem)
            write_json(d / f"{base}.json", meta)
        print(f"page {page:02d} {base} : {'ok' if ok else 'échec'} {rem[:100]}")
