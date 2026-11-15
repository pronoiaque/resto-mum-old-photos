"""Étape 7 : JSON par photo, JPEG de partage, métadonnées XMP (exiftool).
Version de partage : couleur (si ai validé : partage.depuis=couleur) sinon restaurée.
Usage : python s07_metadonnees.py --pages all [--depuis restaure|couleur|visages]"""
import argparse, shutil, subprocess
from pathlib import Path
from PIL import Image

from common import C, journal, page_dir, photos_of, read_json, select_pages, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--pages", default="all")
ap.add_argument("--depuis", default="restaure", choices=["restaure", "couleur", "visages"])
a = ap.parse_args()
EXIF = shutil.which("exiftool")
if not EXIF:
    print("ATTENTION : exiftool absent -> JSON et JPEG seulement (sudo apt install libimage-exiftool-perl)")
idx = {}
import csv
with open(C["out_dir"] / "index_pages.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["page"]:
            idx[int(r["page"])] = Path(r["fichier"]).name

for page in select_pages(a.pages):
    d = page_dir(page)
    assoc = read_json(d / "association.json", [])
    titre_page = " / ".join(b["texte"] for b in assoc if b.get("cible") == "page")
    for ph in photos_of(page):
        base = f"photo_{ph['photo']:02d}"
        meta = read_json(d / f"{base}.json", {})
        if meta.get("role") == "doublon":
            continue
        textes = [b for b in assoc if b.get("cible") == base]
        sure = [b["texte"] for b in textes if b.get("confiance") != "faible"
                and b.get("confiance_association") != "faible"]
        meta.update(textes=textes, titre_page=titre_page)
        choix = {"restaure": f"{base}_restaure.png", "couleur": f"{base}_couleur.png",
                 "visages": f"{base}_restaure_visages.png"}[a.depuis]
        src = d / choix if (d / choix).exists() else d / f"{base}_restaure.png"
        if not src.exists():
            continue
        ia = "couleur" in src.name or "visages" in src.name or "fusion_ia" in meta.get("entree", "")
        im = Image.open(src).convert("RGB")
        im.thumbnail((C["restauration"]["partage_px"],) * 2)
        jpg = d / f"{base}_partage.jpg"
        im.save(jpg, quality=90)
        meta.update(partage=jpg.name, partage_depuis=src.name, contient_ia_generative=ia)
        write_json(d / f"{base}.json", meta)
        if EXIF:
            desc = " | ".join(sure) + (" [colorisation IA]" if "couleur" in src.name else "") \
                + (" [retouche IA]" if ia and "couleur" not in src.name else "")
            args = [EXIF, "-overwrite_original", "-charset", "utf8",
                    f"-XMP-dc:Title=Album page {page:02d} photo {ph['photo']:02d}",
                    f"-XMP-dc:Description={desc.strip(' |')}",
                    f"-XMP-dc:Source=Album, page {page:02d} ({idx.get(page,'')}), photo {ph['photo']:02d}"]
            if titre_page:
                args.append(f"-XMP-dc:Subject={titre_page}")
            if ia:
                args.append("-XMP-iptcExt:DigitalSourceType="
                            "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia")
            cibles = [str(jpg)] + [str(p) for p in d.glob(f"{base}_*.png")
                                   if not p.name.endswith("_raw.png")]
            r = subprocess.run(args + cibles, capture_output=True, text=True)
            journal(page, ph["photo"], "s07_metadonnees", "exiftool",
                    statut="ok" if r.returncode == 0 else "echec", remarque=r.stderr[-200:])
        print(f"page {page:02d} {base} -> {jpg.name} ({src.name})")
