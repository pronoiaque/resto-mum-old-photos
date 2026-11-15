"""Fusion GÉNÉRATIVE multi-prises (optionnelle) : envoie toutes les prises d'un même
tirage à un modèle image multimodal (Gemini par défaut) pour obtenir une seule image
« scan propre ». Résultat marqué IA (peut inventer des détails : à contrôler).
Modèle : gemini.modele_image (config.yaml) ; clé : voir scripts/gemini.py.
Sortie : page_<ref>/photo_YY_fusion_ia.png
Usage : python fusion_generative.py [--groupes 3,7] [--force] [--aussi-uniques]"""
import argparse, os, time
from io import BytesIO
from PIL import Image

from common import C, OUT, journal, page_dir, photos_of, read_json, write_json

G = C["gemini"]
CONSIGNE = """These {n} images are smartphone photos of ONE AND THE SAME physical
analog photographic print, taken from different angles, lighting and framing (some may be
partial, blurry, or show glare or shadows).
Produce ONE image: the most faithful possible reproduction of this print, as if it had
been scanned flat on a good-quality scanner.
Strict rules:
- Combine information from all shots: for each area keep the sharpest, glare-free version;
  remove glare, cast shadows, motion blur and perspective distortion.
- Frame exactly the print (borders included), straightened, without the album background.
- Invent nothing: no faces, objects or text; do not change expressions, do not smooth skin,
  keep the film grain and the period look.
- MAIN GOAL: then COLORIZE it with realistic, natural, period-accurate colors (original tones: {teinte});
  change ONLY the colors.
- Do not retouch scratches or missing parts of the print.
- Output at the highest resolution possible."""


def appeler(images, teinte, objet=""):
    import gemini
    return gemini.appeler(C["gemini"]["modele_image"], CONSIGNE.format(n=len(images), teinte=teinte),
                          images, genre="image", objet=objet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groupes", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aussi-uniques", action="store_true",
                    help="traiter aussi les tirages à une seule prise")
    a = ap.parse_args()
    sel = {int(x) for x in a.groupes.split(",") if x}
    for g in read_json(OUT / "groupes.json", []):
        if sel and g["groupe"] not in sel:
            continue
        if len(g["membres"]) < 2 and not a.aussi_uniques:
            continue
        r = g["reference"]
        out = page_dir(r["page"]) / f"photo_{r['photo']:02d}_fusion_ia.png"
        if out.exists() and not a.force:
            continue
        ims = []
        for m in g["membres"]:
            ph = next(p for p in photos_of(m["page"]) if p["photo"] == m["photo"])
            im = Image.open(page_dir(m["page"]) / ph["fichier"]).convert("RGB")
            im.thumbnail((2048, 2048)); ims.append(im)
        meta = read_json(page_dir(r["page"]) / f"photo_{r['photo']:02d}.json", {})
        teinte = "black & white / sepia" if meta.get("classe", "nb") == "nb" else "period color"
        t = time.time()
        try:
            res = appeler(ims, teinte, objet=f"fusion groupe {g['groupe']}")
            res.save(out)
            write_json(out.with_suffix(".json"), dict(modele=G["modele_image"], prises=g["membres"],
                                                      ia_generative=True))
            journal(r["page"], r["photo"], "fusion_generative", G["modele_image"], "en_ligne",
                    duree=time.time() - t, remarque=f"{len(ims)} prises")
            print("ok", out)
        except Exception as e:
            journal(r["page"], r["photo"], "fusion_generative", G["modele_image"], "en_ligne",
                    duree=time.time() - t, statut="echec", remarque=repr(e)[:300])
            print("échec groupe", g["groupe"], e)


if __name__ == "__main__":
    main()
