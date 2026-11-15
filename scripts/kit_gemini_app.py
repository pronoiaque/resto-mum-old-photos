"""Kit pour l'application Gemini (manuel) : contourne le blocage de génération d'image de l'API.
  preparer : out/kit_gemini/tirage_NNN/ = prises JPEG + consigne.txt (anglais, à coller)
             + out/kit_gemini/index.html (aperçu, consigne copiable en un clic)
  importer : récupère les tirage_NNN/resultat*.(png|jpg|webp) déposés par l'utilisateur
             -> page_<ref>/photo_YY_fusion_ia.png (marqué IA), prêt pour s04 (entree: fusion_ia)
Usage : python kit_gemini_app.py preparer [--tous] [--sans-couleur]   (colorisation des N&B par défaut)
        python kit_gemini_app.py importer"""
import argparse, html, shutil
from PIL import Image

from common import OUT, journal, page_dir, photos_of, read_json, write_json

KIT = OUT / "kit_gemini"
CONSIGNE = """MAIN GOAL: deliver ONE realistic, natural COLOR version of an old photographic print.
These {n} images are smartphone photos of ONE AND THE SAME physical print, taken from different
angles, lighting and framing (some may be partial, blurry, or show glare or shadows).
Steps:
1. Reconstruct the print faithfully from all shots, as if scanned flat on a good scanner:
   for each area keep the sharpest, glare-free version; remove glare, cast shadows, motion
   blur and perspective distortion; frame exactly the print (borders included), straightened,
   without the album background.
2. {couleur}
Strict fidelity rules:
- Invent nothing: no faces, objects or text; do not change expressions, poses or framing;
  do not smooth skin; keep the film grain and the period look.
- Do not retouch scratches or missing parts of the print.
- Output a single image at the highest resolution possible."""
CONSIGNE_1 = CONSIGNE.replace("These {n} images are smartphone photos of ONE AND THE SAME",
                              "This image is a smartphone photo of a").replace(
    "taken from different angles, lighting and framing (some may be partial, blurry, or show glare or shadows).",
    "(it may be tilted, blurry, or show glare or shadows).").replace(
    "Reconstruct the print faithfully from all shots, as if scanned flat on a good scanner:\n   for each area keep the sharpest, glare-free version; remove",
    "Reconstruct the print faithfully, as if scanned flat on a good scanner: remove")


def preparer(tous, colorier):
    KIT.mkdir(parents=True, exist_ok=True)
    cartes = []
    for g in read_json(OUT / "groupes.json", []):
        if len(g["membres"]) < 2 and not tous:
            continue
        r = g["reference"]
        dossier = KIT / f"tirage_{g['groupe']:03d}"
        dossier.mkdir(exist_ok=True)
        imgs = []
        for i, m in enumerate(g["membres"], 1):
            ph = next(p for p in photos_of(m["page"]) if p["photo"] == m["photo"])
            im = Image.open(page_dir(m["page"]) / ph["fichier"]).convert("RGB")
            im.thumbnail((3000, 3000))
            nom = f"prise_{i}_p{m['page']:02d}_ph{m['photo']:02d}.jpg"
            im.save(dossier / nom, quality=92); imgs.append(nom)
        meta = read_json(page_dir(r["page"]) / f"photo_{r['photo']:02d}.json", {})
        nb = meta.get("classe", "nb") == "nb"
        desc = meta.get("description_ia", "")
        periode = meta.get("periode_estimee", "")
        couleur = ("COLORIZE the reconstructed black-and-white print with realistic, natural, "
                   "period-accurate colors" + (f" ({periode})" if periode else "") +
                   ": believable skin tones, fabrics, vehicles, foliage, sky. Change ONLY the colors."
                   + (f" Scene for context: {desc}" if desc else "")
                   if colorier and nb else
                   "Restore the faded colors of this color print (remove yellow/magenta cast, natural "
                   "contrast), without changing anything else." if not nb else
                   "Keep the original black & white / sepia tones; do not colorize.")
        txt = (CONSIGNE if len(imgs) > 1 else CONSIGNE_1).format(n=len(imgs), couleur=couleur)
        (dossier / "consigne.txt").write_text(txt, encoding="utf-8")
        write_json(dossier / "cible.json", dict(groupe=g["groupe"], reference=r, prises=g["membres"],
                                               colorisation=bool(colorier and nb)))
        cartes.append((dossier.name, imgs, txt))
    blocs = []
    for nom, imgs, txt in cartes:
        vign = "".join(f"<img src='{nom}/{i}' loading=lazy>" for i in imgs)
        blocs.append(f"<section><h2>{nom}</h2><div class=v>{vign}</div>"
                     f"<textarea readonly>{html.escape(txt)}</textarea>"
                     f"<button onclick=\"navigator.clipboard.writeText(this.previousElementSibling.value)\">"
                     f"Copier la consigne</button><p>Résultat à enregistrer sous <code>{nom}/resultat.png</code></p></section>")
    (KIT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        "<title>Kit Gemini</title><style>body{font-family:sans-serif;margin:16px;max-width:1100px}"
        "section{border-bottom:1px solid #ccc;padding:12px 0}.v img{height:160px;margin:2px}"
        "textarea{width:100%;height:150px;font-size:12px}</style>"
        "<h1>Kit Gemini (application)</h1><ol><li>Ouvrir Gemini, nouvelle conversation, modèle image.</li>"
        "<li>Joindre toutes les prises du tirage, coller la consigne.</li>"
        "<li>Télécharger l'image produite dans le dossier du tirage sous <b>resultat.png</b>.</li>"
        "<li>Lancer <code>python scripts/kit_gemini_app.py importer</code>.</li></ol>"
        + "".join(blocs), encoding="utf-8")
    print(f"{len(cartes)} tirages préparés -> {KIT/'index.html'}")


def importer():
    n = 0
    for dossier in sorted(KIT.glob("tirage_*")):
        res = sorted([p for p in dossier.iterdir() if p.stem.lower().startswith("resultat")])
        cible = read_json(dossier / "cible.json")
        if not res or not cible:
            continue
        r = cible["reference"]
        d = page_dir(r["page"]); base = f"photo_{r['photo']:02d}"
        suffixe = "couleur" if cible.get("colorisation") else "fusion_ia"
        out = d / f"{base}_{suffixe}.png"
        Image.open(res[-1]).convert("RGB").save(out)
        write_json(out.with_suffix(".json"), dict(source="application Gemini (manuel)", fichier=res[-1].name,
                                                  prises=cible["prises"], ia_generative=True))
        meta = read_json(d / f"{base}.json", {})
        meta.update({f"{suffixe}_gemini_app": True, "contient_ia_generative": True})
        write_json(d / f"{base}.json", meta)
        journal(r["page"], r["photo"], "kit_gemini_import", "gemini app", "manuel", remarque=res[-1].name)
        print(f"{dossier.name} -> {out}"); n += 1
    print(f"{n} résultats importés. Ensuite : contrôle par l'agent album-qualite.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["preparer", "importer"])
    ap.add_argument("--tous", action="store_true", help="aussi les tirages à une seule prise")
    ap.add_argument("--sans-couleur", action="store_true", help="NE PAS demander la colorisation (par défaut : colorisation des N&B)")
    a = ap.parse_args()
    preparer(a.tous, not a.sans_couleur) if a.action == "preparer" else importer()
