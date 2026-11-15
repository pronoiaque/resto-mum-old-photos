"""Tâches sous-traitées à Gemini.
  detecter  : repère les tirages d'une page (4 coins, partielle, masquée, rotation, nb/couleur)
              -> page_XX/boites_gemini.json (repère aperçu), utilisé par s01 si methode=gemini
  texte     : lit textes imprimés/manuscrits -> page_XX/texte_brut.json (format s06)
Usage : python gemini_taches.py detecter|texte --pages 1,5 [--force]"""
import argparse
from PIL import Image

from common import C, journal, page_dir, read_json, select_pages, write_json
import gemini

G = C["gemini"]

CONSIGNE_DETECTION = """The image is a smartphone photo of a photo-album page (or part of one)
holding paper photographic prints. Find EVERY photographic print visible, even partially
(cut off by the image border or covered by another print).
For each print, give the 4 corners of the PAPER print (white / deckled borders included),
in the order top-left, top-right, bottom-right, bottom-left of the print as it lies,
as normalized [x, y] coordinates from 0 to 1000 (x to the right, y downward) over the whole image.
Ignore: background, binding/spiral, printed or handwritten text outside prints, decoration, objects.
partielle = the print extends beyond the image frame; masquee = another print covers part of it;
rotation = clockwise rotation (0, 90, 180, 270) needed to make the print's picture upright;
teinte = "nb" (black & white or sepia) or "couleur" (color);
description = a short English description of the picture."""

SCHEMA_DETECTION = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "coins": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "NUMBER"}}},
    "partielle": {"type": "BOOLEAN"}, "masquee": {"type": "BOOLEAN"},
    "rotation": {"type": "INTEGER"}, "teinte": {"type": "STRING"},
    "description": {"type": "STRING"}},
    "required": ["coins", "partielle", "masquee", "rotation", "teinte"]}}

CONSIGNE_TEXTE = """The image is a photo-album page. Read ALL visible text, printed or
handwritten (usually French), including text written on the print borders.
For each text block: boite = [xmin, ymin, xmax, ymax] normalized 0-1000 over the whole image;
type = "imprime" (printed) or "manuscrit" (handwritten);
texte = faithful transcription in the ORIGINAL language and spelling (do not translate);
put uncertain words in square brackets with a question mark, e.g. [Lucie?];
confiance = "elevee" (high), "moyenne" (medium) or "faible" (low);
portee = "page" if it is a page title, otherwise "photo".
Never invent text; if there is no text, return an empty list."""

SCHEMA_TEXTE = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "boite": {"type": "ARRAY", "items": {"type": "NUMBER"}}, "type": {"type": "STRING"},
    "texte": {"type": "STRING"}, "confiance": {"type": "STRING"}, "portee": {"type": "STRING"}},
    "required": ["boite", "type", "texte", "confiance"]}}


def _apercu(page):
    im = Image.open(page_dir(page) / "source_preview.jpg")
    return im, im.size


def detecter(page, force=False):
    d = page_dir(page)
    out = d / "boites_gemini.json"
    if out.exists() and not force:
        return read_json(out)
    im, (w, h) = _apercu(page)
    rep = gemini.appeler(G["modele_vision"], CONSIGNE_DETECTION, [Image.open(d / "source.png")],
                         json_schema=SCHEMA_DETECTION, objet=f"detection p{page}", temperature=0)
    boites = []
    for i, b in enumerate(rep, 1):
        q = [[x * w / 1000, y * h / 1000] for x, y in b["coins"][:4]]
        if len(q) == 4:
            boites.append(dict(id=i, repere="apercu", quad=q, rotation=int(b.get("rotation", 0)) % 360,
                               masquee=bool(b.get("masquee")), teinte=b.get("teinte", ""),
                               note=b.get("description", ""), source="gemini"))
    write_json(out, boites)
    journal(page, "", "gemini_detection", G["modele_vision"], "en_ligne", remarque=f"{len(boites)} tirages")
    print(f"page {page:02d} : {len(boites)} tirages (Gemini)")
    return boites


def texte(page, force=False):
    d = page_dir(page)
    out = d / "texte_brut.json"
    if out.exists() and not force:
        return
    im, (w, h) = _apercu(page)
    rep = gemini.appeler(G["modele_vision"], CONSIGNE_TEXTE, [Image.open(d / "source.png")],
                         json_schema=SCHEMA_TEXTE, objet=f"texte p{page}", temperature=0)
    blocs = []
    for i, b in enumerate(rep, 1):
        x1, y1, x2, y2 = b["boite"][:4]
        blocs.append(dict(id=i, repere="apercu", boite=[x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000],
                          type=b["type"], texte=b["texte"], confiance=b["confiance"],
                          portee=b.get("portee", "photo"), source="gemini"))
    write_json(out, blocs)
    journal(page, "", "gemini_texte", G["modele_vision"], "en_ligne", remarque=f"{len(blocs)} blocs")
    print(f"page {page:02d} : {len(blocs)} blocs de texte (Gemini)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tache", choices=["detecter", "texte"])
    ap.add_argument("--pages", default="all"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in select_pages(a.pages):
        try:
            (detecter if a.tache == "detecter" else texte)(p, a.force)
        except gemini.Plafond as e:
            raise SystemExit(str(e))
        except Exception as e:
            journal(p, "", f"gemini_{a.tache}", G["modele_vision"], "en_ligne", statut="echec", remarque=repr(e)[:300])
            print(f"page {p:02d} : échec {e}")
