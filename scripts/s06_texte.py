"""Étape 6 : texte imprimé / manuscrit.
  preparer : découpe la page en tuiles agrandies (page_XX/texte_tuiles/) pour lecture
             visuelle par l'agent ; OCR auto en second avis si ai.ocr configuré.
  associer : lit page_XX/texte_brut.json (écrit par l'agent) et associe chaque bloc
             à une photo ou à la page -> association.json, a_relire.json
Format texte_brut.json :
  [{"id":1, "repere":"apercu", "boite":[x1,y1,x2,y2], "type":"manuscrit|imprime",
    "texte":"Annecy, été 1961 [Lucie?]", "confiance":"elevee|moyenne|faible"}]
Usage : python s06_texte.py preparer|associer --pages all"""
import argparse
import cv2
import numpy as np

from common import C, journal, page_dir, photos_of, read_json, run_ai, select_pages, write_json


def preparer(page):
    d = page_dir(page)
    img = cv2.imread(str(d / "source.png"))
    H, W = img.shape[:2]
    td = d / "texte_tuiles"; td.mkdir(exist_ok=True)
    n = 3 if max(H, W) > 3000 else 2
    k = C["apercu_px"] / max(H, W)
    carte = []
    for i in range(n):
        for j in range(n):
            y0, y1 = int(max(0, (i - 0.1) * H / n)), int(min(H, (i + 1.1) * H / n))
            x0, x1 = int(max(0, (j - 0.1) * W / n)), int(min(W, (j + 1.1) * W / n))
            t = img[y0:y1, x0:x1]
            s = 1600 / max(t.shape[:2])
            nom = f"tuile_{i}{j}.jpg"
            # coord. aperçu = x0*k + x_tuile * (k/s)
            carte.append(dict(tuile=nom, x0_apercu=round(x0 * k, 1), y0_apercu=round(y0 * k, 1),
                              facteur_vers_apercu=round(k / s, 5)))
            cv2.imwrite(str(td / nom), cv2.resize(t, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 88])
    write_json(td / "tuiles.json", carte)
    ok, voie, rem = run_ai("ocr", d / "source_preview.jpg", d / "ocr_auto.json")
    print(f"page {page:02d} : tuiles dans {td} ; ocr auto : {'ok' if ok else rem[:60]}")


def associer(page):
    d = page_dir(page)
    blocs = read_json(d / "texte_brut.json")
    if blocs is None:
        print(f"page {page:02d} : texte_brut.json absent (lecture visuelle à faire)"); return
    H, W = cv2.imread(str(d / "source_preview.jpg")).shape[:2]
    img = cv2.imread(str(d / "source.png"), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    k = C["apercu_px"] / (max(img.shape) * 4)
    photos = [(p["photo"], np.array(p["quad"], np.float32) * k) for p in photos_of(page)]
    assoc, relire = [], []
    for b in blocs:
        x1, y1, x2, y2 = b["boite"]
        if b.get("repere") != "apercu":
            x1, y1, x2, y2 = [v * k for v in (x1, y1, x2, y2)]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best = []
        for n, q in photos:
            px1, py1 = q.min(0); px2, py2 = q.max(0)
            dx = max(px1 - x2, x1 - px2, 0); dy = max(py1 - y2, y1 - py2, 0)
            dist = float(np.hypot(dx, dy))
            if cy < py1: pos = "au-dessus"
            elif cy > py2: pos = "en dessous"
            elif cx < px1: pos = "à gauche"
            elif cx > px2: pos = "à droite"
            else: pos = "sur la photo"
            best.append((dist, n, pos, max(px2 - px1, py2 - py1)))
        best.sort()
        rec = dict(b)
        if b.get("portee") == "page" or not best:
            rec.update(cible="page", confiance_association="elevee" if b.get("portee") == "page" else "faible")
        else:
            d1, n, pos, taille = best[0]
            d2 = best[1][0] if len(best) > 1 else 1e9
            if d1 < 0.25 * taille and d2 > 2.5 * max(d1, 5):
                ca = "elevee"
            elif d1 < 0.5 * taille and d2 > 1.4 * max(d1, 5):
                ca = "moyenne"
            else:
                ca = "faible"
            rec.update(cible=f"photo_{n:02d}", position=pos, distance_apercu_px=round(d1),
                       confiance_association=ca)
        assoc.append(rec)
        if rec["confiance_association"] == "faible" or b.get("confiance") == "faible":
            relire.append(rec)
    write_json(d / "association.json", assoc)
    write_json(d / "a_relire.json", relire)
    journal(page, "", "s06_association", "geometrie", remarque=f"{len(assoc)} blocs, {len(relire)} à relire")
    print(f"page {page:02d} : {len(assoc)} blocs, {len(relire)} à relire")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["preparer", "associer"])
    ap.add_argument("--pages", default="all")
    a = ap.parse_args()
    for p in select_pages(a.pages):
        (preparer if a.action == "preparer" else associer)(p)
