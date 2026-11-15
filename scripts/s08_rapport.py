"""Étape 8 : rapport.csv, a_relire.html, planches avant/après (POC).
Usage : python s08_rapport.py [--pages all] [--planches]"""
import argparse, csv, html
from pathlib import Path
import cv2
import numpy as np

from common import OUT, page_dir, photos_of, read_json, select_pages

ap = argparse.ArgumentParser()
ap.add_argument("--pages", default="all"); ap.add_argument("--planches", action="store_true")
a = ap.parse_args()
pages = select_pages(a.pages)


def vignette(p, h=360):
    im = cv2.imread(str(p))
    if im is None:
        im = np.full((h, h, 3), 230, np.uint8)
        cv2.putText(im, "absent", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
    k = h / im.shape[0]
    im = cv2.resize(im, None, fx=k, fy=k)
    cv2.putText(im, Path(p).stem.split("_", 2)[-1], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)
    return im


lignes, relire = [], []
for page in pages:
    d = page_dir(page)
    for ph in photos_of(page):
        base = f"photo_{ph['photo']:02d}"
        m = read_json(d / f"{base}.json", {})
        txt = m.get("textes", [])
        lignes.append(dict(
            page=page, photo=ph["photo"], role=m.get("role", ""), groupe=m.get("groupe", ""),
            doublon_de=(f"p{m['doublon_de']['page']}_ph{m['doublon_de']['photo']}" if m.get("doublon_de") else ""),
            classe=m.get("classe", ""), partielle=ph["partielle"], masquee=ph.get("masquee", False),
            rotation=ph["rotation"], entree=m.get("entree", ""),
            etapes=";".join(m.get("etapes_restauration", [])),
            colorisation_ia=m.get("colorisation_ia", False),
            texte=" | ".join(t["texte"] for t in txt),
            confiance_lecture=";".join(t.get("confiance", "") for t in txt),
            confiance_association=";".join(t.get("confiance_association", "") for t in txt),
            rayures_pct=m.get("rayures_pct", ""),
            fichiers=";".join(sorted(p.name for p in d.glob(f"{base}_*"))),
            a_relire="o" if (ph["partielle"] or any(t.get("confiance") == "faible" or
                             t.get("confiance_association") == "faible" for t in txt)) else "n"))
        if a.planches and m.get("role") != "doublon":
            row = [vignette(d / f"{base}_raw.png")]
            for suf in ("fusion", "fusion_ia", "restaure", "restaure_visages", "couleur"):
                if (d / f"{base}_{suf}.png").exists():
                    row.append(vignette(d / f"{base}_{suf}.png"))
            cv2.imwrite(str(d / f"{base}_avant_apres.jpg"), np.hstack(row),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    for b in read_json(d / "a_relire.json", []):
        relire.append((page, b))

if lignes:
    with open(OUT / "rapport.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(lignes[0].keys())); w.writeheader(); w.writerows(lignes)

rows = []
for page, b in relire:
    img = f"page_{page:02d}/source_preview.jpg"
    rows.append(f"<tr><td>{page}</td><td><a href='{img}'><img src='{img}' loading=lazy></a></td>"
                f"<td contenteditable>{html.escape(b.get('texte',''))}</td>"
                f"<td>{b.get('type','')}</td><td>{b.get('confiance','')}</td>"
                f"<td>{html.escape(str(b.get('cible','')))} {b.get('position','')}</td>"
                f"<td><input type=checkbox></td></tr>")
(OUT / "a_relire.html").write_text(
    "<!doctype html><meta charset=utf-8><title>Textes à relire</title>"
    "<style>body{font-family:sans-serif;margin:16px}img{max-width:220px}"
    "td{border-bottom:1px solid #ccc;padding:6px;vertical-align:top}</style>"
    "<h1>Textes à relire</h1><p>Corriger dans la colonne texte (éditable), cocher quand vu, "
    "puis reporter dans texte_brut.json.</p><table><tr><th>Page</th><th>Aperçu</th><th>Texte</th>"
    "<th>Type</th><th>Lecture</th><th>Association</th><th>Vu</th></tr>"
    + "".join(rows) + "</table>", encoding="utf-8")
print(f"rapport.csv : {len(lignes)} lignes ; a_relire.html : {len(rows)} blocs")
