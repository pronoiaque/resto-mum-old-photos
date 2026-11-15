"""Étape 4 : restauration fidèle (sans IA générative) + options upscale / visages.
Entrée par photo : fusion_ia (si restauration.entree=fusion_ia et présent) > fusion > raw.
Les prises « doublon » d'un groupe ne sont pas restaurées (seule la référence l'est).
Sorties : photo_YY_restaure.png, [photo_YY_restaure_visages.png], photo_YY.json
Usage : python s04_restauration.py --pages all [--force]"""
import argparse, time
import cv2
import numpy as np

from common import C, OUT, journal, page_dir, photos_of, read_json, run_ai, select_pages, write_json

R = C["restauration"]


def role(page, photo, groupes):
    for g in groupes:
        for m in g["membres"]:
            if (m["page"], m["photo"]) == (page, photo):
                r = g["reference"]
                if (r["page"], r["photo"]) == (page, photo):
                    return "reference", g
                return "doublon", g
    return "unique", None


def classer(img):
    """nb (N&B ou sépia : chroma peu dispersée) ou couleur."""
    lab = cv2.cvtColor(cv2.resize(img, (400, 400)), cv2.COLOR_BGR2LAB).astype(np.float32)
    disp = float(np.sqrt(lab[..., 1].var() + lab[..., 2].var()))
    return ("nb" if disp < R["saturation_nb_max"] else "couleur"), round(disp, 1)


def eclairage(L):
    f = R["eclairage_flou_frac"]
    if not f:
        return L
    sig = max(L.shape) * f
    fond = cv2.GaussianBlur(L.astype(np.float32), (0, 0), sig) + 1
    out = L.astype(np.float32) / fond * fond.mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def etirer(c, lo=0.5, hi=99.5):
    # percentiles mesurés au centre (le liseré blanc du tirage fausserait l'étirement)
    h, w = c.shape[:2]
    a, b = np.percentile(c[int(h * .12):int(h * .88), int(w * .12):int(w * .88)], (lo, hi))
    b = max(b, a + 1) + 0.04 * (255 - b)     # garder de la marge dans les hautes lumières
    return np.clip((c.astype(np.float32) - a) * 255.0 / max(1, b - a), 0, 255).astype(np.uint8)


def nettete(img):
    flou = cv2.GaussianBlur(img, (0, 0), R["nettete_rayon"])
    return cv2.addWeighted(img, 1 + R["nettete_force"], flou, -R["nettete_force"], 0)


def restaurer(img, classe):
    clahe = cv2.createCLAHE(R["clahe_clip"], (R["clahe_grille"],) * 2)
    if classe == "nb":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = eclairage(g); g = etirer(g); g = clahe.apply(g)
        g = cv2.fastNlMeansDenoising(g, None, R["debruitage_h"], 7, 21)
        return nettete(g)
    # couleur délavée : balance des blancs (gris moyen) + étirement par canal
    f = img.astype(np.float32)
    f *= f.reshape(-1, 3).mean(0).mean() / (f.reshape(-1, 3).mean(0) + 1e-3)
    f = np.clip(f, 0, 255).astype(np.uint8)
    f = cv2.merge([etirer(c, 0.5, 99.5) for c in cv2.split(f)])
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = clahe.apply(eclairage(L))
    f = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    f = cv2.fastNlMeansDenoisingColored(f, None, R["debruitage_h"], R["debruitage_h"], 7, 21)
    return nettete(f)


def rayures(g):
    """Estimation grossière des rayures/poussières (traits fins clairs) en % de surface."""
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    return round(100 * float((th > 40).mean()), 2)


def traiter(page, force, groupes):
    d = page_dir(page)
    for ph in photos_of(page):
        n = ph["photo"]; base = f"photo_{n:02d}"
        meta_p = d / f"{base}.json"
        meta = read_json(meta_p, {})
        rl, g = role(page, n, groupes)
        meta.update(page=page, photo=n, role=rl, groupe=g["groupe"] if g else None,
                    partielle=ph["partielle"], masquee=ph.get("masquee", False),
                    rotation=ph["rotation"])
        if rl == "doublon":
            meta["doublon_de"] = g["reference"]
            write_json(meta_p, meta); continue
        out = d / f"{base}_restaure.png"
        if out.exists() and not force:
            continue
        t = time.time()
        src = d / ph["fichier"]
        if C["restauration"].get("entree") == "fusion_ia" and (d / f"{base}_fusion_ia.png").exists():
            src = d / f"{base}_fusion_ia.png"
        elif (d / f"{base}_fusion.png").exists():
            src = d / f"{base}_fusion.png"
        img = cv2.imread(str(src))
        classe, disp = classer(img)
        res = restaurer(img, classe)
        cv2.imwrite(str(out), res)
        etapes = ["classement", "eclairage" if R["eclairage_flou_frac"] else None,
                  "niveaux", "clahe", "debruitage", "nettete"]
        if max(res.shape[:2]) < R["agrandir_si_petit_px"]:
            ok, voie, rem = run_ai("upscale", out, d / f"{base}_up.png")
            journal(page, n, "s04_upscale", "upscale", voie, statut="ok" if ok else "echec", remarque=rem)
            if ok:
                (d / f"{base}_up.png").replace(out); etapes.append("upscale_x2")
        ok, voie, rem = run_ai("visages", out, d / f"{base}_restaure_visages.png")
        if ok:
            etapes.append("visages(fichier séparé)")
        meta.update(classe=classe, dispersion_chroma=disp, entree=src.name,
                    etapes_restauration=[e for e in etapes if e],
                    rayures_pct=rayures(res), params_restauration=R)
        write_json(meta_p, meta)
        journal(page, n, "s04_restauration", "opencv", parametres=R, duree=time.time() - t,
                remarque=f"{classe} entree={src.name} rayures≈{meta['rayures_pct']}%")
        print(f"page {page:02d} photo {n:02d} : {classe} ({disp}) <- {src.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="all"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gr = read_json(OUT / "groupes.json", [])
    for p in select_pages(a.pages):
        traiter(p, a.force, gr)
