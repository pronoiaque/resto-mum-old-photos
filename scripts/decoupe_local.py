"""Découpage + redressement 100 % LOCAL des tirages photo sur des prises de pages d'album.

Méthode :
  1. FastSAM (Ultralytics, CPU) segmente tout ce qui ressemble à un objet ;
  2. OpenCV garde les masques « tirage » : quasi rectangulaires, taille plausible,
     ne contenant pas eux-mêmes plusieurs tirages (= page entière), sans doublon ;
  3. quadrilatère (4 coins) ajusté au contour -> correction de perspective ;
  4. remise à l'endroit (détection de visages sur les 4 rotations) ;
  5. drapeaux : partielle (touche le bord de l'image), masquee (rectangle peu rempli).
Sorties par image : <dst>/<nom>/photo_NN.png, controle.jpg, boites.json
  (boites.json éditable à la main, puis --depuis-boites pour redécouper sans détection)

Usage : python decoupe_local.py [--src ~/albums/pngnatives] [--dst ~/albums/decoupes]
                                [--images 20261115_155120,...] [--modele FastSAM-x.pt]
                                [--depuis-boites] [--force]"""
import argparse, json, time
from pathlib import Path

import cv2
import numpy as np

FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FACE2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
P = dict(imgsz=1024, conf=0.25, iou=0.7,
         aire_min=0.004, aire_max=0.92,      # fraction de l'image
         remplissage_min=0.84, remplissage_plausible=0.70,               # part du quadrilatère couverte par le masque
         remplissage_masquee=0.93,           # en dessous : probablement recouverte
         ratio_max=3.5, bord_px=0.004, marge_rognage=0.004)


def ordonner(q):
    q = np.asarray(q, np.float32).reshape(4, 2)
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]                      # tri angulaire : HG, HD, BD, BG
    i = np.argmin(q.sum(1))
    return np.roll(q, -i, axis=0)


def quad_du_masque(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    ra = rect[1][0] * rect[1][1]
    if ra <= 0:
        return None, 0
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.04):        # 4 coins réels si possible (perspective)
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            quad = ap.reshape(4, 2).astype(np.float32)
            break
    if quad is None or cv2.contourArea(quad) < 0.85 * ra:
        quad = cv2.boxPoints(rect)
    return ordonner(quad), a / ra


def masques_papier(img, w, h):
    """Candidats « bord papier » : zones claires et peu saturées (liseré des tirages),
    trous bouchés -> une composante = un tirage potentiel. Complète FastSAM, qui
    fusionne parfois les grands tirages avec leur voisinage."""
    sm = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(sm, (5, 5), 0), cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    sorties = []
    for vmin in (np.percentile(V, 70), np.percentile(V, 82)):
        papier = ((V >= vmin) & (S < 60)).astype(np.uint8)
        papier = cv2.morphologyEx(papier, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, hier = cv2.findContours(papier, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        for i, c in enumerate(cnts):
            if hier[0][i][3] != -1:            # seulement les contours extérieurs
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [c], -1, 1, -1)  # remplissage = trous bouchés
            sorties.append(m)
    return sorties


def detecter(img, modele, debug=False):
    H, W = img.shape[:2]
    res = modele(img, device="cpu", retina_masks=False, imgsz=P["imgsz"], conf=P["conf"],
                 iou=P["iou"], verbose=False)[0]
    if res.masks is None:
        return []
    masks = res.masks.data.cpu().numpy().astype(np.uint8)       # N,h,w
    h, w = masks.shape[1:]
    masks = list(masks) + masques_papier(img, w, h)
    sx, sy = W / w, H / h
    cands = []
    for m in masks:
        a = int(m.sum())
        if not (P["aire_min"] * h * w < a < P["aire_max"] * h * w):
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        q, _ = quad_du_masque(m)
        if q is None:
            continue
        qm = np.zeros_like(m); cv2.fillPoly(qm, [q.astype(np.int32)], 1)
        rempl = (m & qm).sum() / max(1, qm.sum())                 # masque dans le quad
        deborde = (m & (1 - qm)).sum() / max(1, m.sum())          # masque hors du quad
        cote = sorted([np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[1] - q[2])])
        if cote[0] < 8 or cote[1] / cote[0] > P["ratio_max"] or deborde > 0.12:
            continue
        cands.append(dict(quad=q, rempl=float(rempl), aire=a, masque=m, qmasque=qm))
    if debug:
        for c in cands:
            print("  cand aire=%.3f rempl=%.2f" % (c["aire"] / (h * w), c["rempl"]))
    # 1) tirages : nets (bords droits) ou « plausibles » (recouverts, mal segmentés)
    nets = [c for c in cands if c["rempl"] >= P["remplissage_plausible"]]
    for c in nets:
        c["masquee"] = c["rempl"] < P["remplissage_min"]
    def dans(petit, grand):
        return (petit["masque"] & grand["qmasque"]).sum() / max(1, petit["masque"].sum())
    # 3) écarter d'abord les « pages » : un quad qui contient ≥ 2 autres tirages nets
    francs = [c for c in nets if not c["masquee"]]
    pages = [c for c in nets if sum(1 for o in francs if o is not c and o["aire"] < 0.6 * c["aire"]
                                    and dans(o, c) > 0.8) >= 2]
    nets = [c for c in nets if not any(c is p for p in pages)]
    # 4) le plus grand d'abord ; écarter ce qui est contenu dans un tirage retenu
    #    (image intérieure d'un tirage à bord blanc, personnes, objets...)
    nets.sort(key=lambda c: -c["aire"])
    final = []
    for c in nets:
        if any(dans(c, f) > 0.8 for f in final):
            continue
        final.append(c)
    out = []
    for c in final:
        q = c["quad"] * [sx, sy]
        b = P["bord_px"] * max(W, H)
        partielle = bool((q[:, 0] < b).any() or (q[:, 1] < b).any() or
                         (q[:, 0] > W - b).any() or (q[:, 1] > H - b).any())
        out.append(dict(quad=q.round(1).tolist(), partielle=partielle,
                        masquee=bool(c.get("masquee", False)), remplissage=round(c["rempl"], 3)))
    return out


def redresser(img, quad):
    q = ordonner(quad)
    W = int(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    H = int(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m = P["marge_rognage"]
    return out[int(H * m):H - int(H * m), int(W * m):W - int(W * m)]


ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def orientation(ph):
    g = cv2.cvtColor(ph, cv2.COLOR_BGR2GRAY)
    k = 900 / max(g.shape)
    g = cv2.equalizeHist(cv2.resize(g, None, fx=k, fy=k) if k < 1 else g)
    sc = {}
    for r in (0, 90, 180, 270):
        gi = g if r == 0 else cv2.rotate(g, ROT[r])
        f = FACE.detectMultiScale(gi, 1.08, 6, minSize=(28, 28))
        f2 = FACE2.detectMultiScale(gi, 1.08, 6, minSize=(28, 28))
        sc[r] = len(f) + 0.5 * len(f2)
    best = max(sc, key=lambda r: (sc[r], r == 0))
    return (best if sc[best] > sc[0] else 0), sc


def traiter(src, dst, modele, depuis_boites, force):
    d = dst / src.stem
    if (d / "boites.json").exists() and not (force or depuis_boites):
        return f"{src.stem} : déjà fait"
    d.mkdir(parents=True, exist_ok=True)
    t = time.time()
    img = cv2.imread(str(src))
    H, W = img.shape[:2]
    if depuis_boites:
        boites = json.loads((d / "boites.json").read_text())
    else:
        boites = detecter(img, modele)
        cy = lambda b: np.mean([p[1] for p in b["quad"]])
        cx = lambda b: np.mean([p[0] for p in b["quad"]])
        bande = H / 6
        boites.sort(key=lambda b: (int(cy(b) // bande), cx(b)))
    for p in d.glob("photo_*.png"):
        p.unlink()
    ctrl = img.copy()
    ep = max(4, W // 500)
    for i, b in enumerate(boites, 1):
        ph = redresser(img, b["quad"])
        if "rotation" not in b:
            b["rotation"], _ = orientation(ph)
        if b["rotation"]:
            ph = cv2.rotate(ph, ROT[b["rotation"]])
        b["id"] = i
        cv2.imwrite(str(d / f"photo_{i:02d}.png"), ph)
        q = np.array(b["quad"], np.int32)
        coul = (0, 165, 255) if b.get("partielle") or b.get("masquee") else (0, 200, 0)
        cv2.polylines(ctrl, [q], True, coul, ep)
        cv2.putText(ctrl, f"{i}{' R' + str(b['rotation']) if b['rotation'] else ''}",
                    tuple(q.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, W / 900, (0, 0, 255), ep)
    (d / "boites.json").write_text(json.dumps(boites, indent=1))
    k = 1600 / max(H, W)
    cv2.imwrite(str(d / "controle.jpg"), cv2.resize(ctrl, None, fx=k, fy=k), [cv2.IMWRITE_JPEG_QUALITY, 85])
    np_ = sum(b.get("partielle", False) for b in boites)
    nm = sum(b.get("masquee", False) for b in boites)
    nr = sum(1 for b in boites if b.get("rotation"))
    return f"{src.stem} : {len(boites)} photos (partielles {np_}, masquées {nm}, pivotées {nr}) {time.time() - t:.1f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/albums/pngnatives")
    ap.add_argument("--dst", default="~/albums/decoupes")
    ap.add_argument("--images", default="")
    ap.add_argument("--modele", default="FastSAM-x.pt")
    ap.add_argument("--depuis-boites", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    fichiers = sorted(src.glob("*.png"))
    if a.images:
        voulu = set(a.images.split(","))
        fichiers = [f for f in fichiers if f.stem in voulu]
    modele = None
    if not a.depuis_boites:
        from ultralytics import FastSAM
        poids = Path(__file__).resolve().parent.parent / "modeles" / a.modele
        poids.parent.mkdir(exist_ok=True)
        modele = FastSAM(str(poids) if poids.exists() else a.modele)
        if not poids.exists() and Path(a.modele).exists():
            Path(a.modele).replace(poids)
    for f in fichiers:
        print(traiter(f, dst, modele, a.depuis_boites, a.force), flush=True)


if __name__ == "__main__":
    main()
