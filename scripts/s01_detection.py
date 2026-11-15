"""Étape 1 : détection, découpe, redressement, orientation des photos d'une page.

Priorité des sources de boîtes :
  1. page_XX/boites_manuelles.json  (écrit/corrigé par l'agent après inspection visuelle)
  2. segmentation IA (ai.segmentation) si methode=segmentation
  3. OpenCV (deux indices combinés : contraste au fond + contours)
Format boîtes : [{"id":1, "quad":[[x,y]x4] en pixels de source.png, "note":""}]

Usage : python s01_detection.py --pages all|1,3|2-5 [--force]"""
import argparse, time
import cv2
import numpy as np

from common import C, journal, page_dir, read_json, write_json, select_pages

D = C["detection"]
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def ordonner(q):
    q = np.array(q, dtype=np.float32).reshape(4, 2)
    s, d = q.sum(1), np.diff(q, axis=1).ravel()
    return np.array([q[s.argmin()], q[d.argmin()], q[s.argmax()], q[d.argmax()]], np.float32)


def candidats_opencv(img):
    H, W = img.shape[:2]
    k = D["travail_px"] / max(H, W)
    sm = cv2.resize(img, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
    h, w = sm.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(sm, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    masques = []
    # Indice 1 : distance colorimétrique au fond (médiane de la bordure)
    bord = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3),
                           lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(bord, 0), axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m1 = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masques.append(m1)
    # Indice 2 : contours (Canny multi-canal) refermés
    e = np.zeros((h, w), np.uint8)
    for c in cv2.split(lab.astype(np.uint8)):
        e |= cv2.Canny(c, 30, 90)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    masques.append(e)
    cands = []
    for m in masques:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (D["aire_min"] * h * w < a < D["aire_max"] * h * w):
                continue
            rect = cv2.minAreaRect(c)
            ra = rect[1][0] * rect[1][1]
            if ra <= 0 or a / ra < D["rectangularite_min"]:
                continue
            ar = max(rect[1]) / max(1, min(rect[1]))
            if ar > 4:
                continue
            quad = cv2.boxPoints(rect) / k
            cands.append((a / ra * ra, ordonner(quad)))
    return nms(cands)


def iou(q1, q2, shape):
    m1 = np.zeros(shape, np.uint8); m2 = np.zeros(shape, np.uint8)
    cv2.fillPoly(m1, [q1.astype(np.int32)], 1); cv2.fillPoly(m2, [q2.astype(np.int32)], 1)
    inter = (m1 & m2).sum(); union = (m1 | m2).sum()
    return inter / max(1, union), inter / max(1, min(m1.sum(), m2.sum()))


def nms(cands):
    cands.sort(key=lambda x: -x[0])
    garde = []
    for s, q in cands:
        shape = (int(q[:, 1].max()) + 2, int(q[:, 0].max()) + 2)
        ok = True
        for g in garde:
            sh = (max(shape[0], int(g[:, 1].max()) + 2), max(shape[1], int(g[:, 0].max()) + 2))
            u, inc = iou(q / 8, g / 8, (sh[0] // 8 + 2, sh[1] // 8 + 2))
            if u > 0.5 or inc > 0.85:
                ok = False; break
        if ok:
            garde.append(q)
    return garde


def redresser(img, quad):
    q = ordonner(quad)
    wA = np.linalg.norm(q[2] - q[3]); wB = np.linalg.norm(q[1] - q[0])
    hA = np.linalg.norm(q[1] - q[2]); hB = np.linalg.norm(q[0] - q[3])
    W, H = int(max(wA, wB)), int(max(hA, hB))
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m = D["marge_rognage"]
    dy, dx = int(H * m), int(W * m)
    return out[dy:H - dy, dx:W - dx]


def orientation(photo):
    """Rotation (0/90/180/270) qui maximise la détection de visages ; 0 si aucun."""
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    k = 800 / max(g.shape)
    g = cv2.resize(g, None, fx=k, fy=k) if k < 1 else g
    best, score0, scores = 0, 0, {}
    for r, code in ((0, None), (90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180),
                    (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
        gi = g if code is None else cv2.rotate(g, code)
        f = FACE.detectMultiScale(gi, 1.1, 6, minSize=(24, 24))
        scores[r] = len(f)
    best = max(scores, key=lambda r: (scores[r], r == 0))
    if scores[best] == 0 or scores[best] <= scores[0]:
        return 0, scores
    return best, scores


ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def traiter(page, force=False):
    d = page_dir(page)
    if (d / "photos.json").exists() and not force:
        print(f"page {page:02d} : déjà découpée (--force pour refaire)"); return
    t = time.time()
    img = cv2.imread(str(d / "source.png"))
    H, W = img.shape[:2]
    manu = read_json(d / "boites_manuelles.json")
    if not manu and D["methode"] == "gemini":
        from gemini_taches import detecter
        manu = detecter(page)
    seg = (C.get("ai") or {}).get("segmentation", {}).get("backend", "none")
    if manu:
        kp = max(H, W) / C["apercu_px"]
        boites = [dict(b, quad=ordonner(np.array(b["quad"], np.float32) *
                                        (kp if b.get("repere") == "apercu" else 1)))
                  for b in manu]
        src = "manuel"
    elif D["methode"] in ("auto", "segmentation") and seg != "none":
        from common import run_ai
        ok, voie, rem = run_ai("segmentation", d / "source.png", d / "segmentation.json")
        quads = read_json(d / "segmentation.json", []) if ok else []
        if not ok:
            print(f"  segmentation indisponible ({rem[:120]}), repli OpenCV")
            quads = candidats_opencv(img)
            src = "opencv(repli)"
        else:
            src = f"segmentation({voie})"
        boites = [dict(id=i + 1, quad=ordonner(q), note="") for i, q in enumerate(quads)]
    else:
        quads = candidats_opencv(img)
        boites, src = [dict(id=i + 1, quad=q, note="") for i, q in enumerate(quads)], "opencv"
    # ordre de lecture : haut->bas puis gauche->droite (par bandes)
    bande = H / 8
    boites.sort(key=lambda b: (int(b["quad"][:, 1].mean() // bande), b["quad"][:, 0].mean()))
    photos, ctrl = [], img.copy()
    for i, b in enumerate(boites, 1):
        q = b["quad"]
        bp = D["bord_partielle_px"]
        partielle = bool((q[:, 0] < bp).any() or (q[:, 1] < bp).any() or
                         (q[:, 0] > W - bp).any() or (q[:, 1] > H - bp).any())
        ph = redresser(img, q)
        rot = b.get("rotation")
        if rot is None:
            rot, sc = orientation(ph)
        if rot:
            ph = cv2.rotate(ph, ROT[rot])
        nom = f"photo_{i:02d}_raw.png"
        cv2.imwrite(str(d / nom), ph)
        info = dict(photo=i, fichier=nom, quad=q.round(1).tolist(), source_boite=src,
                    largeur=ph.shape[1], hauteur=ph.shape[0], rotation=int(rot),
                    partielle=partielle, masquee=bool(b.get("masquee", False)),
                    note=b.get("note", ""))
        photos.append(info)
        coul = (0, 165, 255) if partielle or info["masquee"] else (0, 200, 0)
        cv2.polylines(ctrl, [q.astype(np.int32)], True, coul, max(4, W // 600))
        c = q.mean(0).astype(int)
        cv2.putText(ctrl, str(i), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, W / 1000, (0, 0, 255),
                    max(3, W // 500))
    write_json(d / "photos.json", photos)
    # boîtes éditables par l'agent (copie de travail)
    write_json(d / "boites_detectees.json",
               [dict(id=p["photo"], quad=p["quad"], rotation=p["rotation"],
                     masquee=p["masquee"], note=p["note"]) for p in photos])
    k = C["apercu_px"] / max(H, W)
    cv2.imwrite(str(d / "controle_decoupe.jpg"), cv2.resize(ctrl, None, fx=k, fy=k),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    journal(page, "", "s01_detection", src, parametres=D, duree=time.time() - t,
            statut="ok" if photos else "aucune_photo", remarque=f"{len(photos)} photos")
    print(f"page {page:02d} : {len(photos)} photos ({src}) "
          f"partielles={sum(p['partielle'] for p in photos)} "
          f"pivotées={sum(1 for p in photos if p['rotation'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in select_pages(a.pages):
        traiter(p, a.force)
