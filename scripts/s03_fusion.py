"""Étape 3 : fusion multi-prises d'un même tirage (déterministe, sans IA générative).
Pour chaque groupe à ≥ 2 prises : recalage de chaque prise sur la référence
(ORB + homographie, affinage ECC), égalisation photométrique, puis empilement
pondéré par la netteté locale, en rejetant reflets et pixels aberrants.
Effet : moins de bruit, reflets/ombres d'une prise compensés par les autres,
zones floues remplacées par les plus nettes. Ne crée aucun détail inexistant.
Sortie : page_<ref>/photo_YY_fusion.png + photo_YY_fusion.json
Usage : python s03_fusion.py [--force]"""
import argparse, time
import cv2
import numpy as np

from common import C, OUT, journal, page_dir, photos_of, read_json, write_json

F = C["fusion"]
ORB = cv2.ORB_create(6000)


def lire(page, photo):
    ph = next(p for p in photos_of(page) if p["photo"] == photo)
    return cv2.imread(str(page_dir(page) / ph["fichier"]))


def recaler(mov, ref):
    """Homographie mov -> ref (images pleine taille), ou None."""
    def pts(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k = 1600 / max(g.shape)
        gs = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        kp, des = ORB.detectAndCompute(gs, None)
        return kp, des, k
    k1, d1, s1 = pts(mov); k2, d2, s2 = pts(ref)
    if d1 is None or d2 is None:
        return None, 0
    m = [x[0] for x in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
         if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(m) < 20:
        return None, 0
    a = np.float32([k1[x.queryIdx].pt for x in m]) / s1
    b = np.float32([k2[x.trainIdx].pt for x in m]) / s2
    H, mask = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
    if H is None:
        return None, 0
    # affinage ECC à résolution réduite
    try:
        k = 1200 / max(ref.shape[:2])
        S = np.diag([k, k, 1.0])
        r = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None, fx=k, fy=k).astype(np.float32)
        w0 = cv2.warpPerspective(mov, H, (ref.shape[1], ref.shape[0]))
        mvs = cv2.resize(cv2.cvtColor(w0, cv2.COLOR_BGR2GRAY), None, fx=k, fy=k).astype(np.float32)
        Wm = np.eye(3, dtype=np.float32)
        _, Wm = cv2.findTransformECC(r, mvs, Wm, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                                     None, 5)
        H = np.linalg.inv(S) @ np.linalg.inv(Wm) @ S @ H
    except cv2.error:
        pass
    return H, int(mask.sum())


def flux_local(ref, wv, val):
    """Recalage fin non rigide (papier gondolé, objectif grand angle) par flux optique DIS."""
    if not F.get("flux_optique", True):
        return wv, val
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(wv, cv2.COLOR_BGR2GRAY)
    g2 = cv2.equalizeHist(g2); g1 = cv2.equalizeHist(g1)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(g1, g2, None)
    flow = cv2.GaussianBlur(flow, (0, 0), 3)
    h, w = g1.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + flow[..., 0], gy + flow[..., 1]
    return (cv2.remap(wv, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT),
            cv2.remap(val, mx, my, cv2.INTER_LINEAR, borderValue=0))


def egaliser(src, ref, valid):
    """Ajuste moyenne/écart-type par canal de src sur ref dans la zone commune."""
    out = src.astype(np.float32)
    v = valid > 0
    for c in range(3):
        ms, ss = out[..., c][v].mean(), out[..., c][v].std() + 1e-3
        mr, sr = ref[..., c][v].mean(), ref[..., c][v].std() + 1e-3
        out[..., c] = (out[..., c] - ms) * (sr / ss) + mr
    return np.clip(out, 0, 255)


def nettete_locale(img):
    g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    return cv2.GaussianBlur(lap, (0, 0), 6) + 1.0


def fusionner(groupe, force=False):
    r = groupe["reference"]
    d = page_dir(r["page"])
    out = d / f"photo_{r['photo']:02d}_fusion.png"
    if out.exists() and not force:
        return
    t = time.time()
    ref = lire(r["page"], r["photo"])
    k = min(1.0, F["taille_ref_max"] / max(ref.shape[:2]))
    if k < 1:
        ref = cv2.resize(ref, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    h, w = ref.shape[:2]
    pile, poids, details = [ref.astype(np.float32)], [], []
    masques = [np.ones((h, w), np.float32)]
    for m in groupe["membres"]:
        if (m["page"], m["photo"]) == (r["page"], r["photo"]):
            continue
        mov = lire(m["page"], m["photo"])
        H, n = recaler(mov, ref)
        if H is None:
            details.append(dict(page=m["page"], photo=m["photo"], statut="recalage_echoue"))
            continue
        wv = cv2.warpPerspective(mov, H, (w, h), flags=cv2.INTER_CUBIC)
        val = cv2.warpPerspective(np.ones(mov.shape[:2], np.float32), H, (w, h))
        wv, val = flux_local(ref, wv, val)
        val = cv2.erode(val, np.ones((9, 9), np.uint8))
        if val.mean() < 0.15:
            details.append(dict(page=m["page"], photo=m["photo"], statut="recouvrement_faible"))
            continue
        pile.append(egaliser(wv, ref.astype(np.float32), val))
        masques.append(val)
        details.append(dict(page=m["page"], photo=m["photo"], statut="ok", inliers=n,
                            recouvrement=round(float(val.mean()), 2)))
    if len(pile) < 2:
        write_json(out.with_suffix(".json"), dict(statut="une_seule_prise_utilisable", prises=details))
        journal(r["page"], r["photo"], "s03_fusion", "opencv", duree=time.time() - t,
                statut="ignore", remarque="recalage impossible")
        return
    P = np.stack(pile)                                   # N,h,w,3
    Mv = np.stack(masques)[..., None]
    med = np.median(np.where(Mv > 0, P, np.nan), axis=0)
    med = np.where(np.isnan(med), P[0], med)
    W = []
    for i in range(len(pile)):
        s = nettete_locale(P[i]) ** F.get("exposant_nettete", 4)
        lum = P[i].mean(2)
        s *= np.where(lum > 245, 0.05, 1.0)                # reflets
        ecart = np.abs(P[i] - med).mean(2)
        s *= np.exp(-(ecart / 25.0) ** 2)                  # pixels aberrants
        W.append(s * Mv[i, ..., 0])
    W = np.stack(W)[..., None] + 1e-6
    res = (P * W).sum(0) / W.sum(0)
    cv2.imwrite(str(out), np.clip(res, 0, 255).astype(np.uint8))
    write_json(out.with_suffix(".json"), dict(statut="ok", reference=r, prises=details,
                                              methode="recalage ORB+ECC, empilement pondéré netteté"))
    journal(r["page"], r["photo"], "s03_fusion", "opencv", parametres=F, duree=time.time() - t,
            remarque=f"{len(pile)} prises fusionnées")
    print(f"groupe {groupe['groupe']} -> {out.name} ({len(pile)} prises)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gs = read_json(OUT / "groupes.json", [])
    if not F.get("classique", True):
        print("fusion classique désactivée"); raise SystemExit
    for g in gs:
        if len(g["membres"]) > 1:
            fusionner(g, a.force)
