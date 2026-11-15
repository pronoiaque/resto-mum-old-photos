"""Étape 2 : regroupe les prises de vue d'un même tirage (toutes pages confondues).
Indices : pHash (photos entières) + appariement ORB/homographie (gère les vues partielles).
Sorties : out/groupes.json, out/doublons.csv
Usage : python s02_doublons.py [--pages all]"""
import argparse, csv, itertools, time
import cv2
import imagehash
import numpy as np
from PIL import Image

from common import C, OUT, journal, page_dir, photos_of, select_pages, write_json

P = C["doublons"]
ORB = cv2.ORB_create(3000)


def charger(pages):
    items = []
    for pg in pages:
        for ph in photos_of(pg):
            f = page_dir(pg) / ph["fichier"]
            im = cv2.imread(str(f))
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            k = 1000 / max(g.shape)
            gs = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            kp, des = ORB.detectAndCompute(gs, None)
            items.append(dict(page=pg, photo=ph["photo"], partielle=ph["partielle"],
                              masquee=ph.get("masquee", False),
                              aire=im.shape[0] * im.shape[1],
                              nettete=float(cv2.Laplacian(gs, cv2.CV_64F).var()),
                              phash=imagehash.phash(Image.fromarray(gs)), kp=kp, des=des))
    return items


def inliers(a, b):
    if a["des"] is None or b["des"] is None or len(a["kp"]) < 20 or len(b["kp"]) < 20:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    m = [x[0] for x in bf.knnMatch(a["des"], b["des"], k=2)
         if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(m) < 12:
        return 0
    pa = np.float32([a["kp"][x.queryIdx].pt for x in m])
    pb = np.float32([b["kp"][x.trainIdx].pt for x in m])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 6.0)
    if H is None:
        return 0
    n = int(mask.sum())
    det = np.linalg.det(H[:2, :2])
    # rejet des homographies dégénérées et des appariements trop dispersés
    if not (0.05 < det < 20) or n / len(m) < 0.25 or abs(H[2, 0]) + abs(H[2, 1]) > 0.003:
        return 0
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pages", default="all")
    a = ap.parse_args()
    t = time.time()
    it = charger(select_pages(a.pages))
    parent = list(range(len(it)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    liens = []
    for i, j in itertools.combinations(range(len(it)), 2):
        A, B = it[i], it[j]
        if A["page"] == B["page"]:
            continue                     # deux tirages d'une même prise sont distincts
        dh = A["phash"] - B["phash"]
        n = inliers(A, B)
        if n >= P["orb_inliers_min"] or (dh <= P["phash_max"] and n >= P["orb_inliers_min"] // 2):
            parent[find(i)] = find(j)
            liens.append((i, j, dh, n))
    groupes = {}
    for i in range(len(it)):
        groupes.setdefault(find(i), []).append(i)
    res, lignes = [], []
    for gid, (_, mem) in enumerate(sorted(groupes.items()), 1):
        # référence : entière, non masquée, puis la plus grande × la plus nette
        ref = max(mem, key=lambda k: (not it[k]["partielle"], not it[k]["masquee"],
                                      it[k]["aire"] * it[k]["nettete"]))
        g = dict(groupe=gid, reference=dict(page=it[ref]["page"], photo=it[ref]["photo"]),
                 membres=[dict(page=it[k]["page"], photo=it[k]["photo"],
                               partielle=it[k]["partielle"]) for k in mem])
        res.append(g)
        for k in mem:
            lignes.append(dict(groupe=gid, page=it[k]["page"], photo=it[k]["photo"],
                               role="reference" if k == ref else "doublon",
                               partielle=it[k]["partielle"], nb_prises=len(mem)))
    write_json(OUT / "groupes.json", res)
    with open(OUT / "doublons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(lignes[0].keys()) if lignes else ["groupe"])
        w.writeheader(); w.writerows(lignes)
    multi = [g for g in res if len(g["membres"]) > 1]
    journal("*", "", "s02_doublons", "phash+orb", parametres=P, duree=time.time() - t,
            remarque=f"{len(it)} photos, {len(multi)} groupes multi-prises")
    print(f"{len(it)} photos -> {len(res)} tirages distincts, {len(multi)} avec plusieurs prises")
    for g in multi:
        print("  groupe", g["groupe"], [(m["page"], m["photo"]) for m in g["membres"]])


if __name__ == "__main__":
    main()
