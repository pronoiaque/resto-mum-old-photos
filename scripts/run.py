"""Orchestrateur : enchaîne les étapes automatiques.
Usage : python run.py --pages 1,19,47 --etapes 1-5,7-8 [--force]
Étapes : 0 inventaire | 1 détection | 2 doublons | 3 fusion | 4 restauration
         5 colorisation | 6 texte (préparer+associer) | 7 métadonnées | 8 rapport
Les étapes « agent » (correction des boîtes, lecture du texte, fusion générative)
se font entre deux appels — voir PROMPT.md."""
import argparse, subprocess, sys
from pathlib import Path

H = Path(__file__).parent
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--pages", default="all")
ap.add_argument("--etapes", default="0-8")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
sel = set()
for part in a.etapes.split(","):
    if "-" in part:
        x, y = map(int, part.split("-")); sel |= set(range(x, y + 1))
    else:
        sel.add(int(part))
F = ["--force"] if a.force else []
P = ["--pages", a.pages]
CMD = {
    0: ["s00_inventaire.py"] + F,
    1: ["s01_detection.py"] + P + F,
    2: ["s02_doublons.py"],
    3: ["s03_fusion.py"] + F,
    4: ["s04_restauration.py"] + P + F,
    5: ["s05_colorisation.py"] + P + F,
    6: ["s06_texte.py", "preparer"] + P,
    7: ["s07_metadonnees.py"] + P,
    8: ["s08_rapport.py", "--planches"] + P,
}
for e in sorted(sel):
    print(f"\n===== étape {e} : {CMD[e][0]} =====", flush=True)
    if e == 6:
        r = subprocess.run([PY, str(H / CMD[e][0])] + CMD[e][1:])
        r = subprocess.run([PY, str(H / "s06_texte.py"), "associer"] + P)
    else:
        r = subprocess.run([PY, str(H / CMD[e][0])] + CMD[e][1:])
    if r.returncode:
        sys.exit(f"étape {e} en échec (code {r.returncode})")
