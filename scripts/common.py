"""Outils communs : configuration, chemins, journal, appels IA génériques."""
import csv, json, os, shlex, shutil, subprocess, sys, time, hashlib
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = Path(os.environ.get("ALBUMS_CONFIG", ROOT / "config.yaml"))


def cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    for k in ("source_dir", "out_dir"):
        c[k] = Path(os.path.expanduser(c[k]))
    return c


C = cfg()
OUT = C["out_dir"]
LOG = OUT / "logs" / "journal.csv"
LOG_COLS = ["horodatage", "page", "photo", "etape", "outil", "voie",
            "parametres", "duree_s", "statut", "remarque"]


def page_dir(page):
    d = OUT / f"page_{int(page):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def journal(page, photo, etape, outil, voie="local", parametres="",
            duree=0.0, statut="ok", remarque=""):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), page, photo,
                    etape, outil, voie,
                    parametres if isinstance(parametres, str) else json.dumps(parametres, ensure_ascii=False),
                    f"{duree:.1f}", statut, remarque])


def read_json(p, default=None):
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def write_json(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pages_index():
    """Liste des pages retenues : [{page, fichier, ...}]"""
    idx = OUT / "index_pages.csv"
    if not idx.exists():
        sys.exit("index_pages.csv absent : lancer d'abord s00_inventaire.py")
    with open(idx, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["retenu"] == "o"]


def select_pages(arg):
    """'all' | '1,3,5' | '2-6' -> liste d'entiers."""
    allp = [int(r["page"]) for r in pages_index()]
    if not arg or arg == "all":
        return allp
    res = []
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            res += list(range(int(a), int(b) + 1))
        else:
            res.append(int(part))
    return [p for p in res if p in allp]


def photos_of(page):
    """Photos découpées d'une page : liste de dicts depuis photos.json."""
    return read_json(page_dir(page) / "photos.json", [])


# ---------------------------------------------------------------- IA générique
def run_ai(kind, inp, out, extra=None):
    """Exécute le traitement IA `kind` (upscale, visages, colorisation, ocr,
    segmentation) selon config.yaml -> ai.<kind> :
      backend: none | cmd | space
      cmd:    gabarit shell avec {in} {out} (ex. realesrgan-ncnn-vulkan -i {in} -o {out} -s 2)
      space:  identifiant Space HF, api_name, args (liste ; "{in}" = fichier image)
    Retourne (ok, voie, remarque)."""
    a = (C.get("ai") or {}).get(kind) or {}
    backend = a.get("backend", "none")
    inp, out = str(inp), str(out)
    if backend == "none":
        return False, "aucune", f"{kind} désactivé"
    if backend == "cmd":
        cmd = a["cmd"].format(**{"in": shlex.quote(inp), "out": shlex.quote(out)}, **(extra or {}))
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ok = r.returncode == 0 and Path(out).exists()
        return ok, "local", "" if ok else (r.stderr or r.stdout)[-400:]
    if backend == "gemini":
        import gemini
        from PIL import Image
        consignes = {
            "colorisation": "Colorize this black-and-white photograph realistically, with natural "
                            "period-accurate colors. Keep every detail, face, expression and the "
                            "framing strictly identical; change only the colors.",
            "visages": "Restore the faces in this old photograph subtly: reduce blur and noise "
                       "only. Keep identity, expressions and everything else identical. No colorization.",
            "upscale": "Upscale this photograph to twice the resolution, faithfully, without "
                       "adding any content. Keep grain and tones. No colorization.",
        }
        try:
            im = gemini.appeler(C["gemini"]["modele_image"], a.get("consigne", consignes.get(kind, "")),
                                [inp], genre="image", objet=f"{kind} {Path(inp).name}")
            im.save(out)
            return True, "en_ligne", C["gemini"]["modele_image"]
        except Exception as e:
            return False, "en_ligne", repr(e)[:300]
    if backend == "space":
        from gradio_client import Client, handle_file
        tries = int(C.get("retries", 3))
        last = ""
        for i in range(tries):
            try:
                client = Client(a["space"], token=os.environ.get("HF_TOKEN") or None, verbose=False)
                envoi = inp
                if a.get("max_px"):                       # réduire avant envoi
                    from PIL import Image
                    im = Image.open(inp); im.thumbnail((a["max_px"],) * 2)
                    envoi = str(Path(out).with_suffix(".envoi.png")); im.save(envoi)
                args = []
                for x in a.get("args", ["{in}"]):
                    if x == "{in}":
                        args.append(handle_file(envoi))
                    elif isinstance(x, str) and x.startswith("{") and extra and x[1:-1] in extra:
                        args.append(extra[x[1:-1]])
                    else:
                        args.append(x)
                res = client.predict(*args, api_name=a.get("api_name", "/predict"))
                path = _first_file(res, a.get("output_index"))
                if not path:
                    return False, "en_ligne", f"sortie inattendue : {str(res)[:300]}"
                if isinstance(path, str) and path.startswith("http"):
                    import urllib.request
                    urllib.request.urlretrieve(path, out)
                else:
                    shutil.copy(path, out)
                if envoi != inp:
                    Path(envoi).unlink(missing_ok=True)
                return True, "en_ligne", a["space"]
            except Exception as e:  # quota, file d'attente, Space en veille…
                last = repr(e)[:400]
                time.sleep(10 * (i + 1))
        return False, "en_ligne", "en_attente : " + last
    return False, "?", f"backend inconnu {backend}"


def _first_file(res, idx=None):
    if idx is not None and isinstance(res, (list, tuple)):
        res = res[idx]
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for k in ("path", "url", "image", "value"):
            if k in res:
                return _first_file(res[k])
    if isinstance(res, (list, tuple)):
        for r in res:
            p = _first_file(r)
            if p:
                return p
    return None
