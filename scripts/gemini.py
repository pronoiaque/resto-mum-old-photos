"""Client Gemini minimal (API REST, sans SDK) avec garde-fou de crédits.

Clé : variable GEMINI_API_KEY, sinon fichier ~/.config/albums/gemini.env (GEMINI_API_KEY=...).
JAMAIS dans le projet ni dans le zip.
Chaque appel est compté dans out/logs/gemini_couts.csv ; au-delà des plafonds de
config.yaml (gemini.plafond_*), l'appel est refusé (exception Plafond).
"""
import base64, csv, io, json, os, time, urllib.error, urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image

from common import C, OUT

G = C.get("gemini", {})
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
COUTS = OUT / "logs" / "gemini_couts.csv"


class Plafond(Exception):
    pass


def cle():
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    f = Path(os.path.expanduser(G.get("fichier_cle", "~/.config/albums/gemini.env")))
    if f.exists():
        for l in f.read_text().splitlines():
            if l.startswith("GEMINI_API_KEY="):
                return l.split("=", 1)[1].strip()
    raise SystemExit("Clé Gemini introuvable (GEMINI_API_KEY ou ~/.config/albums/gemini.env)")


def compteur():
    n = {"texte": 0, "image": 0, "tokens": 0}
    if COUTS.exists():
        with open(COUTS, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["statut"] == "ok":
                    n[r["genre"]] += 1
                    n["tokens"] += int(r["tokens_total"] or 0)
    return n


def _noter(modele, genre, usage, duree, statut, objet):
    COUTS.parent.mkdir(parents=True, exist_ok=True)
    new = not COUTS.exists()
    with open(COUTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["horodatage", "modele", "genre", "objet", "tokens_entree",
                        "tokens_sortie", "tokens_total", "duree_s", "statut"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), modele, genre, objet,
                    usage.get("promptTokenCount", ""), usage.get("candidatesTokenCount", ""),
                    usage.get("totalTokenCount", ""), f"{duree:.1f}", statut])


def _img_part(im, max_px):
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.convert("RGB")
    im.thumbnail((max_px, max_px))
    b = io.BytesIO(); im.save(b, "JPEG", quality=92)
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(b.getvalue()).decode()}}


def appeler(modele, consigne, images=(), genre="texte", json_schema=None, objet="",
            max_px=None, temperature=None, max_tokens=None):
    """genre='texte' -> renvoie str (ou objet JSON si json_schema) ; genre='image' -> PIL.Image"""
    plaf = G.get(f"plafond_appels_{genre}", 0)
    n = compteur()
    if plaf and n[genre] >= plaf:
        raise Plafond(f"plafond {genre} atteint ({n[genre]}/{plaf}) : relever gemini.plafond_appels_{genre}")
    max_px = max_px or G.get("max_px", 2048)
    parts = [{"text": consigne}] + [_img_part(i, max_px) for i in images]
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {}}
    if max_tokens:
        body["generationConfig"]["maxOutputTokens"] = max_tokens
    if temperature is not None:
        body["generationConfig"]["temperature"] = temperature
    if json_schema:
        body["generationConfig"]["responseMimeType"] = "application/json"
        if json_schema is not True:
            body["generationConfig"]["responseSchema"] = json_schema
    if genre == "image":
        body["generationConfig"]["responseModalities"] = ["TEXT", "IMAGE"]
    req = urllib.request.Request(URL.format(m=modele), json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-goog-api-key": cle()})
    t = time.time()
    for essai in range(int(C.get("retries", 3))):
        try:
            with urllib.request.urlopen(req, timeout=G.get("timeout_s", 300)) as r:
                rep = json.load(r)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:400]
            if e.code in (429, 500, 503) and essai < 2:
                time.sleep(20 * (essai + 1)); continue
            _noter(modele, genre, {}, time.time() - t, f"http{e.code}", objet)
            raise RuntimeError(f"Gemini HTTP {e.code}: {msg}")
        except (TimeoutError, OSError) as e:          # socket.timeout, coupure réseau
            if essai < 1:
                continue
            _noter(modele, genre, {}, time.time() - t, "timeout", objet)
            raise RuntimeError(f"Gemini délai dépassé : {e}")
    usage = rep.get("usageMetadata", {})
    parts = ((rep.get("candidates") or [{}])[0].get("content") or {}).get("parts", [])
    if genre == "image":
        for p in parts:
            d = p.get("inlineData") or p.get("inline_data")
            if d and d.get("data"):
                _noter(modele, genre, usage, time.time() - t, "ok", objet)
                return Image.open(io.BytesIO(base64.b64decode(d["data"])))
        _noter(modele, genre, usage, time.time() - t, "sans_image", objet)
        raise RuntimeError("pas d'image renvoyée : " + json.dumps(rep)[:400])
    txt = "".join(p.get("text", "") for p in parts)
    _noter(modele, genre, usage, time.time() - t, "ok", objet)
    if json_schema:
        brut = OUT / "logs" / "gemini_derniere_reponse.txt"
        brut.write_text(txt, encoding="utf-8")       # pour diagnostic si JSON invalide
        txt = txt.strip().removeprefix("```json").removesuffix("```") if hasattr(txt, "removeprefix") else txt
        txt = txt.strip().strip("`")
        if txt.startswith("json"):
            txt = txt[4:]
        return json.loads(txt)
    return txt
