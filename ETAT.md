# État du projet — point d'étape du 2026-11-15 (soir)

## Où sont les choses
| Quoi | actob (portable, CPU) | pcob (GTX 960 2 Go) |
|---|---|---|
| Code | `~/albums_workflow` | `~/albums_workflow` (copie rsync, pas encore un clone git) |
| Python | venv 3.12 via uv | venv 3.12 `--system-site-packages` (réutilise le torch 2.14.1+cu126 du système) |
| Originaux HEIC (47) + zip | `~/albums/` | `~/albums/` |
| PNG natifs (47, 1,4 Go) | `~/albums/pngnatives/` | `~/albums/pngnatives/` |
| Résultats pipeline (pages 1-4, 47) | `~/albums/out/` | `~/albums/out/` |
| Clé Gemini | `~/.config/albums/gemini.env` (600) | idem |
| Dépôt | github.com/pronoiaque/resto-mum-old-photos (public) | — |

Particularités pcob : Python 3.12 compilé sans `lzma` → stub `_lzma.py` dans le venv
(corriger proprement : `sudo apt install liblzma-dev` + recompiler) ; `torchvision 0.29.1+cu126`
installé dans le venv (--no-deps). Lancer avec `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Ce qui marche (testé)
- Conversion HEIC → PNG natifs (`scripts/convert_heic_png.py`).
- Regroupement des prises multiples (pHash + ORB) et **fusion classique** (recalage
  homographie + ECC + flux optique, empilement pondéré netteté) : 4 prises de la 2CV.
- Restauration fidèle (s04), JPEG de partage (s07, sans exiftool), rapport (s08).
- **Gemini (API, texte seulement)** : détection des tirages (16/16 sur le collage),
  description/époque/lieu. Consignes en anglais.
- **Colorisation** : Space HF `fffiloni/text-guided-image-colorization` guidé par la
  description Gemini (meilleur rendu) ; `colorisation_guidee.py` (Gemini propose
  zones+couleurs en polygones, peinture locale ; fidèle mais grossier, ~9 k tokens).
- **Visages** : Space HF `sczhou/CodeFormer` (fidélité 0,7).
- **Kit application Gemini** (`kit_gemini_app.py`) : consigne avec COLORISATION prioritaire.

## Bloqué / limites
- API Gemini : **toute sortie image refusée** (« not available in your country »),
  quel que soit le modèle (gemini-4-hydrogen, 3.8-flash…). → kit manuel dans l'appli.
- `gemini-3.1-flash-image` : quota dépassé.

## Découpage local (`scripts/decoupe_local.py`) — chantier en cours
Candidats = FastSAM-x + SAM 2.1 tiny (grille de points, image encodée une fois, fp16,
image réduite à 1280 px pour tenir dans 2 Go) + masques « bord papier » OpenCV,
puis tri : rectangularité, tirages recouverts, élimination des « pages », inclusion.
Résultats sur 4 pages test (GPU pcob, 30-70 s/page) :
| Page | Réel | Trouvé |
|---|---|---|
| 20261115_155120 collage IDENTITES | 16 | 14 bons + 2 faux (papillon, tissu) |
| 20261115_154238 6 tirages fond bleu | 5 | 5 ✅ |
| 20261115_154456 photo seule | 1 | 1 ✅ (cadre un peu large) |
| 20261115_153859 gros plan 2CV | 1 + 2 partiels | 6 cadres : à vérifier (morceaux intérieurs probables) |

## Prochaines étapes proposées
1. Regarder `~/albums/decoupes_mix/*/controle.jpg` sur pcob (résultats du dernier correctif).
2. Filtrer les faux positifs hors album (papillon, tissu) : candidats touchant le bord
   de l'image avec texture non photographique.
3. **Hybride** : contrôle automatique de cohérence par page ; pages douteuses →
   `gemini_taches.py detecter` (≈ 6 k tokens/page) ou correction de `boites.json`.
4. Brancher `decoupe_local.py` comme étape 1 du pipeline (aujourd'hui autonome,
   sorties dans `~/albums/decoupes*`), puis lot complet des 47 pages sur pcob (nohup).
5. Option : Claude Code sur pcob (`claude` dans `~/albums_workflow`) pour travailler
   sans actob.

## Reprise
- Depuis actob : `ssh taiji@192.168.1.53` (clé `~/.ssh/id_ed25519` autorisée).
- Sur pcob : `cd ~/albums_workflow && git init && git remote add origin
  https://github.com/pronoiaque/resto-mum-old-photos && git fetch && git reset origin/main`
  pour transformer la copie en clone (les fichiers locaux sont identiques).
