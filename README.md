# albums_workflow

Prises smartphone de pages d'album → une image restaurée par tirage, fusionnée à partir
de toutes ses prises, avec légendes et métadonnées.

## Démarrage sur une autre machine
```bash
unzip albums_workflow.zip && cd albums_workflow
bash setup.sh            # CPU ; machine NVIDIA avec torch CUDA déjà installé : bash setup_gpu.sh
# copier les HEIC dans ~/albums (ou modifier source_dir / out_dir dans config.yaml)
mkdir -p ~/.config/albums && echo 'GEMINI_API_KEY=...' > ~/.config/albums/gemini.env && chmod 600 ~/.config/albums/gemini.env
claude --model opus      # puis /effort low  (orchestrateur)
```
Puis demander : « Lance le workflow de PROMPT.md ». Les agents Sonnet (effort moyen)
sont dans `.claude/agents/` et sont lancés par l'orchestrateur.

## Étapes
| # | Script | Rôle | Sortie principale |
|---|--------|------|-------------------|
| 0 | s00_inventaire | sources, doublons fichiers, conversion, diagnostic | index_pages.csv, page_XX/source.png |
| 1 | s01_detection | boîtes (manuel > SAM > OpenCV), redressement, orientation | photo_YY_raw.png, controle_decoupe.jpg |
| – | outil_grille | aperçu quadrillé pour saisir les boîtes à la main | apercu_grille.jpg |
| 2 | s02_doublons | regroupe les prises d'un même tirage (pHash + ORB) | groupes.json, doublons.csv |
| 3 | s03_fusion | fusion multi-prises déterministe | photo_YY_fusion.png |
| – | fusion_generative | fusion multi-prises par Gemini (IA) | photo_YY_fusion_ia.png |
| 4 | s04_restauration | restauration fidèle + upscale / visages optionnels | photo_YY_restaure*.png, photo_YY.json |
| 5 | s05_colorisation | colorisation IA des N&B | photo_YY_couleur.png |
| 6 | s06_texte | tuiles pour lecture, association texte ↔ photo | texte_brut.json (agent), association.json |
| 7 | s07_metadonnees | JPEG de partage + XMP | photo_YY_partage.jpg |
| 8 | s08_rapport | rapport, page de relecture, planches | rapport.csv, a_relire.html |

Backends IA (`config.yaml` → `ai.*`) : `none` | `cmd` (outil local, gabarit `{in}` `{out}`)
| `space` (Space Hugging Face via gradio_client). Test : `scripts/tester_ai.py`.

## GPU (testé sur GTX 960, 2 Go)
`bash setup_gpu.sh` réutilise le torch CUDA du python3.12 système (venv `--system-site-packages`),
ajoute torchvision assorti (--no-deps), ultralytics, OpenCV 4.x, et un remplaçant `_lzma` si Python
a été compilé sans liblzma. `bash setup_gpu.sh --test` vérifie la pile.
Découpage : `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True venv/bin/python scripts/decoupe_local.py`
(FastSAM-x + SAM 2.1 tiny, 20-70 s/page, ~200 Mo à 1,5 Go de mémoire GPU).

## Gemini
`scripts/gemini.py` (REST, sans SDK, compte des appels, plafonds), `scripts/gemini_taches.py`
(détection des tirages, lecture du texte), `fusion_generative.py`, backend `gemini` pour `ai.*`.
Lister les modèles du compte :
`curl -s -H "x-goog-api-key: $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1beta/models | grep '"name"'`

## Contournements du blocage image Gemini
| Besoin | Solution | Script |
|---|---|---|
| Fusion IA des prises | application Gemini, kit préparé + import | kit_gemini_app.py |
| Colorisation | Space HF guidé par la description Gemini | s05_colorisation.py |
| Colorisation sans génération | zones + couleurs par Gemini (texte), peinture locale | colorisation_guidee.py |
| Visages | Space HF CodeFormer | s04 (ai.visages) |

## État des tests (machine d'origine : CPU Intel, pas de GPU)
- Testés sur des images réelles : 0, 1 (boîtes manuelles + OpenCV), 2, 3, 4, 7 (sans exiftool), 8.
  Fusion vérifiée sur 4 prises d'un même tirage (recalage OK, bruit réduit).
- Gemini détection (gemini-3.8-flash) : testé, 16/16 sur la page collage, ~107 s/page.
- Génération d'image Gemini (gemini-4-hydrogen) : REFUSÉE par l'API depuis la machine
  d'origine (« not available in your country ») ; code prêt, à retester ailleurs.
- Spaces HF testés OK : sczhou/CodeFormer, fffiloni/text-guided-image-colorization
  (client Gradio ≥ 2 → Python ≥ 3.10, setup.sh utilise uv).
- Gemini analyse (description, époque, lieu) : testé. Kit application : préparation testée.
- Écrits mais NON testés : seg_sam_local.py, gemini_taches texte, backends `space`
  (upscale, visages, colorisation, OCR), étape 6 complète, écriture XMP.
- Détection OpenCV seule : insuffisante sur les pages collage → SAM ou boîtes manuelles.
