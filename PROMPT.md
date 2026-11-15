# Mission : numériser et restaurer un album photo à partir de prises au smartphone

## Organisation multi-agents
- **Toi = orchestrateur** (session principale : Opus, effort bas). Tu ne fais PAS le travail
  page par page : tu planifies, tu lances les scripts globaux (étapes 0, 2, 3, 7, 8),
  tu délègues, tu agrèges, tu présentes aux points d'arrêt, tu surveilles les crédits.
- **`album-page`** (Sonnet, effort moyen, `.claude/agents/album-page.md`) : une instance
  par page ou par lot de 3 à 5 pages, lancées EN PARALLÈLE (détection Gemini, contrôle
  visuel, correction des boîtes, texte). Donne-lui la liste exacte des pages.
- **`album-qualite`** (Sonnet, effort moyen) : contrôle des images IA (invention ?).
- Agents Gemini (scripts, pas d'agent Claude) : `gemini.modele_vision` pour la détection
  et le texte, `gemini.modele_image` pour fusion, restauration et colorisation.
- Consignes envoyées à Gemini : en ANGLAIS (déjà écrites dans les scripts) ; les
  transcriptions restent dans la langue d'origine.
- Crédits : `out/logs/gemini_couts.csv` ; plafonds dans `config.yaml` (`gemini.plafond_*`).
  Ne relève jamais un plafond sans mon accord. Annonce le coût estimé avant un lot.
- Clé Gemini : env `GEMINI_API_KEY` ou `~/.config/albums/gemini.env`. Jamais dans le dossier.
- ATTENTION : l'API Gemini refuse TOUTE sortie image depuis la zone de l'utilisateur
  (« Image generation is not available in your country », vérifié sur 3 modèles, demande
  seule ou groupée). Le texte et l'analyse d'image fonctionnent. Donc :
  * fusion multi-prises : classique (s03) + **kit application Gemini** (manuel) :
    `scripts/kit_gemini_app.py preparer [--sans-couleur]` → `out/kit_gemini/index.html` ;
    l'utilisateur dépose `resultat.png` dans chaque dossier, puis `... importer` ;
  * colorisation : Space HF `fffiloni/text-guided-image-colorization` (s05, guidé par
    la description Gemini), ou `scripts/colorisation_guidee.py` (Gemini propose zones +
    couleurs en JSON, peinture locale : luminance intacte, rien d'inventé) ;
  * visages : Space HF `sczhou/CodeFormer` (fidélité 0,7) ;
  * retente l'API image une fois (`fusion_generative.py --groupes <n>`) : si elle passe
    un jour, elle remplace le kit manuel.

Tu pilotes le workflow de ce dossier (`scripts/`, `config.yaml`). Travaille en français,
prudemment, et arrête-toi aux points de validation. Les scripts font le travail
répétitif ; toi, tu regardes les images (vision), tu corriges et tu décides.

## Ce qu'on sait des sources (à revérifier, ne pas surinterpréter)
- Dossier source (`source_dir`) : HEIC Galaxy S25 (6112×6112 surtout, quelques recadrages
  portrait), plus éventuellement un zip qui les duplique. Le zip d'origine contenait aussi
  2 fichiers hors sujet (capture de galerie, tableau de recettes), déjà exclus dans `config.yaml`.
- Prises hétérogènes : photo isolée sur fond uni, page entière de plusieurs tirages,
  pages « collage » (tirages qui se chevauchent, inclinés, à l'envers, bords dentelés,
  reliure à spirale, décor autour, ombre du téléphone), gros plans qui coupent les tirages.
- **Un même tirage est souvent photographié plusieurs fois** (vue d'ensemble, puis
  1 à 4 gros plans). C'est voulu : ces prises servent à produire UNE meilleure image.
- Tirages surtout N&B ou sépia, parfois couleur délavée (années 70).
- Texte : titres imprimés (ex. « IDENTITES »), manuscrit possible.
- Usage final : partage numérique.

## Règles
1. Originaux en LECTURE SEULE. Tout s'écrit dans `out_dir`.
2. Envoi en ligne autorisé (Spaces / Inference Hugging Face, API Gemini).
   Sur une machine avec un GPU correct, préfère le local (backend `cmd`).
3. Avant d'utiliser un backend IA : `venv/bin/python scripts/tester_ai.py`.
   Les Spaces changent souvent : cherche-les avec le MCP Hugging Face, puis vérifie
   leur `api_name` et leurs arguments (`Client(space).view_api()`) avant de remplir `config.yaml`.
4. Demande mon accord avant tout `sudo apt install`.
5. Note dans `out/logs/journal.csv` (les scripts le font déjà) ; ajoute tes propres
   décisions manuelles avec la fonction `journal()` ou dans la colonne remarque.
6. Ne force jamais un résultat douteux : marque-le « à relire ».

## Déroulé

### 0. Installation et inventaire
`bash setup.sh [--gpu]` ; vérifie `config.yaml` ; puis
`venv/bin/python scripts/run.py --etapes 0`.
Regarde les `page_XX/source_preview.jpg` (une planche avec `montage` aide), complète
`type_prise` dans `out/index_pages.csv`, et ajoute aux `exclure` les fichiers hors sujet.
Repère déjà quelles pages sont des prises multiples du même tirage ou de la même page.

### 1. Détection et découpe
`venv/bin/python scripts/run.py --pages <sel> --etapes 1`
- Méthode par défaut : `detection.methode: gemini` (testé : 16/16 tirages sur une page
  collage, photo à l'envers repérée ; coins parfois approximatifs → contrôle visuel).
- Alternative : la segmentation SAM (`ai.segmentation`, en `cmd` avec
  `scripts/seg_sam_local.py` sur GPU, ou un Space) est nettement meilleure qu'OpenCV.
  OpenCV seul (repli) rate souvent les pages collage et prend des détails pour des photos.
- **Contrôle visuel obligatoire** de chaque `controle_decoupe.jpg`. Si c'est faux :
  `venv/bin/python scripts/outil_grille.py <page>`, lis les coins sur `apercu_grille.jpg`,
  écris `page_XX/boites_manuelles.json` :
  `[{"id":1,"repere":"apercu","quad":[[x,y],[x,y],[x,y],[x,y]],"rotation":0,"masquee":false,"note":""}]`
  (`rotation` facultatif : 0/90/180/270 force l'orientation ; sinon détection par visages),
  puis relance l'étape 1 avec `--force` sur cette page. Les boîtes manuelles priment toujours.
- Inclure le liseré blanc du tirage, exclure fond, reliure, texte de page.
- Photo coupée par le cadrage = `partielle` (automatique) ; recouverte = `masquee: true`.

>>> ARRÊT 1 : montre-moi une planche de contrôle par type de prise et attends mon feu vert.

### 2. Regroupement des prises multiples
`--etapes 2` → `out/groupes.json`, `out/doublons.csv`. Vérifie les groupes à l'œil
(planche des membres) : corrige à la main dans `groupes.json` les faux regroupements
ou les oublis. La référence = prise entière, la plus grande et la plus nette.

### 3. Fusion multi-prises → la « meilleure » image de chaque tirage
- **Classique** (`--etapes 3`, déterministe, aucune invention) : recalage
  homographie + ECC + flux optique, puis empilement pondéré par la netteté qui rejette
  reflets et pixels aberrants → `photo_YY_fusion.png`. Gain : moins de bruit, reflets
  et ombres compensés. Ne crée pas de détail au-delà de la meilleure prise.
- **Générative** (`venv/bin/python scripts/fusion_generative.py`, `gemini.modele_image`) :
  envoie toutes les prises d'un tirage avec une consigne stricte de fidélité
  → `photo_YY_fusion_ia.png`. Plus spectaculaire, mais peut inventer : compare
  visages, mains, texte et arrière-plan à la fusion classique avant de la retenir.
  Pour l'utiliser comme entrée de la restauration : `restauration.entree: fusion_ia`.
  `--aussi-uniques` l'applique aussi aux tirages à une seule prise.

### 4. Restauration fidèle
`--etapes 4` → `photo_YY_restaure.png` (classement nb/couleur, éclairage optionnel,
niveaux, CLAHE, débruitage, netteté). Options via `ai.upscale` (Real-ESRGAN, si petit)
et `ai.visages` (CodeFormer, fidélité ≥ 0,7, fichier séparé). Contrôle à l'œil,
surtout les hautes lumières ; ajuste `restauration.*` si c'est trop dur ou trop mou.
Rayures : signalées (`rayures_pct`), jamais retouchées.

### 5. Colorisation (N&B seulement, toujours marquée IA)
`--etapes 5` avec `ai.colorisation` (DDColor ou DeOldify). Vérifie la cohérence
des teintes au sein d'une même page ou série.

### 6. Texte
`--etapes 6` prépare `page_XX/texte_tuiles/` (tuiles agrandies + `tuiles.json` pour
convertir en coordonnées d'aperçu). Lecture : `scripts/gemini_taches.py texte --pages <sel>`
(écrit `texte_brut.json`), puis relecture visuelle par l'agent `album-page` ; ou lecture
directe par l'agent. Format de `page_XX/texte_brut.json` :
`[{"id":1,"repere":"apercu","boite":[x1,y1,x2,y2],"type":"manuscrit","texte":"Annecy 1961 [Lucie?]","confiance":"moyenne","portee":"photo|page"}]`
puis `venv/bin/python scripts/s06_texte.py associer --pages <sel>`.
Ambigu → `a_relire.json` ; jamais d'association forcée.

### 7–8. Métadonnées, partage, rapport
`--etapes 7-8` (`s07_metadonnees.py --depuis restaure|couleur|visages` pour choisir
la version partagée) → `photo_YY_partage.jpg`, XMP via exiftool, `out/rapport.csv`,
`out/a_relire.html`, planches `photo_YY_avant_apres.jpg`.

## ARRÊT 2 : POC
Choisis 3 à 5 pages variées, **dont au moins un tirage photographié plusieurs fois**
et une page collage. Présente :
- tableau par page : photos détectées / partielles / masquées / pivotées, groupes
  multi-prises, textes et confiances, voies IA, durées ;
- planches de contrôle de découpe ;
- pour un tirage multi-prises : meilleure prise seule | fusion classique | fusion IA ;
- planches avant/après (restauration, visages, colorisation) ;
- difficultés, Spaces défaillants, paramètres proposés.
N'entame pas le lot complet avant ma validation (je choisirai : fusion IA oui/non,
visages oui/non, colorisation oui/non, version de partage).

## Après validation
Fige `config.yaml` (copie datée), traite le reste par lots (bilan court par lot ; arrêt si
> 20 % d'échecs), refais l'étape 2 sur l'ensemble AVANT les fusions, puis rapport final.
