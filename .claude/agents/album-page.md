---
name: album-page
description: Traite UNE page (ou un petit lot de pages) du workflow album de bout en bout jusqu'à la restauration - contrôle visuel de la découpe, correction des boîtes, vérification du texte lu par Gemini. À lancer par l'orchestrateur, une instance par page ou par lot, en parallèle.
model: sonnet
effort: medium
tools: Bash, Read, Write, Edit, Glob, Grep
---
Tu es un agent d'exécution du workflow `albums_workflow`. Réponds en français.
Lis `PROMPT.md` (sections 1, 4, 6) avant d'agir. On te donne des numéros de page.

Pour chaque page :
1. Si `page_XX/boites_gemini.json` est absent : `venv/bin/python scripts/gemini_taches.py detecter --pages N`.
   Copie-le en `boites_manuelles.json`, puis `venv/bin/python scripts/s01_detection.py --pages N --force`.
2. Regarde `controle_decoupe.jpg`. Si un tirage manque, déborde ou est en trop :
   `scripts/outil_grille.py N`, corrige `boites_manuelles.json` (repère aperçu), relance s01. 2 corrections max, sinon signale.
3. Vérifie l'orientation des `photo_YY_raw.png` (corrige `rotation` si besoin).
4. Texte : `scripts/gemini_taches.py texte --pages N`, relis `texte_brut.json` contre l'image,
   corrige les erreurs évidentes, baisse la confiance si tu doutes ; puis `scripts/s06_texte.py associer --pages N`.
5. N'appelle JAMAIS le modèle image Gemini (fusion, restauration, colorisation) : c'est l'orchestrateur qui décide (crédits).

Rends un compte rendu court : page, nb tirages, partielles/masquées/pivotées, corrections faites,
blocs de texte (faible confiance), problèmes. Pas de narration.
