# albums_workflow

Workflow de numérisation d'album photo (prises smartphone → photos restaurées, fusionnées,
légendées). Toujours répondre en français.

- Mission complète, règles et points d'arrêt : **lire `PROMPT.md` avant toute action.**
- Installation : `bash setup.sh [--gpu]` ; réglages : `config.yaml`.
- Orchestrateur : `venv/bin/python scripts/run.py --pages <sel> --etapes <n-m>`.
- Les originaux (`source_dir`) sont en lecture seule.
- `README.md` : architecture et fichiers produits.
