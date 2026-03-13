# README Equipe - NoSQL Project (Cloture MVP)

## Objectif
Donner une vue finale de l'etat du MVP, des points restants et de la checklist.

## 1) Etat actuel (termine)
- Cadrage technique corrige: `cahier_des_charges_corrige.md`
- Backend API FastAPI + pipeline asynchrone STT -> NLP -> TTS
- Moteurs disponibles: NLP `rule_based`, `phi3`, `hybrid`; STT `simple`, `whisper`; TTS `simple`, `piper`
- UI web locale connectee aux endpoints (dark/light): `src/nosql_project/web/index.html`
- Endpoints exposes: `GET /`, `GET /health`, `POST /chat`, `POST /chat/stream`, `POST /voice`, `GET /exports/conversations.csv`
- Historique de conversation en memoire + export CSV; persistance Mongo optionnelle
- Ingestion OpenSubtitles vers MongoDB: `src/nosql_project/ingestion.py`, `src/nosql_project/mongo_ingestion.py`, `run_ingestion.py`
- Containerisation MVP: `Dockerfile` (image minimale)
- Tests existants: `tests/` (API, pipeline, ingestion, Mongo ingestion, engines)
- Outils locaux: Piper portable et FFmpeg disponibles dans `tools/`

## 2) Ce qui reste a faire
### A. Bloquant pour cloture demo finale
- Validation terrain moteurs reels (audio wav)
- Mesurer latence moyenne de `/voice` sur 10 a 20 essais
- Verifier qualite STT et TTS (transcription et audio)

### B. Non bloquant MVP (Phase 2)
- Produire une image Docker complete (Whisper, Piper, Phi-3 et modeles)
- Ajouter un bus Kafka entre ingestion et preprocessing (architecture distribuee)
- Ajouter monitoring (metrics, traces)

## 3) Checklist de cloture (a executer)
1. Activer les moteurs reels et lancer l'API
```powershell
.\.venv\Scripts\Activate.ps1
$env:NLP_BACKEND="hybrid"
$env:NLP_MODEL_NAME="google/flan-t5-small"
$env:NLP_MODEL_PATH="D:\Licience 3 IA-BD\No sql\NoSql Project\models\Phi-3-mini-4k-instruct-q4.gguf"
$env:STT_BACKEND="whisper"
$env:TTS_BACKEND="piper"
python run_api.py
```

2. Tester le parcours complet sur l'interface
Ouvrir `http://127.0.0.1:8000/`
Envoyer 5 messages texte
Envoyer 5 audios via upload ou micro
Confirmer reponse texte et audio sur `/voice`

3. Revalider la qualite avant livraison
```powershell
python -m pytest -q
python -m pylint src/nosql_project tests
python -m pyright
```

4. Verrouiller le livrable
Garder `README.md` comme explication globale
Garder `README_MVP.md` comme guide d'execution
Garder ce fichier comme etat final equipe
Tagger une version de livraison (si repo git actif)

## 4) Definition of Done (MVP)
- Backend: `/chat` et `/voice` repondent sans blocage.
- Pipeline: STT/NLP/TTS actifs avec indicateurs visibles dans UI.
- Data: ingestion `fr.txt` vers MongoDB fonctionnelle.
- Qualite: tests, lint et type-check executes ou documentes.
- Demo: utilisateur peut parler et recevoir une reponse texte + audio.

## 5) Repartition finale
Contexte actuel: execution centralisee par Daniel pour finalisation complete.

- Backend/API/Pipeline: Daniel
- IA (NLP/STT/TTS): Daniel
- Data/Mongo ingestion: Daniel
- Frontend integration: Daniel

## 6) Commandes utiles
- Demarrer l'API:
`python run_api.py`

- Ingestion MongoDB:
`python run_ingestion.py --file "Fr/fr.txt" --batch-size 1000 --limit 50000`

- Tests:
`python -m pytest -q`

- Qualite:
`python -m pylint src/nosql_project tests`
`python -m pyright`
