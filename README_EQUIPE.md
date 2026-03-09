# README Equipe - NoSQL Project (Cloture MVP)

## Objectif

Donner une vue unique et finale de:

- ce qui est termine,
- ce qui reste (bloquant vs non bloquant),
- la checklist de cloture projet.

## 1) Etat actuel (termine)

- Cadrage technique corrige:
  - `cahier_des_charges_corrige.md`

- Backend API + pipeline asynchrone:
  - API FastAPI: `src/nosql_project/api.py`
  - Orchestrateur STT -> NLP -> TTS: `src/nosql_project/pipeline.py`
  - Schemas: `src/nosql_project/schemas.py`
  - Moteurs: `src/nosql_project/engines.py`

- Interface utilisateur connectee aux endpoints:
  - UI web sur `GET /`
  - conversation texte (`/chat`) + voix (`/voice`)
  - export CSV session (`/exports/conversations.csv`) pour analyse hallucinations
  - indicateurs pipeline + logs dev
  - refonte style ChatGPT simplifiee (chat + micro + upload audio)
  - fichier: `src/nosql_project/web/index.html`

- Containerisation:
  - `Dockerfile` ajoute (launch via `python run_api.py`)
  - URL de lancement documentee: `http://127.0.0.1:8000/`
  - fichier: `Dockerfile`

- Ingestion data:
  - ingestion streaming OpenSubtitles: `src/nosql_project/ingestion.py`
  - ingestion MongoDB batch + index: `src/nosql_project/mongo_ingestion.py`
  - lanceur: `run_ingestion.py`
  - test reel execute: `attempted=10 inserted=10`

- Qualite code:
  - `python -m pytest -q` -> `22 passed`
  - `python -m pylint src/nosql_project tests` -> `10.00/10`
  - `python -m pyright` -> `0 errors, 0 warnings`

- NLP reel active en TensorFlow:
  - backend `transformers` integre
  - fallback anti-echo ajoute pour reponses conversationnelles propres

## 2) Ce qui reste a faire

### A. Bloquant pour cloture demo finale

- Validation terrain moteurs reels (vrais fichiers audio):
  - verifier qualite STT (erreurs de transcription)
  - verifier qualite TTS (audio audible, clair, non sature)
  - mesurer latence moyenne `/voice` sur 10-20 essais

### B. Non bloquant MVP (Phase 2)

- Kafka entre ingestion et preprocessing:
  - utile pour architecture distribuee
  - non requis pour valider le pipeline asynchrone actuel

## 3) Checklist de cloture (a executer)

1. Activer les moteurs reels et lancer API

```powershell
.\.venv\Scripts\Activate.ps1
$env:NLP_BACKEND="transformers"
$env:NLP_MODEL_NAME="google/flan-t5-small"
$env:NLP_FALLBACK_TO_RULE_BASED="true"
$env:STT_BACKEND="whisper"
$env:TTS_BACKEND="piper"
python run_api.py
```

1. Tester parcours complet sur interface

- Ouvrir `http://127.0.0.1:8000/`
- Envoyer 5 messages texte
- Envoyer 5 audios via upload et/ou micro
- Confirmer retour texte + audio sur `/voice`

1. Revalider qualite code avant livraison

```powershell
python -m pytest -q
python -m pylint src/nosql_project tests
python -m pyright
```

1. Verrouiller le livrable

- garder `README_MVP.md` comme guide execution
- garder ce fichier comme etat final equipe
- tagger une version de livraison (si repo git actif)

## 4) Definition of Done (MVP)

- Backend: `/chat` et `/voice` repondent sans blocage.
- Pipeline: STT/NLP/TTS actifs avec indicateurs visibles dans UI.
- Data: ingestion `fr.txt` vers MongoDB fonctionnelle.
- Qualite: tests + lint + type-check verts.
- Demo: utilisateur peut parler et recevoir une reponse texte + audio.

## 5) Repartition finale

Contexte actuel: execution centralisee par Daniel pour finalisation complete.

- Backend/API/Pipeline: Daniel
- IA (NLP/STT/TTS): Daniel
- Data/Mongo ingestion: Daniel
- Frontend integration: Daniel

## 6) Commandes utiles

- Demarrer API:
  - `python run_api.py`

- Ingestion MongoDB:
  - `python run_ingestion.py --file "Fr/fr.txt" --batch-size 1000 --limit 50000`

- Tests:
  - `python -m pytest -q`

- Qualite:
  - `python -m pylint src/nosql_project tests`
  - `python -m pyright`
