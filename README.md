# NoSQL Project - Chatbot vocal IA (TensorFlow / Big Data)

## Theme

15. TensorFlow IA et Deep Learning sur grands volumes

## Resume

Projet de chatbot vocal asynchrone: STT -> NLP -> TTS, API FastAPI, ingestion OpenSubtitles vers MongoDB, interface web locale, export CSV des interactions.

## Objectifs

- Valider le module NoSQL avec ingestion massive et stockage MongoDB.
- Construire un pipeline IA vocal complet et presentable.
- Poser une base evolutive pour une architecture Big Data (Kafka, monitoring, scaling).

## Architecture globale

OpenSubtitles (Fr/fr.txt) -> Ingestion -> MongoDB (dialogues)
Client (UI web) -> API FastAPI -> Pipeline async (STT -> NLP -> TTS) -> Reponse texte + audio
Interactions -> Export CSV (et Mongo optionnel)

## Demarrage rapide

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_api.py
```

Ouvrir l'interface: http://127.0.0.1:8000/

Note: `requirements.txt` installe TensorFlow/Transformers (lourd). Pour un run minimal, installez `fastapi`, `uvicorn`, `pydantic`, `pymongo`.

## Deploiement Render (gratuit, mode leger)

Le plan gratuit Render met le service en veille apres inactivite et reste limite en RAM/CPU. Utilisez le mode leger:
- `render.yaml` configure le service et des variables d'env pour rester light.
- `requirements-render.txt` evite d'installer TensorFlow/Whisper/Piper.

Steps:
1. Connecter le repo a Render.
2. Render detecte `render.yaml` et cree le service.
3. Verifier que les variables d'env sont en mode leger (rule_based / simple).
4. Lancer la web app via l'URL Render et tester `/health`.

## Modules (format presentation 10-15 min)

### 1) Ingestion OpenSubtitles

- Probleme adresse: transformer un fichier texte massif en paires de dialogue exploitables.
- Architecture interne: streaming lignes -> nettoyage -> paires -> batches -> repository MongoDB.
- Fonctionnement: `iter_clean_lines` + `make_pairs` + `batched` + `insert_many`.
- Cas d'usage (demo): `python run_ingestion.py --file "Fr/fr.txt" --batch-size 1000 --limit 50000`.
- Avantages / limites: streaming efficace; qualite des donnees brute et besoin MongoDB.
- Positionnement Big Data: couche ingestion / ETL.

### 2) Stockage MongoDB

- Probleme adresse: stocker gros volume de dialogues + logs d'interactions.
- Architecture interne: collections `dialogues` et `interaction_logs`, indexes utiles.
- Fonctionnement: `mongo_utils.py` + `mongo_ingestion.py` + export CSV.
- Cas d'usage (demo): export `GET /exports/conversations.csv`.
- Avantages / limites: schema flexible; pas de validation forte par defaut.
- Positionnement Big Data: couche stockage NoSQL.

### 3) NLP (TensorFlow + Phi-3)

- Probleme adresse: produire des reponses conversationnelles en francais.
- Architecture interne: mode `hybrid` = Flan-T5 (NLU) + Phi-3 (generation).
- Fonctionnement: `NLP_BACKEND=rule_based|phi3|hybrid`.
- Cas d'usage (demo): activer `NLP_BACKEND=hybrid` + `NLP_MODEL_PATH`.
- Avantages / limites: meilleure qualite en hybride; lourd sur CPU et dependances.
- Positionnement Big Data: couche IA / inference.

### 4) STT (Whisper)

- Probleme adresse: transformer l'audio utilisateur en texte.
- Architecture interne: moteur `WhisperSttEngine` (faster-whisper) + fallback.
- Fonctionnement: `STT_BACKEND=whisper` et params `STT_MODEL_SIZE`, `STT_DEVICE`.
- Cas d'usage (demo): envoyer un audio via `/voice`.
- Avantages / limites: bonne qualite; temps CPU et besoin de modele.
- Positionnement Big Data: couche pre-traitement audio.

### 5) TTS (Piper)

- Probleme adresse: rendre la reponse IA audible.
- Architecture interne: `PiperTtsEngine` appelle `piper.exe` + modele `.onnx`.
- Fonctionnement: `TTS_BACKEND=piper` et `TTS_PIPER_MODEL_PATH`.
- Cas d'usage (demo): generer un wav puis convertir en mp3 avec FFmpeg.
- Avantages / limites: rapide et CPU friendly; modele local requis.
- Positionnement Big Data: couche sortie audio.

### 6) API FastAPI + pipeline async

- Probleme adresse: orchestrer STT/NLP/TTS avec latence maitrisee.
- Architecture interne: 3 queues async (STT, NLP, TTS) + endpoints REST.
- Fonctionnement: `POST /chat`, `POST /chat/stream`, `POST /voice`.
- Cas d'usage (demo): UI web locale + export CSV.
- Avantages / limites: concurrence simple; scalabilite limitee sans orchestration externe.
- Positionnement Big Data: couche service / orchestration.

### 7) UI web locale

- Probleme adresse: demontrer rapidement le pipeline.
- Architecture interne: SPA statique (HTML/CSS/JS) avec mode dark/light.
- Fonctionnement: appels `fetch` vers l'API + micro navigateur.
- Cas d'usage (demo): tester texte et voix en local.
- Avantages / limites: simple et autonome; pas concu pour production.
- Positionnement Big Data: couche presentation / demo.

## Configuration

Les variables d'environnement sont detaillees dans `README_MVP.md`.

## Fichiers importants

- `src/nosql_project/api.py` (API)
- `src/nosql_project/pipeline.py` (pipeline async)
- `src/nosql_project/engines.py` (STT/NLP/TTS)
- `src/nosql_project/ingestion.py` (ingestion)
- `src/nosql_project/mongo_ingestion.py` (ingestion Mongo)
- `src/nosql_project/web/index.html` (UI)

## Roadmap

- Kafka entre ingestion et preprocessing
- Monitoring et observabilite
- Image Docker complete (Whisper, Piper, Phi-3)
- Scalabilite horizontale (workers)
