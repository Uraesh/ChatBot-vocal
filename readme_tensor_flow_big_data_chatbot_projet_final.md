# TensorFlow - IA & Deep Learning sur Grands Volumes

## Theme
15. TensorFlow IA et Deep Learning sur grands volumes

## Resume
Projet de chatbot vocal IA complet: ingestion OpenSubtitles, stockage MongoDB, pipeline STT/NLP/TTS, API FastAPI, interface web locale.

## Outils (format presentation 10-15 min)

### Ingestion OpenSubtitles
- Probleme adresse: transformer un fichier texte massif en paires de dialogue exploitables.
- Architecture interne: streaming lignes -> nettoyage -> paires -> batches -> repository MongoDB.
- Fonctionnement: `iter_clean_lines` + `make_pairs` + `batched` + `insert_many`.
- Cas d'usage (demo): `python run_ingestion.py --file "Fr/fr.txt" --batch-size 1000 --limit 50000`.
- Avantages / limites: streaming efficace; qualite brute du dataset.
- Positionnement dans une architecture Big Data globale: couche ingestion / ETL.

### Stockage MongoDB
- Probleme adresse: stocker gros volume de dialogues + logs d'interactions.
- Architecture interne: collections `dialogues` et `interaction_logs`, indexes utiles.
- Fonctionnement: `mongo_utils.py` + `mongo_ingestion.py` + export CSV.
- Cas d'usage (demo): `GET /exports/conversations.csv`.
- Avantages / limites: schema flexible; besoin de gouvernance data.
- Positionnement dans une architecture Big Data globale: couche stockage NoSQL.

### NLP (TensorFlow + Phi-3)
- Probleme adresse: produire des reponses naturelles en francais.
- Architecture interne: mode `hybrid` = Flan-T5 (NLU) + Phi-3 (generation).
- Fonctionnement: `NLP_BACKEND=rule_based|phi3|hybrid`.
- Cas d'usage (demo): activer `NLP_BACKEND=hybrid` + `NLP_MODEL_PATH`.
- Avantages / limites: bonne qualite; cout CPU et dependances lourdes.
- Positionnement dans une architecture Big Data globale: couche IA / inference.

### STT (Whisper)
- Probleme adresse: convertir la voix en texte.
- Architecture interne: moteur `WhisperSttEngine` (faster-whisper) + fallback.
- Fonctionnement: `STT_BACKEND=whisper`.
- Cas d'usage (demo): endpoint `/voice`.
- Avantages / limites: bonne transcription; temps CPU.
- Positionnement dans une architecture Big Data globale: couche pre-traitement audio.

### TTS (Piper)
- Probleme adresse: generer une reponse audio.
- Architecture interne: `PiperTtsEngine` appelle `piper.exe` + modele `.onnx`.
- Fonctionnement: `TTS_BACKEND=piper`.
- Cas d'usage (demo): generation wav puis mp3 via FFmpeg.
- Avantages / limites: rapide; modele local requis.
- Positionnement dans une architecture Big Data globale: couche sortie audio.

### API FastAPI + pipeline async
- Probleme adresse: orchestrer STT/NLP/TTS avec latence maitrisee.
- Architecture interne: 3 queues async + endpoints REST.
- Fonctionnement: `POST /chat`, `POST /chat/stream`, `POST /voice`.
- Cas d'usage (demo): UI web locale.
- Avantages / limites: simple et lisible; scalabilite limitee sans orchestration.
- Positionnement dans une architecture Big Data globale: couche service.

### UI web locale
- Probleme adresse: fournir une demo rapide.
- Architecture interne: SPA statique, mode dark/light.
- Fonctionnement: appels `fetch` vers l'API.
- Cas d'usage (demo): conversation texte + voix.
- Avantages / limites: rapide a mettre en place; pas destine production.
- Positionnement dans une architecture Big Data globale: couche presentation.

## Etat actuel (MVP implemente)
- Ingestion streaming OpenSubtitles FR vers MongoDB
- API FastAPI avec endpoints texte, streaming et voix
- Pipeline asynchrone STT -> NLP -> TTS
- Interface web locale avec export CSV des interactions
- Moteurs disponibles: NLP `rule_based`, `phi3`, `hybrid`; STT `simple`, `whisper`; TTS `simple`, `piper`

## Evolutions envisagees (hors MVP)
- Kafka entre ingestion et preprocessing (streaming distribue)
- Fine-tuning controle et export modele (TensorFlow)
- Image Docker complete (Whisper, Piper, Phi-3)
- Monitoring et observabilite (Prometheus, traces)
- Deploiement cloud (AWS ou GCP)

Projet realise dans le cadre du module:
TensorFlow - IA & Deep Learning sur Grands Volumes
