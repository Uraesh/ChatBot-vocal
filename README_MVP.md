# MVP Async Pipeline - Demarrage Rapide

## Structure
- `cahier_des_charges_corrige.md`: cadrage corrige et exploitable.
- `README_EQUIPE.md`: suivi equipe (fait/restant/repartition a 4).
- `src/nosql_project/`: code source API + pipeline asynchrone + ingestion.
- `tests/`: tests unitaires de base.

## Lancer l'API
```powershell
python run_api.py
```

## Lancer avec Docker
1. Construire l'image:
```powershell
docker build -t nosql-chatbot .
```
2. Demarrer le conteneur:
```powershell
docker run --name nosql-chatbot -p 8000:8000 nosql-chatbot
```
3. Ouvrir l'interface:
```text
http://127.0.0.1:8000/
```

Option: changer le port expose:
```powershell
docker run --name nosql-chatbot -p 8080:8000 -e PORT=8000 nosql-chatbot
```
Puis ouvrir:
```text
http://127.0.0.1:8080/
```

## Endpoints
- `GET /` (interface web locale texte + voix)
- `GET /health`
- `POST /chat`
- `POST /voice`
- `GET /exports/conversations.csv` (telechargement CSV des entrees/sorties IA)

## Persistance long terme des logs d'evaluation (MongoDB)
Par defaut, les interactions d'evaluation sont conservees en memoire (perdues au redemarrage).
Pour persister dans MongoDB:
```powershell
$env:INTERACTIONS_USE_MONGO="true"
$env:MONGO_URI="mongodb://localhost:27017"
$env:MONGO_DATABASE="nosql_project"
$env:INTERACTIONS_MONGO_COLLECTION="interaction_logs"
python run_api.py
```
Puis exporter:
```text
http://127.0.0.1:8000/exports/conversations.csv?session_id=<session_id>
```

## Tester via l'interface web (recommande)
1. Demarrer le serveur:
```powershell
python run_api.py
```
2. Ouvrir dans le navigateur:
```text
http://127.0.0.1:8000/
```

Note: `0.0.0.0` est une adresse d'ecoute serveur; dans le navigateur utilise `127.0.0.1` ou `localhost`.

L'interface inclut:
- conversation texte + voix,
- indicateur pipeline (`processing`, `STT`, `NLP`, `TTS`),
- logs techniques (mode developpeur),
- bouton `Export CSV` pour telecharger les echanges de la session courante,
- visualisation simple du flux asynchrone.

## Activer un vrai modele NLP (Flan-T5)
Par defaut, le backend NLP reste `rule_based`.
Pour activer le vrai modele:
```powershell
python -m pip install "tensorflow>=2.15.0" "transformers>=4.44.0,<5.0.0" sentencepiece
python -m pip install tf-keras
$env:NLP_BACKEND="transformers"
$env:NLP_MODEL_NAME="google/flan-t5-small"
$env:NLP_FALLBACK_TO_RULE_BASED="true"
$env:NLP_LOCAL_FILES_ONLY="false"
python run_api.py
```
Note: `transformers` 5.x est oriente PyTorch. Pour ce projet TensorFlow, rester en `<5.0.0`.
Le premier lancement peut telecharger le modele.
Si le modele ne charge pas, le service retombe automatiquement sur `rule_based`.

## Activer STT reel (Whisper via faster-whisper)
Par defaut, le backend STT reste `simple`.
Pour activer Whisper:
```powershell
python -m pip install faster-whisper
$env:STT_BACKEND="whisper"
$env:STT_MODEL_SIZE="small"
$env:STT_DEVICE="cpu"
$env:STT_COMPUTE_TYPE="int8"
$env:STT_BEAM_SIZE="1"
$env:STT_FALLBACK_TO_SIMPLE="true"
python run_api.py
```
Si Whisper ne charge pas, le service retombe automatiquement sur `simple`.

## Activer TTS reel (Piper)
Par defaut, le backend TTS reste `simple`.
Pour activer Piper:
```powershell
$env:TTS_BACKEND="piper"
$env:TTS_PIPER_EXECUTABLE="piper"
$env:TTS_PIPER_MODEL_PATH="C:\\models\\fr_FR-mls-medium.onnx"
$env:TTS_PIPER_SPEAKER_ID="-1"
$env:TTS_FALLBACK_TO_SIMPLE="true"
python run_api.py
```
Notes:
- `TTS_PIPER_MODEL_PATH` doit pointer vers un fichier `.onnx` existant.
- Si Piper ne charge pas, le service retombe automatiquement sur `simple`.

## Exemple payload `/chat`
```json
{
  "session_id": "demo-1",
  "text": "bonjour"
}
```

## Exemple payload `/voice`
```json
{
  "session_id": "demo-1",
  "audio_base64": "Ym9uam91cg=="
}
```

## Ingestion streaming OpenSubtitles
Le module `nosql_project.ingestion` lit de gros fichiers texte ligne par ligne, puis genere des paires dialogue `input/response` prêtes pour MongoDB.

## Ingestion MongoDB (reelle)
```powershell
python -m nosql_project.mongo_ingestion `
  --file "Fr/fr.txt" `
  --uri "mongodb://localhost:27017" `
  --database "nosql_project" `
  --collection "dialogues" `
  --batch-size 1000 `
  --limit 50000
```
Ou via lanceur local:
```powershell
python run_ingestion.py --file "Fr/fr.txt" --batch-size 1000 --limit 50000
```

## Qualite code
```powershell
python -m pytest -q
python -m pylint src/nosql_project tests
python -m pyright
```

## Resultats de validation (actuels)
```text
python -m pytest -q -> 22 passed
python -m pylint src/nosql_project tests -> 10.00/10
python -m pyright -> 0 errors, 0 warnings
```
