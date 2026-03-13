# NoSQL Project - Chatbot vocal asynchrone (MVP)

## Objectif
Mettre a disposition un chatbot vocal asynchrone (STT -> NLP -> TTS) avec une API FastAPI,
une interface web locale et un pipeline d'ingestion NoSQL.

## Prerequis
- Python 3.12 recommande (TensorFlow stable)
- MongoDB (optionnel, requis pour ingestion et persistance long terme)
- Piper CLI (optionnel, pour TTS reel)
- Modele Phi-3 GGUF + `llama-cpp-python` (optionnel, pour NLP reel)
- `faster-whisper` (optionnel, pour STT reel)

## Installation
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Note: `requirements.txt` installe TensorFlow/Transformers (lourd). Si vous voulez un run
minimal de l'API, installez a minima `fastapi`, `uvicorn`, `pydantic`, `pymongo`.

## Demarrer l'API
```powershell
python run_api.py
```

Ouvrir l'interface:
```text
http://127.0.0.1:8000/
```

## Endpoints
- `GET /` interface web locale
- `GET /health` metriques des files du pipeline
- `POST /chat` chat texte
- `POST /chat/stream` streaming SSE de reponse
- `POST /voice` traitement vocal STT -> NLP -> TTS
- `GET /exports/conversations.csv` export CSV des interactions

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

## Configuration (variables d'environnement)
Les valeurs par defaut sont dans `src/nosql_project/config.py`.

### General
| Variable | Defaut | Role |
| --- | --- | --- |
| `APP_NAME` | `NoSQL Async Voice Chatbot` | Titre FastAPI |
| `HOST` | `0.0.0.0` | Adresse d'ecoute |
| `PORT` | `8000` | Port HTTP |
| `DEBUG` | `false` | Logs verbeux |
| `DEFAULT_LANGUAGE` | `fr` | Langue principale |
| `VOICE_TIMEOUT_SECONDS` | `30.0` | Timeout traitement vocal |
| `QUEUE_MAX_SIZE` | `128` | Taille des files asynchrones |
| `CHAT_HISTORY_MAX_TURNS` | `6` | Tours conserves en memoire |
| `CHAT_STREAM_CHUNK_CHARS` | `24` | Taille des chunks SSE |
| `INTERACTIONS_EXPORT_LIMIT` | `50000` | Limite d'export CSV |

### MongoDB (ingestion et interactions)
| Variable | Defaut | Role |
| --- | --- | --- |
| `MONGO_URI` | `mongodb://localhost:27017` | URI MongoDB |
| `MONGO_DATABASE` | `nosql_project` | Base MongoDB |
| `MONGO_COLLECTION` | `dialogues` | Collection ingestion |
| `MONGO_BATCH_SIZE` | `1000` | Taille de batch ingestion |
| `SUBTITLES_FILE_PATH` | `Fr/fr.txt` | Fichier OpenSubtitles |
| `INTERACTIONS_USE_MONGO` | `false` | Persister les interactions |
| `INTERACTIONS_MONGO_COLLECTION` | `interaction_logs` | Collection interactions |
| `INTERACTIONS_MONGO_TIMEOUT_MS` | `5000` | Timeout Mongo interactions |
| `INTERACTIONS_MEMORY_MAX_RECORDS` | `10000` | Max interactions en memoire |

### NLP
| Variable | Defaut | Role |
| --- | --- | --- |
| `NLP_BACKEND` | `rule_based` | `rule_based`, `phi3`, `hybrid` |
| `NLP_MODEL_NAME` | `google/flan-t5-small` | Modele Flan-T5 (hybrid) |
| `NLP_MODEL_PATH` | vide | Chemin vers Phi-3 GGUF |
| `NLP_MAX_NEW_TOKENS` | `150` | Max tokens generes |
| `NLP_TEMPERATURE` | `0.7` | Temperature |
| `NLP_N_THREADS` | `3` | Threads llama.cpp |
| `NLP_LOCAL_FILES_ONLY` | `false` | Pas de telechargement HF |
| `NLP_FALLBACK_TO_RULE_BASED` | `true` | Fallback en cas d'echec |

### STT
| Variable | Defaut | Role |
| --- | --- | --- |
| `STT_BACKEND` | `simple` | `simple` ou `whisper` |
| `STT_MODEL_SIZE` | `base` | Taille Whisper |
| `STT_DEVICE` | `cpu` | Device Whisper |
| `STT_COMPUTE_TYPE` | `int8` | Precision Whisper |
| `STT_BEAM_SIZE` | `5` | Beam size Whisper |
| `STT_FALLBACK_TO_SIMPLE` | `true` | Fallback en cas d'echec |

### TTS
| Variable | Defaut | Role |
| --- | --- | --- |
| `TTS_BACKEND` | `simple` | `simple` ou `piper` |
| `TTS_PIPER_EXECUTABLE` | `piper` | Binaire Piper |
| `TTS_PIPER_MODEL_PATH` | vide | Modele Piper `.onnx` |
| `TTS_PIPER_SPEAKER_ID` | `-1` | Id de locuteur |
| `TTS_FALLBACK_TO_SIMPLE` | `true` | Fallback en cas d'echec |

## Activer un vrai modele NLP (Phi-3)
```powershell
python -m pip install llama-cpp-python
$env:NLP_BACKEND="phi3"
$env:NLP_MODEL_PATH="D:\Licience 3 IA-BD\No sql\NoSql Project\models\Phi-3-mini-4k-instruct-q4.gguf"
python run_api.py
```

## Activer un NLP hybride (Flan-T5 + Phi-3)
```powershell
python -m pip install "transformers>=4.44.0,<5.0.0" "tensorflow>=2.15.0" sentencepiece tf-keras
python -m pip install llama-cpp-python
$env:NLP_BACKEND="hybrid"
$env:NLP_MODEL_NAME="google/flan-t5-small"
$env:NLP_MODEL_PATH="D:\Licience 3 IA-BD\No sql\NoSql Project\models\Phi-3-mini-4k-instruct-q4.gguf"
python run_api.py
```

## Activer STT reel (Whisper via faster-whisper)
```powershell
$env:STT_BACKEND="whisper"
$env:STT_MODEL_SIZE="base"
$env:STT_DEVICE="cpu"
$env:STT_COMPUTE_TYPE="int8"
$env:STT_BEAM_SIZE="5"
python run_api.py
```

## Activer TTS reel (Piper)
```powershell
$env:TTS_BACKEND="piper"
$env:TTS_PIPER_EXECUTABLE="D:\Licience 3 IA-BD\No sql\NoSql Project\tools\piper\piper\piper.exe"
$env:TTS_PIPER_MODEL_PATH="D:\Licience 3 IA-BD\No sql\NoSql Project\models\fr_FR-siwis-medium.onnx"
$env:TTS_PIPER_SPEAKER_ID="-1"
python run_api.py
```

## Persistance long terme des interactions (MongoDB)
Par defaut, les interactions d'evaluation sont conservees en memoire.
Pour persister dans MongoDB:

```powershell
$env:INTERACTIONS_USE_MONGO="true"
$env:MONGO_URI="mongodb://localhost:27017"
$env:MONGO_DATABASE="nosql_project"
$env:INTERACTIONS_MONGO_COLLECTION="interaction_logs"
python run_api.py
```

Export CSV pour une session:
```text
http://127.0.0.1:8000/exports/conversations.csv?session_id=<session_id>
```

## Ingestion MongoDB (OpenSubtitles)
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

## Docker (image MVP minimale)
```powershell
docker build -t nosql-chatbot .
docker run --name nosql-chatbot -p 8000:8000 nosql-chatbot
```

Note: l'image Docker installe uniquement FastAPI/Uvicorn/Pydantic/PyMongo.
Les moteurs Whisper, Piper ou Phi-3 necessitent une image custom avec dependances et modeles.

## Qualite code
```powershell
python -m pytest -q
python -m pylint src/nosql_project tests
python -m pyright
```
