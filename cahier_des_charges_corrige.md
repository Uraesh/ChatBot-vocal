# Cahier Des Charges Corrige - Chatbot Vocal IA Big Data

## 1. Objectif

Construire un systeme vocal conversationnel en francais avec pipeline asynchrone, en validant le module NoSQL (MongoDB) et une architecture de traitement distribuee.

## 2. Portee MVP (Version 1)

- Ingestion de donnees dialogue OpenSubtitles en mode flux (sans chargement RAM complet).
- Nettoyage, structuration en paires `input/response` et stockage MongoDB.
- Pipeline vocal asynchrone: `Interface/API -> STT -> NLP -> TTS -> Interface`.
- API FastAPI avec endpoints texte et voix.
- Execution sur `Windows 11`, `CPU`, `8/16 Go RAM`.

## 3. Contraintes Reelles

- Fichier source OpenSubtitles `fr.txt` volumineux (plus de 3 Go): lecture ligne a ligne obligatoire.
- Pas de GPU: privilegier modeles legers et mode inference d'abord.
- Latence cible MVP (CPU): reponse vocale en moins de 4 secondes en moyenne sur phrase courte.

## 4. Architecture Cible

### 4.1 Training / Data Engineering

`Source OpenSubtitles -> Worker ingestion -> Kafka (optionnel V1.5) -> Worker nettoyage -> MongoDB -> Export dataset`

### 4.2 Inference / Runtime

`Client Web -> FastAPI -> File STT -> File NLP -> File TTS -> Reponse audio + texte`

## 5. Pipeline Asynchrone (Etapes qui se chevauchent)

- Le STT traite la requete `n+1` pendant que le NLP traite `n`.
- Le TTS synthese la reponse `n` pendant que le NLP genere `n+1`.
- Chaque message transporte `request_id`, `session_id`, `timestamp`.
- Tolerance aux erreurs par etape: echec STT/NLP/TTS trace et retour d'erreur explicite.

## 6. Schema Donnees MongoDB (Minimum)

```json
{
  "conversation_id": "uuid",
  "turn_id": 1,
  "input": "Bonjour",
  "response": "Salut !",
  "lang": "fr",
  "source": "OpenSubtitles",
  "split": "train",
  "quality_score": 0.91,
  "created_at": "2026-02-28T10:00:00Z"
}
```

## 7. API

- `POST /chat`: entree texte, sortie texte.
- `POST /voice`: entree audio (base64), sortie texte + audio (base64).
- `GET /health`: etat du service.

## 8. Qualite et Normes

- Python type hints complets (`pylance`).
- Regles lint (`pylint`) appliquees au code source.
- Journalisation structuree (niveau `INFO/ERROR`).
- Fonctions courtes, modules decouples, tests unitaires de base.

## 9. Plan D'execution

### Phase 1 (MVP)

- Socle API + pipeline asynchrone + moteurs mock/STT/NLP/TTS simples.
- Ingestion OpenSubtitles en streaming vers schema MongoDB.
- Tests unitaires de la chaine asynchrone.

### Phase 2

- Integration Kafka entre ingestion et preprocessing.
- Integration modeles reels (Whisper, Flan-T5, Piper).
- Optimisation latence CPU et monitoring minimal.

### Phase 3

- Dockerisation, supervision, charge, securisation.

## 10. Criteres de Validation

- Le systeme repond en texte et voix via API.
- Les etapes du pipeline tournent en parallele (chevauchement verifiable via logs).
- Le parsing de `fr.txt` fonctionne en streaming sans saturation memoire.
- Le code passe les controles `pylance` et `pylint` definis pour le projet.
