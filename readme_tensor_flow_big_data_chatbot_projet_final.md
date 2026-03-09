# TensorFlow – IA & Deep Learning sur Grands Volumes

## 🎯 Objectif du projet

Mettre en place un pipeline Big Data complet permettant :

- L’ingestion de données conversationnelles massives
- Leur stockage dans une base NoSQL (MongoDB)
- Leur transformation et préparation via Kafka
- L’entraînement ou l’adaptation d’un modèle Deep Learning (TensorFlow)
- L’exposition via un chatbot vocal
- La synthèse vocale en temps réel

Ce projet a deux objectifs principaux :

1. ✅ Valider le module NoSQL
2. ✅ Construire un projet industriel valorisable sur CV

---

# 🏗 Architecture Générale

Sources de données → MongoDB → Kafka → Préprocessing → TensorFlow → API Chatbot → STT → TTS

---

# 📦 Sources de Données (Datasets TTS & Dialogue)

## 1️⃣ Dialogue – OpenSubtitles FR

Dataset principal pour génération conversationnelle :

OpenSubtitles v2018 (Français)
https://opus.nlpl.eu/OpenSubtitles-v2018.php

Utilisation :

- Extraction des phrases
- Reconstruction en paires Question/Réponse
- Nettoyage (bruit, balises, phrases trop longues)
- Stockage MongoDB

---

## 2️⃣ Données Voix (TTS / STT)

### 🔹 Mozilla Common Voice – Français

Common Voice 24.0 – French
https://commonvoice.mozilla.org/fr/datasets

Utilisation :

- Amélioration éventuelle STT
- Tests qualité vocale

---

## 3️⃣ Modèle NLP Pré-entraîné

Flan-T5 Small (Google)
https://huggingface.co/google/flan-t5-small

Caractéristiques :

- ~80M paramètres
- Compatible CPU
- Support français via prompting
- Version TensorFlow utilisée

---

# 🧠 Pipeline d’Entraînement

1. Extraction des données OpenSubtitles
2. Nettoyage et filtrage
3. Construction paires (input, response)
4. Stockage MongoDB
5. Export vers dataset TensorFlow
6. Fine-tuning léger (20k – 30k paires max)
7. Sauvegarde modèle personnalisé (.keras)

---

# 🔄 Pipeline d’Inférence (Boucle Temps Réel)

## 1️⃣ Speech-to-Text (STT)

Outil : Whisper (version tiny/base)

- Conversion audio utilisateur → texte
- Optimisé CPU

## 2️⃣ Moteur IA

Modèle : Flan-T5 Small (TensorFlow)

- Génération réponse texte
- Prompt engineering structuré

## 3️⃣ Text-to-Speech (TTS)

Outil : Piper TTS

- Synthèse vocale rapide CPU
- Voix française naturelle

---

# 🗄 Base de Données – MongoDB

Structure document :

{
"input": "Bonjour",
"response": "Salut !",
"source": "OpenSubtitles",
"timestamp": "2026"
}

---

# 🔌 APIs et Exposition

Framework recommandé : FastAPI

Endpoints :

POST /chat
→ reçoit texte
→ renvoie réponse IA

POST /voice
→ reçoit audio
→ renvoie audio synthétisé

---

# ⚙️ Stack Technique

- Python 3.10+
- TensorFlow
- Transformers (HuggingFace)
- MongoDB
- Kafka
- Whisper
- Piper TTS
- FastAPI
- VS Code (Environnement Windows 11, CPU, 8/16 Go RAM)

---

# 📊 Contraintes Matérielles

- 8/16 Go RAM
- CPU uniquement
- Windows 11

Stratégie adoptée :

- Modèles légers
- Fine-tuning limité
- Échantillonnage dataset
- Optimisation latence

---

# 🏆 Résultat Attendu

Un système industriel capable de :

- Ingestion massive
- Stockage distribué
- Traitement streaming
- Modèle Deep Learning opérationnel
- Chatbot vocal temps réel

---

# 💼 Valorisation CV

Projet : Pipeline IA Big Data temps réel

Compétences démontrées :

- Architecture distribuée
- NoSQL (MongoDB)
- Streaming (Kafka)
- NLP (TensorFlow)
- STT / TTS
- Déploiement API

---

# 📌 Évolution Future

- Quantization du modèle
- Dockerisation
- Déploiement Cloud (AWS / GCP)
- Monitoring Prometheus
- Load balancing

---

Projet réalisé dans le cadre du module :
TensorFlow – IA & Deep Learning sur Grands Volumes

Auteur : Atassa
Spécialité : IA & Big Data

