# VoxLab Emotion TTS

Application bureau Python pour générer une voix expressive à partir d'un texte,
avec sélection d'émotion, waveform intégrée et lecteur audio.

## Partie 1 - Architecture du projet

```text
voxlab/
  main.py                         # Point d'entrée: python main.py
  requirements.txt                # Dépendances Python
  README.md                       # Explication du projet
  app/
    gui.py                        # Interface CustomTkinter + Matplotlib
    audio_player.py               # Play/Pause/Stop avec sounddevice
  model/
    config.py                     # Chemins, modèles HF, profils émotionnels
    emotion.py                    # Traitement DSP des émotions
    tts_engine.py                 # MMS-TTS/VITS, génération WAV et waveform PNG
    style_model.py                # Petit réseau PyTorch émotion -> style
    train_style_model.py          # Script d'entraînement du style model
  dataset/
    create_toy_dataset.py         # Génération dataset jouet + metadata.csv
    README.md
  assets/                         # Réservé aux assets graphiques
  outputs/
    audio/                        # WAV générés par l'application
    waveforms/                    # PNG des waveforms générées
```

L'application est une application bureau, pas une application web. Elle utilise
CustomTkinter, qui reste basé sur Tkinter, avec Matplotlib embarqué via
`FigureCanvasTkAgg`.

## Installation et lancement

Python 3.9+ est requis.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Smoke tests:

```bash
python tests/smoke_test.py
python tests/smoke_test.py --with-tts
```

Au premier lancement, Hugging Face télécharge le modèle choisi. Les modèles
utilisés sont:

- Français: `facebook/mms-tts-fra`
- Anglais: `facebook/mms-tts-eng`

## Partie 2 - Dataset et prétraitement

Comme il n'y a pas de dataset d'acteurs, le projet fournit un dataset jouet:

```bash
python dataset/create_toy_dataset.py
```

Le script:

- génère une voix de base avec `pyttsx3` si le moteur vocal système fonctionne;
- bascule sur un signal synthétique de secours si nécessaire;
- applique les profils émotionnels du projet;
- écrit les fichiers dans `dataset/audio/`;
- crée `dataset/metadata.csv` avec texte, émotion, durée, sample rate et
  paramètres de style.

Ce dataset n'est pas destiné à produire une voix de production. Il sert à
démontrer le pipeline complet: données, métadonnées, apprentissage léger,
inférence.

## Partie 3 - Modèle AI et inférence

La solution combine deux couches:

1. Synthèse vocale neurale pré-entraînée avec MMS-TTS/VITS via Transformers.
2. Contrôle émotionnel local par paramètres de style: vitesse, pitch léger,
   gain, brillance, chaleur et saturation douce.

Le petit modèle PyTorch optionnel apprend la correspondance:

```text
émotion one-hot -> [speed, pitch, gain_db, brightness, warmth, drive]
```

Entraînement:

```bash
python dataset/create_toy_dataset.py
python -m model.train_style_model --metadata dataset/metadata.csv
```

Le checkpoint est sauvegardé dans `model/checkpoints/emotion_style.pt`.
L'interface active automatiquement ce checkpoint si le switch
"Utiliser le modèle de style appris" est coché.

Pourquoi cette approche: entraîner StyleTTS2, VITS ou Tacotron2 from scratch
demande un vrai corpus multi-locuteur annoté émotion, beaucoup de données et du
GPU. Ici, MMS-TTS fournit la partie neurale robuste, puis le contrôle émotionnel
reste léger, pédagogique et exécutable localement.

## Partie 4 - Application bureau avec waveform

L'application `python main.py` fournit:

- une page d'accueil avec présentation du workflow et navigation vers le studio;
- une fenêtre redimensionnable au thème sombre moderne;
- une zone de texte multiligne;
- un menu langue et un menu émotion;
- un bouton "Générer l'audio" avec barre de progression;
- une waveform Matplotlib mise à jour à chaque génération;
- une ligne de lecture qui se déplace pendant le playback;
- boutons Play, Pause, Stop;
- sauvegarde d'une copie WAV;
- historique des générations précédentes;
- gestion des erreurs: texte vide, dépendance manquante, fichier introuvable,
  erreur de génération.

Les fichiers générés sont écrits ici:

- audio: `outputs/audio/`
- waveforms PNG: `outputs/waveforms/`

## Notes techniques

Le contrôle émotionnel n'est pas un vrai modèle TTS émotionnel fine-tuné. C'est
une solution pragmatique et légère: modèle TTS pré-entraîné + style émotionnel
post-synthèse. Pour une version recherche/production, il faudrait collecter un
dataset d'acteurs avec labels émotionnels, puis fine-tuner un modèle comme VITS,
StyleTTS2 ou XTTS avec conditionnement émotionnel.
