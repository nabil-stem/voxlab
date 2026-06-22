# Dataset jouet

Ce dossier est généré avec:

```bash
python dataset/create_toy_dataset.py
```

Le script crée:

- `dataset/audio/`: fichiers WAV synthétiques.
- `dataset/metadata.csv`: texte, émotion, langue, durée et paramètres de style.

Sans dataset d'acteurs, ces fichiers servent surtout à démontrer le flux de
prétraitement et l'entraînement du petit modèle de style.
