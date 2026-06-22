"""Script d'entraînement du petit modèle émotionnel.

Exemples:
    python -m model.train_style_model
    python -m model.train_style_model --metadata dataset/metadata.csv --epochs 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model.config import STYLE_CHECKPOINT_PATH
from model.style_model import train_style_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne le modèle de style émotionnel.")
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--out", type=Path, default=STYLE_CHECKPOINT_PATH)
    args = parser.parse_args()

    metadata_path = args.metadata if args.metadata.exists() else None
    checkpoint = train_style_model(
        metadata_path=metadata_path,
        checkpoint_path=args.out,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    print(f"Checkpoint sauvegardé: {checkpoint}")


if __name__ == "__main__":
    main()
