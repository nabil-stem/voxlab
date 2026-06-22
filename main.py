"""Point d'entrée de VoxLab Emotion TTS.

L'application se lance avec:
    python main.py
"""

from app.gui import VoxLabApp


if __name__ == "__main__":
    app = VoxLabApp()
    app.mainloop()
