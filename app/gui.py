"""Interface bureau CustomTkinter avec waveform Matplotlib."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app.audio_player import AudioPlayer
from model.config import EMOTIONS, LANGUAGE_LABELS, STYLE_CHECKPOINT_PATH
from model.tts_engine import SynthesisResult, TTSEngine


COLORS = {
    "bg": "#07111f",
    "panel": "#0f1b2d",
    "panel_2": "#13243a",
    "border": "#26364f",
    "text": "#e5edf7",
    "muted": "#94a3b8",
    "accent": "#22d3ee",
    "accent_dark": "#0e7490",
    "danger": "#e11d48",
    "success": "#2dd4bf",
    "warning": "#facc15",
}


class VoxLabApp(ctk.CTk):
    """Application principale."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("VoxLab - Synthèse vocale expressive")
        self.geometry("1180x760")
        self.minsize(1020, 680)
        self.configure(fg_color=COLORS["bg"])

        self.engine = TTSEngine()
        self.player = AudioPlayer()
        self.current_result: SynthesisResult | None = None
        self.history: list[SynthesisResult] = []
        self.cursor_line = None
        self.is_generating = False
        self._playback_active = False

        self.language_by_label = {label: key for key, label in LANGUAGE_LABELS.items()}
        self.emotion_by_label = {profile.label: key for key, profile in EMOTIONS.items()}

        self._build_ui()
        self._plot_empty_waveform()
        self._render_history()
        self._refresh_status()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._update_playhead)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color=COLORS["panel"], corner_radius=0, height=82)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        header.grid_propagate(False)

        title = ctk.CTkLabel(
            header,
            text="VoxLab Emotion TTS",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color=COLORS["text"],
        )
        title.grid(row=0, column=0, sticky="w", padx=28, pady=(13, 0))

        subtitle = ctk.CTkLabel(
            header,
            text="MMS-TTS/VITS local, contrôle émotionnel, waveform et lecteur intégré",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLORS["muted"],
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=28, pady=(0, 10))

        nav = ctk.CTkFrame(header, fg_color="transparent")
        nav.grid(row=0, column=1, rowspan=2, padx=(10, 8))

        self.home_button = ctk.CTkButton(
            nav,
            text="Accueil",
            width=92,
            fg_color=COLORS["accent_dark"],
            hover_color="#155e75",
            command=self._show_home,
        )
        self.home_button.grid(row=0, column=0, padx=(0, 8))

        self.studio_button = ctk.CTkButton(
            nav,
            text="Studio",
            width=92,
            fg_color=COLORS["panel_2"],
            hover_color=COLORS["border"],
            command=self._show_studio,
        )
        self.studio_button.grid(row=0, column=1)

        self.status_label = ctk.CTkLabel(
            header,
            text="Initialisation",
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.status_label.grid(row=0, column=2, rowspan=2, sticky="e", padx=28)

        self.home_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.home_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=16)
        self.home_frame.grid_columnconfigure(0, weight=1)
        self.home_frame.grid_rowconfigure(0, weight=1)
        self._build_home(self.home_frame)

        main = ctk.CTkFrame(self, fg_color="transparent")
        self.studio_frame = main
        main.grid(row=1, column=0, sticky="nsew", padx=18, pady=16)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(main, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(1, weight=1)
        left.grid_rowconfigure(3, weight=2)
        left.grid_columnconfigure(0, weight=1)

        self._build_editor(left)
        self._build_waveform(left)

        right = ctk.CTkFrame(main, fg_color=COLORS["panel"], corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(5, weight=1)

        self._build_controls(right)
        self._build_history(right)

        footer = ctk.CTkLabel(
            self,
            text="VoxLab - Projet TTS expressif Python",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
        )
        footer.grid(row=2, column=0, pady=(0, 8))
        self._show_home()

    def _build_home(self, parent: ctk.CTkFrame) -> None:
        shell = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=14)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.grid_columnconfigure(0, weight=3)
        shell.grid_columnconfigure(1, weight=2)
        shell.grid_rowconfigure(0, weight=1)

        hero = ctk.CTkFrame(shell, fg_color="transparent")
        hero.grid(row=0, column=0, sticky="nsew", padx=32, pady=32)
        hero.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hero,
            text="Bienvenue dans VoxLab",
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Segoe UI", size=34, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(8, 10))

        ctk.CTkLabel(
            hero,
            text=(
                "Transformez un texte en voix expressive, choisissez une émotion, "
                "visualisez la waveform et écoutez le résultat directement dans l'application."
            ),
            text_color=COLORS["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=16),
            justify="left",
            wraplength=620,
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 22))

        actions = ctk.CTkFrame(hero, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="w", pady=(0, 26))

        ctk.CTkButton(
            actions,
            text="Ouvrir le studio",
            width=170,
            height=42,
            fg_color=COLORS["accent_dark"],
            hover_color="#155e75",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._show_studio,
        ).grid(row=0, column=0, padx=(0, 10))

        ctk.CTkButton(
            actions,
            text="Essayer un exemple",
            width=165,
            height=42,
            fg_color=COLORS["panel_2"],
            hover_color=COLORS["border"],
            command=self._open_demo_example,
        ).grid(row=0, column=1)

        cards = ctk.CTkFrame(hero, fg_color="transparent")
        cards.grid(row=3, column=0, sticky="ew")
        cards.grid_columnconfigure((0, 1, 2), weight=1)

        self._home_card(
            cards,
            0,
            "Synthèse locale",
            "Modèles MMS-TTS/VITS via Hugging Face, exécutés depuis Python.",
            COLORS["accent"],
        )
        self._home_card(
            cards,
            1,
            "Émotions",
            "Joie, tristesse, colère, calme et neutre avec profils de style.",
            COLORS["warning"],
        )
        self._home_card(
            cards,
            2,
            "Waveform",
            "Affichage dynamique Matplotlib et export PNG automatique.",
            COLORS["success"],
        )

        panel = ctk.CTkFrame(shell, fg_color=COLORS["panel_2"], corner_radius=12)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 28), pady=32)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text="Flux de travail",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=22, pady=(22, 12))

        steps = [
            ("1", "Saisir le texte"),
            ("2", "Choisir la langue et l'émotion"),
            ("3", "Générer l'audio"),
            ("4", "Écouter, visualiser et sauvegarder"),
        ]
        for row, (number, text) in enumerate(steps, start=1):
            item = ctk.CTkFrame(panel, fg_color=COLORS["panel"], corner_radius=8)
            item.grid(row=row, column=0, sticky="ew", padx=18, pady=6)
            item.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                item,
                text=number,
                width=34,
                height=34,
                fg_color=COLORS["accent_dark"],
                corner_radius=17,
                text_color="white",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, padx=10, pady=10)
            ctk.CTkLabel(
                item,
                text=text,
                text_color=COLORS["text"],
                anchor="w",
                font=ctk.CTkFont(size=14),
            ).grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ctk.CTkLabel(
            panel,
            text=(
                "Conseil: lancez d'abord le smoke test complet pour vérifier "
                "le modèle TTS et les sorties WAV/PNG."
            ),
            text_color=COLORS["muted"],
            justify="left",
            wraplength=330,
        ).grid(row=6, column=0, sticky="ew", padx=22, pady=(16, 10))

        ctk.CTkButton(
            panel,
            text="Voir le README",
            fg_color=COLORS["panel"],
            hover_color=COLORS["border"],
            command=self._show_readme_hint,
        ).grid(row=7, column=0, sticky="ew", padx=22, pady=(0, 22))

    def _home_card(
        self,
        parent: ctk.CTkFrame,
        column: int,
        title: str,
        body: str,
        accent: str,
    ) -> None:
        card = ctk.CTkFrame(parent, fg_color=COLORS["panel_2"], corner_radius=10)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        ctk.CTkLabel(
            card,
            text=title,
            text_color=accent,
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 4))
        ctk.CTkLabel(
            card,
            text=body,
            text_color=COLORS["muted"],
            justify="left",
            wraplength=170,
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))

    def _show_home(self) -> None:
        self.studio_frame.grid_remove()
        self.home_frame.grid()
        self.home_button.configure(fg_color=COLORS["accent_dark"], hover_color="#155e75")
        self.studio_button.configure(fg_color=COLORS["panel_2"], hover_color=COLORS["border"])

    def _show_studio(self) -> None:
        self.home_frame.grid_remove()
        self.studio_frame.grid()
        self.home_button.configure(fg_color=COLORS["panel_2"], hover_color=COLORS["border"])
        self.studio_button.configure(fg_color=COLORS["accent_dark"], hover_color="#155e75")

    def _open_demo_example(self) -> None:
        self._set_example("joy")
        self._show_studio()

    def _show_readme_hint(self) -> None:
        messagebox.showinfo(
            "README",
            "Le fichier README.md contient les commandes d'installation, de test et d'utilisation.",
        )

    def _build_editor(self, parent: ctk.CTkFrame) -> None:
        label = ctk.CTkLabel(
            parent,
            text="Texte à synthétiser",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.textbox = ctk.CTkTextbox(
            parent,
            height=190,
            fg_color=COLORS["panel"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=10,
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Segoe UI", size=15),
            wrap="word",
        )
        self.textbox.grid(row=1, column=0, sticky="nsew")
        self.textbox.insert(
            "1.0",
            "Bonjour, je suis VoxLab. Choisissez une émotion, puis générez une voix expressive.",
        )

        quick = ctk.CTkFrame(parent, fg_color="transparent")
        quick.grid(row=2, column=0, sticky="ew", pady=10)
        quick.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(
            quick,
            text="Exemple joie",
            width=110,
            fg_color=COLORS["panel_2"],
            hover_color=COLORS["border"],
            command=lambda: self._set_example("joy"),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        ctk.CTkButton(
            quick,
            text="Exemple calme",
            width=120,
            fg_color=COLORS["panel_2"],
            hover_color=COLORS["border"],
            command=lambda: self._set_example("calm"),
        ).grid(row=0, column=1, sticky="w", padx=(0, 8))

        ctk.CTkButton(
            quick,
            text="Effacer",
            width=90,
            fg_color="#3f1d2a",
            hover_color="#5f2235",
            command=lambda: self.textbox.delete("1.0", "end"),
        ).grid(row=0, column=3, sticky="e")

    def _build_waveform(self, parent: ctk.CTkFrame) -> None:
        block = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=10)
        block.grid(row=3, column=0, sticky="nsew")
        block.grid_columnconfigure(0, weight=1)
        block.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            block,
            text="Forme d'onde",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 4))

        self.figure = Figure(figsize=(7.4, 3.2), dpi=100, facecolor=COLORS["panel"])
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=block)
        self.canvas.get_tk_widget().configure(background=COLORS["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        self.path_label = ctk.CTkLabel(
            block,
            text="Aucun fichier généré pour le moment.",
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.path_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

    def _build_controls(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent,
            text="Contrôles",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(18, 10))

        form = ctk.CTkFrame(parent, fg_color="transparent")
        form.grid(row=1, column=0, sticky="ew", padx=18)
        form.grid_columnconfigure(0, weight=1)

        self.language_var = ctk.StringVar(value=LANGUAGE_LABELS["fr"])
        ctk.CTkLabel(form, text="Langue", text_color=COLORS["muted"]).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ctk.CTkOptionMenu(
            form,
            values=list(self.language_by_label.keys()),
            variable=self.language_var,
            fg_color=COLORS["panel_2"],
            button_color=COLORS["accent_dark"],
            button_hover_color=COLORS["accent_dark"],
        ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

        self.emotion_var = ctk.StringVar(value=EMOTIONS["neutral"].label)
        ctk.CTkLabel(form, text="Émotion", text_color=COLORS["muted"]).grid(
            row=2, column=0, sticky="w", pady=(0, 4)
        )
        self.emotion_menu = ctk.CTkOptionMenu(
            form,
            values=[profile.label for profile in EMOTIONS.values()],
            variable=self.emotion_var,
            command=self._on_emotion_change,
            fg_color=COLORS["panel_2"],
            button_color=EMOTIONS["neutral"].color,
            button_hover_color=EMOTIONS["neutral"].color,
        )
        self.emotion_menu.grid(row=3, column=0, sticky="ew")

        self.emotion_chip = ctk.CTkLabel(
            form,
            text="Couleur émotionnelle: Neutre",
            text_color=EMOTIONS["neutral"].color,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.emotion_chip.grid(row=4, column=0, sticky="w", pady=(6, 12))

        self.style_switch = ctk.CTkSwitch(
            form,
            text="Utiliser le modèle de style appris si disponible",
            text_color=COLORS["text"],
        )
        self.style_switch.grid(row=5, column=0, sticky="w", pady=(0, 12))
        self.style_switch.select()

        self.progress = ctk.CTkProgressBar(form, mode="indeterminate", height=8)
        self.progress.grid(row=6, column=0, sticky="ew", pady=(0, 14))
        self.progress.set(0)

        self.generate_button = ctk.CTkButton(
            form,
            text="Générer l'audio",
            height=42,
            fg_color=COLORS["accent_dark"],
            hover_color="#155e75",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_generation,
        )
        self.generate_button.grid(row=7, column=0, sticky="ew", pady=(0, 10))

        audio_row = ctk.CTkFrame(form, fg_color="transparent")
        audio_row.grid(row=8, column=0, sticky="ew")
        audio_row.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(audio_row, text="Play", command=self._play_audio).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ctk.CTkButton(audio_row, text="Pause", command=self._pause_audio).grid(
            row=0, column=1, sticky="ew", padx=3
        )
        ctk.CTkButton(
            audio_row,
            text="Stop",
            fg_color=COLORS["danger"],
            hover_color="#be123c",
            command=self._stop_audio,
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        ctk.CTkButton(
            form,
            text="Sauvegarder une copie WAV",
            fg_color=COLORS["panel_2"],
            hover_color=COLORS["border"],
            command=self._save_copy,
        ).grid(row=9, column=0, sticky="ew", pady=(12, 0))

        self.model_info = ctk.CTkLabel(
            parent,
            text="",
            text_color=COLORS["muted"],
            justify="left",
            anchor="w",
            wraplength=360,
        )
        self.model_info.grid(row=2, column=0, sticky="ew", padx=18, pady=(14, 10))

    def _build_history(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent,
            text="Historique",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=4, column=0, sticky="w", padx=18, pady=(10, 8))

        self.history_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color=COLORS["panel_2"],
            corner_radius=10,
            label_text="",
        )
        self.history_frame.grid(row=5, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.history_frame.grid_columnconfigure(0, weight=1)

    def _refresh_status(self) -> None:
        style_state = "présent" if STYLE_CHECKPOINT_PATH.exists() else "absent"
        sound_state = "audio prêt" if self.player.available else "sounddevice manquant"
        self.model_info.configure(
            text=(
                f"Moteur: MMS-TTS/VITS via Hugging Face\n"
                f"Device: {self.engine.device}\n"
                f"Checkpoint style: {style_state}\n"
                f"Lecteur: {sound_state}\n\n"
                f"Modèles utilisés: facebook/mms-tts-fra et facebook/mms-tts-eng."
            )
        )
        if self.engine.available:
            self.status_label.configure(text="Prêt")
        else:
            self.status_label.configure(text="Dépendances TTS manquantes", text_color=COLORS["danger"])

    def _set_example(self, emotion: str) -> None:
        examples = {
            "joy": "Quelle belle journée pour créer une voix expressive avec Python !",
            "calm": "Respirez tranquillement. La synthèse vocale peut aussi être douce et posée.",
        }
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", examples[emotion])
        self.emotion_var.set(EMOTIONS[emotion].label)
        self._on_emotion_change(EMOTIONS[emotion].label)

    def _on_emotion_change(self, label: str) -> None:
        key = self.emotion_by_label.get(label, "neutral")
        color = EMOTIONS[key].color
        self.emotion_menu.configure(button_color=color, button_hover_color=color)
        self.emotion_chip.configure(text=f"Couleur émotionnelle: {label}", text_color=color)

    def _start_generation(self) -> None:
        if self.is_generating:
            return

        text = self.textbox.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Texte vide", "Veuillez saisir un texte à synthétiser.")
            return

        language = self.language_by_label.get(self.language_var.get(), "fr")
        emotion = self.emotion_by_label.get(self.emotion_var.get(), "neutral")
        use_style = bool(self.style_switch.get())

        self.is_generating = True
        self.status_label.configure(text="Génération en cours", text_color=COLORS["warning"])
        self.generate_button.configure(state="disabled")
        self.progress.start()

        thread = threading.Thread(
            target=self._generation_worker,
            args=(text, language, emotion, use_style),
            daemon=True,
        )
        thread.start()

    def _generation_worker(
        self,
        text: str,
        language: str,
        emotion: str,
        use_style: bool,
    ) -> None:
        try:
            result = self.engine.synthesize(
                text=text,
                emotion_key=emotion,
                language=language,
                use_style_model=use_style,
            )
        except Exception as exc:  # l'erreur est affichée dans le thread UI
            self.after(0, lambda error=exc: self._generation_failed(error))
            return
        self.after(0, lambda: self._generation_succeeded(result))

    def _generation_succeeded(self, result: SynthesisResult) -> None:
        self.is_generating = False
        self.progress.stop()
        self.generate_button.configure(state="normal")
        self.current_result = result
        self.player.set_audio(result.audio, result.sample_rate)
        self.history.insert(0, result)
        self._plot_waveform(result)
        self._render_history()
        self.status_label.configure(
            text=f"Audio généré: {result.duration_seconds:.1f} s",
            text_color=COLORS["success"],
        )
        self.path_label.configure(text=f"WAV: {result.audio_path}")

    def _generation_failed(self, exc: Exception) -> None:
        self.is_generating = False
        self.progress.stop()
        self.generate_button.configure(state="normal")
        self.status_label.configure(text="Erreur de génération", text_color=COLORS["danger"])
        messagebox.showerror("Erreur", str(exc))

    def _render_history(self) -> None:
        for child in self.history_frame.winfo_children():
            child.destroy()

        if not self.history:
            ctk.CTkLabel(
                self.history_frame,
                text="Les générations apparaîtront ici.",
                text_color=COLORS["muted"],
            ).grid(row=0, column=0, padx=10, pady=10)
            return

        for index, result in enumerate(self.history[:12]):
            row = ctk.CTkFrame(self.history_frame, fg_color=COLORS["panel"], corner_radius=8)
            row.grid(row=index, column=0, sticky="ew", padx=4, pady=5)
            row.grid_columnconfigure(0, weight=1)

            label = f"{result.emotion_label} - {result.language.upper()} - {result.duration_seconds:.1f}s"
            ctk.CTkLabel(row, text=label, text_color=COLORS["text"], anchor="w").grid(
                row=0, column=0, sticky="ew", padx=10, pady=(8, 0)
            )
            ctk.CTkLabel(
                row,
                text=result.text[:65] + ("..." if len(result.text) > 65 else ""),
                text_color=COLORS["muted"],
                anchor="w",
            ).grid(row=1, column=0, sticky="ew", padx=10)

            ctk.CTkButton(
                row,
                text="Recharger",
                width=92,
                fg_color=COLORS["panel_2"],
                hover_color=COLORS["border"],
                command=lambda item=result: self._load_history_item(item),
            ).grid(row=0, column=1, rowspan=2, padx=8, pady=8)

    def _load_history_item(self, result: SynthesisResult) -> None:
        if not result.audio_path.exists():
            messagebox.showwarning("Fichier introuvable", str(result.audio_path))
            return
        audio, sample_rate = sf.read(result.audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result.audio = audio
        result.sample_rate = int(sample_rate)
        self.current_result = result
        self.player.set_audio(audio, int(sample_rate))
        self._plot_waveform(result)
        self.path_label.configure(text=f"WAV: {result.audio_path}")

    def _plot_empty_waveform(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(COLORS["panel"])
        self.ax.text(
            0.5,
            0.5,
            "La waveform apparaîtra après la génération.",
            color=COLORS["muted"],
            ha="center",
            va="center",
            transform=self.ax.transAxes,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(COLORS["border"])
        self.canvas.draw_idle()

    def _plot_waveform(self, result: SynthesisResult) -> None:
        audio = np.asarray(result.audio, dtype=np.float32)
        max_points = 9000
        if len(audio) > max_points:
            indices = np.linspace(0, len(audio) - 1, max_points).astype(np.int64)
            plot_audio = audio[indices]
            time_axis = indices / result.sample_rate
        else:
            plot_audio = audio
            time_axis = np.arange(len(audio)) / result.sample_rate

        color = EMOTIONS.get(result.emotion_key, EMOTIONS["neutral"]).color
        self.ax.clear()
        self.ax.set_facecolor(COLORS["panel"])
        self.ax.plot(time_axis, plot_audio, color=color, linewidth=0.9)
        self.ax.fill_between(time_axis, plot_audio, 0, color=color, alpha=0.18)
        self.ax.axhline(0, color=COLORS["border"], linewidth=0.8)
        self.cursor_line = self.ax.axvline(0, color="#f8fafc", linewidth=1.2)
        self.ax.set_xlim(0, max(result.duration_seconds, 0.1))
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.set_xlabel("Temps (secondes)", color=COLORS["muted"])
        self.ax.set_ylabel("Amplitude", color=COLORS["muted"])
        self.ax.tick_params(colors=COLORS["muted"])
        self.ax.set_title(
            f"{result.emotion_label} - {result.language.upper()}",
            color=COLORS["text"],
            pad=10,
        )
        for spine in self.ax.spines.values():
            spine.set_color(COLORS["border"])
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _update_playhead(self) -> None:
        playing = self.player.is_playing()
        if self.cursor_line is not None and self.current_result is not None:
            if playing:
                position = self.player.position_seconds
                self.cursor_line.set_xdata([position, position])
                self.canvas.draw_idle()
            elif self._playback_active:
                # Transition lecture -> arrêt. On distingue une fin naturelle
                # (la tête de lecture a atteint la fin) d'un Stop explicite, qui
                # gère déjà son propre affichage.
                ended_naturally = (
                    self.player.duration_seconds > 0.1
                    and self.player.position_seconds
                    >= self.player.duration_seconds - 0.05
                )
                if ended_naturally:
                    self.cursor_line.set_xdata([0, 0])
                    self.canvas.draw_idle()
                    self.status_label.configure(
                        text="Lecture terminée", text_color=COLORS["muted"]
                    )
        self._playback_active = playing
        self.after(120, self._update_playhead)

    def _play_audio(self) -> None:
        try:
            self.player.play()
            self.status_label.configure(text="Lecture", text_color=COLORS["success"])
        except Exception as exc:
            messagebox.showerror("Lecture impossible", str(exc))

    def _pause_audio(self) -> None:
        self.player.pause()
        self.status_label.configure(text="Lecture en pause", text_color=COLORS["warning"])

    def _stop_audio(self) -> None:
        self.player.stop()
        if self.cursor_line is not None:
            self.cursor_line.set_xdata([0, 0])
            self.canvas.draw_idle()
        self.status_label.configure(text="Lecture arrêtée", text_color=COLORS["muted"])

    def _save_copy(self) -> None:
        if self.current_result is None or not self.current_result.audio_path.exists():
            messagebox.showinfo("Aucun audio", "Générez d'abord un audio.")
            return

        destination = filedialog.asksaveasfilename(
            title="Sauvegarder le WAV",
            defaultextension=".wav",
            filetypes=[("Fichier WAV", "*.wav")],
            initialfile=self.current_result.audio_path.name,
        )
        if destination:
            shutil.copy2(self.current_result.audio_path, Path(destination))
            self.status_label.configure(text="Copie sauvegardée", text_color=COLORS["success"])

    def _on_close(self) -> None:
        self.player.stop()
        self.destroy()
