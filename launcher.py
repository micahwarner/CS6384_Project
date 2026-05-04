"""
launcher.py — Settings UI for the Facial Expression Music Generator.

Run with:
    python launcher.py

Select a mode and options, then click Start. The launcher hides while
the session runs and reappears when you quit (Q / ESC / window ✕).
"""

import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EMOTION_COLORS = {
    "happy":    "#4caf50",
    "sad":      "#5c85d6",
    "angry":    "#e94560",
    "surprise": "#ffb300",
    "fear":     "#ab47bc",
    "disgust":  "#26a69a",
    "neutral":  "#888888",
}

# ── Colour palette ─────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
BG2       = "#16213e"
BG3       = "#0f3460"
ACCENT    = "#e94560"
FG        = "#eaeaea"
FG_DIM    = "#888888"
FG_GREEN  = "#4caf50"
FG_ORANGE = "#ff9800"


class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Expression Music Generator")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self._configure_style()
        self._build()
        self._stats_win = None

    # ── Style ──────────────────────────────────────────────────────────────────

    def _configure_style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".",          background=BG,  foreground=FG,  font=("Helvetica", 10))
        s.configure("TFrame",     background=BG)
        s.configure("TLabel",     background=BG,  foreground=FG)
        s.configure("TLabelframe",        background=BG,  foreground=FG_DIM)
        s.configure("TLabelframe.Label",  background=BG,  foreground=FG_DIM,
                    font=("Helvetica", 9, "bold"))
        s.configure("TRadiobutton",       background=BG,  foreground=FG)
        s.configure("TCheckbutton",       background=BG,  foreground=FG)
        s.configure("TCombobox",          fieldbackground=BG2, foreground=FG,
                    selectbackground=BG3, selectforeground=FG)
        s.configure("TSpinbox",           fieldbackground=BG2, foreground=FG,
                    selectbackground=BG3, selectforeground=FG)
        s.configure("TScale",             background=BG,  troughcolor=BG3)
        s.configure("Start.TButton",      background=ACCENT, foreground="white",
                    font=("Helvetica", 11, "bold"), padding=(18, 8))

        # Hover / active / disabled state colours — keeps text readable on dark bg
        s.map("TRadiobutton",
              background=[("active", BG3), ("disabled", BG)],
              foreground=[("active", FG),  ("disabled", FG_DIM)])
        s.map("TCheckbutton",
              background=[("active", BG3), ("disabled", BG)],
              foreground=[("active", FG),  ("disabled", FG_DIM)])
        s.map("TCombobox",
              fieldbackground=[("readonly", BG2)],
              selectbackground=[("readonly", BG3)],
              selectforeground=[("readonly", FG)])
        s.map("Start.TButton",
              background=[("active", "#c73652"), ("disabled", "#555555")],
              foreground=[("disabled", "#999999")])

        # Combobox dropdown listbox colours (uses tk option_add, not ttk style)
        self.root.option_add("*TCombobox*Listbox.background",       BG2)
        self.root.option_add("*TCombobox*Listbox.foreground",       FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BG3)
        self.root.option_add("*TCombobox*Listbox.selectForeground", FG)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build(self):
        root = self.root

        # Header
        hdr = tk.Frame(root, bg=BG2, pady=16)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Expression Music Generator",
                 font=("Helvetica", 17, "bold"), fg=FG, bg=BG2).pack()
        tk.Label(hdr, text="Real-time facial emotion → music",
                 font=("Helvetica", 10), fg=FG_DIM, bg=BG2).pack(pady=(2, 0))

        body = ttk.Frame(root, padding=(18, 14, 18, 14))
        body.pack(fill="both")

        # ── Mode ──────────────────────────────────────────────────────────────
        mode_lf = ttk.LabelFrame(body, text="MODE", padding=(12, 8))
        mode_lf.pack(fill="x", pady=(0, 10))

        self.mode = tk.StringVar(value="full")
        modes = [
            ("Full  –  vision + audio",             "full"),
            ("Audio test  –  no camera required",   "audio"),
            ("Vision only  –  no audio output",     "vision"),
            ("Evaluation  –  ground-truth labelling","eval"),
        ]
        for label, val in modes:
            ttk.Radiobutton(mode_lf, text=label, variable=self.mode, value=val,
                            command=self._on_mode_change).pack(anchor="w", pady=1)

        # ── Camera ────────────────────────────────────────────────────────────
        self.cam_lf = ttk.LabelFrame(body, text="CAMERA", padding=(12, 8))
        self.cam_lf.pack(fill="x", pady=(0, 10))

        r1 = ttk.Frame(self.cam_lf)
        r1.pack(fill="x", pady=(0, 6))

        ttk.Label(r1, text="Camera index:").pack(side="left")
        self.camera_idx = tk.IntVar(value=0)
        ttk.Spinbox(r1, from_=0, to=8, width=4,
                    textvariable=self.camera_idx).pack(side="left", padx=(6, 24))

        ttk.Label(r1, text="Max FPS:").pack(side="left")
        self.fps = tk.StringVar(value="30")
        ttk.Combobox(r1, textvariable=self.fps, values=["15", "24", "30", "60"],
                     width=5, state="readonly").pack(side="left", padx=(6, 0))
        tk.Label(r1, text="(upper cap)", fg=FG_DIM, bg=BG,
                 font=("Helvetica", 8)).pack(side="left", padx=(4, 0))

        r2 = ttk.Frame(self.cam_lf)
        r2.pack(fill="x", pady=(6, 0))

        tk.Label(r2, text="Facial feature smoothing:", bg=BG, fg=FG).pack(side="left")
        self.smooth_val = tk.Label(r2, text="0.30", width=4, bg=BG, fg=FG_DIM,
                                   font=("Helvetica", 9))
        self.smooth_val.pack(side="right")
        self.smooth = tk.DoubleVar(value=0.3)
        ttk.Scale(r2, from_=0.05, to=0.95, variable=self.smooth,
                  orient="horizontal", length=180,
                  command=lambda v: self.smooth_val.config(
                      text=f"{float(v):.2f}")).pack(side="left", padx=(8, 4))

        tk.Label(self.cam_lf,
                 text="↑ lower = smoother/slower response   higher = more reactive/jittery",
                 fg=FG_DIM, bg=BG, font=("Helvetica", 8)).pack(anchor="w", pady=(2, 0))

        # ── Audio ─────────────────────────────────────────────────────────────
        audio_lf = ttk.LabelFrame(body, text="AUDIO BACKEND", padding=(12, 8))
        audio_lf.pack(fill="x", pady=(0, 10))

        self.synth = tk.BooleanVar(value=False)
        ttk.Radiobutton(audio_lf, text="System MIDI  (recommended, low latency)",
                        variable=self.synth, value=False).pack(anchor="w", pady=1)
        ttk.Radiobutton(audio_lf, text="Software synth  (no MIDI device required)",
                        variable=self.synth, value=True).pack(anchor="w", pady=1)

        # ── Options ───────────────────────────────────────────────────────────
        opt_lf = ttk.LabelFrame(body, text="OPTIONS", padding=(12, 8))
        opt_lf.pack(fill="x", pady=(0, 14))

        self.latency  = tk.BooleanVar(value=False)
        self.show_conf = tk.BooleanVar(value=True)

        ttk.Checkbutton(opt_lf, text="Print latency stats every 5 s",
                        variable=self.latency).pack(anchor="w", pady=1)
        ttk.Checkbutton(opt_lf, text="Show emotion confidence bars",
                        variable=self.show_conf).pack(anchor="w", pady=1)

        # ── Start + status ────────────────────────────────────────────────────
        bot = ttk.Frame(body)
        bot.pack(fill="x")

        self.start_btn = ttk.Button(bot, text="▶  Start Session",
                                    style="Start.TButton", command=self._start)
        self.start_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = tk.Label(bot, textvariable=self.status_var,
                                   fg=FG_DIM, bg=BG, font=("Helvetica", 9),
                                   anchor="w")
        self.status_lbl.pack(side="left", padx=14)

        self._on_mode_change()

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_mode_change(self):
        """Disable camera settings when audio-only mode is selected."""
        state = "disabled" if self.mode.get() == "audio" else "normal"
        self._set_children_state(self.cam_lf, state)

    def _set_children_state(self, widget, state):
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_children_state(child, state)

    def _start(self):
        stats_path = os.path.join(SCRIPT_DIR, "session_stats.json")
        try:
            os.remove(stats_path)
        except FileNotFoundError:
            pass

        mode = self.mode.get()

        if mode == "audio":
            cmd = [sys.executable, "test_audio.py"]
            if self.synth.get():
                cmd.append("--synth")
        else:
            cmd = [sys.executable, "main.py"]
            if mode == "vision":
                cmd.append("--no-audio")
            elif mode == "eval":
                cmd.append("--eval")
            if self.synth.get():
                cmd.append("--synth")
            if self.latency.get():
                cmd.append("--latency")
            cmd += ["--camera", str(self.camera_idx.get())]
            cmd += ["--fps",    self.fps.get()]
            cmd += ["--smooth", f"{self.smooth.get():.2f}"]

        self.start_btn.configure(state="disabled")
        self._set_status("Session running…", FG_ORANGE)
        self.root.withdraw()

        def _run():
            try:
                proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
                proc.wait()
                returncode = proc.returncode
            except Exception as e:
                returncode = -1
                self.root.after(0, lambda: self._set_status(f"Error: {e}", ACCENT))
            self.root.after(0, lambda: self._on_end(returncode))

        threading.Thread(target=_run, daemon=True).start()

    def _on_end(self, _returncode: int):
        self.root.deiconify()
        self.start_btn.configure(state="normal")
        # Non-zero codes on WSL are normal (signal termination) — don't alarm the user
        self._set_status("Session ended — ready", FG_GREEN)

        stats_path = os.path.join(SCRIPT_DIR, "session_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path) as f:
                    self._show_stats(json.load(f))
            except Exception as e:
                print(f"[Launcher] Could not load stats: {e}")

    def _set_status(self, text: str, colour: str):
        self.status_var.set(text)
        self.status_lbl.configure(fg=colour)

    # ── Stats popup ────────────────────────────────────────────────────────────

    def _show_stats(self, stats: dict):
        if self._stats_win and self._stats_win.winfo_exists():
            self._stats_win.destroy()

        win = tk.Toplevel(self.root)
        win.title("Session Statistics")
        win.configure(bg=BG)
        win.resizable(True, True)
        self._stats_win = win

        # Position to the right of the launcher
        self.root.update_idletasks()
        x = self.root.winfo_x() + self.root.winfo_width() + 12
        y = self.root.winfo_y()
        win.geometry(f"+{x}+{y}")

        # Header
        dur = stats.get("session_duration", 0)
        mode = stats.get("mode", "full").capitalize()
        hdr = tk.Frame(win, bg=BG2, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Session Statistics",
                 font=("Helvetica", 14, "bold"), fg=FG, bg=BG2).pack()
        tk.Label(hdr, text=f"{mode} mode  ·  {int(dur//60)}m {int(dur%60)}s",
                 font=("Helvetica", 9), fg=FG_DIM, bg=BG2).pack()

        try:
            self._build_stats_chart(win, stats)
        except Exception:
            self._build_stats_text(win, stats)

    def _build_stats_chart(self, win, stats):
        import matplotlib
        matplotlib.use("TkAgg")
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        emotions_data = stats.get("emotion_durations", {})
        has_eval      = bool(stats.get("eval"))

        if has_eval:
            fig = Figure(figsize=(10, 5.6), facecolor=BG2)
            gs  = GridSpec(2, 2, figure=fig,
                           left=0.12, right=0.97, top=0.92, bottom=0.10,
                           hspace=0.55, wspace=0.50)
        else:
            fig = Figure(figsize=(5, 3.2), facecolor=BG2)
            gs  = GridSpec(1, 1, figure=fig,
                           left=0.22, right=0.96, top=0.92, bottom=0.12)

        # ── Chart 1: time per emotion ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(BG2)
        if emotions_data:
            labels  = list(emotions_data.keys())
            values  = list(emotions_data.values())
            colours = [EMOTION_COLORS.get(e, "#888888") for e in labels]
            ax1.barh(labels, values, color=colours, alpha=0.88, height=0.55)
            ax1.set_xlabel("seconds", color=FG_DIM, fontsize=8)
        ax1.set_title("Time per Emotion", color=FG, fontsize=9,
                      fontweight="bold", pad=6)
        ax1.tick_params(colors=FG_DIM, labelsize=8)
        for sp in ax1.spines.values():
            sp.set_color(BG3)
        ax1.xaxis.label.set_color(FG_DIM)

        if not has_eval:
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
            return

        ev  = stats["eval"]
        pc  = ev.get("per_class", {})
        acc = ev.get("accuracy", 0)

        # ── Chart 2: per-class F1 ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(BG2)
        cls      = list(pc.keys())
        f1s      = [pc[c]["f1"] for c in cls]
        colours2 = [EMOTION_COLORS.get(c, "#888888") for c in cls]
        ax2.barh(cls, f1s, color=colours2, alpha=0.88, height=0.55)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("F1 score", color=FG_DIM, fontsize=8)
        ax2.set_title(f"Per-class F1  (accuracy {acc:.1%})",
                      color=FG, fontsize=9, fontweight="bold", pad=6)
        ax2.tick_params(colors=FG_DIM, labelsize=8)
        for sp in ax2.spines.values():
            sp.set_color(BG3)

        # ── Chart 3: confusion matrix (spans full right column) ────────────
        cm_data = ev.get("confusion_matrix")
        if cm_data:
            cm      = np.array(cm_data, dtype=float)
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

            ax3 = fig.add_subplot(gs[:, 1])
            ax3.set_facecolor(BG2)
            im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

            short = [c[:3] for c in cls]
            ax3.set_xticks(range(len(cls)))
            ax3.set_yticks(range(len(cls)))
            ax3.set_xticklabels(short, fontsize=7, color=FG_DIM,
                                rotation=45, ha="right")
            ax3.set_yticklabels(short, fontsize=7, color=FG_DIM)
            ax3.set_xlabel("Predicted", color=FG_DIM, fontsize=8)
            ax3.set_ylabel("Ground Truth", color=FG_DIM, fontsize=8)
            ax3.set_title("Confusion Matrix (row-normalised)",
                          color=FG, fontsize=9, fontweight="bold", pad=6)
            ax3.tick_params(colors=FG_DIM)
            for sp in ax3.spines.values():
                sp.set_color(BG3)

            for i in range(len(cls)):
                for j in range(len(cls)):
                    count = int(cm[i, j])
                    if count > 0:
                        txt_color = "white" if cm_norm[i, j] > 0.5 else FG_DIM
                        ax3.text(j, i, str(count), ha="center", va="center",
                                 fontsize=6, color=txt_color)

            cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelcolor=FG_DIM, labelsize=7)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)

    def _build_stats_text(self, win, stats):
        """Plain-text fallback when matplotlib is not available."""
        body = tk.Frame(win, bg=BG, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        emotions_data = stats.get("emotion_durations", {})
        if emotions_data:
            tk.Label(body, text="Time per emotion:", fg=FG, bg=BG,
                     font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
            total = sum(emotions_data.values()) or 1
            for emo, sec in emotions_data.items():
                pct = sec / total * 100
                color = EMOTION_COLORS.get(emo, FG_DIM)
                tk.Label(body, text=f"  {emo:<10}  {sec:5.1f}s  ({pct:.0f}%)",
                         fg=color, bg=BG, font=("Courier", 10)).pack(anchor="w")

        ev = stats.get("eval")
        if ev:
            tk.Label(body, text=f"\nOverall accuracy: {ev['accuracy']:.1%}",
                     fg=FG, bg=BG, font=("Helvetica", 10, "bold")).pack(anchor="w")
            for cls, m in ev.get("per_class", {}).items():
                tk.Label(body,
                         text=f"  {cls:<10}  F1 {m['f1']:.2f}  "
                              f"(prec {m['precision']:.2f}  rec {m['recall']:.2f})",
                         fg=EMOTION_COLORS.get(cls, FG_DIM), bg=BG,
                         font=("Courier", 9)).pack(anchor="w")

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Launcher().run()
