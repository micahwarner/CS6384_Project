"""
mapping.py: Translates FaceFeatures into MusicParameters.

Emotion sets the tonal context (mode, key, tempo, timbre, sustain, register)
Facial geometry then modulates within that context in real time:

  mouth_openness  → volume
  smile_width     → tempo / energy
  eyebrow_raise   → pitch contour
  eye_openness    → octave lift/drop
  head_tilt       → stereo pan / pitch bend

"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
import random
import time
from vision import FaceFeatures


# output

@dataclass
class MusicParameters:
    """Everything the audio engine needs to generate one note event"""

    # Pitch
    midi_note: int = 60
    pitch_bend: float = 0.0

    # Time
    bpm: float = 90.0
    note_duration: float = 0.5

    # Dynamics
    velocity: int = 80

    # Timbre / Instrument
    program: int = 0
    instrument_name: str = "Acoustic Grand Piano"

    # Harmony
    scale: List[int] = field(default_factory=list)
    root_note: int = 60
    chord_voicing: List[int] = field(default_factory=list)
    chord_size: int = 3

    # Spatial
    pan: float = 0.0

    # Meta
    emotion: str = "neutral"
    trigger_note: bool = True
    sustain: float = 0.5


# Scale Definitions

SCALES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_maj": [0, 2, 4, 7, 9],
    "pentatonic_min": [0, 3, 5, 7, 10],
    "blues":          [0, 3, 5, 6, 7, 10],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "lydian":         [0, 2, 4, 6, 7, 9, 11],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],
    "whole_tone":     [0, 2, 4, 6, 8, 10],
    "diminished":     [0, 2, 3, 5, 6, 8, 9, 11],
}


# General MIDI program numbers
INSTRUMENTS = {
    "piano":      (0,  "Acoustic Grand Piano"),
    "strings":    (48, "String Ensemble 1"),
    "flute":      (73, "Flute"),
    "pad":        (88, "New Age Pad"),
    "organ":      (19, "Church Organ"),
    "guitar":     (25, "Acoustic Guitar (Steel)"),
    "marimba":    (12, "Marimba"),
    "celesta":    (8,  "Celesta"),
    "trumpet":    (56, "Trumpet"),
    "synth_lead": (80, "Square Lead"),
    "choir":      (52, "Voice Oohs"),
    "music_box":  (10, "Music Box"),
}


# emotion presets

EMOTION_PRESETS = {
    "happy": {
        "scale": "lydian",
        "root": 60,
        "bpm_range": (105, 155),
        "velocity_range": (75, 115),
        "instrument": "piano",
        "note_duration": 0.28,
        "description": "Bright, energetic, lifted and consonant",
        "sustain": 0.45,
        "register_shift": 12,
        "max_leap": 2,
        "repeat_bias": 0.08,
        "rest_bias": 0.04,
        "chord_size": 3,
        "activity": 0.82,
        "brightness": 0.90,
    },
    "sad": {
        "scale": "harmonic_minor",
        "root": 57,
        "bpm_range": (45, 78),
        "velocity_range": (30, 68),
        "instrument": "strings",
        "note_duration": 1.05,
        "description": "Slow, low, sparse, sustained melancholy",
        "sustain": 1.40,
        "register_shift": -12,
        "max_leap": 1,
        "repeat_bias": 0.40,
        "rest_bias": 0.22,
        "chord_size": 2,
        "activity": 0.28,
        "brightness": 0.18,
    },
    "angry": {
        "scale": "phrygian",
        "root": 52,
        "bpm_range": (70, 110),
        "velocity_range": (95, 127),
        "instrument": "organ",
        "note_duration": 0.75,
        "description": "Dark, threatening, heavy, ominous",
        "sustain": 0.9,
        "register_shift": -12,
        "max_leap": 2,
        "repeat_bias": 0.18,
        "rest_bias": 0.22,
        "chord_size": 4,
        "activity": 0.32,
        "brightness": 0.08,
    },
    "surprise": {
        "scale": "whole_tone",
        "root": 62,
        "bpm_range": (95, 155),
        "velocity_range": (78, 118),
        "instrument": "celesta",
        "note_duration": 0.24,
        "description": "Sparkly, sudden, upward and unstable",
        "sustain": 0.95,
        "register_shift": 12,
        "max_leap": 4,
        "repeat_bias": 0.10,
        "rest_bias": 0.08,
        "chord_size": 2,
        "activity": 0.78,
        "brightness": 0.95,
    },
    "fear": {
        "scale": "diminished",
        "root": 59,
        "bpm_range": (70, 118),
        "velocity_range": (42, 82),
        "instrument": "pad",
        "note_duration": 0.42,
        "description": "Thin, tense, unstable and eerie",
        "sustain": 1.60,
        "register_shift": -5,
        "max_leap": 4,
        "repeat_bias": 0.14,
        "rest_bias": 0.28,
        "chord_size": 2,
        "activity": 0.42,
        "brightness": 0.25,
    },
    "disgust": {
        "scale": "blues",
        "root": 53,
        "bpm_range": (62, 95),
        "velocity_range": (48, 88),
        "instrument": "guitar",
        "note_duration": 0.50,
        "description": "Dry, gritty, uneven, bluesy tension",
        "sustain": 0.05,
        "register_shift": -7,
        "max_leap": 2,
        "repeat_bias": 0.26,
        "rest_bias": 0.16,
        "chord_size": 2,
        "activity": 0.48,
        "brightness": 0.22,
    },
    "neutral": {
        "scale": "pentatonic_maj",
        "root": 60,
        "bpm_range": (78, 108),
        "velocity_range": (52, 92),
        "instrument": "marimba",
        "note_duration": 0.38,
        "description": "Balanced, simple, centered, unobtrusive",
        "sustain": 0.12,
        "register_shift": 0,
        "max_leap": 2,
        "repeat_bias": 0.18,
        "rest_bias": 0.10,
        "chord_size": 2,
        "activity": 0.55,
        "brightness": 0.50,
    },
}


FER_ALIAS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    "neutral": "neutral",
}


# Main Mapper

class ExpressionMusicMapper:
    """
    Stateful mapper: tracks the current emotion context and melodic cursor
    so successive calls produce a coherent, emotion aware melody
    """

    def __init__(self, note_trigger_interval: float = 0.4):
        self.note_interval = note_trigger_interval
        self._last_note_time = 0.0
        self._scale_cursor = 0
        self._current_emotion = "neutral"
        self._step_dir = 1
        self._pending_emotion = None
        self._emotion_hold_count = 0
        self._emotion_hold_frames = 4
        self._start_time = time.time()

    # Public

    def map(self, features: FaceFeatures) -> MusicParameters:
        now = time.time()

        emotion_key = FER_ALIAS.get(features.emotion, "neutral")

        # Smooth emotion transitions
        if emotion_key != self._current_emotion:
            if self._pending_emotion != emotion_key:
                self._pending_emotion = emotion_key
                self._emotion_hold_count = 0
            else:
                self._emotion_hold_count += 1
                if self._emotion_hold_count >= self._emotion_hold_frames:
                    self._current_emotion = emotion_key
                    self._pending_emotion = None
                    self._scale_cursor = 0
                    self._step_dir = 1
        else:
            self._pending_emotion = None
            self._emotion_hold_count = 0

        emotion_key = self._current_emotion
        preset = EMOTION_PRESETS[emotion_key]

        params = MusicParameters(emotion=features.emotion)

        scale_offsets = SCALES[preset["scale"]]
        params.scale = scale_offsets
        params.root_note = preset["root"]
        params.chord_size = preset.get("chord_size", 3)

        prog_num, prog_name = INSTRUMENTS[preset["instrument"]]
        params.program = prog_num
        params.instrument_name = prog_name
        params.sustain = preset.get("sustain", 0.3)

        # BPM
        bpm_lo, bpm_hi = preset["bpm_range"]
        bpm_modulation = np.clip(features.smile_width, 0, 1)
        params.bpm = bpm_lo + bpm_modulation * (bpm_hi - bpm_lo)

        # Velocity
        vel_lo, vel_hi = preset["velocity_range"]
        params.velocity = int(vel_lo + features.mouth_openness * (vel_hi - vel_lo))
        params.velocity = int(np.clip(params.velocity, 0, 127))

        #Pitch selection
        brow_shift = round((features.eyebrow_raise - 0.5) * 10)
        cursor = (self._scale_cursor + brow_shift) % len(scale_offsets)
        base_offset = scale_offsets[cursor]

        if features.eye_openness > 0.78:
            octave_shift = 12
        elif features.eye_openness < 0.22:
            octave_shift = -12
        else:
            octave_shift = 0

        register_shift = preset.get("register_shift", 0)

        params.midi_note = int(np.clip(
            preset["root"] + register_shift + base_offset + octave_shift, 24, 108
        ))

        # Expression
        params.pitch_bend = float(np.clip((features.head_tilt - 0.5) * 3.0, -2.0, 2.0))
        params.pan = float(np.clip((features.head_tilt - 0.5) * 2.0, -1, 1))
        params.note_duration = preset["note_duration"]

        # Timing / density
        elapsed = now - self._last_note_time
        beat_duration = 60.0 / max(params.bpm, 1.0)

        activity = preset.get("activity", 0.5)
        density_factor = np.interp(activity, [0.0, 1.0], [1.35, 0.35])
        trigger_every = max(0.12, self.note_interval * density_factor, beat_duration * 0.30)

        if elapsed >= trigger_every:
            if random.random() < preset.get("rest_bias", 0.0):
                params.trigger_note = False
                self._last_note_time = now
            else:
                params.trigger_note = True
                self._last_note_time = now
                self._advance_cursor(scale_offsets, preset)
        else:
            params.trigger_note = False

        return params

    # Helpers

    def _advance_cursor(self, scale: list, preset: dict):
        repeat_bias = preset.get("repeat_bias", 0.2)
        max_leap = max(1, int(preset.get("max_leap", 2)))
        brightness = preset.get("brightness", 0.5)

        roll = random.random()

        if roll < repeat_bias:
            step = 0
        else:
            leap = random.randint(1, max_leap)
            direction = self._step_dir

            if brightness > 0.65 and random.random() < 0.60:
                direction = 1
            elif brightness < 0.35 and random.random() < 0.60:
                direction = -1

            step = direction * leap

        self._scale_cursor = (self._scale_cursor + step) % len(scale)

        if random.random() < 0.18:
            self._step_dir *= -1

    def get_preset_info(self, emotion: str) -> str:
        key = FER_ALIAS.get(emotion, "neutral")
        preset = EMOTION_PRESETS[key]
        return (
            f"{emotion.upper()} → {preset['description']} | "
            f"BPM: {preset['bpm_range'][0]}–{preset['bpm_range'][1]} | "
            f"Instr: {preset['instrument']} | Sustain: {preset['sustain']}"
        )