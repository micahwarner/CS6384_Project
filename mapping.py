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
    chord_root_note: int = 60       # MIDI note of the active chord root (for bass)
    chord_voicing: List[int] = field(default_factory=list)
    chord_size: int = 3

    # Spatial
    pan: float = 0.0

    # Meta
    emotion: str = "neutral"
    trigger_note: bool = True
    sustain: float = 0.5

    # Background pad layer
    bg_program: int = 88        # MIDI program for background instrument
    bg_velocity: int = 35       # volume of background chord
    bg_chord_changed: bool = False  # signals audio engine to update bg chord

    # Percussion (MIDI channel 9) – list of (note, velocity) pairs for this slot
    drum_notes: list = field(default_factory=list)


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


_TRANSITION_SECS = 1.5   # seconds to ramp between emotion presets
_PHRASE_BEATS    = 16    # melodic phrase length in beats
_SLOT_BEATS      = 0.5   # one 8th-note per rhythm slot

# Per-emotion chord progressions: list of (scale_degree_index, duration_in_beats).
# Degrees are 0-based indices into the emotion's scale array.
# Chord tones are built as a triad: [deg, deg+2, deg+4] (mod scale length).
CHORD_PROGRESSIONS = {
    "happy":    [(0, 4), (3, 4), (4, 4), (0, 4)],           # I – IV – V – I  (Lydian)
    "sad":      [(0, 6), (5, 2), (6, 4), (0, 4)],           # i – VI – VII – i (Harmonic minor)
    "angry":    [(0, 4), (1, 2), (6, 2), (5, 4), (0, 4)],   # i – ♭II – VII – v – i (Phrygian)
    "surprise": [(0, 2), (2, 2), (4, 2), (2, 2)],           # Whole-tone walk
    "fear":     [(0, 4), (6, 4), (3, 4), (0, 4)],           # i – VII° – iv° – i (Diminished)
    "disgust":  [(0, 4), (4, 2), (0, 2), (3, 4)],           # Blues  i – V – i – IV
    "neutral":  [(0, 4), (2, 4), (4, 4), (2, 4)],           # Pentatonic loop
}

# 16-step rhythmic patterns (1=play, 0=rest) at the trigger subdivision.
# Each slot fires once per trigger_every interval; the pattern determines whether
# that slot produces a note or a rest.
# 8-step patterns (one bar of 4/4 at 8th-note resolution, indexed by _SLOT_BEATS).
# Each slot fires on a beat-synchronized grid so the pattern always sounds in time.
# GM percussion note numbers used in drum patterns.
DRUM_NOTE_VELOCITY = {
    36: 90,  # Bass Drum 1
    38: 78,  # Acoustic Snare
    42: 58,  # Closed Hi-Hat
    46: 65,  # Open Hi-Hat
    49: 88,  # Crash Cymbal
}

# 8-step drum patterns for emotions that have percussion.
# Each step is a list of GM percussion note numbers (empty = rest).
# Only happy, angry, and surprise have drums — slower/tense emotions stay silent.
DRUM_PATTERNS = {
    "happy":    [[36, 42], [42], [38, 42], [42], [36, 42], [42], [38, 42], [42]],
    "angry":    [[36],     [36], [38],     [],   [36],     [],   [38],     [36]],
    "surprise": [[36, 49], [],   [],       [38], [36],     [],   [42],     [38]],
}

RHYTHM_PATTERNS = {
    "happy":    [1, 1, 0, 1, 1, 0, 1, 0],  # syncopated, energetic
    "sad":      [1, 0, 0, 0, 1, 0, 0, 0],  # sparse quarter notes
    "angry":    [1, 1, 0, 1, 0, 1, 1, 0],  # driving, punchy
    "surprise": [1, 0, 0, 1, 1, 0, 0, 1],  # off-beat, erratic
    "fear":     [1, 0, 0, 1, 0, 0, 0, 1],  # 3+3+2 grouping, tense
    "disgust":  [1, 0, 1, 0, 0, 1, 0, 1],  # uneven, bluesy
    "neutral":  [1, 0, 1, 0, 1, 0, 1, 0],  # steady quarter notes
}

# Melodic motifs per emotion.
# Each motif is a short sequence of chord-relative indices:
#   0 = chord root, 1 = chord third, 2 = chord fifth
#  -1 = one scale step below root (passing tone)
#   3 = one scale step above fifth (passing tone)
# Motifs cycle in order; 50% chance to advance to the next motif when one finishes.
MELODIC_MOTIFS = {
    "happy":    [[0, 2, 1, 0],       # root → fifth → third → root
                 [0, 1, 2, 3, 2],    # stepwise ascent with passing tone
                 [2, 1, 0, 2]],      # from fifth back home
    "sad":      [[2, 1, 0],          # sigh: fifth → third → root
                 [0, -1, 0, 1]],     # oscillate with chromatic drooping
    "angry":    [[0, 0, 2, 0],       # root hammer with fifth accent
                 [2, 1, 0, 0]],      # descend to heavy root landing
    "surprise": [[0, 2, 3, 1],       # leap up then scatter
                 [2, 0, 1, 3]],      # surprise zigzag
    "fear":     [[0, -1, 0, 1],      # chromatic neighbor motion
                 [2, 1, 0, -1]],     # creeping descent
    "disgust":  [[0, -1, 0, 1, 0],   # blue-note feel
                 [0, 0, 2, 1, 0]],   # root emphasis with resolution
    "neutral":  [[0, 1, 2, 1],       # simple arc up
                 [2, 1, 0, 1]],      # simple arc down
}

# Background pad instrument per emotion: (MIDI program, velocity)
BG_INSTRUMENTS = {
    "happy":    (89, 40),   # Warm Pad — full warmth behind piano
    "sad":      (91, 30),   # Choir Pad — ethereal sadness
    "angry":    (48, 50),   # String Ensemble — heavy bed under organ
    "surprise": (94, 38),   # Halo Pad — sparkly shimmer
    "fear":     (92, 28),   # Bowed Pad — eerie glass-like texture
    "disgust":  (95, 25),   # Sweep Pad — dark murk
    "neutral":  (89, 28),   # Warm Pad — gentle warmth
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
        self._from_preset: dict = EMOTION_PRESETS["neutral"]
        self._transition_start: float = 0.0
        self._total_beats: float = 0.0
        self._last_map_time: float = time.time()
        self._rhythm_idx: int = 0
        self._last_trigger_beat: float = 0.0
        self._motif_pos: int = 0
        self._motif_idx: int = 0
        self._current_motifs: list = MELODIC_MOTIFS["neutral"]
        self._fixed_bpm: float = 93.0  # locked per emotion, random within range
        self._last_bpm: float = 93.0   # smoothed toward _fixed_bpm each frame
        self._last_chord_root_deg: int = -1

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
                    self._from_preset = EMOTION_PRESETS[self._current_emotion]
                    self._transition_start = now
                    self._total_beats = 0.0
                    self._last_map_time = now
                    self._rhythm_idx = 0
                    self._last_trigger_beat = 0.0
                    self._motif_pos = 0
                    self._motif_idx = 0
                    new_range = EMOTION_PRESETS[emotion_key]["bpm_range"]
                    self._fixed_bpm = random.uniform(new_range[0], new_range[1])
                    self._last_chord_root_deg = -1
                    self._current_emotion = emotion_key
                    self._pending_emotion = None
                    self._scale_cursor = 0
                    self._step_dir = 1
        else:
            self._pending_emotion = None
            self._emotion_hold_count = 0

        emotion_key = self._current_emotion
        preset = EMOTION_PRESETS[emotion_key]
        self._current_motifs = MELODIC_MOTIFS[emotion_key]

        params = MusicParameters(emotion=features.emotion)

        scale_offsets = SCALES[preset["scale"]]
        params.scale = scale_offsets
        params.root_note = preset["root"]
        params.chord_size = preset.get("chord_size", 3)

        params.sustain = preset.get("sustain", 0.3)

        # Transition alpha: smoothstep curve over _TRANSITION_SECS
        raw_alpha = np.clip((now - self._transition_start) / _TRANSITION_SECS, 0.0, 1.0)
        alpha = raw_alpha * raw_alpha * (3.0 - 2.0 * raw_alpha)
        fp = self._from_preset

        # BPM: fixed per emotion (random within range on switch), EMA for smooth transition
        self._last_bpm = self._last_bpm * 0.97 + self._fixed_bpm * 0.03
        params.bpm = self._last_bpm

        # Beat / chord / phrase tracking
        dt = now - self._last_map_time
        self._last_map_time = now
        self._total_beats += dt * params.bpm / 60.0

        progression = CHORD_PROGRESSIONS[emotion_key]
        prog_total = sum(b for _, b in progression)
        beat_in_prog = self._total_beats % prog_total
        cumulative, chord_root_deg = 0.0, 0
        for deg, dur in progression:
            cumulative += dur
            if beat_in_prog < cumulative:
                chord_root_deg = deg
                break
        n_scale = len(scale_offsets)
        chord_tones = [chord_root_deg % n_scale,
                       (chord_root_deg + 2) % n_scale,
                       (chord_root_deg + 4) % n_scale]
        phrase_phase = (self._total_beats % _PHRASE_BEATS) / _PHRASE_BEATS
        params.chord_root_note = preset["root"] + scale_offsets[chord_root_deg % n_scale]

        # Signal background layer to update whenever the chord changes
        if chord_root_deg != self._last_chord_root_deg:
            self._last_chord_root_deg = chord_root_deg
            params.bg_chord_changed = True
        bg_prog, bg_vel = BG_INSTRUMENTS[emotion_key]
        params.bg_program  = bg_prog
        params.bg_velocity = bg_vel

        prog_num, prog_name = INSTRUMENTS[preset["instrument"]]
        params.program = prog_num
        params.instrument_name = prog_name

        # Velocity: interpolated range, then scaled by emotion confidence
        # Low confidence → stays near vel_lo; high confidence → full dynamic range
        confidence = np.clip(features.emotion_scores.get(emotion_key, 100) / 100, 0.0, 1.0)
        vel_lo = fp["velocity_range"][0] + alpha * (preset["velocity_range"][0] - fp["velocity_range"][0])
        vel_hi = fp["velocity_range"][1] + alpha * (preset["velocity_range"][1] - fp["velocity_range"][1])
        base_vel = vel_lo + features.mouth_openness * (vel_hi - vel_lo)
        params.velocity = int(np.clip(vel_lo + confidence * (base_vel - vel_lo), 0, 127))

        # Phrase arc: 75% at start → swell to full → resolve to 80% at end
        if phrase_phase < 0.5:
            phrase_factor = 0.75 + 0.25 * (phrase_phase / 0.5)
        else:
            phrase_factor = 1.0 - 0.20 * ((phrase_phase - 0.5) / 0.5)
        params.velocity = int(np.clip(params.velocity * phrase_factor, 0, 127))

        #Pitch selection
        brow_shift = int(np.clip(round((features.eyebrow_raise - 0.5) * 4), -1, 1))
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
        params.note_duration = fp["note_duration"] + alpha * (preset["note_duration"] - fp["note_duration"])

        # Timing: advance one slot per 8th note; fire only if the pattern says so.
        # The while loop catches up cleanly if frames are slow or BPM is very high.
        params.trigger_note = False
        params.drum_notes = []
        while self._total_beats >= self._last_trigger_beat + _SLOT_BEATS:
            self._last_trigger_beat += _SLOT_BEATS
            pattern = RHYTHM_PATTERNS[emotion_key]
            slot_idx = self._rhythm_idx % len(pattern)
            slot_is_note = pattern[slot_idx] == 1

            drum_pat = DRUM_PATTERNS.get(emotion_key, [])
            if drum_pat:
                params.drum_notes = [
                    (note, DRUM_NOTE_VELOCITY.get(note, 70))
                    for note in drum_pat[slot_idx % len(drum_pat)]
                ]

            self._rhythm_idx = (self._rhythm_idx + 1) % len(pattern)
            if slot_is_note and random.random() >= preset.get("rest_bias", 0.0):
                params.trigger_note = True
                self._advance_cursor(scale_offsets, preset, chord_tones, phrase_phase)
                break

        return params

    # Helpers

    def _advance_cursor(self, scale: list, preset: dict,
                        chord_tones: list, phrase_phase: float):
        n = len(scale)

        if random.random() < preset.get("repeat_bias", 0.2):
            return

        if phrase_phase > 0.75 and random.random() < 0.65:
            # Phrase cadence: resolve to tonic
            self._scale_cursor = 0
            self._motif_pos = 0
            return

        # Advance through the current motif sequentially
        motif = self._current_motifs[self._motif_idx % len(self._current_motifs)]
        el = motif[self._motif_pos % len(motif)]
        self._motif_pos += 1
        if self._motif_pos >= len(motif):
            self._motif_pos = 0
            if random.random() < 0.5:   # 50% chance to pick the next motif
                self._motif_idx = (self._motif_idx + 1) % len(self._current_motifs)

        # Resolve motif element to a scale cursor position
        if el == 0:
            self._scale_cursor = chord_tones[0]
        elif el == 1:
            self._scale_cursor = chord_tones[1] if len(chord_tones) > 1 else chord_tones[0]
        elif el == 2:
            self._scale_cursor = chord_tones[-1]
        elif el == -1:
            self._scale_cursor = (chord_tones[0] - 1) % n
        elif el == 3:
            self._scale_cursor = (chord_tones[-1] + 1) % n

    def get_preset_info(self, emotion: str) -> str:
        key = FER_ALIAS.get(emotion, "neutral")
        preset = EMOTION_PRESETS[key]
        return (
            f"{emotion.upper()} → {preset['description']} | "
            f"BPM: {preset['bpm_range'][0]}–{preset['bpm_range'][1]} | "
            f"Instr: {preset['instrument']} | Sustain: {preset['sustain']}"
        )