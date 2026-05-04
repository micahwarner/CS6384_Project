"""
audio.py: Real-time audio engine using pygame.midi with a numpy/pygame
fallback synthesizer

Backends
────────
1. MidiPlayer: primary, low-latency system MIDI output
2. SynthPlayer: fallback, pure Python additive synthesis

"""

import heapq
import itertools
import threading
import time
from typing import Optional

import numpy as np

from mapping import MusicParameters


# MIDI Player

class MidiPlayer:
    """
    Sends MIDI to the system synthesizer via pygame.midi

    """

    def __init__(self, device_id: int = -1):
        import pygame.midi

        pygame.midi.init()
        if device_id < 0:
            device_id = pygame.midi.get_default_output_id()
        if device_id < 0:
            raise RuntimeError("No MIDI output device found")

        self.out = pygame.midi.Output(device_id, latency=1)

        self._current_channel = 0
        self._bass_channel = 1
        self._current_program = -1
        self._current_chord = []

        self._lock = threading.Lock()
        self._closing = False

        self._bass_counter = 0
        self._sustain_token = 0

        # Scheduler state
        self._events = []
        self._event_counter = itertools.count()
        self._event_cv = threading.Condition(self._lock)
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()

        # Bass instrument: Acoustic Bass
        self.out.set_instrument(32, self._bass_channel)

        self._bg_channel = 2
        self._bg_chord: list = []
        self._bg_program: int = -1

    # helpers

    @staticmethod
    def _clip_note(note: int) -> int:
        return int(np.clip(note, 24, 108))

    @staticmethod
    def _circular_distance(a: int, b: int) -> int:
        d = abs(a - b) % 12
        return min(d, 12 - d)

    def _schedule_event(self, when: float, event_type: str, payload: dict):
        with self._event_cv:
            if self._closing:
                return
            heapq.heappush(
                self._events,
                (when, next(self._event_counter), event_type, payload)
            )
            self._event_cv.notify()

    def _scheduler_loop(self):
        while True:
            with self._event_cv:
                while not self._events and not self._closing:
                    self._event_cv.wait()

                if self._closing:
                    break

                when, _, event_type, payload = self._events[0]
                now = time.time()
                delay = when - now

                if delay > 0:
                    self._event_cv.wait(timeout=delay)
                    continue

                heapq.heappop(self._events)

            # Execute event outside condition wait, but still under MIDI lock
            with self._lock:
                if self._closing:
                    break
                try:
                    if event_type == "note_off":
                        notes = payload["notes"]
                        channel = payload["channel"]
                        for note in notes:
                            self.out.note_off(int(note), 0, channel)

                    elif event_type == "sustain_off":
                        channel = payload["channel"]
                        token = payload["token"]
                        if token == self._sustain_token:
                            self.out.write_short(0xB0 | channel, 64, 0)
                except Exception:
                    pass

    def _send_pan(self, pan: float, channel: int):
        pan_val = int(np.clip((pan + 1.0) * 63.5, 0, 127))
        self.out.write_short(0xB0 | channel, 10, pan_val)

    def _send_pitch_bend(self, semitones: float, channel: int):
        semitones = float(np.clip(semitones, -2.0, 2.0))
        value = int(8192 + (semitones / 2.0) * 8191)
        value = int(np.clip(value, 0, 16383))
        lsb = value & 0x7F
        msb = (value >> 7) & 0x7F
        self.out.write_short(0xE0 | channel, lsb, msb)

    # chord builder

    @classmethod
    def _build_chord(cls, root: int, scale: list, scale_root: int, chord_size: int = 3) -> list:
        """
        Build a scale-aware chord around the melody note.

        chord_size:
            1 -> single note
            2 -> dyad / open interval
            3 -> triad
            4 -> seventh chord
        """
        if not scale:
            return [cls._clip_note(root)]

        offset = (root - scale_root) % 12
        scale_mod = [s % 12 for s in scale]

        idx = min(
            range(len(scale_mod)),
            key=lambda i: cls._circular_distance(scale_mod[i], offset)
        )

        if chord_size <= 1:
            steps = [0]
        elif chord_size == 2:
            steps = [0, 4] if len(scale) >= 5 else [0, 2]
        elif chord_size == 3:
            steps = [0, 2, 4]
        else:
            steps = [0, 2, 4, 6]

        chord_notes = []
        for step in steps:
            target_idx = idx + step
            scale_degree = scale[target_idx % len(scale)] + 12 * (target_idx // len(scale))
            note = scale_root + scale_degree

            while note < root - 5:
                note += 12
            while note > root + 12:
                note -= 12

            chord_notes.append(cls._clip_note(note))

        chord_notes = sorted(set(chord_notes))
        if not chord_notes:
            chord_notes = [cls._clip_note(root)]
        return chord_notes

    # Background chord

    def _update_bg_chord(self, params: MusicParameters):
        """Swap in a new sustained background chord. Call with self._lock held."""
        if params.bg_program != self._bg_program:
            self.out.write_short(0xB0 | self._bg_channel, 64, 0)
            for note in self._bg_chord:
                self.out.note_off(note, 0, self._bg_channel)
            self.out.set_instrument(params.bg_program, self._bg_channel)
            self._bg_program = params.bg_program
        else:
            for note in self._bg_chord:
                self.out.note_off(note, 0, self._bg_channel)

        # Place chord root in a middle-low register, below the melody
        bg_root = params.chord_root_note
        while bg_root > 60:
            bg_root -= 12
        while bg_root < 36:
            bg_root += 12

        chord = self._build_chord(bg_root, params.scale, params.root_note, 3)
        self._bg_chord = chord

        self.out.write_short(0xB0 | self._bg_channel, 64, 127)   # sustain on
        for note in chord:
            self.out.note_on(note, params.bg_velocity, self._bg_channel)

    # Playback

    def play(self, params: MusicParameters):
        if self._closing:
            return

        bass_note = None
        now = time.time()

        with self._lock:
            if self._closing:
                return

            if params.bg_chord_changed:
                self._update_bg_chord(params)

            # Percussion on GM channel 9 — fires every slot, independent of melody
            # Channel 9 drums decay naturally so note_off is not needed
            for note, vel in params.drum_notes:
                self.out.note_on(note, vel, 9)

            if not params.trigger_note:
                return

            self._sustain_token += 1
            sustain_token = self._sustain_token

            # Instrument change: clear pedal and old notes for clean transition
            if params.program != self._current_program:
                self.out.write_short(0xB0 | self._current_channel, 64, 0)
                for note in self._current_chord:
                    self.out.note_off(note, 0, self._current_channel)
                self.out.set_instrument(params.program, self._current_channel)
                self._current_program = params.program

            # Expression controls
            self._send_pan(params.pan, self._current_channel)
            self._send_pan(params.pan * 0.35, self._bass_channel)
            self._send_pitch_bend(params.pitch_bend, self._current_channel)

            # If dry note, explicitly lift sustain
            if params.sustain <= 0:
                self.out.write_short(0xB0 | self._current_channel, 64, 0)

            # Release current chord's note-ons; sustained notes will naturally continue if pedal is down
            for note in self._current_chord:
                self.out.note_off(note, 0, self._current_channel)

            # Build new chord
            chord = self._build_chord(
                params.midi_note,
                params.scale,
                params.root_note,
                getattr(params, "chord_size", 3)
            )
            self._current_chord = chord

            # Play chord tones
            for i, note in enumerate(chord):
                if i == 0:
                    vel = params.velocity
                elif i == 1:
                    vel = int(params.velocity * 0.72)
                else:
                    vel = int(params.velocity * 0.58)

                self.out.note_on(note, int(np.clip(vel, 1, 127)), self._current_channel)

            # Sustain pedal
            if params.sustain > 0:
                self.out.write_short(0xB0 | self._current_channel, 64, 127)
            else:
                self.out.write_short(0xB0 | self._current_channel, 64, 0)

            # Bass every other trigger
            self._bass_counter += 1
            if self._bass_counter % 2 == 0:
                bass_note = self._clip_note(params.chord_root_note - 12)
                bass_vel = int(np.clip(params.velocity * 0.50, 1, 110))
                self.out.note_on(bass_note, bass_vel, self._bass_channel)

        # Schedule note-offs
        self._schedule_event(
            now + max(0.03, params.note_duration * 0.95),
            "note_off",
            {"notes": chord, "channel": self._current_channel}
        )

        if bass_note is not None:
            self._schedule_event(
                now + max(0.03, params.note_duration * 1.15),
                "note_off",
                {"notes": [bass_note], "channel": self._bass_channel}
            )

        if params.sustain > 0:
            self._schedule_event(
                now + float(params.sustain),
                "sustain_off",
                {"channel": self._current_channel, "token": sustain_token}
            )

    # Cleanup

    def close(self):
        import pygame.midi

        # Signal scheduler to stop — fast, no blocking
        with self._event_cv:
            self._closing = True
            self._events.clear()
            self._event_cv.notify_all()

        # All blocking MIDI teardown runs on a daemon thread so a frozen
        # pygame.midi driver or a scheduler thread stuck in note_off() can
        # never hang the main thread (and therefore the launcher).
        def _shutdown():
            try:
                if self._scheduler_thread.is_alive():
                    self._scheduler_thread.join(timeout=0.3)
            except Exception:
                pass
            try:
                with self._lock:
                    for ch in [self._current_channel,
                                self._bass_channel,
                                self._bg_channel]:
                        for msg in [
                            (0xB0 | ch, 64,  0),    # sustain off
                            (0xB0 | ch, 123, 0),    # all notes off
                            (0xB0 | ch, 120, 0),    # all sound off
                        ]:
                            try:
                                self.out.write_short(*msg)
                            except Exception:
                                pass
                    try:
                        self.out.close()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                pygame.midi.quit()
            except Exception:
                pass

        t = threading.Thread(target=_shutdown, daemon=True)
        t.start()
        t.join(timeout=0.5)   # give up after 500 ms; daemon dies with the process


# Synthesizer Player (Fallback)

class SynthPlayer:
    """
    Pure numpy/pygame additive synthesizer fallback

    Supports:
    - Chords via params.chord_size
    - Extended perceived sustain
    - Quiet bass reinforcement
    - Stereo panning
    """

    SAMPLE_RATE = 44100
    CHANNELS = 2
    CHUNK = 2048

    TIMBRES = {
        "piano":      [(1, 1.0), (2, 0.6), (3, 0.3), (4, 0.1)],
        "strings":    [(1, 1.0), (2, 0.5), (3, 0.4), (4, 0.3), (5, 0.2)],
        "flute":      [(1, 1.0), (2, 0.15), (3, 0.05)],
        "pad":        [(1, 1.0), (2, 0.3), (3, 0.1)],
        "organ":      [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4)],
        "guitar":     [(1, 1.0), (2, 0.7), (3, 0.5), (4, 0.2)],
        "marimba":    [(1, 1.0), (2, 0.45), (4, 0.2)],
        "celesta":    [(1, 1.0), (2, 0.3), (6, 0.15)],
        "trumpet":    [(1, 1.0), (2, 0.9), (3, 0.7), (4, 0.5), (5, 0.3)],
        "synth_lead": [(1, 1.0), (3, 0.5), (5, 0.3), (7, 0.2)],
        "choir":      [(1, 1.0), (2, 0.2), (3, 0.1)],
        "music_box":  [(1, 1.0), (3, 0.3), (5, 0.1)],
    }

    def __init__(self):
        import pygame
        import pygame.mixer

        pygame.mixer.pre_init(self.SAMPLE_RATE, -16, self.CHANNELS, self.CHUNK)
        pygame.mixer.init()

        self._channel = pygame.mixer.Channel(0)
        self._current_params: Optional[MusicParameters] = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def play(self, params: MusicParameters):
        if params.trigger_note:
            with self._lock:
                self._current_params = params

    def _audio_loop(self):
        import pygame

        while self._running:
            with self._lock:
                p = self._current_params
                self._current_params = None  # consume once so we don't replay stale notes

            if p is None:
                time.sleep(0.02)
                continue

            try:
                samples = self._synthesize(p)
                sound = pygame.sndarray.make_sound(samples)
                self._channel.play(sound)
            except Exception as e:
                print(f"[SynthPlayer] Audio error: {e}")
                time.sleep(0.05)
                continue

            wait_time = max(0.05, p.note_duration * 0.80)
            time.sleep(wait_time)

    def _synthesize(self, p: MusicParameters) -> np.ndarray:
        effective_duration = max(p.note_duration, p.note_duration + p.sustain * 0.55)
        n_samples = int(self.SAMPLE_RATE * effective_duration)
        t = np.linspace(0, effective_duration, n_samples, endpoint=False)

        instr_key = self._program_to_timbre_key(p.program)
        harmonics = self.TIMBRES.get(instr_key, self.TIMBRES["piano"])

        chord = MidiPlayer._build_chord(
            p.midi_note,
            p.scale,
            p.root_note,
            getattr(p, "chord_size", 3)
        )

        wave = np.zeros(n_samples, dtype=np.float64)

        # Chord / melody
        for i, note in enumerate(chord):
            bend = p.pitch_bend if i == 0 else 0.0
            freq = self._midi_to_freq(note + bend)
            partial = np.zeros(n_samples, dtype=np.float64)

            for harmonic, amp in harmonics:
                partial += amp * np.sin(2 * np.pi * freq * harmonic * t)

            partial /= (np.max(np.abs(partial)) + 1e-9)

            if i == 0:
                gain = 1.0
            elif i == 1:
                gain = 0.68
            else:
                gain = 0.50

            wave += partial * gain

        # Bass reinforcement
        bass_freq = self._midi_to_freq(max(24, p.chord_root_note - 12))
        wave += 0.18 * np.sin(2 * np.pi * bass_freq * t)

        peak = np.max(np.abs(wave)) + 1e-9
        wave = wave / peak

        wave = self._apply_adsr(wave, self.SAMPLE_RATE, effective_duration, p.sustain)
        wave *= (p.velocity / 127.0) * 0.88

        pan = np.clip((p.pan + 1.0) / 2.0, 0, 1)
        left_gain = np.sqrt(1 - pan)
        right_gain = np.sqrt(pan)

        stereo = np.zeros((n_samples, 2), dtype=np.int16)
        stereo[:, 0] = (wave * left_gain * 32767).astype(np.int16)
        stereo[:, 1] = (wave * right_gain * 32767).astype(np.int16)
        return stereo

    @staticmethod
    def _apply_adsr(wave: np.ndarray, sr: int, dur: float, sustain: float) -> np.ndarray:
        n = len(wave)

        attack = min(int(0.03 * sr), max(8, n // 8))
        decay = min(int(0.08 * sr), max(8, n // 8))
        sustain_level = 0.72 if sustain > 0 else 0.60

        release_time = min(0.20 + sustain * 0.30, dur * 0.5)
        release = min(int(release_time * sr), max(8, n // 3))

        env = np.ones(n, dtype=np.float64)
        env[:attack] = np.linspace(0, 1, attack)

        decay_end = min(attack + decay, n)
        if decay_end > attack:
            env[attack:decay_end] = np.linspace(1, sustain_level, decay_end - attack)

        sustain_start = decay_end
        sustain_end = max(sustain_start, n - release)
        env[sustain_start:sustain_end] = sustain_level

        if release > 0:
            env[-release:] = np.linspace(sustain_level, 0, release)

        return wave * env

    @staticmethod
    def _midi_to_freq(note: float) -> float:
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    @staticmethod
    def _program_to_timbre_key(prog: int) -> str:
        mapping = {
            0: "piano",
            8: "celesta",
            10: "music_box",
            12: "marimba",
            19: "organ",
            25: "guitar",
            48: "strings",
            52: "choir",
            56: "trumpet",
            73: "flute",
            80: "synth_lead",
            88: "pad",
        }
        return mapping.get(prog, "piano")

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)


#AudioEngine — Auto-Selects Backend

class AudioEngine:
    """
    Picks MidiPlayer when system MIDI is available,
    otherwise falls back to SynthPlayer.
    """

    def __init__(self, force_synth: bool = False):
        self.backend_name = "none"
        self._player = None

        if not force_synth:
            try:
                self._player = MidiPlayer()
                self.backend_name = "midi"
                print("[AudioEngine] Using system MIDI backend.")
            except Exception as e:
                print(f"[AudioEngine] MIDI unavailable ({e}), falling back to software synth.")

        if self._player is None:
            try:
                self._player = SynthPlayer()
                self.backend_name = "synth"
                print("[AudioEngine] Using software synthesizer backend.")
            except Exception as e:
                print(f"[AudioEngine] SynthPlayer failed: {e}")
                print("[AudioEngine] Audio disabled.")

    def play(self, params: MusicParameters):
        if self._player:
            self._player.play(params)

    def close(self):
        if not self._player:
            return
        try:
            self._player.close()
        except Exception as e:
            print(f"[AudioEngine] close warning: {e}")

    @property
    def is_active(self) -> bool:
        return self._player is not None