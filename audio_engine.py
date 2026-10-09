"""
audio_engine.py
Low-latency real-time audio synthesizer driven by hand position.

Pitched instruments (piano/guitar/flute/saxophone) are rendered by loading
one real recorded note per instrument and pitch-shifting it live via
varispeed resampling — reading through the sample buffer faster or slower
depending on how far the target pitch is from the note it was recorded at.
This is the same principle as changing pitch by changing tape/turntable
speed, and is what makes these sound like the real instrument rather than
a synthesized approximation.

Drums remain procedural one-shot hits (noise burst + thump), since
percussion doesn't have a "pitch" to varispeed against.
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading

from instruments import SAMPLES, INSTRUMENT_NAMES, FALLBACK_HARMONICS

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024  # larger block = fewer callbacks = far less risk of underrun/crackle

NOTE_ON_THRESHOLD = 0.03  # volume level above which a note is considered "sounding"


def _load_and_resample(filepath, target_sr):
    """Load an audio file as mono float32, resampled to target_sr if needed."""
    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix to mono if stereo
    if sr != target_sr:
        # Simple linear-interpolation resample. Good enough for one-time
        # sample loading (not real-time critical) and needs no extra deps.
        duration = len(mono) / sr
        n_target = int(round(duration * target_sr))
        src_idx = np.linspace(0, len(mono) - 1, num=n_target)
        mono = np.interp(src_idx, np.arange(len(mono)), mono)
    # Normalize so the loudest sample is close to full scale, so quiet vs.
    # loud source recordings don't end up wildly different volumes.
    peak = np.max(np.abs(mono)) or 1.0
    mono = (mono / peak) * 0.9
    return mono.astype(np.float32)


class SampleVoice:
    """
    One pitched instrument voice: holds a loaded note sample and renders
    it pitch-shifted to a (possibly gliding) target frequency array.
    """

    def __init__(self, filepath, base_freq, sample_rate, loop=False):
        self.sample_rate = sample_rate
        self.base_freq = base_freq
        self.loop = loop
        self.buffer = _load_and_resample(filepath, sample_rate)
        self.n = len(self.buffer)
        if loop:
            # Loop the middle section, skipping the sharp attack transient
            # and any trailing silence, so looping doesn't repeat a click
            # or a pluck/breath-attack over and over.
            self.loop_start = int(self.n * 0.15)
            self.loop_end = int(self.n * 0.85)
            self.loop_len = max(1, self.loop_end - self.loop_start)
        self.read_pos = 0.0

    def render(self, freqs: np.ndarray, vols: np.ndarray, note_on: bool) -> np.ndarray:
        if note_on:
            self.read_pos = 0.0  # fresh attack — restart from the beginning of the sample

        rate = freqs / self.base_freq
        positions = self.read_pos + np.cumsum(rate)

        if self.loop:
            past_loop = positions >= self.loop_end
            positions = np.where(
                past_loop,
                self.loop_start + np.mod(positions - self.loop_end, self.loop_len),
                positions,
            )
            valid = np.ones_like(positions, dtype=bool)
        else:
            valid = positions < (self.n - 1)

        idx0 = np.clip(np.floor(positions).astype(np.int64), 0, self.n - 2)
        idx1 = idx0 + 1
        frac = positions - idx0
        samples = self.buffer[idx0] * (1 - frac) + self.buffer[idx1] * frac
        samples = np.where(valid, samples, 0.0) * vols

        self.read_pos = positions[-1]
        return samples


class SynthVoice:
    """
    Fallback pitched voice used when a real sample file isn't available.
    Additive synthesis: sums several harmonics (overtones) on top of the
    fundamental frequency, shaped by an ADSR amplitude envelope (attack/
    decay/sustain/release) and optional vibrato (a slow pitch wobble).
    The envelope is what actually gives instruments a distinct character
    beyond their harmonic mix — e.g. a fast attack + fast decay reads as a
    "pluck" (piano/guitar), a slow attack + high sustain reads as a
    "breath tone" (flute/sax). Same render() shape as SampleVoice so
    AudioEngine can use either interchangeably.
    """

    # Envelope stages
    IDLE, ATTACK, DECAY, SUSTAIN, RELEASE = range(5)

    def __init__(self, profile, sample_rate):
        self.harmonics = profile["harmonics"]
        self.harmonic_sum = sum(self.harmonics) or 1.0
        self.attack_t = max(profile["attack"], 1e-4)
        self.decay_t = max(profile["decay"], 1e-4)
        self.sustain_level = profile["sustain"]
        self.release_t = max(profile["release"], 1e-4)
        self.vibrato_rate = profile.get("vibrato_rate", 0.0)
        self.vibrato_depth = profile.get("vibrato_depth", 0.0)

        self.sample_rate = sample_rate
        self.phase = 0.0
        self.vibrato_phase = 0.0

        self.env_stage = self.IDLE
        self.env_level = 0.0

    def _step_envelope(self, n_frames, gate_on: bool) -> np.ndarray:
        """
        Generate n_frames of the ADSR envelope, one sample at a time. Small
        per-sample loop (same technique already used for the drum's noise
        filter) — cheap at audio block sizes and needed because the
        envelope's behaviour depends on its own running state.
        """
        out = np.empty(n_frames)
        attack_step = 1.0 / (self.attack_t * self.sample_rate)
        decay_step = 1.0 / (self.decay_t * self.sample_rate)
        release_step = 1.0 / (self.release_t * self.sample_rate)

        for i in range(n_frames):
            if gate_on and self.env_stage == self.IDLE:
                self.env_stage = self.ATTACK

            if self.env_stage == self.ATTACK:
                self.env_level += attack_step
                if self.env_level >= 1.0:
                    self.env_level = 1.0
                    self.env_stage = self.DECAY
            elif self.env_stage == self.DECAY:
                self.env_level -= decay_step
                if self.env_level <= self.sustain_level:
                    self.env_level = self.sustain_level
                    self.env_stage = self.SUSTAIN
            elif self.env_stage == self.SUSTAIN:
                self.env_level = self.sustain_level
            elif self.env_stage == self.RELEASE:
                self.env_level -= release_step
                if self.env_level <= 0.0:
                    self.env_level = 0.0
                    self.env_stage = self.IDLE

            if not gate_on and self.env_stage not in (self.IDLE, self.RELEASE):
                self.env_stage = self.RELEASE

            out[i] = self.env_level
        return out

    def render(self, freqs: np.ndarray, vols: np.ndarray, note_on: bool) -> np.ndarray:
        n = len(freqs)
        gate_on = bool(vols[-1] > NOTE_ON_THRESHOLD) if n else False
        if note_on:
            self.phase = 0.0  # fresh attack starts the waveform cleanly

        if self.vibrato_depth > 0:
            t = np.arange(n)
            vib_inc = 2 * np.pi * self.vibrato_rate / self.sample_rate
            vib_phases = self.vibrato_phase + np.cumsum(np.full(n, vib_inc))
            freqs = freqs + self.vibrato_depth * np.sin(vib_phases)
            self.vibrato_phase = vib_phases[-1] % (2 * np.pi) if n else self.vibrato_phase

        phase_inc = 2 * np.pi * freqs / self.sample_rate
        phases = self.phase + np.cumsum(phase_inc)
        wave = np.zeros(n)
        for h, amp in enumerate(self.harmonics):
            if amp <= 0:
                continue
            wave += amp * np.sin((h + 1) * phases)
        wave = wave / self.harmonic_sum

        envelope = self._step_envelope(n, gate_on)
        wave = wave * envelope * vols

        self.phase = phases[-1] % (2 * np.pi) if n else self.phase
        return wave


class DrumHit:
    """A single percussive hit: filtered noise burst + a low sine 'thump',
    both with a fast exponential decay."""

    def __init__(self, sample_rate, velocity=1.0):
        self.sample_rate = sample_rate
        self.age = 0
        self.velocity = velocity
        self.thump_freq = 90.0
        self.noise_decay = 18.0
        self.thump_decay = 9.0
        self._lp_state = 0.0

    def render(self, n_frames):
        t = (self.age + np.arange(n_frames)) / self.sample_rate
        noise = np.random.uniform(-1, 1, n_frames) * np.exp(-self.noise_decay * t)
        filtered = np.empty(n_frames)
        state = self._lp_state
        for i in range(n_frames):
            state = state + 0.5 * (noise[i] - state)
            filtered[i] = state
        self._lp_state = state

        thump = np.sin(2 * np.pi * self.thump_freq * t) * np.exp(-self.thump_decay * t)
        out = (filtered * 0.6 + thump * 0.8) * self.velocity
        self.age += n_frames
        return out

    def is_done(self):
        return (self.age / self.sample_rate) > 0.6


class AudioEngine:
    """
    Thread-safe theremin voice. Call set_target(freq, volume) every video
    frame from the main loop; the audio callback smoothly glides toward
    those targets and renders the current instrument's sample, pitch-shifted
    live. Call trigger_drum() for one-shot percussion hits.
    """

    def __init__(self, instrument="piano", glide=0.15):
        self.sample_rate = SAMPLE_RATE
        self.instrument_idx = INSTRUMENT_NAMES.index(instrument)
        self.glide = glide

        self._voices = {}
        self._using_fallback = {}
        for name, meta in SAMPLES.items():
            try:
                self._voices[name] = SampleVoice(
                    meta["file"], meta["base_freq"], self.sample_rate, loop=meta["loop"]
                )
                self._using_fallback[name] = False
            except Exception as e:
                print(f"[AudioEngine] '{name}' sample not found ({meta['file']}) — "
                      f"using synthetic fallback until a real sample is added. ({e})")
                self._voices[name] = SynthVoice(FALLBACK_HARMONICS[name], self.sample_rate)
                self._using_fallback[name] = True

        self._current_freq = 220.0
        self._current_vol = 0.0
        self._target_freq = 220.0
        self._target_vol = 0.0
        self._lock = threading.Lock()

        self._note_active_prev = False
        self._active_instrument_prev = None

        self._drum_hits = []

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=BLOCK_SIZE,
            channels=1,
            latency="high",
            callback=self._callback,
        )

    def set_target(self, freq_hz: float, volume01: float):
        with self._lock:
            self._target_freq = max(20.0, float(freq_hz))
            self._target_vol = float(np.clip(volume01, 0.0, 1.0))

    def set_instrument(self, name: str):
        with self._lock:
            self.instrument_idx = INSTRUMENT_NAMES.index(name)

    def cycle_instrument(self):
        with self._lock:
            self.instrument_idx = (self.instrument_idx + 1) % len(INSTRUMENT_NAMES)
            return INSTRUMENT_NAMES[self.instrument_idx]

    @property
    def instrument(self):
        return INSTRUMENT_NAMES[self.instrument_idx]

    def current_instrument_is_fallback(self):
        return self._using_fallback.get(self.instrument, False)

    def trigger_drum(self, velocity=1.0):
        with self._lock:
            if len(self._drum_hits) < 8:
                self._drum_hits.append(DrumHit(self.sample_rate, velocity))

    def _render_instrument(self, frames):
        with self._lock:
            target_freq = self._target_freq
            target_vol = self._target_vol
            name = INSTRUMENT_NAMES[self.instrument_idx]

        if name == "drum" or name not in self._voices:
            self._current_freq = target_freq
            self._current_vol = 0.0
            self._note_active_prev = False
            self._active_instrument_prev = name
            return np.zeros(frames)

        t = np.arange(frames)
        freqs = self._current_freq + (target_freq - self._current_freq) * (
            1 - np.exp(-self.glide * (t + 1))
        )
        vols = self._current_vol + (target_vol - self._current_vol) * (
            1 - np.exp(-self.glide * (t + 1))
        )

        is_active_now = target_vol > NOTE_ON_THRESHOLD
        note_on = is_active_now and (
            not self._note_active_prev or name != self._active_instrument_prev
        )

        voice = self._voices[name]
        wave = voice.render(freqs, vols, note_on)

        self._current_freq = freqs[-1]
        self._current_vol = vols[-1]
        self._note_active_prev = is_active_now
        self._active_instrument_prev = name
        return wave

    def _render_drums(self, frames):
        with self._lock:
            hits = self._drum_hits
        if not hits:
            return np.zeros(frames)
        mix = np.zeros(frames)
        still_active = []
        for hit in hits:
            mix += hit.render(frames)
            if not hit.is_done():
                still_active.append(hit)
        with self._lock:
            self._drum_hits = still_active
        return mix

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print("Audio status:", status)

        instrument_wave = self._render_instrument(frames)
        drum_wave = self._render_drums(frames)

        mixed = instrument_wave + drum_wave
        mixed = np.tanh(mixed * 1.2)  # soft-clip against combined peaks
        outdata[:, 0] = mixed.astype(np.float32)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
