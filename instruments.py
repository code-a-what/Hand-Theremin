"""
instruments.py
Metadata for the real recorded note samples used by each instrument voice.
Each entry points at a single-note audio file plus the exact pitch (in Hz)
that note was recorded at — the audio engine pitch-shifts this recording
in real time by resampling it at a rate proportional to
(target_frequency / base_freq), which is the same principle as speeding up
or slowing down a tape to change its pitch.

Source: University of Iowa Electronic Music Studios Musical Instrument
Samples (https://theremin.music.uiowa.edu/MIS.html) — freely usable for
any project without restriction.

"loop": True means the recording is treated as a sustained tone and the
engine loops its middle section for as long as the note is held (flute,
sax — real wind instruments sustain as long as you blow). "loop": False
means it's left to decay/ring out naturally like the real instrument
(piano, guitar — a held key/string still fades over time).
"""

SAMPLES = {
    "piano": {
        "file": "samples/piano.wav",
        "base_freq": 440.00,   # A4
        "loop": False,
    },
    "guitar": {
        "file": "samples/guitar.wav",
        "base_freq": 246.94,   # B3, open B string
        "loop": False,
    },
    "flute": {
        "file": "samples/flute.wav",
        "base_freq": 246.94,   # B3 (first note of the trimmed run)
        "loop": True,
    },
    "saxophone": {
        "file": "samples/saxophone.wav",
        "base_freq": 261.63,   # C4 (first note of the trimmed run)
        "loop": True,
    },
}

INSTRUMENT_NAMES = list(SAMPLES.keys()) + ["drum"]  # drum is procedural, not sample-based

# Synthetic fallback used automatically when a real sample file for an
# instrument isn't present yet (e.g. you haven't trimmed/added it). Built
# via additive synthesis: each instrument is a set of harmonic (overtone)
# amplitudes stacked on the fundamental, which is what actually gives a
# rough sense of timbre without needing a real recording.
# Synthetic fallback used automatically when a real sample file for an
# instrument isn't present yet (e.g. you haven't trimmed/added it). Built
# via additive synthesis: each instrument is a set of harmonic (overtone)
# amplitudes stacked on the fundamental, PLUS an ADSR envelope (attack/
# decay/sustain/release) and optional vibrato — the envelope shape is at
# least as important as the harmonic mix for making instruments sound
# distinct from each other (a pluck vs. a sustained breath tone), and pure
# steady vibrato-free tones read as robotic/synthetic, hence a touch of it
# on flute/sax.
#
# attack/decay/release are in seconds. sustain is the level (0-1) the tone
# settles to after decay, while the note is still held.
# vibrato_rate is in Hz (wobbles per second), vibrato_depth is in Hz
# (how far the pitch wobbles up/down); 0 means no vibrato.
FALLBACK_HARMONICS = {
    "piano": {
        "harmonics": [1.0, 0.55, 0.35, 0.20, 0.12, 0.08, 0.04, 0.02],
        "attack": 0.004,
        "decay": 0.6,
        "sustain": 0.15,
        "release": 0.3,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
    },
    "guitar": {
        "harmonics": [1.0, 0.7, 0.5, 0.35, 0.22, 0.14, 0.08, 0.05],
        "attack": 0.008,
        "decay": 0.8,
        "sustain": 0.1,
        "release": 0.35,
        "vibrato_rate": 0.0,
        "vibrato_depth": 0.0,
    },
    "flute": {
        "harmonics": [1.0, 0.12, 0.06, 0.03, 0.0, 0.0, 0.0, 0.0],
        "attack": 0.12,
        "decay": 0.05,
        "sustain": 0.9,
        "release": 0.15,
        "vibrato_rate": 5.0,
        "vibrato_depth": 3.0,
    },
    "saxophone": {
        "harmonics": [1.0, 0.6, 0.7, 0.45, 0.5, 0.3, 0.25, 0.15],
        "attack": 0.05,
        "decay": 0.15,
        "sustain": 0.8,
        "release": 0.2,
        "vibrato_rate": 5.5,
        "vibrato_depth": 4.0,
    },
}
