"""
scales.py
Maps a continuous 0-1 value (e.g. normalized hand height) to a musical
frequency, quantized onto a scale so the theremin sounds musical instead
of a raw sweeping siren.
"""

import numpy as np

# Semitone offsets from the root, within one octave, for common scales.
SCALES = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
}


class ScaleMapper:
    def __init__(self, root_note_hz=220.0, scale_name="minor_pentatonic", octaves=2):
        """
        root_note_hz: frequency of the lowest note in the range (A3 = 220Hz by default)
        scale_name: key into SCALES
        octaves: how many octaves the mapped range should span
        """
        self.root = root_note_hz
        self.set_scale(scale_name)
        self.octaves = octaves
        self._build_note_table()

    def set_scale(self, scale_name):
        if scale_name not in SCALES:
            raise ValueError(f"Unknown scale '{scale_name}'. Choose from {list(SCALES)}")
        self.scale_name = scale_name
        self.intervals = SCALES[scale_name]

    def _build_note_table(self):
        """Precompute every note frequency across the configured octave range."""
        notes = []
        for octave in range(self.octaves):
            for semitone in self.intervals:
                freq = self.root * (2 ** ((octave * 12 + semitone) / 12))
                notes.append(freq)
        self.note_table = np.array(sorted(notes))

    def map_value(self, value01: float) -> float:
        """
        value01: float in [0, 1], e.g. 1 - (y / frame_height) so higher hand = higher pitch.
        Returns the nearest scale-quantized frequency in Hz.
        """
        value01 = float(np.clip(value01, 0.0, 1.0))
        idx = int(round(value01 * (len(self.note_table) - 1)))
        return float(self.note_table[idx])

    def cycle_scale(self):
        names = list(SCALES.keys())
        next_idx = (names.index(self.scale_name) + 1) % len(names)
        self.set_scale(names[next_idx])
        self._build_note_table()
        return self.scale_name
