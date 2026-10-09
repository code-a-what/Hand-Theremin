"""
main.py
Hand-Controlled Theremin â€” Image/Video Processing course project.

Right hand height  -> pitch (quantized to a musical scale)
Right hand pinch    -> volume (thumb-index distance, wide = loud)
Left hand present   -> mute toggle (no left hand = silence, like lifting
                        your hand off a real theremin's volume antenna)
Left hand closes into a fist -> triggers a drum hit (percussive, one-shot,
                        works alongside whatever pitched instrument is active)

Colourful musical notes float up from your fingertip whenever a note is
actually sounding (coloured by pitch), and a warm-coloured burst pops from
your fist on every drum hit â€” purely visual flair via note_particles.py,
no effect on the audio.

Keys:
  i  - cycle instrument (piano / guitar / flute / saxophone / drum)
  s  - cycle musical scale
  q / ESC - quit
"""

import cv2
import numpy as np
import time

from hand_tracker import HandTracker, WRIST, INDEX_TIP
from audio_engine import AudioEngine
from scales import ScaleMapper
from note_particles import NoteParticleSystem, hue_to_bgr

WINDOW_NAME = "Hand Theremin"

# Minimum time between drum hits from the same hand, so holding a fist
# doesn't spam dozens of hits per second.
DRUM_RETRIGGER_COOLDOWN = 0.22

# How often (seconds) a new floating note spawns while a note is actively
# sounding. Matches audio_engine's own NOTE_ON_THRESHOLD for "is this loud
# enough to count as a played note" so notes only appear when you'd
# actually hear something, not during silence or the mute gap.
NOTE_ON_THRESHOLD = 0.03
NOTE_SPAWN_INTERVAL = 0.09


def draw_pitch_bar(frame, value01, freq_hz):
    h, w = frame.shape[:2]
    bar_x = w - 60
    bar_top, bar_bottom = 40, h - 40
    fill_y = int(bar_bottom - value01 * (bar_bottom - bar_top))

    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 30, bar_bottom), (80, 80, 80), 2)
    cv2.rectangle(frame, (bar_x, fill_y), (bar_x + 30, bar_bottom), (0, 220, 255), -1)
    cv2.putText(frame, f"{freq_hz:.0f} Hz", (bar_x - 40, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_hud(frame, instrument, is_fallback, scale_name, volume01, muted, drum_flash):
    instrument_label = f"{instrument} (synth)" if is_fallback else instrument
    lines = [
        f"Instrument: {instrument_label}   (i to cycle)",
        f"Scale: {scale_name}   (s to cycle)",
        f"Volume: {'MUTED' if muted else f'{volume01:.2f}'}",
        "Right hand height = pitch | pinch = volume",
        "Left hand shown = unmute | left fist = drum hit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if drum_flash:
        h, w = frame.shape[:2]
        cv2.putText(frame, "HIT!", (w // 2 - 60, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 100, 255), 4, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    tracker = HandTracker(num_hands=2)
    scale_mapper = ScaleMapper(root_note_hz=220.0, scale_name="minor_pentatonic", octaves=2)
    audio = AudioEngine(instrument="piano")
    audio.start()
    notes = NoteParticleSystem()

    was_fist = False
    last_drum_time = 0.0
    drum_flash_until = 0.0
    last_note_spawn = 0.0
    last_frame_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now_frame = time.time()
            dt = min(now_frame - last_frame_time, 0.1)  # clamp so a stutter/pause doesn't fling particles
            last_frame_time = now_frame
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands = tracker.process(frame_rgb)
            # MediaPipe's own Left/Right handedness label gets unreliable once
            # we've flipped the frame for a mirror view. Instead, identify
            # hands by their actual x-position on screen: whichever hand is
            # further right on screen is "right_hand", the other is "left_hand".
            right_hand = None
            left_hand = None
            if len(hands) == 1:
                right_hand = hands[0]
            elif len(hands) >= 2:
                def wrist_x(h):
                    x, _, _ = h["landmarks"][WRIST]
                    return x
                sorted_hands = sorted(hands, key=wrist_x)
                left_hand = sorted_hands[0]
                right_hand = sorted_hands[-1]

            muted = left_hand is None
            value01 = 0.0
            freq = scale_mapper.note_table[0]
            volume01 = 0.0

            index_x, index_y = None, None
            if right_hand is not None:
                wx, wy, _ = HandTracker.get_point(right_hand, WRIST, frame_w, frame_h)
                value01 = 1.0 - np.clip(wy / frame_h, 0.0, 1.0)
                freq = scale_mapper.map_value(value01)

                pinch = HandTracker.pinch_distance(right_hand)
                volume01 = np.clip((pinch - 0.02) / (0.25 - 0.02), 0.0, 1.0)

                cv2.circle(frame, (wx, wy), 10, (0, 220, 255), -1)

                index_x, index_y, _ = HandTracker.get_point(right_hand, INDEX_TIP, frame_w, frame_h)

            audio.set_target(freq, 0.0 if muted else volume01)

            # Spawn floating musical notes from the playing fingertip while a
            # note is actually audible (unmuted and loud enough to hear) â€”
            # tying this to the same threshold the audio engine uses to gate
            # a note "on" keeps the visual honest: notes only appear when
            # you'd actually hear something.
            now = time.time()
            note_sounding = (not muted) and (volume01 > NOTE_ON_THRESHOLD) and (index_x is not None)
            if note_sounding and (now - last_note_spawn) > NOTE_SPAWN_INTERVAL:
                notes.spawn(index_x, index_y, value01, count=1)
                last_note_spawn = now

            # Drum trigger: left hand closing into a fist. Edge-triggered
            # (fires once per close, not continuously) with a cooldown so
            # a held fist doesn't machine-gun hits.
            drum_flash = now < drum_flash_until
            if left_hand is not None:
                is_fist_now = HandTracker.is_fist(left_hand)
                if is_fist_now and not was_fist and (now - last_drum_time) > DRUM_RETRIGGER_COOLDOWN:
                    audio.trigger_drum(velocity=1.0)
                    last_drum_time = now
                    drum_flash_until = now + 0.25
                    drum_flash = True
                    # A bright warm-coloured burst at the fist, visually
                    # distinct from the pitch-coloured single notes, so a
                    # drum hit reads as a percussive "pop" rather than
                    # another floating note.
                    fx, fy, _ = HandTracker.get_point(left_hand, WRIST, frame_w, frame_h)
                    notes.spawn(fx, fy, 0.0, count=6, size_range=(16, 26),
                                lifetime_range=(0.5, 0.9), speed_scale=1.8,
                                color_override=hue_to_bgr(0.02, sat=0.9, val=1.0))
                was_fist = is_fist_now
            else:
                was_fist = False

            notes.update(dt)
            notes.draw(frame)

            draw_pitch_bar(frame, value01, freq)
            is_fallback = (audio.instrument == "drum" and not getattr(audio, "drum_is_real", lambda: True)()) \
                or (audio.instrument != "drum" and getattr(audio, "current_instrument_is_fallback", lambda: False)())
            draw_hud(frame, audio.instrument, is_fallback,
                     scale_mapper.scale_name, volume01, muted, drum_flash)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("i"):
                audio.cycle_instrument()
            elif key == ord("s"):
                scale_mapper.cycle_scale()

    finally:
        audio.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
