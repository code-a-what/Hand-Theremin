"""
note_particles.py
A lightweight particle system that draws colourful musical notes (filled
notehead + stem, some with a flag) drifting upward and fading out from
wherever a note is currently being played. Pure OpenCV drawing — no extra
dependencies, and cheap enough to run every frame since particle counts
stay small (tens, not hundreds).

Design notes:
- Each particle fades via real alpha blending (not just colour fade), done
  per-particle over just its small bounding box so it stays fast even with
  many particles on screen (drawing onto a full-frame overlay and blending
  the whole frame every particle would be far more expensive).
- Colour is derived from the note's pitch (0-1, same value the pitch bar
  uses), cycled through hue space — so moving your hand up/down doesn't
  just change pitch, it visibly changes colour too, which is what makes
  this feel "alive" rather than just decorative.
"""

import cv2
import numpy as np
import random
import math


def hue_to_bgr(hue01: float, sat=0.85, val=1.0):
    """Map a 0-1 value to a vivid BGR colour by walking around the hue wheel."""
    hsv = np.uint8([[[int(hue01 * 179) % 180, int(sat * 255), int(val * 255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "age", "lifetime", "size", "color",
                 "angle", "spin", "has_flag", "wobble_phase")

    def __init__(self, x, y, color, size, lifetime, speed_scale=1.0):
        self.x = x
        self.y = y
        # Drift mostly upward, like a note floating off a staff, with a
        # little random sideways kick so a burst of particles fans out
        # instead of stacking in a single vertical line.
        self.vx = random.uniform(-18, 18) * speed_scale
        self.vy = random.uniform(-70, -40) * speed_scale
        self.age = 0.0
        self.lifetime = lifetime
        self.size = size
        self.color = color
        self.angle = random.uniform(-15, 15)
        self.spin = random.uniform(-25, 25)  # degrees/sec, gentle tumble
        self.has_flag = random.random() < 0.5
        self.wobble_phase = random.uniform(0, 2 * math.pi)

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt + math.sin(self.age * 3 + self.wobble_phase) * 10 * dt
        self.y += self.vy * dt
        self.vy *= (1 - 0.6 * dt)  # ease the rise, like gentle buoyancy rather than a constant launch
        self.angle += self.spin * dt

    @property
    def alpha(self):
        # Fade in quickly, hold, then fade out over the back half of life.
        t = self.age / self.lifetime
        if t < 0.15:
            return t / 0.15
        return max(0.0, 1.0 - (t - 0.15) / 0.85)

    @property
    def is_dead(self):
        return self.age >= self.lifetime


def _draw_note_shape(img, cx, cy, size, color, angle_deg, has_flag):
    """Draws one musical-note glyph (tilted filled notehead + stem, with an
    optional flag) centred at (cx, cy) in the given image, in place."""
    theta = math.radians(angle_deg)
    cos_a, sin_a = math.cos(theta), math.sin(theta)

    # Notehead: a small filled ellipse, tilted like real notation (~20deg).
    head_rx, head_ry = int(size * 0.62), int(size * 0.46)
    head_angle = angle_deg - 20
    cv2.ellipse(img, (int(cx), int(cy)), (head_rx, head_ry), head_angle,
                0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx), int(cy)), (head_rx, head_ry), head_angle,
                0, 360, (255, 255, 255), 1, cv2.LINE_AA)

    # Stem: a line from the right edge of the notehead going up.
    stem_len = size * 2.6
    stem_base = (cx + head_rx * cos_a * 0.9, cy - head_ry * 0.1)
    stem_top = (stem_base[0] + stem_len * sin_a * 0.3,
                stem_base[1] - stem_len * cos_a)
    cv2.line(img, (int(stem_base[0]), int(stem_base[1])),
              (int(stem_top[0]), int(stem_top[1])), color, max(2, int(size * 0.14)), cv2.LINE_AA)

    if has_flag:
        # A small curved flag near the top of the stem (eighth-note look),
        # approximated with a short ellipse arc rather than a true bezier.
        flag_center = (int(stem_top[0] + size * 0.5), int(stem_top[1] + size * 0.35))
        cv2.ellipse(img, flag_center, (int(size * 0.5), int(size * 0.7)),
                    angle_deg + 20, -40, 140, color, max(2, int(size * 0.14)), cv2.LINE_AA)


def _blend_region(frame, cx, cy, half_extent, draw_fn, alpha):
    """Draws onto a copy of just the particle's bounding box and alpha-blends
    that small region back into the frame, so fading looks like real
    transparency instead of a colour shift, without touching the whole frame."""
    h, w = frame.shape[:2]
    x0, y0 = int(cx - half_extent), int(cy - half_extent)
    x1, y1 = int(cx + half_extent), int(cy + half_extent)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    draw_fn(overlay, cx - x0, cy - y0)
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    frame[y0:y1, x0:x1] = blended


class NoteParticleSystem:
    """Owns and animates all currently-alive note particles."""

    def __init__(self, max_particles=120):
        self.particles = []
        self.max_particles = max_particles

    def spawn(self, x, y, pitch01, count=1, size_range=(14, 22), lifetime_range=(0.9, 1.5),
              speed_scale=1.0, color_override=None):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            color = color_override or hue_to_bgr(pitch01)
            size = random.uniform(*size_range)
            lifetime = random.uniform(*lifetime_range)
            jitter_x = x + random.uniform(-8, 8)
            jitter_y = y + random.uniform(-8, 8)
            self.particles.append(_Particle(jitter_x, jitter_y, color, size, lifetime, speed_scale))

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if not p.is_dead]

    def draw(self, frame):
        for p in self.particles:
            alpha = p.alpha
            if alpha <= 0.02:
                continue
            half_extent = p.size * 3.2  # covers stem + flag + a margin
            _blend_region(
                frame, p.x, p.y, half_extent,
                lambda img, lx, ly, p=p: _draw_note_shape(img, lx, ly, p.size, p.color, p.angle, p.has_flag),
                alpha,
            )

    def clear(self):
        self.particles.clear()
