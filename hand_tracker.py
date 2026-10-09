"""
hand_tracker.py
Thin wrapper around MediaPipe's HandLandmarker (Tasks API) — same API family
used in the invisibility-cloak reference project. Tracks up to 2 hands and
exposes the landmarks needed for the theremin: wrist/index-tip for position,
thumb-tip/index-tip distance for pinch (volume) control.
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

MODEL_PATH = "models/hand_landmarker.task"

# Landmark indices we care about (MediaPipe hand landmark spec)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
MIDDLE_MCP = 9  # base knuckle of middle finger, used as a hand-size reference


class HandTracker:
    def __init__(self, model_path=MODEL_PATH, num_hands=2,
                 min_detection_confidence=0.5, min_presence_confidence=0.5,
                 min_tracking_confidence=0.4):
        # Lowered from an earlier 0.6-for-everything default. 0.6 was
        # rejecting too many frames under ordinary indoor lighting, which
        # reads as hands "not being detected" or flickering in and out.
        # Tracking confidence in particular can sit lower than detection
        # confidence (0.4 here) since it only needs to confirm a hand
        # already found, not find one from scratch frame-to-frame.
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_rgb: np.ndarray):
        """
        frame_rgb: an RGB numpy frame (convert from BGR before calling).
        Returns a list of dicts, one per detected hand:
            {"handedness": "Left"/"Right", "landmarks": [(x, y, z), ...] normalized 0-1}
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 1
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands = []
        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                label = "Unknown"
                if result.handedness and len(result.handedness) > i:
                    label = result.handedness[i][0].category_name
                points = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                hands.append({"handedness": label, "landmarks": points})
        return hands

    @staticmethod
    def get_point(hand, idx, frame_w, frame_h):
        """Convert a normalized landmark to pixel coordinates."""
        x, y, z = hand["landmarks"][idx]
        return int(x * frame_w), int(y * frame_h), z

    @staticmethod
    def pinch_distance(hand):
        """Normalized (0-1 ish) distance between thumb tip and index tip."""
        tx, ty, _ = hand["landmarks"][THUMB_TIP]
        ix, iy, _ = hand["landmarks"][INDEX_TIP]
        return float(np.hypot(tx - ix, ty - iy))

    @staticmethod
    def is_fist(hand):
        """
        Rough fist/closed-hand detector: average distance from the four
        fingertips (index/middle/ring/pinky) to the wrist, normalized by
        the wrist-to-middle-knuckle distance (a stand-in for hand size so
        this works regardless of how close the hand is to the camera).
        A closed fist pulls all fingertips back near the wrist, so this
        ratio drops sharply compared to an open hand.
        """
        wx, wy, _ = hand["landmarks"][WRIST]
        mx, my, _ = hand["landmarks"][MIDDLE_MCP]
        hand_size = float(np.hypot(mx - wx, my - wy)) or 1e-6

        tip_ids = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        dists = []
        for tid in tip_ids:
            fx, fy, _ = hand["landmarks"][tid]
            dists.append(np.hypot(fx - wx, fy - wy))
        avg_ratio = (sum(dists) / len(dists)) / hand_size
        return avg_ratio < 1.6  # open hand is typically well above 2.0

    def close(self):
        self.landmarker.close()
