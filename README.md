# Hand Theremin: A Gesture-Controlled Musical Instrument

**Course:** Image and Video Processing, SVKM's NMIMS Indore (B.Tech Computer Engineering, Sem 5, Div A)
**Faculty:** Dr. Raj Gaurav Mishra
**Team:** Anaya Doneriya (F034), Khushi Patel (F027), Anuva Rathi (F014)

Play music with your bare hands and a webcam. MediaPipe tracks your hand, OpenCV handles the video, and a real-time audio engine plays six instruments (piano, guitar, violin, flute, saxophone, drums). Floating music notes appear on screen whenever a note sounds, and their colour follows the pitch.

## How to play
- **Right-side hand (pitch hand):** move it up and down for pitch, open and close thumb and index finger for volume.
- **Other hand (control hand):** show it to mute/unmute; close it into a fist for a drum hit.
- Keyboard keys switch instrument and scale (see `main.py`).

## Setup (Windows, Python 3.10)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Project files
| File | Purpose |
|---|---|
| main.py | Webcam loop, gesture features, on-screen display |
| hand_tracker.py | MediaPipe HandLandmarker wrapper |
| audio_engine.py | Real-time audio and pitch-shifted sample playback |
| instruments.py | Instrument settings |
| scales.py | Musical scales and note snapping |
| note_particles.py | Floating music-note effect |

The full report is in `Hand_Theremin_Project_Report.docx`.
