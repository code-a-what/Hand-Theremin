# Hand Theremin: A Gesture-Controlled Musical Instrument

Play music with your bare hands and a webcam. The program tracks your hands with MediaPipe, processes the video with OpenCV, and plays sound in real time. Small music notes float up on screen whenever a note is sounding, and their colour follows the pitch.

**Course:** Image and Video Processing, SVKM's NMIMS Indore (B.Tech Computer Engineering, Sem 5, Div A)
**Faculty:** Dr. Raj Gaurav Mishra
**Team:**
- Anaya Doneriya (F034)
- Khushi Patel (F027)
- Anuva Rathi (F014)

## Features
- Six instruments: piano, guitar, violin, flute, saxophone, drums
- Pitch snapped to a musical scale, so rough hand movement still sounds good
- Floating music-note visuals (colour follows pitch, warm burst on a drum hit)
- Runs fully on your own laptop. No internet or GPU needed after setup.

## What you need
- A Windows laptop or PC with a **webcam** and speakers or headphones
- **Python 3.10** (newer versions such as 3.14 do not work with MediaPipe yet)
- **Git** (https://git-scm.com)

## How to run it (step by step)

Open **PowerShell** and run these one at a time.

**1. Download the project**
```powershell
git clone https://github.com/code-a-what/Hand-Theremin.git
cd Hand-Theremin
```

**2. Create a virtual environment with Python 3.10**
```powershell
py -3.10 -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the activate script, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**3. Install the libraries**
```powershell
pip install -r requirements.txt
```

**4. Start the instrument**
```powershell
python main.py
```
A camera window opens. Press **q** or **Esc** in that window to quit.

## How to play
1. Stand about an arm's length from the camera in good light, with both hands visible.
2. **Pitch hand (your right hand, the one further right on the screen):** move it up and down to change the note. Open and close your thumb and index finger to change the volume (wide = loud).
3. **Control hand (your left hand):** keep it in view to hear sound. If the left hand is not visible, the instrument is muted, like lifting your hand away from a real theremin. Close it into a **fist** for a drum hit.
4. Floating music notes appear from your index fingertip while a note is sounding, and a warm-coloured burst appears on each drum hit.

| Key | Action |
|---|---|
| i | Switch instrument |
| s | Switch musical scale |
| q or Esc | Quit |

## Troubleshooting
- **Camera does not open:** close other apps that use the webcam (Zoom, Teams, browser tabs), then run again.
- **No sound:** check your volume and output device. Also make sure your left hand is visible, because the instrument is muted without it.
- **Crackling sound:** close heavy programs running in the background.
- **Install errors:** make sure you are on Python 3.10 by running `python --version` inside the activated venv.
- **Hands get mixed up:** keep your pitch hand further right on the screen than your control hand.

## Project files
| File | Purpose |
|---|---|
| main.py | Webcam loop, gesture features, on-screen display |
| hand_tracker.py | MediaPipe HandLandmarker wrapper |
| audio_engine.py | Real-time audio and pitch-shifted sample playback |
| instruments.py | Instrument settings |
| scales.py | Musical scales and note snapping |
| note_particles.py | Floating music-note effect |
| models/ | MediaPipe hand landmark model |
| samples/ | Audio recordings used for the instruments |

The full project report is in `Hand_Theremin_Project_Report.docx`.