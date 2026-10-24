# Gaze Communication Project: Interface Contract

**Batch 1: Eye-Gaze Interface + Gaze Data Collection**
**Module owner of this document:** Team 4, Dwell Detection & Button Selection: Tanmay Gupta (26BIT0206)

Please read this once and tell me (or raise a pull request) if anything needs to change.
Everyone codes against these formats so the pieces plug together.

## 1. Pipeline and owners

| Team | Student | Module | Hands over to the next module |
|---|---|---|---|
| 1 | 26BCB0165 | Camera & Tobii Input | raw gaze X, Y, timestamp, blink/valid flag |
| 2 | 26BCE0341 | Face & Eye Detection | eye/pupil position, blink indication |
| 3 | 26BME0222 | Calibration & Gaze Mapping | **screen-coordinate gaze samples** |
| 4 | 26BIT0206 | Dwell & Button Selection (this repo folder `dwell/`) | **BUTTON_SELECTED events + CSV log** |
| 5 | 26BIT0140 | Dataset & Evaluation | final dataset, accuracy report |

Batch 2 connects at the end: `BUTTON_SELECTED` -> Message mapping + TTS (26BEC0335).

## 2. Input to the dwell module (from Team 3, 26BME0222)

One sample per call, in **screen/canvas pixels** (origin top-left, x to the right, y downward):

```python
gaze_x: float | None
gaze_y: float | None
t:      float        # seconds, monotonic clock (time.monotonic()) - optional, defaults to now
```

- Send `None, None` for blinks or lost tracking. Do **not** send `0, 0`, which would be read as a real gaze position.
- Sample rate: ideally 30 Hz or more. Tobii (about 33-90 Hz) and webcam both work.
- Please apply light smoothing (e.g. moving average of 3-5 samples) before sending.

```python
from dwell_selection import DwellDetector
events = detector.update(gaze_x, gaze_y, t)   # returns a list of DwellEvent
# or, with the GUI:  app.feed_gaze(gaze_x, gaze_y, t)
```

### 2.1 Gaze Source Adapters (`GazeSource`)

To cleanly decouple hardware and gaze estimation from the interface, `dwell_selection.py` provides the `GazeSource` adapter interface:

```python
from dwell_selection import GazeSource, MouseGazeSource, WebcamGazeSource, TobiiGazeSource

# 1. Mouse Simulation (Default)
source = MouseGazeSource()

# 2. Webcam Stream (Team 1 + Team 3)
# Implement get_sample() to return calibrated screen (x, y, timestamp)
source = WebcamGazeSource(camera_id=0)

# 3. Tobii Eye Tracker (Team 1)
# Connects to Tobii Stream Engine / Research SDK
source = TobiiGazeSource()

# Pass directly to GazeApp:
app = GazeApp(gaze_source=source)
```

## 3. Output events (from Team 4)

| Event | Meaning |
|---|---|
| `GAZE_ENTER` | gaze entered a button, timer started |
| `DWELL_PROGRESS` | gaze still on the button; `progress` is 0.0 to 1.0 |
| `GAZE_LEAVE` | gaze left the button, timer reset |
| `BUTTON_SELECTED` | dwell completed, button is selected |

Each event carries: `type`, `button`, `fixation_time` (s), `timestamp`, `progress`.

**Button names are exactly these 9 strings** (same as the Batch 2 GUI):
`WATER, FOOD, HELP, CALL, EMERGENCY, MESSAGE, ENTERTAIN, SPEAK, EXIT`

### 3.1 Hook for Batch 2 (Message Mapping & Text-To-Speech)
Teammate 26BEC0335 can plug in their TTS or phrase generation function with a single line:

```python
from dwell_gui import GazeApp

def speak_button(button_name: str):
    print(f"[TTS Module] Speaking phrase for: {button_name}")

# Option A: In constructor
app = GazeApp(on_selected=speak_button)

# Option B: Plug in dynamically
app = GazeApp()
app.set_on_selected(speak_button)
```

## 4. Dwell settings

- Dwell threshold: **3 s, 4 s or 5 s** (selectable in the GUI, default 3 s).
- Timer resets when gaze leaves the button, but a short break of up to 0.15 s (blink or noise) is tolerated.
- After a selection, that button is locked for 1 s cooldown to avoid accidental repeat triggers.

## 5. CSV log for Team 5 (`dwell_log.csv`)

| Column | Type | Description |
|---|---|---|
| `user_id` | string | e.g. U001 |
| `timestamp` | float | Unix time in seconds |
| `gaze_x`, `gaze_y` | float | screen pixels (average over attempt or sample) |
| `target_button` | string | button under gaze, empty if none |
| `fixation_time` | float | seconds on the current button |
| `blink` | 0/1 | 1 if blink occurred during fixation, else 0 |
| `successful_selection` | 0/1 | 1 on row where selection completed, 0 if abandoned |
| `dwell_setting` | float | 3.0, 4.0 or 5.0 |

### 5.1 Logging Modes (Reduced File Size vs Raw Samples)
- **Attempt Mode (`mode='attempt'`, Default)**: Logs **one row per selection attempt** (`enter -> leave/select`). This keeps the dataset concise and readable, eliminating redundant 30 ms samples while retaining all required metrics for Team 5.
- **Raw Mode (`mode='raw'`, Optional)**: Logs every single 30 ms sample for detailed gaze-path trajectory analysis.

```python
logger = DwellLogger("dwell_log.csv", user_id="U001", mode="attempt")
```

## 6. Button layout (shared with the Batch 2 GUI)

```
| WATER     | FOOD      | HELP   |
| CALL      | EMERGENCY | MESSAGE|
| ENTERTAIN | SPEAK     | EXIT   |
```

If the final GUI changes size or position, send the new button rectangles as
`{name: (x, y, width, height)}`, and the detector will use them.

## 7. Experiment Mode (Evaluation Protocol)

The GUI includes a built-in automated Experiment Mode:
1. Prompts for participant / user ID (e.g. `U001`).
2. Presents randomized targets across dwell thresholds (**3 s, 4 s, and 5 s**).
3. Highlights the target button with a high-contrast amber border.
4. Automatically records target accuracy (`correct`), completion duration, and writes attempt rows to the CSV log.
5. Displays a session performance summary upon completion.

Run directly via GUI button **[Experiment Mode]** or CLI:
```bash
python dwell_gui.py --experiment --user U001
```

## 8. Results Analysis (`analyze_results.py`)

Team 5 and project evaluators can run `analyze_results.py` (built with pandas + matplotlib) to compute:
- Selection success rate (%) per dwell setting (3 s, 4 s, 5 s)
- Average selection time (s) per dwell setting
- Aggregated CSV export: `analysis_summary.csv`
- Publication-quality bar charts: `analysis_chart.png`

```bash
python analyze_results.py
```

## 9. How to run

### Step 0: Setup Environment & Dependencies
Create a virtual environment and install dependencies:

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

### Option A: One-Click GUI Launcher
You can launch everything from an interactive control window:
```bash
python launcher.py
```
*(On Mac/Linux: `.venv/bin/python launcher.py` | On Windows: `.venv\Scripts\python.exe launcher.py`)*

---

### Option B: Direct Terminal Commands

#### On Mac / Linux:
1. **Run Unit Tests (all 13 edge-case test suites):**
   ```bash
   .venv/bin/python test_dwell.py
   ```
2. **Launch Interactive GUI (mouse simulates gaze):**
   ```bash
   .venv/bin/python dwell_gui.py
   ```
3. **Run Automated Experiment Mode (dwell = 3s, 4s, 5s):**
   ```bash
   .venv/bin/python dwell_gui.py --experiment --user U001
   ```
   *(For a quick automated dry-run test without display interaction: `.venv/bin/python dwell_gui.py --experiment --user TEST --dry-run`)*
4. **Generate Evaluation Statistics & Bar Charts:**
   ```bash
   .venv/bin/python analyze_results.py
   ```

#### On Windows:
1. **Run Unit Tests:**
   ```powershell
   .venv\Scripts\python.exe test_dwell.py
   ```
2. **Launch Interactive GUI:**
   ```powershell
   .venv\Scripts\python.exe dwell_gui.py
   ```
3. **Run Automated Experiment Mode:**
   ```powershell
   .venv\Scripts\python.exe dwell_gui.py --experiment --user U001
   ```
4. **Generate Evaluation Statistics & Bar Charts:**
   ```powershell
   .venv\Scripts\python.exe analyze_results.py
   ```
