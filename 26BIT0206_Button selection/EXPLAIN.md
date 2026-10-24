# Dwell-Based Gaze Interaction — Codebase & Viva Explanation Guide

**Student:** Tanmay Gupta (26BIT0206)  
**Batch & Team:** Batch 1, Team 4  
**Project Title:** Multidisciplinary Eye-Gaze Communication System  
**Module:** Dwell Detection & Button Selection (`/dwell`)

---

## 1. Project Context & Pipeline Position

This project is an assistive eye-gaze communication interface designed for individuals with severe speech and motor disabilities (such as ALS, locked-in syndrome, or cerebral palsy). Users cannot move a physical mouse or speak; they communicate entirely by directing their eye gaze at on-screen targets.

The full multidisciplinary pipeline connects across student teams:

```
[Team 1: Camera / Tobii Hardware Input] (26BCB0165)
                  │  (raw eye image / eye coordinates + timestamps)
                  ▼
[Team 2: Face & Eye Landmark Detection] (26BCE0341)
                  │  (pupil position, blink flags)
                  ▼
[Team 3: Calibration & Gaze Screen Mapping] (26BME0222)
                  │  (screen-coordinate gaze samples: x, y, t)
                  ▼
[TEAM 4 (MY MODULE): Dwell Detection & Button Selection] (26BIT0206)
                  │
       ┌──────────┴─────────────────────────┐
       ▼                                    ▼
[BUTTON_SELECTED Event]              [dwell_log.csv]
       │                                    │
       ▼                                    ▼
[Batch 2: TTS & Phrase Output]       [Team 5: Dataset & Evaluation]
(26BEC0335)                          (26BIT0140 / analyze_results.py)
```

---

## 2. File-by-File Summary & Key Components

### 2.1 `dwell_selection.py` (Core Logic & Data Handling)
- **Plain-English Summary:** Contains the pure mathematical and algorithmic detection logic, CSV logger, and hardware input adapters. It has **no GUI or OS dependencies**, meaning it can run headless on servers, microcontrollers, or in unit test scripts.
- **Key Classes & Functions:**
  - `DwellEvent`: A structured dataclass carrying event metadata: `type` (`GAZE_ENTER`, `DWELL_PROGRESS`, `GAZE_LEAVE`, `BUTTON_SELECTED`), `button`, `fixation_time`, `timestamp`, and normalized `progress` ($0.0 \to 1.0$).
  - `DwellDetector`: The finite-state engine tracking continuous gaze. Manages the dwell accumulation timer, grace period tolerance, boundary detection, and selection cooldown.
    - `update(gaze_x, gaze_y, t=None) -> List[DwellEvent]`: The central entry point called for each gaze sample.
    - `button_at(x, y)`: Checks whether coordinate `(x, y)` falls inside any button bounding box.
    - `set_dwell_time(seconds)`: Reconfigures the threshold ($3.0\text{ s}, 4.0\text{ s}, 5.0\text{ s}$) with positive-value validation.
  - `GazeSource`: Abstract adapter base class enforcing `get_sample() -> (x, y, t)`.
    - `MouseGazeSource`: Default adapter simulating eye gaze with mouse coordinates.
    - `WebcamGazeSource`: Placeholder with TODOs for Team 1 & Team 3 webcam/OpenCV input.
    - `TobiiGazeSource`: Placeholder with TODOs for Team 1 official Tobii SDK integration.
  - `DwellLogger`: Manages structured data logging matching Team 5's schema (`user_id, timestamp, gaze_x, gaze_y, target_button, fixation_time, blink, successful_selection, dwell_setting`). Supports:
    - **`mode='attempt'` (Default):** Logs exactly one row per selection attempt (`enter -> leave` or `enter -> select`), keeping files compact.
    - **`mode='raw'`:** Logs every 30 ms sample for full gaze path trajectory tracking.

---

### 2.2 `dwell_gui.py` (High-Contrast Assistive User Interface)
- **Plain-English Summary:** A Tkinter graphical user interface providing a $3 \times 3$ communication grid (`WATER, FOOD, HELP, CALL, EMERGENCY, MESSAGE, ENTERTAIN, SPEAK, EXIT`). It dynamically renders circular progress rings, provides audiovisual feedback, connects to the `GazeSource` adapter, and runs automated experiments.
- **Key Classes & Functions:**
  - `GazeApp`: The primary GUI controller.
    - `build()`: Calculates responsive grid dimensions based on window size and renders button cards, rings, and text labels.
    - `feed_gaze(x, y, t)`: Feeds coordinates into `DwellDetector` and updates card colors, cursor, and progress rings.
    - `tick()`: Polling loop running every $30\text{ ms}$ ($\approx 33\text{ Hz}$) that pulls samples from `self.gaze_source`.
    - `set_on_selected(callback)`: Allows Batch 2 teammates to register a speech/phrase callback with a single line.
    - `start_experiment()` / `handle_experiment_selection()`: Manages automated testing trials across 3 s, 4 s, and 5 s dwell times with randomized target prompts and duration measurement.
    - `beep()`: Cross-platform sound confirmation upon button selection.

---

### 2.3 `launcher.py` (One-Click Control Window)
- **Plain-English Summary:** A control panel with dedicated buttons to launch unit tests, open the interactive GUI, run the experiment protocol with a custom User ID, and run the analysis script.
- **Key Functions:**
  - `find_python()`: Automatically detects and prioritizes `.venv/bin/python` (Mac/Linux) or `.venv/Scripts/python.exe` (Windows) over the system Python.
  - `run(args, wait)`: Spawns subprocesses either synchronously (displaying terminal output inside a GUI window) or asynchronously (launching the full GUI).

---

### 2.4 `test_dwell.py` (Unit Test Suite)
- **Plain-English Summary:** Standalone verification script testing the core logic across 13 test categories without launching Tkinter windows.
- **Test Coverage:**
  1. Dwell completion across 3 s, 4 s, and 5 s.
  2. Gating: no selection occurs before dwell threshold.
  3. Timer reset when gaze leaves before threshold.
  4. Short blinks ($< 0.15\text{ s}$) tolerated without resetting.
  5. Long blinks ($> 0.15\text{ s}$) reset timer.
  6. Button switching selects target button and cancels origin button.
  7. Cooldown period prevents duplicate triggers.
  8. Boundary edge cases (gaze exactly on boundaries, corners, and shared edges).
  9. Non-monotonic / duplicate timestamps handled gracefully.
  10. Rapid target switching between adjacent buttons.
  11. Invalid input coordinates (`NaN`, `inf`, non-numeric types, `None`).
  12. Validation: negative or zero dwell times raise `ValueError`.
  13. `GazeSource` adapters and `DwellLogger` attempt vs raw mode row validation.

---

### 2.5 `analyze_results.py` (Statistical Evaluation & Visualization)
- **Plain-English Summary:** Evaluates `dwell_log.csv` using `pandas` and `matplotlib` (using the headless `Agg` backend).
- **Outputs:**
  - `analysis_summary.csv`: Aggregated table containing total attempts, successful selections, abandoned attempts, success rate (%), average selection time (s), standard deviation, and min/max times grouped by dwell threshold (3 s, 4 s, 5 s).
  - `analysis_chart.png`: High-resolution dual bar chart visualizing success rates and selection times with error bars.
  - `--demo` flag: Generates synthetic trial data for instant demonstrations when no live data has been collected yet.

---

## 3. Step-by-Step Execution Walkthrough

Here is exactly what happens in the code from the moment a user looks at a button until selection confirmation is logged:

```
[User directs eye gaze onto on-screen button (e.g., 'HELP')]
                       │
                       ▼
1. Polling Tick:
   - GazeApp.tick() runs every 30 ms.
   - Calls self.gaze_source.get_sample(), returning (gaze_x, gaze_y, timestamp).
   - Coordinates are passed to GazeApp.feed_gaze(x, y, t).
                       │
                       ▼
2. Spatial Hit-Testing:
   - DwellDetector.update(x, y, t) runs.
   - button_at(x, y) tests if coordinates fall within button rectangles:
     bx <= x <= bx + bw and by <= y <= by + bh
   - Target is identified as 'HELP'.
                       │
                       ▼
3. Dwell Start (GAZE_ENTER):
   - Detector sees self.current is None and target is 'HELP'.
   - Checks cooldown: t - last_selected['HELP'] >= 1.0s.
   - Sets self.current = 'HELP', self.start_t = t.
   - Emits DwellEvent(type="GAZE_ENTER", button="HELP", fixation_time=0.0, progress=0.0).
   - In GUI: button card border lights up with active color (#89b4fa).
                       │
                       ▼
4. Dwell Accumulation (DWELL_PROGRESS):
   - Subsequent samples arrive every 30 ms.
   - fixation_time = t - self.start_t increases (e.g. 0.5s, 1.2s, 2.5s).
   - progress = fixation_time / dwell_time (0.0 to 1.0).
   - Emits DwellEvent(type="DWELL_PROGRESS", progress=progress).
   - In GUI: circular arc expands dynamically (-359.9 * progress degrees).
   - Countdown timer text updates ("1.5 s", "0.8 s", etc.).
   - In DwellLogger (attempt mode): gaze coordinates and sample counts are accumulated.
                       │
                       ▼
5. Blink / Noise Check (Grace Period):
   - If user blinks, sample arrives as (None, None, t).
   - Detector notes target != current and records self._lost_since = t.
   - If user opens eyes within 0.15s, self._lost_since resets to None.
   - Dwell accumulation resumes seamlessly without losing elapsed time!
                       │
                       ▼
6. Threshold Reached (BUTTON_SELECTED):
   - fixation_time >= dwell_time (e.g., reaches 3.0s).
   - Emits DwellEvent(type="BUTTON_SELECTED", button="HELP", fixation_time=3.0, progress=1.0).
   - Sets cooldown timer: self._last_selected['HELP'] = t.
   - Resets detector state for the next interaction.
                       │
                       ▼
7. Multimodal Feedback & Handoff:
   - GUI flashes card green (#1f3a2a background, #a6e3a1 outline) for 850 ms.
   - Ring completes to full circle, and label changes to "SELECTED".
   - beep(root) emits an auditory confirmation chime.
   - Batch 2 hook is invoked: self.on_selected("HELP") triggers speech synthesizer.
                       │
                       ▼
8. Data Logging:
   - DwellLogger.log(...) writes a single, structured row to dwell_log.csv:
     ["U001", 1791999115.9, 510.0, 377.5, "HELP", 3.05, 0, 1, 3.0]
   - The attempt is recorded and file buffer flushed immediately.
```

---

## 4. Top 10 Likely Viva Questions & Model Answers

### Q1: Why did you choose dwell time for selection rather than eye blinks?
> **Answer:** Involuntary blinks occur naturally every 2 to 10 seconds. Using blinks for selection causes frequent accidental selections and eye fatigue. Dwell selection leverages intentional, continuous fixation on a target, making it much more reliable and natural for motor-impaired users.

### Q2: What is the "Midas Touch" problem, and how does your module address it?
> **Answer:** In gaze tracking, the "Midas Touch" problem refers to the dilemma where anywhere the user looks might inadvertently trigger an action, because eyes are primarily perceptual organs, not output devices. Our module solves this by requiring deliberate dwell fixation (3, 4, or 5 seconds) before confirming selection, plus a 1.0-second post-selection cooldown to prevent re-triggering.

### Q3: What is the purpose of the 0.15-second grace period?
> **Answer:** Low-cost webcams and eye trackers suffer from tracking noise, pupil detection jitter, and natural micro-saccades. Furthermore, physiological blinks last between 100 ms and 150 ms. The 0.15 s grace period prevents the dwell timer from immediately resetting during a momentary loss of tracking or brief blink, preventing user frustration.

### Q4: How does the system differentiate between a blink and the user intentionally looking away?
> **Answer:** Duration. If gaze is lost or leaves the button for less than or equal to 0.15 seconds, the timer remains intact (tolerating a blink). If gaze stays off the button for longer than 0.15 seconds, the detector determines the user looked away, emits `GAZE_LEAVE`, logs an abandoned attempt, and resets the timer to zero.

### Q5: Why is a cooldown period (1.0 s) implemented after a button selection?
> **Answer:** After a button is selected, the user's gaze is still resting on that button while they comprehend the confirmation flash or hear the spoken phrase. Without a cooldown period, the detector would immediately start accumulating dwell time again, resulting in repeated selections of the same button.

### Q6: Why test 3 s, 4 s, and 5 s dwell thresholds? What is the usability trade-off?
> **Answer:** There is an inherent trade-off between communication speed and cognitive effort/accuracy:
> - **3.0 s:** Faster communication rate, but higher risk of inadvertent triggers if the user hesitates.
> - **5.0 s:** Near-zero accidental triggers, but slower communication and higher physical strain.
> - **4.0 s:** Provides an optimal balance of intentionality, confidence, and speed. Testing all three allows clinicians to tailor the threshold to an individual user's motor control level.

### Q7: How does your module reduce CSV file size, and why does that matter?
> **Answer:** Gaze trackers sample at 30 to 90 Hz. Logging every sample creates over 2,000 CSV rows per minute, resulting in multi-megabyte files filled with redundant progress steps. Our `DwellLogger` defaults to **Attempt Mode**, writing exactly one row when an attempt concludes (either `successful_selection=1` upon dwell completion or `successful_selection=0` if abandoned). This reduces file size by over 98% while providing Team 5 with all necessary metrics.

### Q8: How does your software architecture support both webcam and Tobii eye-tracking hardware?
> **Answer:** We implemented the **Adapter Pattern** via the `GazeSource` base class. The core `DwellDetector` and `GazeApp` only interact with `get_sample() -> (x, y, t)`. Whether coordinates originate from a mouse (`MouseGazeSource`), Team 3's webcam mapping (`WebcamGazeSource`), or Tobii hardware (`TobiiGazeSource`), the detection algorithms and GUI remain completely unchanged.

### Q9: What happens if gaze lands exactly on the boundary between two adjacent buttons?
> **Answer:** The bounding box check uses inclusive comparisons (`bx <= x <= bx + bw` and `by <= y <= by + bh`). If two buttons share a common edge, `button_at()` returns the first button encountered in deterministic dictionary order. Crucially, once gaze enters a button, the detector locks onto that button until gaze leaves past the grace period, preventing rapid jitter or flipping between buttons.

### Q10: How does your module integrate with Batch 2 (Text-to-Speech) and Team 5 (Evaluation)?
> **Answer:**
> - **Batch 2 Integration:** Provided via a one-line callback hook `app.set_on_selected(callback)`. When `BUTTON_SELECTED` fires, it passes the button name (e.g., `"HELP"`) directly to Batch 2's phrase-mapping and TTS audio engine.
> - **Team 5 Integration:** Every selection attempt is automatically appended to `dwell_log.csv`. Team 5 can run `analyze_results.py` to immediately compute success rates, average selection times, and produce publication-ready charts and summary tables.
