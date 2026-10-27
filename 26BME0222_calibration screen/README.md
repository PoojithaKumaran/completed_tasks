# Calibration & Gaze Mapping

Part of the eye-gaze interaction project for people who communicate with their eyes.
This part of the project turns webcam images into **a live gaze position in screen pixels**, with a
separate saved calibration profile for each user.

```
Webcam -> face & iris landmarks -> eye readings -> per-user mapping -> gaze (x, y) in pixels
```

## What it does

- Reads the laptop webcam (OpenCV) and finds the face and iris (MediaPipe Face Landmarker).
- Turns the iris position inside each eye into two numbers, `raw_x` and `raw_y`.
- **Calibration:** shows 9 dots; the user looks at each one and presses SPACE. Readings are
  averaged for about 1.5 s per dot, and a mapping from eye readings to screen pixels is fitted
  by least squares (`screen = a·raw_x + b·raw_y + c`, separately for x and y).
- **User profiles:** each user (User 1 – User 6) has their own saved calibration. Returning
  users can reuse their profile or recalibrate; profiles can be deleted.
- **Live gaze:** after calibration, a green cursor follows the gaze, the coordinates are shown
  on screen and printed in the console, and other code can read them with `get_current_gaze()`.
- Logs calibration results and live gaze to CSV files, tagged with the user id.

## Requirements

- Python 3.9 – 3.12 (developed with 3.12). Tkinter must be included in your Python install
  (on Windows, keep "tcl/tk and IDLE" ticked in the installer).
- A webcam.
- Internet on the **first run only**, to download the small MediaPipe face model
  (`face_landmarker.task`, saved next to the script).

```
py -m pip install -r requirements.txt
```

## Run

```
py stage2_user_profiles.py
```

If the model download fails, download `face_landmarker.task` manually (the link is printed in
the error message) and save it in the same folder as the script.

## Controls

| Screen | Keys |
|---|---|
| User selection | click a card or press **1–6**; **D** = delete mode; **ESC** = quit |
| Calibration | look at the black centre of the red dot, keep your head still, press **SPACE**; **U** = back to user selection |
| Saved profile found | **ENTER** = use it; **R** = recalibrate; **D** = delete; **B** = back |
| Delete confirmation | **Y** = delete the profile; **N** = keep it |
| Live gaze | **U** = switch user; **ESC** = quit |

Tips for a good calibration: sit about 50–60 cm from the screen, light your face evenly from
the front (a bright window behind you hurts tracking), keep your head still and move only your
eyes, and calibrate at the seat and distance you will use later.

## Output files

All files are created next to the script.

| File | Contents |
|---|---|
| `profiles/user_<id>.json` | one profile per user: the 9 eye readings, the fitted mapping, accuracy, and the screen size it was made on |
| `calibration_data.csv` | every calibration run (new rows are appended): `user_id, calibrated_at, point, raw_x, raw_y, target_x, target_y, mapped_x, mapped_y, fit_error_px, loo_error_px, samples_used` |
| `gaze_log.csv` | live gaze, about 30 rows per second (rows are appended): `timestamp, elapsed_s, user_id, gaze_x, gaze_y, raw_x, raw_y` |

- `gaze_x`, `gaze_y` are screen pixels with the origin at the top-left, x to the right, y downward.
- `fit_error_px` is how well the fitted rule matches the same 9 dots (optimistic).
- `loo_error_px` is a leave-one-out error: the rule is fitted on 8 dots and tested on the 9th.
  It is the more realistic accuracy estimate.
- A profile only works on the **same screen size** it was made on; otherwise that user is
  asked to recalibrate.
- Deleting a profile does **not** delete anything from the CSV files.
- If an older CSV with different columns already exists, it is kept under a new name
  (`..._old_<date>.csv`) instead of being mixed in.

## Using the live gaze from other code

`get_current_gaze()` returns the latest `(x, y)` in pixels, or `None` when there is no fresh
reading (calibration not finished, face lost, eyes closed for more than 0.5 s). Treat `None`
as "no gaze" so that an old position is never used. `get_current_user()` returns the user id.

The gaze is produced by the running window, so other code must run in the **same program**
and the same Tkinter loop, and should draw its buttons on `app.canvas`:

```python
import tkinter as tk
import stage2_user_profiles as gaze

tracker = gaze.GazeTracker()
root = tk.Tk()
app = gaze.ProfileApp(root, tracker)      # app.canvas is the window to draw buttons on

def check_gaze():
    pos = gaze.get_current_gaze()         # (x, y) in pixels, or None
    user = gaze.get_current_user()        # user id, or None
    if pos is not None:
        x, y = pos
        # ... test whether (x, y) is inside a button ...
    root.after(30, check_gaze)

check_gaze()
root.mainloop()
```

## Settings

Near the top of `stage2_user_profiles.py`:

| Setting | Meaning |
|---|---|
| `USER_IDS` | the users that can be selected |
| `FULLSCREEN`, `WINDOWED_SIZE` | fullscreen or a normal window |
| `CAMERA_INDEX` | which camera to use (0 = built-in webcam) |
| `SHOW_PREVIEW` | small camera picture in the corner |
| `SETTLE_S`, `CAPTURE_S`, `MIN_SAMPLES` | how long each calibration dot is captured |
| `SHOW_GAZE_READOUT`, `PRINT_GAZE_EVERY_S` | on-screen and console gaze numbers (`0` turns the console print off) |
| `GAZE_TIMEOUT_S` | how long before a gaze reading counts as stale |
| `RAW_SMOOTHING`, `CURSOR_SMOOTHING` | smoothing of the readings and of the cursor |
| `BLINK_OPENNESS` | eyelid-gap threshold below which an eye counts as closed |

## Limitations

- Webcam gaze tracking is far less accurate than dedicated eye-tracking hardware. It is
  sensitive to head movement, lighting and glasses, so interface buttons need to be large.
- Users are chosen on the selection screen; the system does **not** recognise a user
  automatically.
- Profiles and logs contain personal data (eye measurements recorded from real people).
  Do not commit them to the repository, and get each user's consent before recording.

## Status

Implemented: webcam gaze input, 9-point calibration, per-user profiles (save, reuse, recalibrate,
delete), live gaze readout and the `get_current_gaze()` feed.
