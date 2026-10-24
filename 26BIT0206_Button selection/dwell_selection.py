"""
Batch 1 - Team 4: Dwell-Based Gaze Interaction
Student : Tanmay Gupta (26BIT0206)
Deliverable: Dwell Detection & Button Selection Module

Pipeline position:
    Calibration & Gaze Mapping (26BME0222)  -->  [THIS MODULE]  -->  Dataset + Evaluation (26BIT0140)

Rules implemented
  * gaze enters a button        -> start timer
  * gaze stays on the button    -> dwell time increases
  * gaze leaves the button      -> timer resets
  * dwell time >= threshold     -> BUTTON_SELECTED event
  * dwell thresholds to test    -> 3 s, 4 s, 5 s

The core (DwellDetector) has no GUI/hardware dependency, so Team 3 can feed it
screen-coordinate gaze samples and Team 5 can read the CSV log it produces.
Run this file directly for a demo GUI (mouse pointer simulates gaze).
"""
import csv
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
GAZE_ENTER = "GAZE_ENTER"
GAZE_LEAVE = "GAZE_LEAVE"
DWELL_PROGRESS = "DWELL_PROGRESS"
BUTTON_SELECTED = "BUTTON_SELECTED"


@dataclass
class DwellEvent:
    type: str
    button: str
    fixation_time: float      # seconds the gaze has stayed on this button
    timestamp: float          # time of the event (same clock as the input samples)
    progress: float = 0.0     # 0.0 - 1.0 fraction of the dwell threshold


# --------------------------------------------------------------------------
# Gaze Input Adapters
# --------------------------------------------------------------------------
class GazeSource:
    """
    Abstract base class for gaze input providers.
    Returns (x, y, timestamp) in screen/canvas pixels, or (None, None, timestamp)
    for invalid samples (blinks, lost tracking, out-of-screen).
    """

    def get_sample(self) -> Tuple[Optional[float], Optional[float], float]:
        """Return (x, y, timestamp) or (None, None, timestamp) if lost/blink."""
        raise NotImplementedError("Subclasses must implement get_sample()")


class MouseGazeSource(GazeSource):
    """
    Default adapter: simulates gaze using mouse pointer coordinates.
    Can either poll a coordinate getter callback or use manually updated coordinates.
    """

    def __init__(self, coord_getter: Optional[Callable[[], Tuple[Optional[float], Optional[float]]]] = None):
        self._coord_getter = coord_getter
        self._x: Optional[float] = 0.0
        self._y: Optional[float] = 0.0

    def set_position(self, x: Optional[float], y: Optional[float]):
        """Manually push mouse/cursor coordinates."""
        self._x = x
        self._y = y

    def get_sample(self) -> Tuple[Optional[float], Optional[float], float]:
        now = time.monotonic()
        if self._coord_getter is not None:
            gx, gy = self._coord_getter()
            return gx, gy, now
        return self._x, self._y, now


class WebcamGazeSource(GazeSource):
    """
    Placeholder adapter for Team 1 (Camera Input: 26BCB0165), Team 2 (Face/Eye Detection: 26BCE0341),
    and Team 3 (Calibration & Gaze Mapping: 26BME0222).
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        # TODO (Team 1 & Team 2): Initialize webcam capture pipeline (OpenCV/MediaPipe)
        # TODO (Team 3): Connect gaze calibration matrix to map pupil coordinates to canvas (x, y)

    def get_sample(self) -> Tuple[Optional[float], Optional[float], float]:
        # TODO (Team 3, 26BME0222): Return (calibrated_screen_x, calibrated_screen_y, timestamp)
        # Return (None, None, timestamp) when blink is detected or face tracking is lost.
        return None, None, time.monotonic()


class TobiiGazeSource(GazeSource):
    """
    Placeholder adapter for Tobii Eye Tracker hardware (Team 1, 26BCB0165).
    Connects to official Tobii SDK (e.g. tobii_research / stream engine).
    """

    def __init__(self, device_url: Optional[str] = None):
        self.device_url = device_url
        # TODO (Team 1, 26BCB0165): Initialize Tobii eye tracker via official SDK:
        #   import tobii_research as tr
        #   eyetrackers = tr.find_all_eyetrackers()
        #   self.tracker = eyetrackers[0] if eyetrackers else None
        #   self._latest_gaze = (None, None)

    def get_sample(self) -> Tuple[Optional[float], Optional[float], float]:
        # TODO (Team 1, 26BCB0165): Retrieve latest gaze sample from Tobii subscription callback.
        # Return (screen_x, screen_y, timestamp) or (None, None, timestamp) if validity flag is False.
        return None, None, time.monotonic()


# --------------------------------------------------------------------------
# Core dwell detector
# --------------------------------------------------------------------------
class DwellDetector:
    """
    buttons       : {name: (x, y, width, height)} in screen coordinates
    dwell_time    : seconds of continuous gaze needed to select (3, 4 or 5)
    grace_period  : seconds of lost/off-target gaze tolerated before the timer
                    resets (absorbs blinks and tracker noise). Set 0 for strict reset.
    cooldown      : seconds after a selection during which that button
                    cannot be selected again (prevents accidental repeats).
    """

    def __init__(self, buttons: Dict[str, Tuple[float, float, float, float]],
                 dwell_time: float = 3.0, grace_period: float = 0.15,
                 cooldown: float = 1.0):
        self.buttons = dict(buttons)
        self.dwell_time = dwell_time
        self.grace_period = grace_period
        self.cooldown = cooldown
        self._reset_state()
        self._last_selected: Dict[str, float] = {}
        self._last_t: Optional[float] = None

    # ---- configuration ---------------------------------------------------
    def set_dwell_time(self, seconds: float):
        if not (isinstance(seconds, (int, float)) and seconds > 0):
            raise ValueError("dwell time must be positive")
        self.dwell_time = float(seconds)
        self._reset_state()

    def _reset_state(self):
        self.current: Optional[str] = None
        self.start_t: Optional[float] = None
        self._lost_since: Optional[float] = None

    # ---- helpers ---------------------------------------------------------
    def button_at(self, x: float, y: float) -> Optional[str]:
        """
        Check if point (x, y) is inside any button rectangle.
        Boundary rule: inclusive on top/left, inclusive on bottom/right.
        First matching button in dictionary iteration order is returned.
        """
        for name, (bx, by, bw, bh) in self.buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return name
        return None

    @staticmethod
    def _is_valid_coordinate(val: Optional[float]) -> bool:
        """Validate that coordinate is a real, finite number."""
        if val is None or not isinstance(val, (int, float)):
            return False
        # Reject NaN and infinite values
        if val != val or val == float("inf") or val == float("-inf"):
            return False
        return True

    # ---- main entry point ------------------------------------------------
    def update(self, gaze_x: Optional[float], gaze_y: Optional[float],
               t: Optional[float] = None) -> List[DwellEvent]:
        """Feed one gaze sample (None/NaN/inf = invalid, e.g. blink). Returns events."""
        t = time.monotonic() if t is None else float(t)
        # Guard against backwards-ticking clocks or non-monotonic timestamps
        if self._last_t is not None and t < self._last_t:
            t = self._last_t
        self._last_t = t

        events: List[DwellEvent] = []

        is_valid = self._is_valid_coordinate(gaze_x) and self._is_valid_coordinate(gaze_y)
        target = self.button_at(gaze_x, gaze_y) if is_valid else None

        # Gaze is off the current button (or invalid): apply grace period
        if self.current is not None and target != self.current:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since > self.grace_period:
                fixation = max(0.0, self._lost_since - (self.start_t or self._lost_since))
                events.append(DwellEvent(GAZE_LEAVE, self.current, fixation, t))
                self._reset_state()          # gaze left -> timer resets
                # If gaze immediately landed on another valid button, start tracking it
                if target is not None:
                    in_cooldown = (t - self._last_selected.get(target, -1e9)) < self.cooldown
                    if not in_cooldown:
                        self.current, self.start_t, self._lost_since = target, t, None
                        events.append(DwellEvent(GAZE_ENTER, target, 0.0, t))
            else:
                return events                # still within tolerance, keep timer
        elif self.current is not None:
            self._lost_since = None          # gaze came back in time

        # Gaze entered a (new) button -> start timer
        if self.current is None and target is not None:
            in_cooldown = (t - self._last_selected.get(target, -1e9)) < self.cooldown
            if not in_cooldown:
                self.current, self.start_t, self._lost_since = target, t, None
                events.append(DwellEvent(GAZE_ENTER, target, 0.0, t))

        # Continuous gaze -> dwell grows; complete -> BUTTON_SELECTED
        if self.current is not None and self._lost_since is None and self.start_t is not None:
            fixation = max(0.0, t - self.start_t)
            progress = min(fixation / self.dwell_time, 1.0)
            if fixation >= self.dwell_time:
                events.append(DwellEvent(BUTTON_SELECTED, self.current, fixation, t, 1.0))
                self._last_selected[self.current] = t
                self._reset_state()
            else:
                events.append(DwellEvent(DWELL_PROGRESS, self.current, fixation, t, progress))
        return events


# --------------------------------------------------------------------------
# CSV logger (feeds Team 5's dataset) - fields from the project brief
# --------------------------------------------------------------------------
class DwellLogger:
    """
    Logs dwell data to CSV matching Team 5's dataset schema:
    [user_id, timestamp, gaze_x, gaze_y, target_button, fixation_time, blink, successful_selection, dwell_setting]

    Supports two modes:
      - mode='attempt' (default): logs ONE row per selection attempt (enter -> leave/select),
        preventing bloated CSV files while maintaining all evaluation metrics.
      - mode='raw': logs every 30 ms sample.
    """

    FIELDS = ["user_id", "timestamp", "gaze_x", "gaze_y", "target_button",
              "fixation_time", "blink", "successful_selection", "dwell_setting"]

    def __init__(self, path="dwell_log.csv", user_id="U001", mode="attempt"):
        self.path = path
        self.user_id = user_id
        self.mode = mode
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        if new:
            self._w.writeheader()
            self._f.flush()

        # In-flight attempt state (used in 'attempt' mode)
        self._active_target: Optional[str] = None
        self._attempt_start_t: Optional[float] = None
        self._attempt_gaze_sum: List[float] = [0.0, 0.0]
        self._attempt_gaze_count: int = 0
        self._attempt_blink_count: int = 0
        self._last_valid_gaze: Tuple[Optional[float], Optional[float]] = (None, None)

    def set_user_id(self, user_id: str):
        self.user_id = str(user_id).strip() or "U001"

    def set_mode(self, mode: str):
        if mode not in ("attempt", "raw"):
            raise ValueError("mode must be 'attempt' or 'raw'")
        self.mode = mode

    def log(self, gaze_x, gaze_y, target, fixation, selected, dwell_setting, blink=0, timestamp=None):
        """
        Direct logging method (compatible with legacy sample-by-sample calls).
        In 'raw' mode, writes immediately.
        In 'attempt' mode, handles attempt lifecycle automatically.
        """
        ts = round(time.time(), 3) if timestamp is None else round(float(timestamp), 3)
        gx = round(float(gaze_x), 1) if gaze_x is not None else None
        gy = round(float(gaze_y), 1) if gaze_y is not None else None

        if self.mode == "raw":
            self._write_row(ts, gx, gy, target, fixation, blink, selected, dwell_setting)
            return

        # 'attempt' mode logic:
        # If selection completed, write the successful row immediately
        if selected:
            # Use average gaze if available, else current gaze
            final_x = round(self._attempt_gaze_sum[0] / self._attempt_gaze_count, 1) if self._attempt_gaze_count > 0 else gx
            final_y = round(self._attempt_gaze_sum[1] / self._attempt_gaze_count, 1) if self._attempt_gaze_count > 0 else gy
            had_blink = 1 if (blink or self._attempt_blink_count > 0) else 0
            self._write_row(ts, final_x, final_y, target or self._active_target,
                            fixation, had_blink, 1, dwell_setting)
            self._reset_attempt()
        else:
            # Track ongoing attempt
            if target and self._active_target != target:
                # Switched button without clean leave event: conclude previous attempt
                if self._active_target:
                    self.conclude_attempt(False, dwell_setting, timestamp=ts)
                self._active_target = target
                self._attempt_start_t = ts
                self._attempt_gaze_sum = [0.0, 0.0]
                self._attempt_gaze_count = 0
                self._attempt_blink_count = 0

            if gx is not None and gy is not None:
                self._attempt_gaze_sum[0] += gx
                self._attempt_gaze_sum[1] += gy
                self._attempt_gaze_count += 1
                self._last_valid_gaze = (gx, gy)
            else:
                self._attempt_blink_count += 1

    def conclude_attempt(self, successful: bool, dwell_setting: float,
                         fixation_time: float = 0.0, target: Optional[str] = None,
                         timestamp: Optional[float] = None):
        """Record conclusion of an attempt (e.g. when gaze leaves or dwell completes)."""
        if self._active_target is None and target is None:
            return
        ts = round(time.time(), 3) if timestamp is None else round(float(timestamp), 3)
        tgt = target or self._active_target
        avg_x = round(self._attempt_gaze_sum[0] / self._attempt_gaze_count, 1) if self._attempt_gaze_count > 0 else self._last_valid_gaze[0]
        avg_y = round(self._attempt_gaze_sum[1] / self._attempt_gaze_count, 1) if self._attempt_gaze_count > 0 else self._last_valid_gaze[1]
        had_blink = 1 if self._attempt_blink_count > 0 else 0
        fix = round(float(fixation_time), 3)
        self._write_row(ts, avg_x, avg_y, tgt, fix, had_blink, int(successful), dwell_setting)
        self._reset_attempt()

    def _reset_attempt(self):
        self._active_target = None
        self._attempt_start_t = None
        self._attempt_gaze_sum = [0.0, 0.0]
        self._attempt_gaze_count = 0
        self._attempt_blink_count = 0

    def _write_row(self, ts, gx, gy, target, fixation, blink, selected, dwell_setting):
        self._w.writerow({
            "user_id": self.user_id,
            "timestamp": ts,
            "gaze_x": "" if gx is None else gx,
            "gaze_y": "" if gy is None else gy,
            "target_button": target or "",
            "fixation_time": round(float(fixation), 3),
            "blink": int(blink),
            "successful_selection": int(selected),
            "dwell_setting": float(dwell_setting),
        })
        self._f.flush()

    def close(self):
        if self._f and not self._f.closed:
            self._f.close()


# --------------------------------------------------------------------------
# Demo GUI (Tkinter). Mouse simulates gaze. Replace with Team 3's gaze stream
# by calling app.feed_gaze(x, y) instead of using the <Motion> binding.
# --------------------------------------------------------------------------
LABELS = ["WATER", "FOOD", "HELP", "CALL", "EMERGENCY",
          "MESSAGE", "ENTERTAIN", "SPEAK", "EXIT"]


def run_demo(user_id="U001", on_selected: Optional[Callable[[str], None]] = None):
    import tkinter as tk

    root = tk.Tk()
    root.title("Dwell-Based Gaze Selection - 26BIT0206")
    W, H, TOP = 900, 640, 60
    canvas = tk.Canvas(root, width=W, height=H, bg="#1e1e2e", highlightthickness=0)
    canvas.pack()

    # build 3x3 grid
    pad, bw, bh = 15, (W - 4 * 15) / 3, (H - TOP - 4 * 15) / 3
    buttons, rects, bars = {}, {}, {}
    for i, name in enumerate(LABELS):
        r, c = divmod(i, 3)
        x, y = pad + c * (bw + pad), TOP + pad + r * (bh + pad)
        buttons[name] = (x, y, bw, bh)
        rects[name] = canvas.create_rectangle(x, y, x + bw, y + bh, fill="#313244", outline="#585b70", width=3)
        bars[name] = canvas.create_rectangle(x, y + bh - 14, x, y + bh, fill="#a6e3a1", width=0)
        canvas.create_text(x + bw / 2, y + bh / 2, text=name, fill="white", font=("Arial", 22, "bold"))

    status = canvas.create_text(W / 2, 30, text="Look at a button (move mouse) and hold...",
                                fill="white", font=("Arial", 14))
    cursor = canvas.create_oval(0, 0, 14, 14, fill="#f38ba8", outline="")

    detector = DwellDetector(buttons, dwell_time=3.0)
    logger = DwellLogger(user_id=user_id)
    dwell_var = tk.DoubleVar(value=3.0)

    ctrl = tk.Frame(root)
    ctrl.pack()
    tk.Label(ctrl, text="Dwell time (s):").pack(side="left")
    for s in (3, 4, 5):
        tk.Radiobutton(ctrl, text=f"{s} s", variable=dwell_var, value=s,
                       command=lambda: detector.set_dwell_time(dwell_var.get())).pack(side="left")

    def reset_colors():
        for n in LABELS:
            canvas.itemconfig(rects[n], fill="#313244")
            x, y, w, h = buttons[n]
            canvas.coords(bars[n], x, y + h - 14, x, y + h)

    def feed_gaze(x, y):
        """Entry point for gaze samples in SCREEN/CANVAS coordinates."""
        canvas.coords(cursor, x - 7, y - 7, x + 7, y + 7)
        for ev in detector.update(x, y):
            bx, by, bw_, bh_ = buttons[ev.button]
            if ev.type == DWELL_PROGRESS or ev.type == GAZE_ENTER:
                canvas.itemconfig(rects[ev.button], fill="#45475a")
                canvas.coords(bars[ev.button], bx, by + bh_ - 14, bx + bw_ * ev.progress, by + bh_)
                canvas.itemconfig(status, text=f"{ev.button}: {ev.fixation_time:.1f}s / {detector.dwell_time:.0f}s")
                logger.log(x, y, ev.button, ev.fixation_time, False, detector.dwell_time)
            elif ev.type == GAZE_LEAVE:
                reset_colors()
                canvas.itemconfig(status, text=f"Gaze left {ev.button} - timer reset")
                logger.conclude_attempt(False, detector.dwell_time, fixation_time=ev.fixation_time, target=ev.button)
            elif ev.type == BUTTON_SELECTED:
                reset_colors()
                canvas.itemconfig(rects[ev.button], fill="#a6e3a1")
                canvas.itemconfig(status, text=f"BUTTON_SELECTED: {ev.button}")
                logger.log(x, y, ev.button, ev.fixation_time, True, detector.dwell_time)
                print("BUTTON_SELECTED:", ev.button)
                if on_selected:
                    on_selected(ev.button)           # hook for Batch 2 (message + TTS)
                root.after(700, reset_colors)

    canvas.bind("<Motion>", lambda e: feed_gaze(e.x, e.y))

    # keep the timer ticking even if the mouse is held perfectly still
    def tick():
        x, y = root.winfo_pointerx() - canvas.winfo_rootx(), root.winfo_pointery() - canvas.winfo_rooty()
        feed_gaze(x, y)
        root.after(30, tick)
    tick()

    root.protocol("WM_DELETE_WINDOW", lambda: (logger.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    run_demo()
