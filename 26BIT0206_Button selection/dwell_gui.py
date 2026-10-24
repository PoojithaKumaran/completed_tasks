"""
Polished dwell-selection GUI  -  Batch 1, Team 4 (Tanmay Gupta, 26BIT0206)

Multidisciplinary Project: Eye-Gaze Communication System
Pipeline Position:
  Camera/Tobii Input -> Face & Eye Detection -> Calibration & Gaze Mapping
  -> [THIS MODULE: Dwell Detection & Button Selection] -> Dataset & Evaluation
  (Feeds Batch 2 Message Mapping + Text-to-Speech)

Pure Tkinter (no extra GUI frameworks). Features:
  * 3x3 high-contrast communication grid with responsive auto-scaling
  * Circular progress ring filling dynamically with gaze fixation
  * Audio + visual confirmation on BUTTON_SELECTED
  * Selectable dwell settings: 3 s, 4 s, 5 s
  * GazeSource adapter pattern: MouseGazeSource (default), WebcamGazeSource, TobiiGazeSource
  * Batch 2 hook: on_selected callback can be plugged in with one line
  * Size-efficient CSV logging (1 row per attempt by default, optional raw mode)
  * Experiment Mode: prompts for User ID, tests dwell settings 3s, 4s, 5s with random targets,
    measures correctness and selection time, and feeds evaluation.
"""
import argparse
import os
import random
import sys
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import Callable, Dict, List, Optional, Tuple

from dwell_selection import (
    DwellDetector,
    DwellLogger,
    GazeSource,
    MouseGazeSource,
    WebcamGazeSource,
    TobiiGazeSource,
    GAZE_ENTER,
    GAZE_LEAVE,
    DWELL_PROGRESS,
    BUTTON_SELECTED,
)

# --------------------------------------------------------------------------
# Button Grid & Constants
# --------------------------------------------------------------------------
LABELS = [
    "WATER", "FOOD", "HELP",
    "CALL", "EMERGENCY", "MESSAGE",
    "ENTERTAIN", "SPEAK", "EXIT",
]

# Modern high-contrast color palette (Catppuccin inspired)
BG = "#11111b"
CARD = "#1e1e2e"
CARD_ACTIVE = "#2a2b3d"
BORDER = "#45475a"
TEXT = "#cdd6f4"
SUBTEXT = "#9399b2"
RING = "#89b4fa"
RING_TRACK = "#313244"
OK = "#a6e3a1"
ALERT = "#f38ba8"
TARGET_HIGHLIGHT = "#fab387"
CURSOR = "#f9e2af"

# Cross-platform font fallback
if sys.platform == "darwin":
    FONT_FAMILY = "Helvetica Neue"
elif sys.platform.startswith("win"):
    FONT_FAMILY = "Segoe UI"
else:
    FONT_FAMILY = "DejaVu Sans"


def beep(root: tk.Tk):
    """Auditory confirmation on button selection."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(880, 180)
        else:
            root.bell()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Main GUI Application
# --------------------------------------------------------------------------
class GazeApp:
    TOP, BOTTOM, PAD = 75, 60, 14

    def __init__(
        self,
        user_id: str = "U001",
        dwell_time: float = 3.0,
        gaze_source: Optional[GazeSource] = None,
        on_selected: Optional[Callable[[str], None]] = None,
        log_path: str = "dwell_log.csv",
        log_mode: str = "attempt",
        start_experiment: bool = False,
        dry_run: bool = False,
    ):
        """
        user_id        : Participant identifier (e.g. U001)
        dwell_time     : Default dwell threshold (3.0, 4.0, or 5.0)
        gaze_source    : GazeSource adapter instance (defaults to MouseGazeSource)
        on_selected    : Callback hook for Batch 2 (e.g. tts_function)
        log_path       : CSV file path for Team 5 dataset
        log_mode       : 'attempt' (1 row per attempt) or 'raw' (every sample)
        start_experiment: If True, immediately launches experiment mode
        dry_run        : If True, simulates gaze on targets and exits automatically
        """
        self.dry_run = dry_run or (os.environ.get("DWELL_DRY_RUN") == "1")
        self.root = tk.Tk()
        self.root.title("Gaze Communication - Dwell Selection (Team 4: 26BIT0206)")
        self.root.geometry("1020x740")
        self.root.configure(bg=BG)
        self.root.minsize(720, 540)

        self.on_selected = on_selected
        self.detector = DwellDetector({}, dwell_time=dwell_time, grace_period=0.15, cooldown=1.0)
        self.logger = DwellLogger(path=log_path, user_id=user_id, mode=log_mode)
        self.dwell_choice = tk.IntVar(value=int(dwell_time))

        # Setup gaze source adapter
        if gaze_source is not None:
            self.gaze_source = gaze_source
        else:
            self.gaze_source = MouseGazeSource()

        # Canvas & state
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.last_selected = "-"
        self.flash_until: Dict[str, str] = {}
        self.items: Dict[str, Dict] = {}
        self.gaze: Tuple[float, float] = (0.0, 0.0)
        self.pointer: Tuple[float, float] = (0.0, 0.0)

        # Resize debounce tracking
        self._last_w = 0
        self._last_h = 0
        self._resize_timer = None

        # Experiment Mode state
        self.experiment_active = False
        self.experiment_trials: List[Dict] = []
        self.current_trial_idx = 0
        self.trial_target_button: Optional[str] = None
        self.trial_start_time: Optional[float] = None
        self.experiment_results: List[Dict] = []

        # Event bindings
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Initial layout build
        self.build()

        # Start gaze polling loop (30 ms ~ 33 Hz)
        self.tick()

        if start_experiment or self.dry_run:
            delay = 50 if self.dry_run else 250
            self.root.after(delay, lambda: self.start_experiment(prompt_user=False))

    # ------------------------------------------------------------
    # Public Callback Hook for Batch 2 (Message Mapping & TTS)
    # ------------------------------------------------------------
    def set_on_selected(self, callback: Callable[[str], None]):
        """
        Plug in a teammate's function with one line:
            app.set_on_selected(my_tts_speak_function)
        """
        self.on_selected = callback

    # ------------------------------------------------------------
    # Layout and Rendering
    # ------------------------------------------------------------
    def _on_configure(self, event):
        """Debounce resize events to prevent canvas flicker and widget churn."""
        if abs(event.width - self._last_w) > 8 or abs(event.height - self._last_h) > 8:
            self._last_w = event.width
            self._last_h = event.height
            if self._resize_timer is not None:
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(60, self.build)

    def build(self):
        """Construct or reconstruct GUI widgets based on current window dimensions."""
        c = self.canvas
        # Cancel any active flash timeouts before clearing items
        for timer_id in self.flash_until.values():
            try:
                self.root.after_cancel(timer_id)
            except Exception:
                pass
        self.flash_until.clear()

        c.delete("all")
        W, H = max(c.winfo_width(), 720), max(c.winfo_height(), 540)
        gw = W - 2 * self.PAD
        gh = H - self.TOP - self.BOTTOM
        bw = (gw - 2 * self.PAD) / 3
        bh = (gh - 2 * self.PAD) / 3

        # Header Title
        title_text = "Look at a button to dwell & select"
        if self.experiment_active and self.trial_target_button:
            dwell_s = self.detector.dwell_time
            title_text = (
                f"EXPERIMENT [{self.current_trial_idx + 1}/{len(self.experiment_trials)}] "
                f"Dwell: {dwell_s:.0f}s  ->  TARGET: {self.trial_target_button}"
            )
        self.header_title = c.create_text(
            self.PAD + 6, 36, anchor="w", text=title_text,
            fill=TARGET_HIGHLIGHT if self.experiment_active else TEXT,
            font=(FONT_FAMILY, 17, "bold"),
        )

        # Header Controls: Experiment Mode Toggle Button
        exp_bg = "#f38ba8" if self.experiment_active else "#313244"
        exp_txt = "Stop Test" if self.experiment_active else "Experiment Mode"
        self.exp_btn = c.create_rectangle(W - self.PAD - 330, 18, W - self.PAD - 200, 50,
                                          fill=exp_bg, outline=BORDER, width=2)
        self.exp_btn_txt = c.create_text(W - self.PAD - 265, 34, text=exp_txt,
                                         fill=TEXT, font=(FONT_FAMILY, 11, "bold"))
        for item in (self.exp_btn, self.exp_btn_txt):
            c.tag_bind(item, "<Button-1>", lambda e: self.toggle_experiment())

        # Header Controls: Dwell Time Selector (3s, 4s, 5s)
        self.seg = {}
        for i, s in enumerate((3, 4, 5)):
            x1 = W - self.PAD - (3 - i) * 62
            r = c.create_rectangle(x1, 18, x1 + 56, 50, outline=BORDER, width=2)
            t = c.create_text(x1 + 28, 34, text=f"{s} s", font=(FONT_FAMILY, 12, "bold"))
            self.seg[s] = (r, t)
            for item in (r, t):
                c.tag_bind(item, "<Button-1>", lambda e, s=s: self.choose_dwell(s))

        self.refresh_selector()

        # 3x3 Button Grid
        buttons, self.items = {}, {}
        for i, name in enumerate(LABELS):
            r, col = divmod(i, 3)
            x = self.PAD + col * (bw + self.PAD)
            y = self.TOP + r * (bh + self.PAD)
            buttons[name] = (x, y, bw, bh)

            is_target = self.experiment_active and (name == self.trial_target_button)
            border_color = TARGET_HIGHLIGHT if is_target else BORDER
            border_w = 4 if is_target else 2

            card = c.create_rectangle(x, y, x + bw, y + bh, fill=CARD, outline=border_color, width=border_w)
            rad = min(bw, bh) * 0.28
            cx, cy = x + bw / 2, y + bh / 2 - 8
            box = (cx - rad, cy - rad, cx + rad, cy + rad)

            c.create_oval(*box, outline=RING_TRACK, width=8)
            arc = c.create_arc(*box, start=90, extent=0, style="arc", outline=RING, width=8)
            label = c.create_text(
                cx, cy, text=name, fill=TEXT,
                font=(FONT_FAMILY, max(13, int(bh / 10)), "bold"),
            )
            timer = c.create_text(cx, y + bh - 20, text="", fill=SUBTEXT, font=(FONT_FAMILY, 11))
            self.items[name] = dict(card=card, arc=arc, label=label, timer=timer)

        self.detector.buttons = buttons

        # Footer Status Bar
        self.footer = c.create_text(
            self.PAD + 6, H - 28, anchor="w", fill=SUBTEXT,
            font=(FONT_FAMILY, 12), text="",
        )
        self.cursor = c.create_oval(-20, -20, -6, -6, fill=CURSOR, outline="")
        self.update_footer()

    def refresh_selector(self):
        """Highlight active dwell time in the segmented control."""
        for s, (r, t) in self.seg.items():
            on = (s == self.dwell_choice.get())
            self.canvas.itemconfig(r, fill=RING if on else CARD)
            self.canvas.itemconfig(t, fill=BG if on else TEXT)

    def choose_dwell(self, s: int):
        """Set dwell threshold (3, 4, or 5 seconds)."""
        self.dwell_choice.set(s)
        self.detector.set_dwell_time(float(s))
        self.refresh_selector()
        self.reset_visuals()
        self.update_footer()

    def update_footer(self):
        """Render live gaze coordinates and current interaction status."""
        x, y = self.gaze
        mode_str = f"Log: {self.logger.mode}"
        user_str = f"User: {self.logger.user_id}"
        self.canvas.itemconfig(
            self.footer,
            text=f"{user_str}  |  Gaze: ({x:.0f}, {y:.0f})  |  Dwell: {self.detector.dwell_time:.0f} s  "
                 f"|  Last: {self.last_selected}  |  {mode_str}  |  [Esc] quit",
        )

    # ------------------------------------------------------------
    # Visual Feedback & Confirmation
    # ------------------------------------------------------------
    def reset_visuals(self, only: Optional[str] = None):
        """Reset button styles when gaze leaves."""
        for n, it in self.items.items():
            if only and n != only:
                continue
            if n in self.flash_until:
                continue
            is_target = self.experiment_active and (n == self.trial_target_button)
            border_col = TARGET_HIGHLIGHT if is_target else BORDER
            border_w = 4 if is_target else 2
            self.canvas.itemconfig(it["card"], fill=CARD, outline=border_col, width=border_w)
            self.canvas.itemconfig(it["arc"], extent=0, outline=RING)
            self.canvas.itemconfig(it["timer"], text="", fill=SUBTEXT)

    def flash(self, name: str):
        """Selection confirmation: green flash, full ring, audio cue."""
        if name not in self.items:
            return
        it = self.items[name]
        self.canvas.itemconfig(it["card"], fill="#1f3a2a", outline=OK, width=3)
        self.canvas.itemconfig(it["arc"], extent=-359.9, outline=OK)
        self.canvas.itemconfig(it["timer"], text="SELECTED", fill=OK)
        self.flash_until[name] = self.root.after(850, lambda: self.end_flash(name))

    def end_flash(self, name: str):
        """Restore button appearance after flash duration."""
        self.flash_until.pop(name, None)
        if name in self.items:
            self.reset_visuals(only=name)

    # ------------------------------------------------------------
    # Gaze Input Processing
    # ------------------------------------------------------------
    def _on_mouse_motion(self, event):
        """Update simulated gaze position from mouse movements."""
        self.pointer = (event.x, event.y)
        if isinstance(self.gaze_source, MouseGazeSource):
            self.gaze_source.set_position(event.x, event.y)

    def feed_gaze(self, x: Optional[float], y: Optional[float], t: Optional[float] = None):
        """
        Core entry point for gaze samples in canvas/screen coordinates.
        Called 30 times per second by tick() via the GazeSource adapter.
        """
        if x is not None and y is not None:
            self.gaze = (x, y)
            self.canvas.coords(self.cursor, x - 7, y - 7, x + 7, y + 7)
        else:
            # Blink or tracking lost: move cursor off-screen
            self.canvas.coords(self.cursor, -20, -20, -6, -6)

        events = self.detector.update(x, y, t)
        for ev in events:
            it = self.items.get(ev.button)
            if it is None:
                continue

            if ev.type in (GAZE_ENTER, DWELL_PROGRESS) and ev.button not in self.flash_until:
                is_target = self.experiment_active and (ev.button == self.trial_target_button)
                border_color = TARGET_HIGHLIGHT if is_target else RING
                self.canvas.itemconfig(it["card"], fill=CARD_ACTIVE, outline=border_color, width=3)
                self.canvas.itemconfig(it["arc"], extent=-359.9 * ev.progress)
                remaining = max(self.detector.dwell_time - ev.fixation_time, 0.0)
                self.canvas.itemconfig(it["timer"], text=f"{remaining:.1f} s", fill=SUBTEXT)
                self.logger.log(x, y, ev.button, ev.fixation_time, False, self.detector.dwell_time)

            elif ev.type == GAZE_LEAVE:
                self.reset_visuals(only=ev.button)
                self.logger.conclude_attempt(False, self.detector.dwell_time,
                                            fixation_time=ev.fixation_time, target=ev.button)

            elif ev.type == BUTTON_SELECTED:
                self.last_selected = ev.button
                self.flash(ev.button)
                beep(self.root)
                self.logger.log(x, y, ev.button, ev.fixation_time, True, self.detector.dwell_time)
                print(f"[BUTTON_SELECTED] -> {ev.button} (dwell={self.detector.dwell_time:.1f}s)")

                # Batch 2 callback hook: message mapping & TTS
                if self.on_selected:
                    try:
                        self.on_selected(ev.button)
                    except Exception as err:
                        print(f"Error in on_selected callback: {err}")

                # Experiment mode handler
                if self.experiment_active:
                    self.handle_experiment_selection(ev.button)

        self.update_footer()

    def tick(self):
        """Periodic loop polling GazeSource adapter (approx 33 Hz)."""
        try:
            if self.dry_run and self.experiment_active and self.trial_target_button:
                btn_name = self.trial_target_button
                if btn_name in self.detector.buttons:
                    bx, by, bw, bh = self.detector.buttons[btn_name]
                    cx, cy = bx + bw / 2, by + bh / 2
                    t0 = time.monotonic()
                    dwell = self.detector.dwell_time
                    self.feed_gaze(cx, cy, t=t0)
                    self.feed_gaze(cx, cy, t=t0 + dwell + 0.05)
            else:
                gx, gy, gt = self.gaze_source.get_sample()
                self.feed_gaze(gx, gy, t=gt)
        except Exception as exc:
            print(f"Error polling gaze source: {exc}")
        self.root.after(30, self.tick)

    # ------------------------------------------------------------
    # Experiment Mode
    # ------------------------------------------------------------
    def toggle_experiment(self):
        """Toggle experiment mode on/off from the UI."""
        if self.experiment_active:
            self.stop_experiment()
        else:
            self.start_experiment(prompt_user=True)

    def start_experiment_dialog(self):
        """Backwards-compatible alias for starting experiment."""
        self.start_experiment(prompt_user=True)

    def start_experiment(self, prompt_user: bool = False):
        """Initialize experiment trials across 3s, 4s, 5s (prompts for User ID if requested)."""
        if prompt_user:
            user_id = simpledialog.askstring(
                "Experiment Mode Setup",
                "Enter Participant / User ID:",
                initialvalue=self.logger.user_id,
                parent=self.root,
            )
            if not user_id:
                return
            self.logger.set_user_id(user_id)

        # Trials: 1 per dwell setting in dry-run, 3 per dwell setting in normal experiment
        reps = 1 if self.dry_run else 3
        trials = []
        for dwell_val in (3, 4, 5):
            for _ in range(reps):
                target = random.choice(LABELS)
                trials.append({"dwell": float(dwell_val), "target": target})
        if not self.dry_run:
            random.shuffle(trials)

        self.experiment_trials = trials
        self.current_trial_idx = 0
        self.experiment_results = []
        self.experiment_active = True
        self.start_next_trial()

    def start_next_trial(self):
        """Prepare and display the next trial target."""
        if self.current_trial_idx >= len(self.experiment_trials):
            self.finish_experiment()
            return

        trial = self.experiment_trials[self.current_trial_idx]
        self.trial_target_button = trial["target"]
        self.choose_dwell(int(trial["dwell"]))
        self.trial_start_time = time.monotonic()
        self.build()

    def handle_experiment_selection(self, selected_button: str):
        """Evaluate selection correctness and timing for the current trial."""
        if not self.experiment_active or not self.trial_target_button:
            return
        duration = time.monotonic() - (self.trial_start_time or time.monotonic())
        target = self.trial_target_button
        self.trial_target_button = None  # Clear during pause
        dwell_s = self.detector.dwell_time
        is_correct = (selected_button == target)

        self.experiment_results.append({
            "trial": self.current_trial_idx + 1,
            "dwell_setting": dwell_s,
            "target": target,
            "selected": selected_button,
            "correct": is_correct,
            "time_taken": round(duration, 3),
        })
        print(f"Trial {self.current_trial_idx + 1}/{len(self.experiment_trials)}: "
              f"Target={target}, Selected={selected_button}, Correct={is_correct}, Time={duration:.2f}s")

        self.current_trial_idx += 1
        # Pause between trials
        delay = 60 if self.dry_run else 900
        self.root.after(delay, self.start_next_trial)

    def finish_experiment(self):
        """Display experiment summary statistics and return to normal mode."""
        self.experiment_active = False
        self.trial_target_button = None
        self.build()

        total = len(self.experiment_results)
        if total == 0:
            if self.dry_run:
                self.close()
            return
        correct_count = sum(1 for r in self.experiment_results if r["correct"])
        acc = (correct_count / total) * 100
        avg_time = sum(r["time_taken"] for r in self.experiment_results) / total

        summary_msg = (
            f"Experiment Completed for {self.logger.user_id}!\n\n"
            f"Total Trials: {total}\n"
            f"Success Rate: {acc:.1f}% ({correct_count}/{total})\n"
            f"Average Selection Time: {avg_time:.2f} s\n\n"
            f"Data saved to '{self.logger.path}'.\n"
            f"Run 'python analyze_results.py' to generate charts and summary tables."
        )
        print(f"\n{summary_msg}\n")
        if self.dry_run:
            self.close()
        else:
            messagebox.showinfo("Experiment Results", summary_msg, parent=self.root)

    def stop_experiment(self):
        """Abort active experiment session."""
        self.experiment_active = False
        self.trial_target_button = None
        self.build()
        messagebox.showinfo("Experiment Stopped", "Experiment mode cancelled.", parent=self.root)

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def close(self):
        """Clean up resources on application close."""
        self.logger.close()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        """Launch the GUI event loop."""
        self.root.mainloop()


# --------------------------------------------------------------------------
# CLI Launcher
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dwell-Based Gaze Selection GUI - Team 4 (26BIT0206)")
    parser.add_argument("--user", default="U001", help="Participant User ID (default: U001)")
    parser.add_argument("--dwell", type=float, default=3.0, choices=[3.0, 4.0, 5.0],
                        help="Initial dwell time in seconds (default: 3.0)")
    parser.add_argument("--mode", default="attempt", choices=["attempt", "raw"],
                        help="CSV logging mode: 'attempt' (default) or 'raw'")
    parser.add_argument("--log", default="dwell_log.csv", help="CSV log filename")
    parser.add_argument("--experiment", action="store_true",
                        help="Launch directly into experiment mode (dwell = 3s, 4s, 5s)")
    parser.add_argument("--dry-run", "--simulated", dest="dry_run", action="store_true",
                        help="Run quick automated experiment trials and exit")
    args = parser.parse_args()

    # Demo Batch 2 hook: print simulated TTS output
    def demo_tts_hook(btn: str):
        phrases = {
            "WATER": "Please give me some water.",
            "FOOD": "I am hungry, I want food.",
            "HELP": "Please help me.",
            "CALL": "Please call my caregiver.",
            "EMERGENCY": "Emergency! I need immediate assistance!",
            "MESSAGE": "I want to send a message.",
            "ENTERTAIN": "Turn on entertainment.",
            "SPEAK": "I would like to speak.",
            "EXIT": "Goodbye.",
        }
        print(f"[Batch 2 TTS Hook] Button '{btn}' -> Speaking: \"{phrases.get(btn, btn)}\"")

    app = GazeApp(
        user_id=args.user,
        dwell_time=args.dwell,
        log_path=args.log,
        log_mode=args.mode,
        on_selected=demo_tts_hook,
        start_experiment=args.experiment,
        dry_run=args.dry_run,
    )
    app.run()


if __name__ == "__main__":
    main()

