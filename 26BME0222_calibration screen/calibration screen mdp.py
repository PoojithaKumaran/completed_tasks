import base64
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
import tkinter as tk

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError as err:
    print("A required library is missing:", err)
    print("Install everything with:  py -m pip install opencv-python mediapipe numpy")
    sys.exit(1)


# ======================================================================
# SETTINGS (change these to tune the program)
# ======================================================================
FULLSCREEN = True               # False = use a normal window instead
WINDOWED_SIZE = (1200, 720)     # only used when FULLSCREEN is False
CAMERA_INDEX = 0                # 0 = the built-in laptop webcam
SHOW_PREVIEW = True             # small camera picture in the top-right corner

MARGIN = 200                    # distance of the outer dots from the screen edge
DOT_RADIUS = 16

SETTLE_S = 0.7                  # ignore the first part (eyes are still moving)
CAPTURE_S = 1.5                 # then average readings for this long
MIN_SAMPLES = 10                # need at least this many good readings per dot

TICK_MS = 30                    # main loop repeats every 30 ms
RAW_SMOOTHING = 0.5             # 0..1, higher = less smoothing of raw readings
CURSOR_SMOOTHING = 0.35         # 0..1, higher = cursor reacts faster but jitters
CURSOR_RADIUS = 12
BLINK_OPENNESS = 0.10           # eyelid gap / eye width below this = eye closed

SHOW_GAZE_READOUT = True        # show the live gaze x, y numbers on the screen
PRINT_GAZE_EVERY_S = 0.25       # also print them in the console (0 = off)
GAZE_TIMEOUT_S = 0.5            # get_current_gaze() returns None if the last
                                # good reading is older than this

PREVIEW_W, PREVIEW_H = 200, 150

MODEL_FILE = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# Landmark numbers in MediaPipe's 478-point face model.
#   a / b  = the two eye corners (a = left in the picture, b = right)
#   top / bottom = eyelids,   iris = centre of the iris
EYES = (
    dict(a=33,  b=133, top=159, bottom=145, iris=468),   # the person's right eye
    dict(a=362, b=263, top=386, bottom=374, iris=473),   # the person's left eye
)

HERE = os.path.dirname(os.path.abspath(__file__))

USER_IDS = [1, 2, 3, 4, 5, 6]   # users that can be selected (number keys 1-9)
PROFILE_DIR = os.path.join(HERE, "profiles")


# ======================================================================
# 1. FEATURES: from face landmarks to (raw_x, raw_y)   (pure maths)
# ======================================================================
def eye_features(pts, eye):
    """Return (rx, ry, openness) for one eye, or None if it can't be measured.

    rx : where the iris sits ALONG the eye (0 = corner a, 1 = corner b)
    ry : how far the iris sits ABOVE/BELOW the line between the corners,
         measured in eye-widths (+ = down)
    Both are measured along the eye's own direction, so tilting your head
    does not change them much.
    """
    A, B, I = pts[eye["a"]], pts[eye["b"]], pts[eye["iris"]]
    v = B - A
    width = float(np.linalg.norm(v))
    if width < 1e-6:
        return None
    u = v / width                          # unit vector along the eye
    n = np.array([-u[1], u[0]])            # unit vector perpendicular (down)
    mid = (A + B) / 2.0
    rx = float(np.dot(I - A, u) / width)
    ry = float(np.dot(I - mid, n) / width)
    lid_gap = float(np.linalg.norm(pts[eye["top"]] - pts[eye["bottom"]]))
    return rx, ry, lid_gap / width


def compute_raw(pts):
    """Average both eyes into one (raw_x, raw_y). None if an eye is closed."""
    values = []
    for eye in EYES:
        feat = eye_features(pts, eye)
        if feat is None or feat[2] < BLINK_OPENNESS:
            return None
        values.append(feat[:2])
    rx, ry = np.mean(values, axis=0)
    return float(rx), float(ry)


# ======================================================================
# 2. WEBCAM + FACE MODEL
# ======================================================================
def ensure_model():
    """Return the path of the MediaPipe face model, downloading it once."""
    path = os.path.join(HERE, MODEL_FILE)
    if os.path.exists(path):
        return path
    print("Downloading the face model (one time only, a few MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as err:
        if os.path.exists(path):
            os.remove(path)                # remove a half-downloaded file
        raise RuntimeError(
            "Could not download the face model automatically.\n"
            f"  Reason: {err}\n"
            "Fix: download the file from this link in your browser\n"
            f"  {MODEL_URL}\n"
            f"and save it as '{MODEL_FILE}' in the same folder as this script."
        )
    return path


class GazeTracker:
    """Reads the webcam and returns (raw_x, raw_y) for every frame."""

    def __init__(self, camera_index=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the webcam.\n"
                "Close other apps that use the camera (Zoom, Teams, Camera app, "
                "browser tabs) and check Settings > Privacy & security > Camera."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.status = "starting"
        self.preview_b64 = None
        self._last_ts = 0
        self._smooth = None

    def read(self):
        """Grab one frame. Returns (raw_x, raw_y) or None if no good reading."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status = "camera error"
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if ts <= self._last_ts:            # MediaPipe needs increasing times
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)

        pts, raw = None, None
        if not result.face_landmarks:
            self.status = "no face detected"
        else:
            landmarks = result.face_landmarks[0]
            if len(landmarks) < 478:
                self.status = "iris points missing"
            else:
                pts = np.array([(p.x * w, p.y * h) for p in landmarks])
                raw = compute_raw(pts)
                self.status = "tracking OK" if raw is not None else "eyes closed"

        # light smoothing of the raw readings (webcam data jitters)
        if raw is None:
            self._smooth = None
        elif self._smooth is None:
            self._smooth = raw
        else:
            a = RAW_SMOOTHING
            self._smooth = (a * raw[0] + (1 - a) * self._smooth[0],
                            a * raw[1] + (1 - a) * self._smooth[1])

        if SHOW_PREVIEW:
            self._make_preview(frame, pts)
        return self._smooth

    def _make_preview(self, frame, pts):
        img = frame.copy()
        if pts is not None:                # mark corners (blue) and iris (green)
            for eye in EYES:
                for key, colour in (("a", (255, 200, 0)), ("b", (255, 200, 0)),
                                    ("iris", (0, 255, 0))):
                    x, y = pts[eye[key]]
                    cv2.circle(img, (int(x), int(y)), 3, colour, -1)
        img = cv2.flip(img, 1)             # mirror view, like a mirror
        img = cv2.resize(img, (PREVIEW_W, PREVIEW_H))
        ok, png = cv2.imencode(".png", img)
        if ok:
            self.preview_b64 = base64.b64encode(png.tobytes()).decode("ascii")

    def close(self):
        self.cap.release()
        self.landmarker.close()


# ======================================================================
# 3. MAPPING: fit a rule from the 9 calibration pairs
# ======================================================================
def fit_mapping(pairs):
    """Least-squares fit of  screen = a*raw_x + b*raw_y + c  (x and y separately).

    pairs = list of (raw_x, raw_y, target_x, target_y)
    """
    A = np.array([[rx, ry, 1.0] for rx, ry, _, _ in pairs])
    tx = np.array([p[2] for p in pairs], dtype=float)
    ty = np.array([p[3] for p in pairs], dtype=float)
    cx, *_ = np.linalg.lstsq(A, tx, rcond=None)
    cy, *_ = np.linalg.lstsq(A, ty, rcond=None)
    return {"x": cx, "y": cy}


def apply_mapping(mapping, raw_x, raw_y):
    ax, bx, cx = mapping["x"]
    ay, by, cy = mapping["y"]
    return ax * raw_x + bx * raw_y + cx, ay * raw_x + by * raw_y + cy


def leave_one_out_errors(pairs):
    """Honest accuracy estimate: for each dot, fit on the OTHER 8 dots and
    see how far off the prediction for the left-out dot is."""
    errors = []
    for i in range(len(pairs)):
        mapping = fit_mapping(pairs[:i] + pairs[i + 1:])
        rx, ry, tx, ty = pairs[i]
        mx, my = apply_mapping(mapping, rx, ry)
        errors.append(float(np.hypot(mx - tx, my - ty)))
    return errors


def report_calibration(pairs, counts, mapping):
    """Print the calibration table and return its rows (for the CSV)."""
    loo = leave_one_out_errors(pairs)
    rows, fit_errors = [], []
    print("\nCalibration result")
    print(" pt |  raw_x   raw_y  |   target (px)  |   mapped (px)   | fit err | LOO err | n")
    for i, ((rx, ry, tx, ty), n, lo) in enumerate(zip(pairs, counts, loo), start=1):
        mx, my = apply_mapping(mapping, rx, ry)
        err = float(np.hypot(mx - tx, my - ty))
        fit_errors.append(err)
        rows.append((i, rx, ry, tx, ty, mx, my, err, lo, n))
        print(f" {i:>2} | {rx:7.3f} {ry:7.3f} | ({tx:5d},{ty:5d}) | "
              f"({mx:7.1f},{my:7.1f}) | {err:7.1f} | {lo:7.1f} | {n}")
    print(f"Average fit error: {np.mean(fit_errors):.1f} px   "
          f"(how well the rule matches the 9 dots)")
    print(f"Average LOO error: {np.mean(loo):.1f} px   "
          f"(realistic accuracy estimate - use this one in your report)")

    rx_all = [p[0] for p in pairs]
    ry_all = [p[1] for p in pairs]
    x_range, y_range = max(rx_all) - min(rx_all), max(ry_all) - min(ry_all)
    print(f"Eye-reading spread across the 9 dots: raw_x {x_range:.3f}, raw_y {y_range:.3f}")
    if x_range < 0.02 or y_range < 0.01:
        print("WARNING: your eye readings barely changed between dots, so the "
              "calibration will be unreliable. Sit closer, light your face "
              "evenly, and move your EYES (not your head) to each dot.")
    return rows


# ======================================================================
# 4. SAVING: calibration CSV and user profiles
# ======================================================================
CALIBRATION_HEADER = ["user_id", "calibrated_at", "point", "raw_x", "raw_y",
                      "target_x", "target_y", "mapped_x", "mapped_y",
                      "fit_error_px", "loo_error_px", "samples_used"]
GAZE_LOG_HEADER = ["timestamp", "elapsed_s", "user_id", "gaze_x", "gaze_y",
                   "raw_x", "raw_y"]


def open_csv_append(path, header):
    """Open a CSV for appending (all users share one file).

    If an older file has different columns (for example the Stage 1 file),
    it is kept under a new name so the columns never get mixed up.
    Returns (writer, file) - close the file when you are done.
    """
    if os.path.exists(path):
        with open(path, newline="") as f:
            first_line = next(csv.reader(f), None)
        if first_line != header:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(path)
            os.rename(path, f"{base}_old_{stamp}{ext}")
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(header)
    return writer, f


def save_calibration_csv(user_id, rows):
    """Append this user's calibration (9 rows) to calibration_data.csv."""
    path = os.path.join(HERE, "calibration_data.csv")
    writer, f = open_csv_append(path, CALIBRATION_HEADER)
    stamp = datetime.now().isoformat(timespec="seconds")
    for i, rx, ry, tx, ty, mx, my, err, lo, n in rows:
        writer.writerow([user_id, stamp, i, round(rx, 4), round(ry, 4), tx, ty,
                         round(mx, 1), round(my, 1), round(err, 2),
                         round(lo, 2), n])
    f.close()
    return path


def profile_path(user_id):
    return os.path.join(PROFILE_DIR, f"user_{user_id}.json")


def save_profile(user_id, pairs, counts, mapping, fit_err, loo_err, w, h):
    """Write the user's profile to profiles/user_<id>.json."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    data = {
        "user_id": user_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "screen_width": int(w),
        "screen_height": int(h),
        "pairs": [[float(v) for v in p] for p in pairs],
        "samples_per_point": [int(n) for n in counts],
        "mapping": {"x": [float(v) for v in mapping["x"]],
                    "y": [float(v) for v in mapping["y"]]},
        "fit_error_px": float(fit_err),
        "loo_error_px": float(loo_err),
    }
    path = profile_path(user_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


_warned_files = set()


def load_profile(user_id):
    """Return the saved profile as a dict, or None if missing or unreadable."""
    path = profile_path(user_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        mapping = {"x": np.array(data["mapping"]["x"], dtype=float),
                   "y": np.array(data["mapping"]["y"], dtype=float)}
        if mapping["x"].shape != (3,) or mapping["y"].shape != (3,):
            raise ValueError("the mapping must have 3 numbers per axis")
        return {
            "user_id": data["user_id"],
            "created": str(data["created"]),
            "screen": (int(data["screen_width"]), int(data["screen_height"])),
            "pairs": [tuple(p) for p in data["pairs"]],
            "mapping": mapping,
            "fit_error_px": float(data["fit_error_px"]),
            "loo_error_px": float(data["loo_error_px"]),
        }
    except Exception as err:
        if path not in _warned_files:              # warn only once per file
            _warned_files.add(path)
            print(f"WARNING: could not read {path} ({err}). "
                  f"Treating User {user_id} as having no profile.")
        return None


# ======================================================================
# 5. LIVE FEED for other modules
# ======================================================================
_latest_gaze = None
_latest_time = 0.0
_current_user = None


def get_current_gaze():
    """Latest gaze position (gaze_x, gaze_y) in pixels, or None.

    None means: not live yet, or no good reading for GAZE_TIMEOUT_S seconds
    (face lost, eyes closed). Other modules should treat None as "no gaze",
    so that an old position is never used by mistake.
    """
    if _latest_gaze is None or time.time() - _latest_time > GAZE_TIMEOUT_S:
        return None
    return _latest_gaze


def get_current_user():
    """The user id of the person currently using the cursor, or None."""
    return _current_user


# ======================================================================
# 6. THE APP (Tkinter screen)
# ======================================================================
class ProfileApp:
    def __init__(self, root, tracker):
        self.root = root
        self.tracker = tracker

        root.title("Gaze calibration - Module 3")
        root.configure(bg="black")
        if FULLSCREEN:
            root.attributes("-fullscreen", True)
        else:
            root.geometry(f"{WINDOWED_SIZE[0]}x{WINDOWED_SIZE[1]}")
        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        root.update()
        self.w = int(self.canvas.winfo_width())
        self.h = int(self.canvas.winfo_height())
        root.focus_force()

        # 9 targets: 3x3 grid, row by row
        xs = [MARGIN, self.w // 2, self.w - MARGIN]
        ys = [MARGIN, self.h // 2, self.h - MARGIN]
        self.targets = [(x, y) for y in ys for x in xs]

        # state:  select -> (confirm) -> waiting <-> capturing -> live
        self.state = "select"
        self.user_id = None
        self.profile = None             # saved profile of the selected user
        self.delete_mode = False        # selection screen: next pick = delete
        self.delete_target = None       # user whose profile may be deleted
        self.delete_origin = "select"   # screen to return to if cancelled
        self.index = 0
        self.pairs = []                 # (raw_x, raw_y, target_x, target_y)
        self.counts = []
        self.samples = []
        self.capture_start = 0.0
        self.mapping = None
        self.dot_items = []
        self.ui_items = []              # cards of the selection screen
        self.cards = []                 # (user_id, x1, y1, x2, y2) for clicks
        self.cursor_item = None
        self.smooth_pos = None
        self.log_file = None
        self.log_writer = None
        self.log_start = 0.0
        self.preview_image = None
        self.frame_count = 0

        # text on screen
        self.instruction = self.canvas.create_text(
            self.w // 2, 50, fill="white", font=("Arial", 20), text="")
        self.progress = self.canvas.create_text(
            self.w // 2, 88, fill="gray", font=("Arial", 14), text="")
        self.status_item = self.canvas.create_text(
            30, self.h - 30, anchor="w", fill="orange", font=("Arial", 14),
            text="starting camera...")
        self.readout_item = self.canvas.create_text(
            self.w // 2, 130, fill="yellow", font=("Consolas", 22, "bold"),
            text="")
        self.last_print = 0.0
        self.preview_item = None
        if SHOW_PREVIEW:
            self.preview_item = self.canvas.create_image(
                self.w - 20, 20, anchor="ne")

        root.bind("<space>", self.on_space)
        root.bind("<Key>", self.on_key)
        root.bind("<Button-1>", self.on_click)
        root.bind("<Escape>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.go_to_select()
        self.tick()

    # ---------------- small helpers ----------------
    def say(self, text, colour="white"):
        self.canvas.itemconfig(self.instruction, text=text, fill=colour)

    def sub(self, text):
        self.canvas.itemconfig(self.progress, text=text)

    def readout(self, text, colour="yellow"):
        if SHOW_GAZE_READOUT:
            self.canvas.itemconfig(self.readout_item, text=text, fill=colour)

    def clear_ui(self):
        for item in self.ui_items:
            self.canvas.delete(item)
        self.ui_items = []
        self.cards = []

    def clear_screen(self):
        self.clear_ui()
        for item in self.dot_items:
            self.canvas.delete(item)
        self.dot_items = []
        if self.cursor_item is not None:
            self.canvas.delete(self.cursor_item)
            self.cursor_item = None

    def close_log(self):
        if self.log_file is not None:
            self.log_file.close()
        self.log_file = None
        self.log_writer = None

    def update_status_and_preview(self):
        status = self.tracker.status
        colour = "lime" if status == "tracking OK" else "orange"
        self.canvas.itemconfig(self.status_item, text=f"Camera: {status}",
                               fill=colour)
        self.frame_count += 1
        if (SHOW_PREVIEW and self.tracker.preview_b64
                and self.frame_count % 2 == 0):
            self.preview_image = tk.PhotoImage(data=self.tracker.preview_b64)
            self.canvas.itemconfig(self.preview_item, image=self.preview_image)

    # ---------------- main loop ----------------
    def tick(self):
        raw = self.tracker.read()
        self.update_status_and_preview()
        if self.state == "capturing":
            self.handle_capture(raw)
        elif self.state == "live":
            self.handle_live(raw)
        self.root.after(TICK_MS, self.tick)

    # ---------------- keyboard and mouse ----------------
    def on_key(self, event):
        key = (event.keysym or "").lower()
        ch = event.char or ""
        if self.state == "select":
            if key == "d":
                self.toggle_delete_mode()
            elif ch.isdigit() and int(ch) in USER_IDS:
                self.pick_user(int(ch))
        elif self.state == "confirm":
            if key in ("return", "kp_enter"):
                self.use_saved_profile()
            elif key == "r":
                self.start_calibration()
            elif key == "d":
                self.ask_delete(self.user_id, "confirm")
            elif key in ("b", "u", "backspace"):
                self.go_to_select()
        elif self.state == "delete_confirm":
            if key == "y":
                self.do_delete()
            elif key in ("n", "b", "backspace"):
                self.cancel_delete()
        elif key == "u":                       # waiting / capturing / live
            self.go_to_select()

    def on_click(self, event):
        if self.state != "select":
            return
        for uid, x1, y1, x2, y2 in self.cards:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.pick_user(uid)
                return

    # ---------------- user selection ----------------
    def go_to_select(self):
        global _latest_gaze, _latest_time, _current_user
        self.close_log()
        self.clear_screen()
        _latest_gaze = None
        _current_user = None
        self.user_id = None
        self.profile = None
        self.delete_mode = False
        self.delete_target = None
        self.mapping = None
        self.smooth_pos = None
        self.index = 0
        self.pairs, self.counts, self.samples = [], [], []
        self.state = "select"
        self.say("Who is using the system?  Click a card or press its number")
        self.sub("D = delete a profile      ESC = quit")
        self.readout("")
        _latest_time = 0.0
        self.draw_user_cards()

    def draw_user_cards(self):
        self.clear_ui()
        gap, card_h, cols = 40, 150, 3
        card_w = min(380, (self.w - 4 * gap) // cols)
        rows = (len(USER_IDS) + cols - 1) // cols
        total_h = rows * card_h + (rows - 1) * gap
        top = (self.h - total_h) // 2
        left = (self.w - (cols * card_w + (cols - 1) * gap)) // 2
        for k, uid in enumerate(USER_IDS):
            r, c = divmod(k, cols)
            x1 = left + c * (card_w + gap)
            y1 = top + r * (card_h + gap)
            x2, y2 = x1 + card_w, y1 + card_h
            profile = load_profile(uid)
            if profile is None and os.path.exists(profile_path(uid)):
                line, colour = "profile file is damaged", "orange"
            elif profile is None:
                line, colour = "no profile yet", "gray"
            elif profile["screen"] != (self.w, self.h):
                line, colour = "saved on a different screen size", "orange"
            else:
                line = f"profile saved - accuracy ~{profile['loo_error_px']:.0f} px"
                colour = "lime"
            cx = (x1 + x2) // 2
            outline = "white"
            if self.delete_mode:                    # show what can be deleted
                outline = ("tomato" if os.path.exists(profile_path(uid))
                           else "gray30")
            self.ui_items += [
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline=outline,
                    width=3 if self.delete_mode else 2),
                self.canvas.create_text(cx, y1 + 50, text=f"User {uid}",
                                        fill="white", font=("Arial", 26, "bold")),
                self.canvas.create_text(cx, y1 + 105, text=line, fill=colour,
                                        font=("Arial", 13)),
            ]
            self.cards.append((uid, x1, y1, x2, y2))

    def select_user(self, uid):
        self.user_id = uid
        self.clear_ui()
        self.profile = load_profile(uid)
        if self.profile is None:
            self.start_calibration()
        elif self.profile["screen"] != (self.w, self.h):
            print(f"User {uid}: the saved profile was made on a "
                  f"{self.profile['screen'][0]}x{self.profile['screen'][1]} screen "
                  f"(this one is {self.w}x{self.h}), so it must be recalibrated.")
            self.start_calibration()
        else:
            self.state = "confirm"
            when = self.profile["created"].replace("T", " ")
            self.say(f"User {uid}: saved profile found  (calibrated {when}, "
                     f"accuracy ~{self.profile['loo_error_px']:.0f} px)")
            self.sub("ENTER = use it      R = recalibrate      D = delete      B = back")

    def pick_user(self, uid):
        """A card was clicked / a number pressed on the selection screen."""
        if self.delete_mode:
            self.ask_delete(uid, "select")
        else:
            self.select_user(uid)

    # ---------------- deleting profiles ----------------
    def toggle_delete_mode(self):
        self.delete_mode = not self.delete_mode
        if self.delete_mode:
            self.say("DELETE MODE - click or press the number of the profile "
                     "to delete", "tomato")
            self.sub("D = cancel      ESC = quit")
        else:
            self.say("Who is using the system?  Click a card or press its number")
            self.sub("D = delete a profile      ESC = quit")
        self.draw_user_cards()

    def ask_delete(self, uid, origin):
        if not os.path.exists(profile_path(uid)):
            self.say(f"User {uid} has no saved profile to delete", "orange")
            return
        self.delete_target = uid
        self.delete_origin = origin
        self.clear_ui()
        self.state = "delete_confirm"
        self.say(f"Delete User {uid}'s profile?  This cannot be undone.", "tomato")
        self.sub("Y = yes, delete it      N = no, keep it      "
                 "(saved CSV data is not deleted)")

    def do_delete(self):
        uid = self.delete_target
        path = profile_path(uid)
        try:
            os.remove(path)
            _warned_files.discard(path)
            print(f"Deleted the profile of User {uid}.")
            message, colour = f"User {uid}'s profile was deleted", "lime"
        except OSError as err:
            print(f"Could not delete the profile of User {uid}: {err}")
            message, colour = f"Could not delete User {uid}'s profile", "orange"
        self.go_to_select()
        self.say(message, colour)

    def cancel_delete(self):
        uid, origin = self.delete_target, self.delete_origin
        self.delete_target = None
        if origin == "confirm":
            self.select_user(uid)           # back to "saved profile found"
        else:
            self.go_to_select()

    def use_saved_profile(self):
        self.mapping = self.profile["mapping"]
        print(f"User {self.user_id}: using the saved profile "
              f"(made {self.profile['created']}).")
        self.start_live()

    # ---------------- calibration ----------------
    def start_calibration(self):
        self.index = 0
        self.pairs, self.counts, self.samples = [], [], []
        self.state = "waiting"
        self.show_target()

    def show_target(self):
        for item in self.dot_items:
            self.canvas.delete(item)
        x, y = self.targets[self.index]
        r = DOT_RADIUS
        outer = self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill="red", outline="")
        inner = self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                        fill="black", outline="")
        self.dot_items = [outer, inner]
        self.say("Look at the black centre of the red dot, keep your head "
                 "still, then press SPACE")
        self.sub(f"User {self.user_id}  -  point {self.index + 1} of "
                 f"{len(self.targets)}      (U = back to user selection)")

    def on_space(self, event=None):
        if self.state != "waiting" or self.index >= len(self.targets):
            return
        self.state = "capturing"
        self.capture_start = time.time()
        self.samples = []
        self.say("Keep looking at the dot...")

    def handle_capture(self, raw):
        elapsed = time.time() - self.capture_start
        if elapsed < SETTLE_S:
            return                                  # eyes still settling
        if elapsed < SETTLE_S + CAPTURE_S:
            if raw is not None:
                self.samples.append(raw)
            return

        # capture time is over
        if len(self.samples) < MIN_SAMPLES:
            self.state = "waiting"
            self.say(f"Only {len(self.samples)} good readings - make sure your "
                     "face is visible, then press SPACE again")
            return

        raw_x, raw_y = np.median(np.array(self.samples), axis=0)
        tx, ty = self.targets[self.index]
        self.pairs.append((float(raw_x), float(raw_y), tx, ty))
        self.counts.append(len(self.samples))
        self.index += 1
        if self.index < len(self.targets):
            self.state = "waiting"
            self.show_target()
        else:
            self.finish_calibration()

    def finish_calibration(self):
        for item in self.dot_items:
            self.canvas.delete(item)
        self.dot_items = []
        self.mapping = fit_mapping(self.pairs)
        print(f"\n===== User {self.user_id} =====")
        rows = report_calibration(self.pairs, self.counts, self.mapping)
        fit_err = float(np.mean([r[7] for r in rows]))
        loo_err = float(np.mean([r[8] for r in rows]))
        print("Calibration rows added to:", save_calibration_csv(self.user_id, rows))
        print("Profile saved to:", save_profile(
            self.user_id, self.pairs, self.counts, self.mapping,
            fit_err, loo_err, self.w, self.h))
        self.start_live()

    # ---------------- live gaze cursor ----------------
    def start_live(self):
        global _current_user
        path = os.path.join(HERE, "gaze_log.csv")
        self.log_writer, self.log_file = open_csv_append(path, GAZE_LOG_HEADER)
        self.log_start = time.time()
        print("Logging live gaze to:", path)

        self.smooth_pos = None
        self.cursor_item = self.canvas.create_oval(0, 0, 0, 0, fill="lime",
                                                   outline="")
        _current_user = self.user_id
        self.say(f"User {self.user_id} - live gaze cursor")
        self.sub("U = switch user      ESC = quit")
        self.readout("Gaze: waiting for a reading...", "orange")
        self.last_print = 0.0
        self.state = "live"

    def handle_live(self, raw):
        global _latest_gaze, _latest_time
        now = time.time()
        if raw is None:                             # no good reading this frame
            if now - _latest_time > GAZE_TIMEOUT_S:
                self.readout("Gaze: no reading (face lost or eyes closed)",
                             "orange")
            return                                  # cursor stays where it was
        x, y = apply_mapping(self.mapping, raw[0], raw[1])
        x = min(max(x, 0), self.w)                  # keep inside the screen
        y = min(max(y, 0), self.h)

        if self.smooth_pos is None:
            self.smooth_pos = (x, y)
        else:
            a = CURSOR_SMOOTHING
            self.smooth_pos = (a * x + (1 - a) * self.smooth_pos[0],
                               a * y + (1 - a) * self.smooth_pos[1])
        sx, sy = self.smooth_pos
        r = CURSOR_RADIUS
        self.canvas.coords(self.cursor_item, sx - r, sy - r, sx + r, sy + r)

        _latest_gaze = (sx, sy)                     # for get_current_gaze()
        _latest_time = now
        self.readout(f"Gaze:  x = {sx:7.1f}   y = {sy:7.1f}   (pixels)")
        if PRINT_GAZE_EVERY_S > 0 and now - self.last_print >= PRINT_GAZE_EVERY_S:
            self.last_print = now
            print(f"Gaze  x={sx:7.1f}  y={sy:7.1f}   (User {self.user_id})")
        self.log_writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            round(time.time() - self.log_start, 3), self.user_id,
            round(sx, 1), round(sy, 1), round(raw[0], 4), round(raw[1], 4)])
        self.log_file.flush()

    # ---------------- quitting ----------------
    def quit(self):
        self.close_log()
        self.tracker.close()
        self.root.destroy()


# ======================================================================
def main():
    try:
        tracker = GazeTracker()
    except RuntimeError as err:
        print("\n" + str(err))
        return
    root = tk.Tk()
    ProfileApp(root, tracker)
    root.mainloop()


if __name__ == "__main__":
    main()
