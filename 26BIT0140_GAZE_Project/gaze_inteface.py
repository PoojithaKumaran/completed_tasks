
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from datetime import datetime
from collections import deque
from gaze_core import (
    DwellTracker, TARGETS, WIDTH, HEIGHT, bound_screen_point, identify_target,
    fit_gaze_model, load_latest_calibration, RAW_FEATURE_COLUMNS,
    RELATIVE_FEATURE_COLUMNS, INTERACTION_COLUMNS, INTERRUPTION_COLUMNS,
)
from gaze_features import extract_gaze_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
LEGACY_CALIBRATION_CSV = os.path.join(BASE_DIR, "calibration_data.csv")
FEATURE_CALIBRATION_CSV = os.path.join(BASE_DIR, "calibration_data_v2.csv")
USE_RELATIVE_FEATURES = os.path.exists(FEATURE_CALIBRATION_CSV)
CALIBRATION_CSV = (FEATURE_CALIBRATION_CSV if USE_RELATIVE_FEATURES
                   else LEGACY_CALIBRATION_CSV)
OUTPUT_CSV = os.path.join(BASE_DIR, "gaze_interactions.csv")

DWELL_SECONDS = 3.0
RADIUS = 42
HORIZONTAL_INVERT = True
VERTICAL_INVERT = True
GAZE_SENSITIVITY = 0.60
VERTICAL_SENSITIVITY = 1.20
SCREEN_SMOOTHING_ALPHA = 0.18
SCREEN_DEADBAND_PX = 2.0
targets = TARGETS

# --------------------------------------------------
# 1. Load the calibration readings
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("face_landmarker.task is missing.")

if not os.path.exists(CALIBRATION_CSV):
    raise FileNotFoundError("Calibration CSV is missing. Run calibration.py first.")

USER_ID = input("Enter User ID (e.g., user_001): ").strip()
if not USER_ID:
    raise ValueError("User ID cannot be empty.")

feature_columns = (*RAW_FEATURE_COLUMNS, *RELATIVE_FEATURE_COLUMNS) if USE_RELATIVE_FEATURES else RAW_FEATURE_COLUMNS
X, Y, target_labels, _ = load_latest_calibration(CALIBRATION_CSV, feature_columns)
target_counts = set(target_labels.tolist())
if target_counts != set(range(1, 10)):
    missing_targets = sorted(set(range(1, 10)) - target_counts)
    raise ValueError(f"Latest calibration session is missing target readings: {missing_targets}")
model = fit_gaze_model(X, Y)

print(f"Calibration samples used: {len(X)}")
print(f"Targets covered: {len(target_counts)}")
print(f"Mapping created from {'eye-relative + raw' if USE_RELATIVE_FEATURES else 'legacy raw'} iris features.")

# --------------------------------------------------
# 2. Set up MediaPipe and webcam
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera.release()
    raise RuntimeError("Could not open webcam. Check camera permission and connection.")

# Create a separate interaction log; never overwrite calibration data.
if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
    with open(OUTPUT_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(INTERACTION_COLUMNS)

# Prompt targets in order so we can measure selection success.
expected_target = 1
dwell = DwellTracker(DWELL_SECONDS)
selected_lock_target = None

# Separate log for interrupted dwell attempts.
INTERRUPTION_CSV = os.path.join(BASE_DIR, "gaze_interruption_log.csv")

if (
    not os.path.exists(INTERRUPTION_CSV)
    or os.path.getsize(INTERRUPTION_CSV) == 0
):
    with open(INTERRUPTION_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(INTERRUPTION_COLUMNS)

# Tracks the last target whose dwell was interrupted.
interrupted_target = None

message = "Look at target 1."
frame_number = 0
gaze_x, gaze_y = None, None

# --------------------------------------------------
# 3. Run gaze interaction
# --------------------------------------------------

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Gaze interaction started. Press Q to quit.")
        print(f"Interaction data will be saved to: {OUTPUT_CSV}")

        
        eyes_were_closed = False
        blink_event = False
        blink_during_dwell = False
        last_valid_gaze_x = None
        last_valid_gaze_y = None
        feature_history = deque(maxlen=7)
        while True:
            current_target = None
            invalid_gaze_features = False
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000
            frame_number += 1

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            now = time.monotonic()
            landmarks = (
                result.face_landmarks[0]
                if result.face_landmarks else None
            )

            if landmarks is not None and len(landmarks) > 477:
                left = landmarks[468]
                right = landmarks[473]
                
                # Estimate eye closure using eyelid landmarks.
                # Face Landmarker eye contour indices:
                LEFT_EYE_TOP = landmarks[159]
                LEFT_EYE_BOTTOM = landmarks[145]
                LEFT_EYE_OUTER = landmarks[33]
                LEFT_EYE_INNER = landmarks[133]

                RIGHT_EYE_TOP = landmarks[386]
                RIGHT_EYE_BOTTOM = landmarks[374]
                RIGHT_EYE_OUTER = landmarks[362]
                RIGHT_EYE_INNER = landmarks[263]

                left_eye_height = np.hypot(
                    LEFT_EYE_TOP.x - LEFT_EYE_BOTTOM.x,
                    LEFT_EYE_TOP.y - LEFT_EYE_BOTTOM.y
                )
                left_eye_width = np.hypot(
                    LEFT_EYE_OUTER.x - LEFT_EYE_INNER.x,
                    LEFT_EYE_OUTER.y - LEFT_EYE_INNER.y
                )

                right_eye_height = np.hypot(
                    RIGHT_EYE_TOP.x - RIGHT_EYE_BOTTOM.x,
                    RIGHT_EYE_TOP.y - RIGHT_EYE_BOTTOM.y
                )
                right_eye_width = np.hypot(
                    RIGHT_EYE_OUTER.x - RIGHT_EYE_INNER.x,
                    RIGHT_EYE_OUTER.y - RIGHT_EYE_INNER.y
                )

                eye_ratio = (
                    left_eye_height / max(left_eye_width, 1e-6)
                    + right_eye_height / max(right_eye_width, 1e-6)
                ) / 2

                # A low ratio suggests the eyes are closed.
                blink_event = eye_ratio < 0.18

                if blink_event and not eyes_were_closed:
                    if dwell.target_id is not None:
                        blink_during_dwell = True

                eyes_were_closed = blink_event


                # Keep the cursor stable during a blink.
                if not blink_event:
                    try:
                        if USE_RELATIVE_FEATURES:
                            live_raw = extract_gaze_features(landmarks)
                        else:
                            live_raw = np.array([left.x, left.y, right.x, right.y], dtype=float)
                        feature_history.append(live_raw)
                        smoothed = np.median(np.asarray(feature_history), axis=0)
                        live_features = smoothed
                        prediction = model.predict(live_features.reshape(1, -1))[0]
                        gaze_x, gaze_y = bound_screen_point(*prediction)
                    except ValueError:
                        invalid_gaze_features = True

                    if not invalid_gaze_features:
                        center_x, center_y = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
                        delta_x = prediction[0] - center_x
                        if HORIZONTAL_INVERT:
                            delta_x = -delta_x
                        delta_y = prediction[1] - center_y
                        if VERTICAL_INVERT:
                            delta_y = -delta_y
                        mapped_x = center_x + GAZE_SENSITIVITY * delta_x
                        mapped_y = center_y + VERTICAL_SENSITIVITY * delta_y
                        mapped_x, mapped_y = bound_screen_point(mapped_x, mapped_y)
                        if last_valid_gaze_x is None or last_valid_gaze_y is None:
                            gaze_x, gaze_y = mapped_x, mapped_y
                        else:
                            dx, dy = mapped_x - last_valid_gaze_x, mapped_y - last_valid_gaze_y
                            if abs(dx) < SCREEN_DEADBAND_PX:
                                dx = 0.0
                            if abs(dy) < SCREEN_DEADBAND_PX:
                                dy = 0.0
                            gaze_x = int(round(last_valid_gaze_x + SCREEN_SMOOTHING_ALPHA * dx))
                            gaze_y = int(round(last_valid_gaze_y + SCREEN_SMOOTHING_ALPHA * dy))
                        last_valid_gaze_x, last_valid_gaze_y = gaze_x, gaze_y
                else:
                    if last_valid_gaze_x is not None:
                        gaze_x = last_valid_gaze_x
                        gaze_y = last_valid_gaze_y


                # Find which target contains the estimated gaze.
                current_target = (identify_target(gaze_x, gaze_y, RADIUS)
                                  if gaze_x is not None and gaze_y is not None
                                  else None)

                # Require looking away after a selection before
                # allowing another selection.
                
                # Prevent selecting the same target repeatedly,
                # while allowing direct movement to the next target.
                if blink_event or invalid_gaze_features:
                    dwell.pause(now)
                    reason = "Blink" if blink_event else "Invalid eye features"
                    message = f"{reason}; dwell paused ({dwell.elapsed(now):.1f}s)."
                else:
                    dwell.resume(now)
                if not blink_event and not invalid_gaze_features and current_target is None:
                    state, elapsed_previous = dwell.update(None, now)
                    if state == "interrupted":
                        with open(INTERRUPTION_CSV, "a", newline="", encoding="utf-8") as file:
                            csv.writer(file).writerow([
                                datetime.now().isoformat(timespec="seconds"), USER_ID,
                                expected_target, "", round(elapsed_previous, 3), "looked_away"
                            ])
                    message = f"Look at target {expected_target}."

                elif not blink_event and not invalid_gaze_features and current_target == selected_lock_target:
                    state, elapsed_previous = dwell.update(None, now)
                    if state == "interrupted":
                        with open(INTERRUPTION_CSV, "a", newline="", encoding="utf-8") as file:
                            csv.writer(file).writerow([
                                datetime.now().isoformat(timespec="seconds"), USER_ID,
                                expected_target, selected_lock_target,
                                round(elapsed_previous, 3), "move_after_selection"
                            ])
                    dwell.reset()
                    message = "Move your gaze to the next target."

                elif not blink_event and not invalid_gaze_features:
                    selected_lock_target = None

                    state, elapsed = dwell.update(current_target, now)
                    if state == "started":
                        blink_during_dwell = False
                    if state == "completed":
                        with open(OUTPUT_CSV, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                datetime.now().isoformat(timespec="seconds"),
                                USER_ID,
                                gaze_x,
                                gaze_y,
                                current_target,
                                expected_target,
                                round(elapsed, 3),
                                "yes" if blink_during_dwell else "no",
                                current_target,
                                "yes" if current_target == expected_target else "no"
                            ])

                        if current_target == expected_target:
                            message = f"Target {expected_target} selected!"
                        else:
                            message = (
                                f"Target {current_target} selected. "
                                f"Expected target was {expected_target}."
                            )

                        selected_lock_target = current_target
                        expected_target = expected_target % len(targets) + 1

            else:
                feature_history.clear()
                gaze_x = gaze_y = None
                last_valid_gaze_x = last_valid_gaze_y = None
                blink_event = False
                eyes_were_closed = False
                interrupted_target = dwell.target_id
                state, elapsed_previous = dwell.update(None, now)
                if state == "interrupted":
                    with open(INTERRUPTION_CSV, "a", newline="", encoding="utf-8") as file:
                        csv.writer(file).writerow([
                            datetime.now().isoformat(timespec="seconds"), USER_ID,
                            expected_target, interrupted_target or "",
                            round(elapsed_previous, 3), "tracking_lost"
                        ])
                message = f"Face not detected. Look at target {expected_target}."

            # Draw the interaction screen.
            screen = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

            cv2.putText(
                screen, "GAZE INTERACTION - 9 TARGETS",
                (240, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (30, 30, 30), 2
            )

            for i, (tx, ty) in enumerate(targets, start=1):
                color = (200, 180, 100)

                if i == expected_target:
                    color = (100, 210, 100)

                cv2.circle(screen, (tx, ty), RADIUS, color, -1)
                cv2.putText(
                    screen, str(i), (tx - 9, ty + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2
                )

            # Draw the red gaze dot
            # Draw the red gaze dot
            if gaze_x is not None and gaze_y is not None:
                cv2.circle(
                    screen, (gaze_x, gaze_y), 9, (0, 0, 255), -1
                )
                cv2.circle(
                    screen, (gaze_x, gaze_y), 14, (0, 0, 255), 2
                )

                nearest_dist = min(
                    np.hypot(gaze_x - tx, gaze_y - ty)
                    for tx, ty in targets
                )

                if dwell.target_id is not None:
                    elapsed = min(dwell.elapsed(now), DWELL_SECONDS)
                    timer_text = (
                        f"Dwell: {elapsed:.1f}/{DWELL_SECONDS:.1f}s"
                    )
                else:
                    timer_text = "Dwell: 0.0/3.0s"

                debug_text = (
                    f"Gaze: ({gaze_x}, {gaze_y}) | "
                    f"Target: {current_target} | "
                    f"Distance: {nearest_dist:.1f} | "
                    f"{timer_text}"
                )
            else:
                debug_text = "Gaze coordinates unavailable"

            cv2.putText(
                screen, debug_text, (30, 570),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 2
            )

            cv2.putText(
                screen, message, (65, 600),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (30, 30, 30), 2
            )

            cv2.putText(
                screen,
                f"Invert X/Y: {'ON' if HORIZONTAL_INVERT else 'OFF'}/{'ON' if VERTICAL_INVERT else 'OFF'} | Sens X/Y: {GAZE_SENSITIVITY:.2f}/{VERTICAL_SENSITIVITY:.2f} | I/V, +/- X, [/] Y",
                (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1
            )

            cv2.putText(
                screen, "Q: quit",
                (440, 630), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (30, 30, 30), 1
            )

            cv2.imshow("Gaze Interaction", screen)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                HORIZONTAL_INVERT = not HORIZONTAL_INVERT
                message = f"Horizontal inversion {'on' if HORIZONTAL_INVERT else 'off'}."
                last_valid_gaze_x = last_valid_gaze_y = None
            elif key == ord("v"):
                VERTICAL_INVERT = not VERTICAL_INVERT
                message = f"Vertical inversion {'on' if VERTICAL_INVERT else 'off'}."
                last_valid_gaze_x = last_valid_gaze_y = None
            elif key in (ord("+"), ord("=")):
                GAZE_SENSITIVITY = min(1.0, GAZE_SENSITIVITY + 0.05)
                message = f"Sensitivity: {GAZE_SENSITIVITY:.2f}"
            elif key in (ord("-"), ord("_")):
                GAZE_SENSITIVITY = max(0.20, GAZE_SENSITIVITY - 0.05)
                message = f"Sensitivity: {GAZE_SENSITIVITY:.2f}"
            elif key == ord("]"):
                VERTICAL_SENSITIVITY = min(2.5, VERTICAL_SENSITIVITY + 0.10)
                message = f"Vertical sensitivity: {VERTICAL_SENSITIVITY:.2f}"
            elif key == ord("["):
                VERTICAL_SENSITIVITY = max(0.30, VERTICAL_SENSITIVITY - 0.10)
                message = f"Vertical sensitivity: {VERTICAL_SENSITIVITY:.2f}"

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Interaction data saved to: {OUTPUT_CSV}")
