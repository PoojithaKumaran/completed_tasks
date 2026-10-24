
import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
from collections import deque
import random
from gaze_features import extract_gaze_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
CSV_PATH = os.path.join(BASE_DIR, "calibration_data_v2.csv")

WIDTH, HEIGHT = 1000, 650

# Nine known screen targets
targets = [
    (150, 120), (500, 120), (850, 120),
    (150, 325), (500, 325), (850, 325),
    (150, 530), (500, 530), (850, 530)
]

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
    raise RuntimeError("Could not open webcam.")

# Create the CSV and write its column headings

# Create the CSV header only if the file doesn't exist or is empty
file_is_empty = (
    not os.path.exists(CSV_PATH)
    or os.path.getsize(CSV_PATH) == 0
)

if file_is_empty:
    with open(CSV_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "target_id",
            "target_x",
            "target_y",
            "left_iris_x",
            "left_iris_y",
            "right_iris_x",
            "right_iris_y",
            "left_eye_rel_x",
            "left_eye_rel_y",
            "right_eye_rel_x",
            "right_eye_rel_y"
        ])

# Per-target reading counters (total across all sessions from CSV)
readings_per_target = {i: 0 for i in range(1, 10)}

# Pre-count existing readings from the CSV if appending
if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
    with open(CSV_PATH, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    tid = int(row[1])
                    if tid in readings_per_target:
                        readings_per_target[tid] += 1
                except ValueError:
                    pass

# Session-only counters for the exit summary
session_readings = {i: 0 for i in range(1, 10)}
feature_history = deque(maxlen=15)

target_order = list(range(len(targets)))
random.shuffle(target_order)
target_index = 0
frame_number = 0
message = "Look at the target, then press SPACE to record."
last_capture = ""

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Calibration started.")
        print("SPACE = record reading | N = next target | Q = quit")

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000
            frame_number += 1

            result = landmarker.detect_for_video(
                image, timestamp_ms
            )

            landmarks = None
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

            if landmarks is not None and len(landmarks) > 477:
                try:
                    feature_history.append(extract_gaze_features(landmarks))
                except ValueError:
                    feature_history.clear()
            else:
                feature_history.clear()

            # Draw webcam preview
            cv2.putText(
                frame,
                "Face detected" if landmarks else "No face detected",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0) if landmarks else (0, 0, 255),
                2
            )

            # Display iris positions on the webcam preview for stability feedback
            if landmarks and len(landmarks) > 477:
                h_cam, w_cam = frame.shape[:2]
                left_iris = landmarks[468]
                right_iris = landmarks[473]
                lx, ly = int(left_iris.x * w_cam), int(left_iris.y * h_cam)
                rx, ry = int(right_iris.x * w_cam), int(right_iris.y * h_cam)
                cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)
                cv2.circle(frame, (rx, ry), 4, (0, 255, 0), -1)

            cv2.imshow("Webcam Preview", frame)

            # Draw the target on a separate experiment window
            screen = 255 * np.ones(
                (HEIGHT, WIDTH, 3), dtype="uint8"
            )

            tid = target_order[target_index] + 1
            tx, ty = targets[tid - 1]
            cv2.circle(screen, (tx, ty), 42, (255, 0, 0), -1)

            # Show reading count on the target circle
            count_text = str(readings_per_target[tid])
            cv2.putText(
                screen, count_text,
                (tx - 10 * len(count_text) // 2, ty + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.putText(
                screen,
                f"Calibration target {tid}/9",
                (300, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (30, 30, 30), 2
            )

            # Show session readings counter
            total_session = sum(session_readings.values())
            cv2.putText(
                screen,
                f"Readings this session: {total_session}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (100, 100, 100), 1
            )

            cv2.putText(
                screen, message, (120, 600),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (40, 40, 40), 1
            )

            cv2.putText(
                screen,
                "SPACE: median of last 15 frames   N: next target   Q: quit",
                (210, 630),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (40, 40, 40), 1
            )

            cv2.imshow("Calibration Targets", screen)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("n"):
                target_index += 1
                if target_index >= len(target_order):
                    target_index = 0
                    random.shuffle(target_order)
                    message = "New randomized target pass. Look at the target."
                else:
                    message = "Look at the target, then press SPACE."
                last_capture = ""

            elif key == ord(" "):
                if len(feature_history) < feature_history.maxlen:
                    message = "Hold still with face visible for 15 frames, then retry."
                    continue
                stable = np.asarray(feature_history, dtype=float)
                if not np.isfinite(stable).all():
                    message = "Invalid iris readings. Hold still and retry."
                    continue
                # Median suppresses occasional noisy frames while retaining the
                # existing four-feature CSV schema and target labels.
                gaze_features = np.median(stable, axis=0)

                tx, ty = targets[target_index]

                row = [
                    time.time(),
                    target_index + 1,
                    tx,
                    ty,
                    *gaze_features
                ]

                with open(CSV_PATH, "a", newline="") as file:
                    csv.writer(file).writerow(row)

                # Update counters
                readings_per_target[tid] += 1
                session_readings[tid] += 1

                last_capture = (
                    f"Target {tid} recorded "
                    f"({readings_per_target[tid]} total)"
                )
                message = last_capture
                print(last_capture, row)

finally:
    camera.release()
    cv2.destroyAllWindows()

# Print summary of readings per target
print(f"\nCalibration readings saved to: {CSV_PATH}")
print("\n--- Session Summary ---")
for tid in range(1, 10):
    print(
        f"  Target {tid}: "
        f"{session_readings[tid]} this session, "
        f"{readings_per_target[tid]} total"
    )
print(f"  Total this session: {sum(session_readings.values())}")
