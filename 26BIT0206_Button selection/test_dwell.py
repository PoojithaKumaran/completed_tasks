"""Unit tests for the dwell logic (no GUI needed). Run: python test_dwell.py"""
from dwell_selection import *

B = {"WATER": (0, 0, 100, 100), "FOOD": (200, 0, 100, 100)}

def run(det, samples):
    out = []
    for t, x, y in samples:
        out += [(e.type, e.button) for e in det.update(x, y, t)]
    return out

def steady(x, y, t0, t1, dt=0.05):
    n = int(round((t1 - t0) / dt))
    return [(t0 + i * dt, x, y) for i in range(n + 1)]

# 1. continuous gaze selects after dwell time
for d in (3, 4, 5):
    ev = run(DwellDetector(B, dwell_time=d), steady(50, 50, 0, d + 0.1))
    assert ("BUTTON_SELECTED", "WATER") in ev, d
# 2. not selected before threshold
ev = run(DwellDetector(B, dwell_time=3), steady(50, 50, 0, 2.9))
assert ("BUTTON_SELECTED", "WATER") not in ev
# 3. leaving resets the timer
det = DwellDetector(B, dwell_time=3, grace_period=0.1)
ev = run(det, steady(50, 50, 0, 2) + steady(150, 150, 2.05, 2.5) + steady(50, 50, 2.55, 4.5))
assert ("GAZE_LEAVE", "WATER") in ev and ("BUTTON_SELECTED", "WATER") not in ev
# 4. short blink (invalid samples) does not reset
det = DwellDetector(B, dwell_time=3, grace_period=0.2)
ev = run(det, steady(50, 50, 0, 1) + [(1.05, None, None), (1.10, None, None)] + steady(50, 50, 1.15, 3.1))
assert ("BUTTON_SELECTED", "WATER") in ev
# 5. switching buttons selects the right one
det = DwellDetector(B, dwell_time=3, grace_period=0.1)
ev = run(det, steady(50, 50, 0, 1) + steady(250, 50, 1.2, 4.5))
assert ("BUTTON_SELECTED", "FOOD") in ev and ("BUTTON_SELECTED", "WATER") not in ev
# 6. cooldown prevents instant re-selection
det = DwellDetector(B, dwell_time=1, cooldown=2)
ev = run(det, steady(50, 50, 0, 2.5))
assert ev.count(("BUTTON_SELECTED", "WATER")) == 1

# 7. edge case: gaze exactly on borders and corners
det_border = DwellDetector(B, dwell_time=1.0)
# Top-left corner (0, 0)
assert det_border.button_at(0, 0) == "WATER"
# Bottom-right corner (100, 100)
assert det_border.button_at(100, 100) == "WATER"
# Center of button
assert det_border.button_at(50, 50) == "WATER"
# Adjacent button boundary test
B_ADJACENT = {"LEFT": (0, 0, 100, 100), "RIGHT": (100, 0, 100, 100)}
det_adj = DwellDetector(B_ADJACENT, dwell_time=1.0)
# Exactly on shared boundary x=100
shared_btn = det_adj.button_at(100, 50)
assert shared_btn in ("LEFT", "RIGHT")
# Continuous dwell on border selects without crashing
ev_border = run(det_adj, steady(100, 50, 0, 1.1))
assert any(t == "BUTTON_SELECTED" for t, _ in ev_border)

# 8. edge case: overlapping and non-monotonic timestamps
det_time = DwellDetector(B, dwell_time=1.0)
overlapping_samples = [
    (0.0, 50, 50),
    (0.0, 50, 50),     # duplicate timestamp
    (0.5, 50, 50),
    (0.4, 50, 50),     # backward / non-monotonic timestamp
    (1.1, 50, 50),
]
ev_time = run(det_time, overlapping_samples)
assert ("BUTTON_SELECTED", "WATER") in ev_time

# 9. edge case: rapid switching between adjacent buttons
det_switch = DwellDetector(B, dwell_time=1.0, grace_period=0.1)
# 0.5s on WATER, then jumps directly to FOOD for 1.2s
rapid_switch_samples = steady(50, 50, 0.0, 0.5) + steady(250, 50, 0.55, 1.8)
ev_switch = run(det_switch, rapid_switch_samples)
assert ("GAZE_LEAVE", "WATER") in ev_switch
assert ("BUTTON_SELECTED", "FOOD") in ev_switch
assert ("BUTTON_SELECTED", "WATER") not in ev_switch

# 10. edge case: invalid values (NaN, inf, None, non-numeric strings)
det_invalid = DwellDetector(B, dwell_time=1.0, grace_period=0.1)
assert det_invalid.button_at(float("nan"), 50) is None
assert det_invalid.button_at(float("inf"), 50) is None
invalid_samples = [
    (0.0, 50, 50),
    (0.1, float("nan"), 50),       # NaN x
    (0.2, 50, float("inf")),       # Inf y
    (0.3, "invalid", 50),          # string
    (0.4, None, None),             # blink / lost
    (0.5, 50, 50),
    (1.5, 50, 50),                 # complete after grace period recovers
]
ev_inv = run(det_invalid, invalid_samples)
# Due to invalid samples spanning > grace_period (0.1 to 0.4 = 0.3s > 0.1s), WATER should leave and reset
assert ("GAZE_LEAVE", "WATER") in ev_inv

# 11. edge case: dwell time parameter validation
det_val = DwellDetector(B)
try:
    det_val.set_dwell_time(-1)
    assert False, "Should raise ValueError for negative dwell time"
except ValueError:
    pass
try:
    det_val.set_dwell_time(0)
    assert False, "Should raise ValueError for 0 dwell time"
except ValueError:
    pass

# 12. GazeSource adapters
mouse_src = MouseGazeSource()
mouse_src.set_position(120, 240)
x, y, t = mouse_src.get_sample()
assert x == 120 and y == 240 and t > 0

cb_mouse = MouseGazeSource(coord_getter=lambda: (300, 400))
cx, cy, ct = cb_mouse.get_sample()
assert cx == 300 and cy == 400

webcam_src = WebcamGazeSource()
wx, wy, wt = webcam_src.get_sample()
assert wx is None and wy is None and wt > 0

tobii_src = TobiiGazeSource()
tx, ty, tt = tobii_src.get_sample()
assert tx is None and ty is None and tt > 0

# 13. CSV Logger verification (Attempt mode vs Raw mode)
import os
import tempfile
import csv

# Test Attempt mode: writes ONE row per attempt
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
    tmp_path = tmp.name

try:
    logger = DwellLogger(path=tmp_path, user_id="TEST_U01", mode="attempt")
    # Simulate a successful selection attempt (multiple samples, then selected)
    logger.log(50, 50, "WATER", 0.5, False, 3.0)
    logger.log(52, 48, "WATER", 1.5, False, 3.0)
    logger.log(51, 49, "WATER", 3.0, True, 3.0)

    # Simulate an abandoned attempt (gaze leaves after 1.2s)
    logger.log(200, 50, "FOOD", 0.5, False, 3.0)
    logger.log(202, 52, "FOOD", 1.2, False, 3.0)
    logger.conclude_attempt(False, 3.0, fixation_time=1.2, target="FOOD")
    logger.close()

    with open(tmp_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2, f"Attempt mode should produce exactly 2 rows, got {len(reader)}"
        assert reader[0]["target_button"] == "WATER"
        assert reader[0]["successful_selection"] == "1"
        assert reader[0]["dwell_setting"] == "3.0"
        assert reader[1]["target_button"] == "FOOD"
        assert reader[1]["successful_selection"] == "0"
        assert reader[1]["fixation_time"] == "1.2"
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

# Test Raw mode: writes every sample
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp2:
    tmp2_path = tmp2.name

try:
    raw_logger = DwellLogger(path=tmp2_path, user_id="TEST_U02", mode="raw")
    raw_logger.log(50, 50, "WATER", 0.1, False, 3.0)
    raw_logger.log(50, 50, "WATER", 0.2, False, 3.0)
    raw_logger.log(50, 50, "WATER", 0.3, True, 3.0)
    raw_logger.close()

    with open(tmp2_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3, f"Raw mode should produce 3 rows, got {len(reader)}"
finally:
    if os.path.exists(tmp2_path):
        os.remove(tmp2_path)

print("All dwell tests passed (including edge cases, adapters, and logger verification)")

