# GAZE Project

Webcam based gaze estimation and dwell selection prototype for a fixed 1000 x 650 screen with nine targets. This is a research demo, not system wide gaze control.

## Run

Use Python 3.10 to 3.12. Install dependencies with `pip install -r requirements.txt`. Put `face_landmarker.task` in this folder (see `MODEL_DOWNLOAD.md`). Connect a webcam.

- Calibrate: `python calibration.py`
- Launch: `python gaze_inteface.py`
- Evaluate: `python evaluate_calibration_v2.py` and `python evaluate_dataset.py`
- Tests: `python -m unittest discover -s tests -v`

## Results and limits

The included data and interaction logs are shared with participant approval. The report and CSVs contain the evaluation details. Current calibration error is much larger than the 42 px target radius, so the observed selection rate does not demonstrate gaze accuracy. Logged completed and interrupted attempts are incomplete measures of all gaze attempts. Live webcam calibration and inference were not tested in this environment.
