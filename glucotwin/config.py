"""Central configuration: paths, simulation and modelling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "synthetic"
FHIR_DIR = RAW_DIR / "fhir"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

SEED = 2026

# Cohort / simulation
N_PATIENTS = 80
N_DAYS = 14
CALIB_DAYS = 10            # days 1-10 calibrate each patient's twin; days 11-14 are held out
SAMPLE_MIN = 5             # CGM / wearable sampling period (minutes)
SIM_DT_MIN = 1             # ground-truth physiology integration step (minutes)
START_DATE = "2026-09-01"  # synthetic calendar start (Monday)

# Prediction targets
HORIZONS_MIN = (30, 60, 120)
HYPER_THRESHOLD = 180      # mg/dL, upper bound of target range
HYPO_THRESHOLD = 70        # mg/dL, lower bound of target range
EVENT_WINDOW_MIN = 120     # alert look-ahead window

# Subject-level split fractions (train / conformal-calibration / test)
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)

# Conformal prediction coverage
INTERVAL_COVERAGE = 0.90


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, FHIR_DIR, ARTIFACT_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
