from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
INTERIM_DATA = DATA_DIR / "interim"
PROCESSED_DATA = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
ARTIFACTS = MODELS_DIR / "artifacts"
REPORTS = MODELS_DIR / "reports"