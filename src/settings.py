from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

rng = 42

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
INTERIM_DATA = DATA_DIR / "interim"
PROCESSED_DATA = DATA_DIR / "processed"

MODELS = ROOT / "models"

model_name = "mlp.pth"

input_size=20
hidden_size=32
output_size=1