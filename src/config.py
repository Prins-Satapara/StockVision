from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

CLEANED_DATA_DIR = DATA_DIR / "cleaned"

IMAGES_DIR = PROJECT_ROOT / "images"

REPORTS_DIR = PROJECT_ROOT / "reports"
