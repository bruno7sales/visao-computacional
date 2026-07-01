from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "modelos"
RESULTS_DIR = BASE_DIR / "resultados"

MODEL_PATH = MODELS_DIR / "modelo_cnn.h5"
CLASSES_PATH = MODELS_DIR / "classes.json"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
RANDOM_SEED = 42
