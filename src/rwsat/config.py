"""Single source of truth for seeds, paths and thresholds (SPEC: Configuration)."""

from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Impact_of_Remote_Work_on_Mental_Health.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TEST_SIZE = 0.2
N_SPLITS = 5
N_UNIVARIATE_BINS = 5
N_BOOT = 2000
N_PERM = 2000
N_PERM_MODEL = 200
N_IMPORTANCE_REPEATS = 30
ALPHA = 0.05
NOISE_PERMUTATION_P = 0.2
DELTA_NEGLIGIBLE = 0.01
BASELINE_SD_MULTIPLIER = 2
CONFIDENCE_DEVIATION_THRESHOLD = 0.10
MIN_CATEGORY_N = 5
