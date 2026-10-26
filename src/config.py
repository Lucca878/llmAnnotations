import os
from dotenv import load_dotenv

# Load .env from project root before reading any env vars
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# ---------------------------------------------------------------------------
# LLM configuration (specifically for OpenRouter)
# OpenRouter: https://openrouter.ai/models
# Change these values to switch the model, global seed, or temperature configurations.
# ---------------------------------------------------------------------------
LLM_ARCHITECTURE = {
    "name": "llama3.3b-instruct",
    "model": "meta-llama/llama-3.3-70b-instruct",
}
SEED = 42
USE_FIXED_TEMPERATURE = True  # If True, use FIXED_TEMPERATURE for every annotation. If False, sample from [TEMPERATURE_MIN, TEMPERATURE_MAX].
FIXED_TEMPERATURE = 0.7
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 1.0

# ---------------------------------------------------------------------------
# Annotation parameters
# ---------------------------------------------------------------------------
ANNOTATION_TYPE = "all"  # "cbca", "reality_monitoring", "verifiability", or "all"
ANNOTATION_TYPES = ["cbca", "reality_monitoring", "verifiability"]
ANNOTATIONS_PER_STATEMENT = 1 # replicate annotations per statement
N_STATEMENTS = 100 # number of statements to sample from the dataset for annotation
CBCA_EVENT_DOMAIN = ""
CBCA_PARTICIPANT = ""
CBCA_CALIBRATION_EXAMPLES = ""
RM_RATING_MIN = 1
RM_RATING_MAX = 7
RM_INCLUDE_C6_C7 = True
RM_CALIBRATION_EXAMPLES = ""
VA_INFORMATION_PROTOCOL = None
VA_IDENTIFIABILITY_THRESHOLD = ""
VA_GRANULARITY = ""
VA_CALIBRATION_EXAMPLES = ""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = "data/HIP_2022_id.csv"
RESULTS_DIR  = "results"

