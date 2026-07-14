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
USE_FIXED_TEMPERATURE = False  # If True, use FIXED_TEMPERATURE for every annotation. If False, sample from [TEMPERATURE_MIN, TEMPERATURE_MAX].
FIXED_TEMPERATURE = 0.7
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 1.0

# ---------------------------------------------------------------------------
# Annotation parameters
# ---------------------------------------------------------------------------
ANNOTATION_TYPE = "specificity"  # "reality_monitoring" or "specificity" for now
ANNOTATION_SCORE_MIN = 0
ANNOTATION_SCORE_MAX = 2
ANNOTATIONS_PER_STATEMENT = 10 # how many times each statement is annotated
N_STATEMENTS = 330 # number of statements to sample from the dataset for annotation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = "data/HIP_2022_id.csv"
RESULTS_DIR  = "results"

