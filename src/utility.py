import os

import pandas as pd
import numpy as np

from config import RESULTS_DIR, ANNOTATION_SCORE_MIN, ANNOTATION_SCORE_MAX


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path):
    return pd.read_csv(path).reset_index(drop=True)


def sample_statements(df, n, seed):
    """Sample statements without replacement; oversize n returns all rows."""
    if n >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=n, replace=False)
    return df.iloc[indices].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

#This is just a placeholder for now and not using any actual codebook.
def _annotation_prompt_reality_monitoring(statement_text: str) -> str:
    """Build a strict JSON prompt for reality monitoring annotations."""
    return (
        "Criterion: Reality Monitoring\n"
        "Definition: Higher scores indicate more perceptual, temporal, and spatial detail indicating lived experience.\n"
        f"Scale: Integer {ANNOTATION_SCORE_MIN}-{ANNOTATION_SCORE_MAX} where "
        f"{ANNOTATION_SCORE_MIN} means very low reality monitoring and {ANNOTATION_SCORE_MAX} means very high reality monitoring.\n\n"
        "Output format requirements:\n"
        "- Return valid JSON only (no markdown, no extra text).\n"
        "- Use exactly these keys: annotation_type, annotation_score, rationale.\n"
        "- annotation_type must be 'reality_monitoring'.\n"
        f"- annotation_score must be an integer from {ANNOTATION_SCORE_MIN} to {ANNOTATION_SCORE_MAX}.\n"
        "- rationale must be a short explanation (max 20 words).\n\n"
        "JSON schema:\n"
        f"{{\"annotation_type\": \"reality_monitoring\", \"annotation_score\": {ANNOTATION_SCORE_MIN}-{ANNOTATION_SCORE_MAX}, \"rationale\": \"string\"}}\n\n"
        "Statement to score:\n"
        f"{statement_text}"
    )

def _annotation_prompt_specificity(statement_text: str) -> str:
    """Build a strict JSON prompt for specificity annotations."""
    return (
        "Criterion: Specificity\n"
        "Definition: Higher scores indicate more specific information.\n"
        f"Scale: Integer {ANNOTATION_SCORE_MIN}-{ANNOTATION_SCORE_MAX} where "
        f"{ANNOTATION_SCORE_MIN} means very low specificity and {ANNOTATION_SCORE_MAX} means very high specificity.\n\n"
        "Output format requirements:\n"
        "- Return valid JSON only (no markdown, no extra text).\n"
        "- Use exactly these keys: annotation_type, annotation_score, rationale.\n"
        "- annotation_type must be 'specificity'.\n"
        f"- annotation_score must be an integer from {ANNOTATION_SCORE_MIN} to {ANNOTATION_SCORE_MAX}.\n"
        "- rationale must be a short explanation (max 20 words).\n\n"
        "JSON schema:\n"
        f"{{\"annotation_type\": \"specificity\", \"annotation_score\": {ANNOTATION_SCORE_MIN}-{ANNOTATION_SCORE_MAX}, \"rationale\": \"string\"}}\n\n"
        "Statement to score:\n"
        f"{statement_text}"
    )

_ANNOTATION_PROMPT_BUILDERS = {
    "reality_monitoring": _annotation_prompt_reality_monitoring,
    "specificity": _annotation_prompt_specificity,  # Placeholder for now
}


def annotation_prompt(annotation_type: str, statement_text: str) -> str:
    """Return the user prompt associated with the requested annotation type."""
    builder = _ANNOTATION_PROMPT_BUILDERS.get(annotation_type)
    if builder is None:
        supported = ", ".join(sorted(_ANNOTATION_PROMPT_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported annotation_type '{annotation_type}'. Supported values: {supported}."
        )
    return builder(statement_text)

def system_prompt():
    """Return the system prompt for the LLM."""
    return (
        "You are a forensic expert trained to annotate deceptive and truthful linguistic cues in statements.\n\n"
        "##TASK##\n"
        "You will be given a statement and you must score deceptive cues and truthful cues on a scale. "
        "Both the scale and criteria will be provided."
    )

# ---------------------------------------------------------------------------
# CSV output (wide format matching the human all_sessions.csv structure)
# ---------------------------------------------------------------------------

def _build_columns():
    cols = [
        "session_id", "statement_id", "original_text", "original_label", "annotation_id", "annotation_number_per_statement", "annotation_score", "annotation_type", "rationale", "llm_architecture", "temperature", "started_at", "received_at", "annotation_duration_ms", "system_prompt", "annotation_prompt"
    ]
    return cols


def init_results_csv(architecture, annotation_type, timestamp, subdir=None):
    """Create an empty results CSV and return its path."""
    out_dir = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{architecture}_{annotation_type}_{timestamp}.csv")
    pd.DataFrame(columns=_build_columns()).to_csv(path, index=False)
    return path


def append_sequence_to_csv(path, sequence):
    """Append one completed annotation as a row to the results CSV."""
    row = {
        "session_id": getattr(sequence, "session_id", ""),
        "statement_id": getattr(sequence, "statement_id", ""),
        "original_text": getattr(sequence, "original_text", ""),
        "original_label": getattr(sequence, "original_label", ""),
        "annotation_id": getattr(sequence, "annotation_id", ""),
        "annotation_number_per_statement": getattr(sequence, "annotation_number_per_statement", ""),
        "annotation_score": getattr(sequence, "annotation_score", ""),
        "annotation_type": getattr(sequence, "annotation_type", ""),
        "rationale": getattr(sequence, "rationale", ""),
        "llm_architecture": getattr(sequence, "llm_architecture", ""),
        "temperature": round(float(getattr(sequence, "temperature", 0.0)), 4),
        "started_at": getattr(sequence, "started_at", getattr(sequence, "session_start", "")),
        "received_at": getattr(sequence, "received_at", getattr(sequence, "session_end", "")),
        "annotation_duration_ms": getattr(
            sequence,
            "annotation_duration_ms",
            getattr(sequence, "total_duration_ms", ""),
        ),
        "system_prompt": getattr(sequence, "system_prompt", ""),
        "annotation_prompt": getattr(sequence, "annotation_prompt", ""),
    }

    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=False)
