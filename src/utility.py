import json
import os

import krippendorff
import numpy as np
import pandas as pd

from config import RESULTS_DIR
from prompts import (
    build_cbca_prompt,
    build_reality_monitoring_prompt,
    build_verifiability_prompt,
)


def load_dataset(path):
    return pd.read_csv(path).reset_index(drop=True)


def sample_statements(df, n, seed):
    """Sample statements without replacement; oversize n returns all rows."""
    if n >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=n, replace=False)
    return df.iloc[indices].reset_index(drop=True)


def annotation_prompt(
    annotation_type: str,
    statement_text: str,
    statement_id=None,
    event_domain: str = "",
    participant: str = "",
    cbca_calibration_examples: str = "",
    rm_rating_min: int = 1,
    rm_rating_max: int = 7,
    rm_include_c6_c7: bool = True,
    rm_calibration_examples: str = "",
    va_information_protocol=None,
    va_identifiability_threshold: str = "",
    va_granularity: str = "",
    va_calibration_examples: str = "",
) -> str:
    """Return the prompt associated with the requested annotation type."""
    builders = {
        "cbca": lambda: build_cbca_prompt(
            statement_text,
            statement_id=statement_id,
            event_domain=event_domain,
            participant=participant,
            calibration_examples=cbca_calibration_examples,
        ),
        "reality_monitoring": lambda: build_reality_monitoring_prompt(
            statement_text,
            statement_id=statement_id,
            rating_min=rm_rating_min,
            rating_max=rm_rating_max,
            include_c6_c7=rm_include_c6_c7,
            calibration_examples=rm_calibration_examples,
        ),
        "verifiability": lambda: build_verifiability_prompt(
            statement_text,
            statement_id=statement_id,
            information_protocol=va_information_protocol,
            identifiability_threshold=va_identifiability_threshold,
            granularity=va_granularity,
            calibration_examples=va_calibration_examples,
        ),
    }
    builder = builders.get(annotation_type)
    if builder is None:
        supported = ", ".join(sorted(builders))
        raise ValueError(
            f"Unsupported annotation_type '{annotation_type}'. Supported values: {supported}."
        )
    return builder()


def _criterion_columns(count):
    columns = []
    for criterion_id in range(1, count + 1):
        columns.extend([
            f"C{criterion_id}_score",
            f"C{criterion_id}_instance_count",
            f"C{criterion_id}_evidence",
            f"C{criterion_id}_justification",
            f"C{criterion_id}_confidence",
        ])
    return columns


def _build_columns(annotation_type):
    cols = [
        "session_id",
        "statement_id",
        "original_text",
        "original_label",
        "annotation_id",
        "annotation_number_per_statement",
    ]
    if annotation_type == "cbca":
        cols.extend(["instrument", "speaker_word_count", "codeable", "codeability_note"])
        cols.extend(_criterion_columns(19))
        cols.extend(f"C{criterion_id}_alpha" for criterion_id in range(1, 20))
    elif annotation_type == "reality_monitoring":
        cols.extend(["instrument", "speaker_word_count", "codeable", "codeability_note"])
        for criterion_id in range(1, 9):
            cols.extend([
                f"C{criterion_id}_score",
                f"C{criterion_id}_instance_count",
            ])
            if criterion_id == 2:
                cols.extend([
                    "C2_visual_count",
                    "C2_auditory_count",
                    "C2_olfactory_count",
                    "C2_gustatory_count",
                    "C2_tactile_count",
                ])
            cols.extend([
                f"C{criterion_id}_evidence",
                f"C{criterion_id}_justification",
                f"C{criterion_id}_confidence",
            ])
    elif annotation_type == "verifiability":
        cols.extend([
            "instrument",
            "speaker_word_count",
            "scenario_verifiable",
            "scenario_note",
            "information_protocol",
            "details",
        ])
    else:
        raise ValueError(f"Unsupported annotation_type '{annotation_type}'.")
    cols.extend([
        "llm_architecture",
        "temperature",
        "started_at",
        "received_at",
        "annotation_duration_ms",
        "annotation_prompt",
    ])
    return cols


def init_results_csv(architecture, annotation_type, timestamp, subdir=None):
    """Create an empty results CSV and return its path."""
    out_dir = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{architecture}_{annotation_type}_{timestamp}.csv")
    pd.DataFrame(columns=_build_columns(annotation_type)).to_csv(path, index=False)
    return path


def _criterion_rows(payload):
    row = {}
    for criterion in payload.get("criteria", []):
        criterion_id = criterion["id"]
        prefix = f"C{criterion_id}"
        score = criterion.get("score", criterion.get("rating", ""))
        row[f"{prefix}_score"] = "" if score is None else score
        row[f"{prefix}_instance_count"] = criterion.get("instance_count", "")
        row[f"{prefix}_evidence"] = json.dumps(criterion.get("evidence", []), ensure_ascii=False)
        row[f"{prefix}_justification"] = criterion.get("justification", "")
        row[f"{prefix}_confidence"] = criterion.get("confidence", "")
        modality_counts = criterion.get("modality_counts")
        if modality_counts:
            for modality, count in modality_counts.items():
                row[f"{prefix}_{modality}_count"] = count
    return row


def append_sequence_to_csv(path, sequence, annotation_type=None):
    """Append one completed annotation as a row to the results CSV."""
    annotation_type = annotation_type or sequence.annotation_type
    row = {
        "session_id": getattr(sequence, "session_id", ""),
        "statement_id": getattr(sequence, "statement_id", ""),
        "original_text": getattr(sequence, "original_text", ""),
        "original_label": getattr(sequence, "original_label", ""),
        "annotation_id": getattr(sequence, "annotation_id", ""),
        "annotation_number_per_statement": getattr(sequence, "annotation_number_per_statement", ""),
        "llm_architecture": getattr(sequence, "llm_architecture", ""),
        "temperature": round(float(getattr(sequence, "temperature", 0.0)), 4),
        "started_at": getattr(sequence, "started_at", ""),
        "received_at": getattr(sequence, "received_at", ""),
        "annotation_duration_ms": getattr(sequence, "annotation_duration_ms", ""),
        "annotation_prompt": getattr(sequence, "annotation_prompt", ""),
    }

    payload = getattr(sequence, "annotation_payload", {})
    if annotation_type in ("cbca", "reality_monitoring"):
        row.update({
            "instrument": payload.get("instrument", ""),
            "speaker_word_count": payload.get("speaker_word_count", ""),
            "codeable": payload.get("codeable", ""),
            "codeability_note": payload.get("codeability_note", ""),
        })
        row.update(_criterion_rows(payload))
    elif annotation_type == "verifiability":
        row.update({
            "instrument": payload.get("instrument", ""),
            "speaker_word_count": payload.get("speaker_word_count", ""),
            "scenario_verifiable": payload.get("scenario_verifiable", ""),
            "scenario_note": payload.get("scenario_note", ""),
            "information_protocol": payload.get("information_protocol", ""),
            "details": json.dumps(payload.get("details", []), ensure_ascii=False),
        })

    columns = _build_columns(annotation_type)
    pd.DataFrame([{column: row.get(column, "") for column in columns}]).to_csv(
        path,
        mode="a",
        index=False,
        header=False,
    )


def add_criterion_alpha(path, n_replicates):
    """Add run-level Krippendorff alpha values for the 19 CBCA criteria."""
    results = "none"
    if n_replicates > 1:
        df = pd.read_csv(path)
        grouped = {}
        for _, row in df.iterrows():
            statement_id = row["statement_id"]
            replicate = int(row["annotation_number_per_statement"])
            scores = {
                criterion_id: row[f"C{criterion_id}_score"]
                for criterion_id in range(1, 20)
            }
            grouped.setdefault(statement_id, {})[replicate] = scores

        alphas = {}
        for criterion_id in range(1, 20):
            reliability_data = []
            complete_units = [
                replicates
                for replicates in grouped.values()
                if len(replicates) == n_replicates
                and all(
                    pd.notna(replicates[replicate][criterion_id])
                    for replicate in range(1, n_replicates + 1)
                )
            ]
            for replicate in range(1, n_replicates + 1):
                reliability_data.append([
                    int(replicates[replicate][criterion_id])
                    for replicates in complete_units
                ])

            if reliability_data and reliability_data[0]:
                try:
                    alpha = krippendorff.alpha(
                        reliability_data=reliability_data,
                        level_of_measurement="ordinal",
                    )
                    alphas[f"C{criterion_id}"] = None if pd.isna(alpha) else round(float(alpha), 2)
                except ValueError:
                    alphas[f"C{criterion_id}"] = None
            else:
                alphas[f"C{criterion_id}"] = None
        results = json.dumps(alphas)

    df = pd.read_csv(path)
    if results == "none":
        for criterion_id in range(1, 20):
            df[f"C{criterion_id}_alpha"] = "none"
    else:
        alphas = json.loads(results)
        for criterion_id in range(1, 20):
            df[f"C{criterion_id}_alpha"] = alphas[f"C{criterion_id}"]
    df.to_csv(path, index=False)
