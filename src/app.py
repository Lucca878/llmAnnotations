import argparse
import datetime
import os
import time
import uuid

import numpy as np

from config import (
    LLM_ARCHITECTURE,
    ANNOTATION_TYPE,
    ANNOTATIONS_PER_STATEMENT,
    N_STATEMENTS,
    USE_FIXED_TEMPERATURE,
    FIXED_TEMPERATURE,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    SEED,
    DATASET_PATH,
)
from dao import AnnotationRecord
from llm_client import call_llm
from utility import (
    load_dataset,
    sample_statements,
    annotation_prompt,
    system_prompt,
    init_results_csv,
    append_sequence_to_csv,
)

def _build_temperatures(n_annotations: int) -> np.ndarray:
    if USE_FIXED_TEMPERATURE:
        return np.full(n_annotations, FIXED_TEMPERATURE, dtype=float)

    return np.random.default_rng(SEED).uniform(
        TEMPERATURE_MIN,
        TEMPERATURE_MAX,
        size=n_annotations,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Structured LLM annotation"
    )
    parser.add_argument(
        "--n-statements",
        type=int,
        default=N_STATEMENTS,
        help=(
            "Number of unique statements to annotate "
            f"(default: {N_STATEMENTS}; values above dataset size use all unique statements)"
        ),
    )
    parser.add_argument(
        "--annotations-per-statement",
        type=int,
        default=ANNOTATIONS_PER_STATEMENT,
        help=(
            "Number of annotations per statement "
            f"(default: {ANNOTATIONS_PER_STATEMENT})"
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test: 3 statements x 2 annotations each in results/test/.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens passed to the LLM API.",
    )
    args = parser.parse_args()

    if args.test:
        args.n_statements = 3
        args.annotations_per_statement = 2

    results_subdir    = "test" if args.test else None

    if args.test:
        print("[TEST MODE] 3 statements x 2 annotations each, output -> results/test/")
    if args.max_tokens is not None:
        print(f"[MAX TOKENS] using max_tokens={args.max_tokens}")
    if USE_FIXED_TEMPERATURE:
        print(f"[FIXED TEMPERATURE] using {FIXED_TEMPERATURE:.3f} for every annotation")
    else:
        print(f"[UNIFORM TEMPERATURE] sampling from [{TEMPERATURE_MIN:.3f}, {TEMPERATURE_MAX:.3f}]")

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    df = load_dataset(DATASET_PATH)
    sampled = sample_statements(df, args.n_statements, seed=SEED)

    args.n_statements = len(sampled)
    architecture = LLM_ARCHITECTURE["name"]
    total_annotations = args.n_statements * args.annotations_per_statement
    temperatures = _build_temperatures(total_annotations)

    print(f"\n=== Architecture: {architecture} | Annotation: {ANNOTATION_TYPE} ===")

    csv_path = init_results_csv(
        architecture,
        ANNOTATION_TYPE,
        timestamp,
        subdir=results_subdir,
    )
    print(f"Results -> {csv_path}")

    global_idx = 0
    for statement_idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        statement_id = int(row.get("statement_id", statement_idx))
        statement_text = str(row.get("text", ""))
        original_label = str(row.get("label_standardized", row.get("label_original", "")))
        prompt_text = annotation_prompt(ANNOTATION_TYPE, statement_text)
        sys_prompt = system_prompt()

        for annotation_number in range(1, args.annotations_per_statement + 1):
            temperature = float(temperatures[global_idx])
            started_at = datetime.datetime.utcnow().isoformat() + "Z"
            t0 = time.time()

            structured = call_llm(
                architecture=architecture,
                prompt=prompt_text,
                temperature=temperature,
                system_prompt=sys_prompt,
                max_tokens=args.max_tokens,
            )

            received_at = datetime.datetime.utcnow().isoformat() + "Z"
            duration_ms = int((time.time() - t0) * 1000)

            record = AnnotationRecord(
                session_id=str(uuid.uuid4()),
                statement_id=statement_id,
                original_text=statement_text,
                original_label=original_label,
                annotation_id=str(uuid.uuid4()),
                annotation_number_per_statement=annotation_number,
                annotation_score=structured["annotation_score"],
                annotation_type=structured["annotation_type"],
                rationale=structured["rationale"],
                llm_architecture=architecture,
                temperature=temperature,
                started_at=started_at,
                received_at=received_at,
                annotation_duration_ms=duration_ms,
                system_prompt=sys_prompt,
                annotation_prompt=prompt_text,
            )

            append_sequence_to_csv(csv_path, record)

            print(
                f"  [{architecture}] {global_idx + 1}/{total_annotations} | "
                f"statement={statement_idx}/{args.n_statements} | "
                f"annotation={annotation_number}/{args.annotations_per_statement} | "
                f"score={record.annotation_score} | "
                f"temp={temperature:.3f} | "
                f"duration={duration_ms}ms"
            )

            global_idx += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
