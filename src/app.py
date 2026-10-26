import argparse
import datetime
import time
import uuid

import numpy as np

from config import (
    ANNOTATION_TYPE,
    ANNOTATION_TYPES,
    ANNOTATIONS_PER_STATEMENT,
    CBCA_EVENT_DOMAIN,
    CBCA_PARTICIPANT,
    CBCA_CALIBRATION_EXAMPLES,
    RM_RATING_MIN,
    RM_RATING_MAX,
    RM_INCLUDE_C6_C7,
    RM_CALIBRATION_EXAMPLES,
    VA_INFORMATION_PROTOCOL,
    VA_IDENTIFIABILITY_THRESHOLD,
    VA_GRANULARITY,
    VA_CALIBRATION_EXAMPLES,
    DATASET_PATH,
    FIXED_TEMPERATURE,
    LLM_ARCHITECTURE,
    N_STATEMENTS,
    SEED,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    USE_FIXED_TEMPERATURE,
)
from dao import AnnotationRecord
from llm_client import call_llm
from utility import (
    add_criterion_alpha,
    annotation_prompt,
    append_sequence_to_csv,
    init_results_csv,
    load_dataset,
    sample_statements,
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
    parser = argparse.ArgumentParser(description="Structured LLM annotation")
    parser.add_argument(
        "--n-statements",
        type=int,
        default=N_STATEMENTS,
        help=f"Number of statements to annotate (default: {N_STATEMENTS})",
    )
    parser.add_argument(
        "--annotations-per-statement",
        type=int,
        default=ANNOTATIONS_PER_STATEMENT,
        help=f"Number of annotations per statement (default: {ANNOTATIONS_PER_STATEMENT})",
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

    results_subdir = "test" if args.test else None
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    df = load_dataset(DATASET_PATH)
    sampled = sample_statements(df, args.n_statements, seed=SEED)
    args.n_statements = len(sampled)
    architecture = LLM_ARCHITECTURE["name"]

    if ANNOTATION_TYPE == "all":
        annotation_types = ANNOTATION_TYPES
    else:
        annotation_types = [ANNOTATION_TYPE]

    print(f"\n=== Architecture: {architecture} | Annotations: {', '.join(annotation_types)} ===")
    csv_paths = {
        annotation_type: init_results_csv(
            architecture,
            annotation_type,
            timestamp,
            subdir=results_subdir,
        )
        for annotation_type in annotation_types
    }
    for annotation_type, csv_path in csv_paths.items():
        print(f"{annotation_type} results -> {csv_path}")

    global_idx = 0
    total_annotations = (
        args.n_statements * args.annotations_per_statement * len(annotation_types)
    )
    temperatures = _build_temperatures(total_annotations)

    for statement_idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        statement_id = int(row.get("statement_id", statement_idx))
        statement_text = str(row.get("text", ""))
        original_label = str(row.get("label_standardized", row.get("label_original", "")))

        for annotation_type in annotation_types:
            prompt_text = annotation_prompt(
                annotation_type,
                statement_text,
                statement_id=statement_id,
                event_domain=CBCA_EVENT_DOMAIN,
                participant=CBCA_PARTICIPANT,
                cbca_calibration_examples=CBCA_CALIBRATION_EXAMPLES,
                rm_rating_min=RM_RATING_MIN,
                rm_rating_max=RM_RATING_MAX,
                rm_include_c6_c7=RM_INCLUDE_C6_C7,
                rm_calibration_examples=RM_CALIBRATION_EXAMPLES,
                va_information_protocol=VA_INFORMATION_PROTOCOL,
                va_identifiability_threshold=VA_IDENTIFIABILITY_THRESHOLD,
                va_granularity=VA_GRANULARITY,
                va_calibration_examples=VA_CALIBRATION_EXAMPLES,
            )

            for annotation_number in range(1, args.annotations_per_statement + 1):
                temperature = float(temperatures[global_idx])
                started_at = datetime.datetime.utcnow().isoformat() + "Z"
                t0 = time.time()

                structured = call_llm(
                    architecture=architecture,
                    annotation_type=annotation_type,
                    prompt=prompt_text,
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                )
                structured["speaker_word_count"] = len(statement_text.split())

                received_at = datetime.datetime.utcnow().isoformat() + "Z"
                duration_ms = int((time.time() - t0) * 1000)

                record = AnnotationRecord(
                    session_id=str(uuid.uuid4()),
                    statement_id=statement_id,
                    original_text=statement_text,
                    original_label=original_label,
                    annotation_id=str(uuid.uuid4()),
                    annotation_number_per_statement=annotation_number,
                    annotation_score="",
                    annotation_type=annotation_type,
                    rationale="",
                    llm_architecture=architecture,
                    temperature=temperature,
                    started_at=started_at,
                    received_at=received_at,
                    annotation_duration_ms=duration_ms,
                    annotation_prompt=prompt_text,
                    annotation_payload=structured,
                )
                append_sequence_to_csv(
                    csv_paths[annotation_type],
                    record,
                    annotation_type=annotation_type,
                )

                print(
                    f"  [{architecture}] {global_idx + 1}/{total_annotations} | "
                    f"statement={statement_idx}/{args.n_statements} | "
                    f"type={annotation_type} | "
                    f"annotation={annotation_number}/{args.annotations_per_statement} | "
                    f"temp={temperature:.3f} | duration={duration_ms}ms"
                )
                global_idx += 1

    for annotation_type, csv_path in csv_paths.items():
        if annotation_type == "cbca":
            add_criterion_alpha(csv_path, args.annotations_per_statement)
            print(f"Criterion alpha saved -> {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
