# LLM Annotations

Structured LLM annotation pipeline for deception-related cues (i.e., `reality_monitoring`) using OpenRouter models.

## What This Project Does

For each run, using [data/HIP_2022_id.csv](data/HIP_2022_id.csv), the app:

1. Samples statements without replacement.
2. For each sampled statement and each repeat annotation, builds the annotation prompt.
3. Sends system + user prompts to OpenRouter.
4. Enforces structured JSON output using OpenRouter `response_format` JSON schema.
5. Writes one row per annotation to a CSV in [results](results).

## Project Structure

- [src/config.py](src/config.py): all runtime configuration
- [src/app.py](src/app.py): main annotation loop
- [src/llm_client.py](src/llm_client.py): OpenRouter request + structured validation call
- [src/utility.py](src/utility.py): prompt builders, CSV helpers
- [src/dao.py](src/dao.py): `AnnotationRecord` data model
- [data/HIP_2022_id.csv](data/HIP_2022_id.csv): input dataset

## Requirements

From [requirements.txt](requirements.txt):

- openrouter
- python-dotenv
- pandas
- numpy
- pydantic

## Setup

### Option A: pip/venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate llm-annotations
```

## API Key

Create a `.env` in project root:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

`config.py` loads `.env` automatically.

## Configuration

Edit [src/config.py](src/config.py):

- `LLM_ARCHITECTURE`: active model name + OpenRouter model id
- `ANNOTATION_TYPE`: selects which user prompt template is used
- `ANNOTATION_SCORE_MIN` / `ANNOTATION_SCORE_MAX`: score range used by prompt and structured-output schema
- `ANNOTATIONS_PER_STATEMENT`: repeat annotations per statement
- `N_STATEMENTS`: number of statements to sample (values above dataset size use all statements)
- `USE_FIXED_TEMPERATURE` and `FIXED_TEMPERATURE` or `TEMPERATURE_MIN` / `TEMPERATURE_MAX`
- `DATASET_PATH` and `RESULTS_DIR`

## Run

### Default run

```bash
python src/app.py
```

### Custom batch size

```bash
python src/app.py --n-statements 50 --annotations-per-statement 5
```

### Annotate all unique statements

If you set `--n-statements` to a value above dataset size. The pipeline will use all unique statements automatically.

### Test mode

```bash
python src/app.py --test
```

In `--test` mode, it caps to 3 statements x 2 annotations and writes under `results/test/`.

### Optional max tokens

```bash
python src/app.py --max-tokens 300
```

## Output

Each run creates a fresh CSV:

- `results/<architecture>_<annotation_type>_<timestamp>.csv`

Columns written include:

- `session_id`, `statement_id`, `original_text`, `original_label`
- `annotation_id`, `annotation_number_per_statement`
- `annotation_score`, `annotation_type`, `rationale`
- `llm_architecture`, `temperature`
- `started_at`, `received_at`, `annotation_duration_ms`
- `system_prompt`, `annotation_prompt`

## Structured Output Enforcement

Enforcement is handled in [src/llm_client.py](src/llm_client.py):

1. The OpenRouter call uses `response_format` -> strict JSON schema.
2. Parses JSON schema into useable object.
3. If the call/parsing fails, it retries up to 3 times.

If it fails 3 times, an error is raised.

## Adding a New Annotation Type

To add another criterion:

1. Add a new user prompt in [src/utility.py](src/utility.py).
2. Register it in `_ANNOTATION_PROMPT_BUILDERS`.
3. Set `ANNOTATION_TYPE` in [src/config.py](src/config.py).
4. If score scale differs, update `ANNOTATION_SCORE_MIN` / `ANNOTATION_SCORE_MAX` in [src/config.py](src/config.py).

## Notes

- I added `statement_id` to the dataset in [data/HIP_2022_id.csv](data/HIP_2022_id.csv).
- Every run writes a new output file.

