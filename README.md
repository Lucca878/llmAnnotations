# LLM annotations

Structured LLM annotation pipeline for CBCA, Reality Monitoring, and Verifiability cues using OpenRouter models.

## Before you run

Create output folders first:

```bash
mkdir -p results results/test
```

General project structure:

```text
deceptionAnnotationsLLM/
├─ data/
│  └─ HIP_2022_id.csv
├─ results/
│  └─ test/
├─ src/
│  ├─ app.py
│  ├─ config.py
│  ├─ dao.py
│  ├─ llm_client.py
│  ├─ prompts.py
│  ├─ schemas.py
│  └─ utility.py
├─ environment.yml
├─ requirements.txt
└─ README.md
```

## How the project works

For each run, using [data/HIP_2022_id.csv](data/HIP_2022_id.csv), the app:

1. Samples statements without replacement.
2. Selects the configured annotation type(s): CBCA, Reality Monitoring, Verifiability, or all three.
3. Builds the exact instrument-specific prompt for each sampled statement.
4. Sends the prompt to OpenRouter with the corresponding structured JSON schema.
5. Saves one output CSV per annotation type.
6. For CBCA, calculates criterion-level Krippendorff alpha after all replicates are collected.

## Project structure

- [src/config.py](src/config.py): all runtime configuration
- [src/app.py](src/app.py): main annotation loop and CLI
- [src/prompts.py](src/prompts.py): exact CBCA, RM, and VA prompt builders
- [src/schemas.py](src/schemas.py): OpenRouter structured-output schemas
- [src/llm_client.py](src/llm_client.py): OpenRouter request and JSON parsing
- [src/utility.py](src/utility.py): dataset sampling, CSV writing, and CBCA alpha
- [src/dao.py](src/dao.py): `AnnotationRecord` data model
- [data/HIP_2022_id.csv](data/HIP_2022_id.csv): input dataset

## Requirements

From [requirements.txt](requirements.txt):

- openrouter
- python-dotenv
- pandas
- numpy
- krippendorff

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

## API key

Create a `.env` in project root:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

`config.py` loads `.env` automatically.

## Configuration

Edit [src/config.py](src/config.py):

- `LLM_ARCHITECTURE`: active model name + OpenRouter model id
- `ANNOTATION_TYPE`: `cbca`, `reality_monitoring`, `verifiability`, or `all`
- `ANNOTATIONS_PER_STATEMENT`: repeat annotations per statement
- `N_STATEMENTS`: number of statements to sample (values above dataset size use all statements)
- `USE_FIXED_TEMPERATURE` and `FIXED_TEMPERATURE` or `TEMPERATURE_MIN` / `TEMPERATURE_MAX`
- `CBCA_EVENT_DOMAIN`: optional domain guidance for CBCA criteria C11 and C19
- `CBCA_PARTICIPANT`: optional participant definition for CBCA criterion C13
- `CBCA_CALIBRATION_EXAMPLES`: optional CBCA-specific calibration examples
- `RM_RATING_MIN` / `RM_RATING_MAX`: Reality Monitoring rating range
- `RM_INCLUDE_C6_C7`: whether Reality Monitoring C6 and C7 are scored or returned as null
- `RM_CALIBRATION_EXAMPLES`: optional RM-specific calibration examples
- `VA_INFORMATION_PROTOCOL`: optional VA metadata, `informed`, `uninformed`, or None
- `VA_IDENTIFIABILITY_THRESHOLD`: optional VA threshold guidance
- `VA_GRANULARITY`: optional VA detail granularity guidance
- `VA_CALIBRATION_EXAMPLES`: optional VA-specific calibration examples
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

### Annotate all statements

Set `--n-statements` to a value above the dataset size. The pipeline will use all statements automatically.

### Test mode

```bash
python src/app.py --test
```

In `--test` mode, it caps to 3 statements x 2 annotations and writes under `results/test/`.

### Optional max tokens

```bash
python src/app.py --max-tokens 300
```

## Replicates and reliability

The model is treated as one coder among several. The default configuration runs three independent
annotations per statement at temperature `0.7`:

```python
ANNOTATIONS_PER_STATEMENT = 3
USE_FIXED_TEMPERATURE = True
FIXED_TEMPERATURE = 0.7
```

Each replicate is saved as a separate CSV row. The `annotation_number_per_statement` column identifies
the replicate, and the `annotation_id` identifies the individual annotation.

The replicate spread is retained for downstream reliability analysis. For CBCA analysis, use:

- the modal or median score across replicates as a stable summary
- ICC(2,k) for the summed CBCA score
- Krippendorff's alpha separately for each criterion
- per-criterion alpha together with per-criterion prevalence

Low-prevalence criteria such as C10, C16, C17, and C18 can have deceptively high percentage agreement
while alpha is poor, so both prevalence and alpha should be reported.

## Output

Each run creates a fresh CSV:

- `results/<architecture>_<annotation_type>_<timestamp>.csv`

All output files share run and statement metadata. CBCA output files include:

- `statement_id`, `speaker_word_count`, `codeable`, `codeability_note`
- Separate columns for each criterion: `C1_score` through `C19_score`
- Separate `C1_instance_count`, `C1_evidence`, `C1_justification`, `C1_confidence`, and `C1_alpha` columns, repeated for C2 through C19
- run metadata such as `session_id`, `annotation_id`, `temperature`, and timestamps

## Structured Output Enforcement

Enforcement is handled in [src/llm_client.py](src/llm_client.py):

1. The OpenRouter call uses `response_format` -> strict JSON schema.
2. Parses JSON schema into useable object.
3. If the call/parsing fails, it retries up to 3 times.

If it fails 3 times, an error is raised.

## Annotation types

Set `ANNOTATION_TYPE` in [src/config.py](src/config.py):

- `cbca`: run Criteria-Based Content Analysis only
- `reality_monitoring`: run Reality Monitoring only
- `verifiability`: run the Verifiability Approach only
- `all`: run all three prompts and write one separate CSV per instrument

CBCA files include flattened `C1_score` through `C19_score` columns plus evidence, justification,
confidence, and criterion alpha columns. Reality Monitoring files include flattened `C1_score`
through `C8_score` columns plus the C2 modality counts. Verifiability files include
`scenario_verifiable`, `scenario_note`, `information_protocol`, and the extracted `details` JSON.

## Notes

- I added `statement_id` to the dataset in [data/HIP_2022_id.csv](data/HIP_2022_id.csv).
- Every run writes a new output file.

