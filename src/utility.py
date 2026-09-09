import os
import json

import pandas as pd
import numpy as np
import krippendorff

from config import (
    RESULTS_DIR,
    ANNOTATION_SCORE_MIN,
    ANNOTATION_SCORE_MAX,
    CBCA_EVENT_DOMAIN,
    CBCA_PARTICIPANT,
    CBCA_CALIBRATION_EXAMPLES,
)


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

def _annotation_prompt_reality_monitoring(statement_text: str) -> str:
    """Build a complete user prompt for reality-monitoring annotation."""
    return f'''You are a trained content analyst annotating a statement for reality monitoring.

Score only the supplied text. Higher scores indicate more perceptual, temporal, and spatial detail.
Do not assess truthfulness, credibility, or whether the event actually occurred.

Use an integer score from {ANNOTATION_SCORE_MIN} to {ANNOTATION_SCORE_MAX}.
Return valid JSON only with exactly these keys:

{{
    "annotation_type": "reality_monitoring",
    "annotation_score": {ANNOTATION_SCORE_MIN},
    "rationale": "maximum 3 words"
}}

Rules:
- `annotation_type` must be `reality_monitoring`.
- `annotation_score` must be an integer from {ANNOTATION_SCORE_MIN} to {ANNOTATION_SCORE_MAX}.
- `rationale` must contain no more than 3 words.

Statement to annotate:
{statement_text}'''

def _annotation_prompt_specificity(statement_text: str) -> str:
    """Build a complete user prompt for specificity annotation."""
    return f'''You are a trained content analyst annotating a statement for specificity.

Score only the supplied text. Higher scores indicate more specific information.
Do not assess truthfulness, credibility, or whether the event actually occurred.

Use an integer score from {ANNOTATION_SCORE_MIN} to {ANNOTATION_SCORE_MAX}.
Return valid JSON only with exactly these keys:

{{
    "annotation_type": "specificity",
    "annotation_score": {ANNOTATION_SCORE_MIN},
    "rationale": "maximum 3 words"
}}

Rules:
- `annotation_type` must be `specificity`.
- `annotation_score` must be an integer from {ANNOTATION_SCORE_MIN} to {ANNOTATION_SCORE_MAX}.
- `rationale` must contain no more than 3 words.

Statement to annotate:
{statement_text}'''


def _annotation_prompt_cbca(statement_text: str, statement_id=None) -> str:
        """Build the supplied CBCA prompt with runtime corpus values inserted."""
        prompt = r'''You are a trained content analyst applying Criteria-Based Content Analysis (CBCA) to a transcribed
statement. Your job is to detect and score the presence of 19 content criteria. You are a coder, not
an evaluator.

## Hard constraints

1. **You do not assess truthfulness.** Never state, imply, hedge toward, or reason about whether the
statement is true, false, credible, or fabricated. CBCA scores are one input to an assessment made
elsewhere. Any veracity language in your output is an error.
2. **You score the text, not the world.** Do not use outside knowledge to check whether the described
events occurred, are plausible, or are internally believable given what you know about the world.
A criterion is present if the linguistic feature is present in this transcript.
3. **Every score above 0 must be anchored in a verbatim span.** If you cannot quote a contiguous
stretch of the speaker's own words that instantiates the criterion, the score is 0. Quote exactly,
including disfluencies and grammatical errors. Do not repair, paraphrase, or normalise quotes.
4. **Absence is the normal finding.** Most criteria are absent from most statements. Several (C10,
C11, C16, C17, C18) are absent from the large majority. Do not manufacture evidence to avoid a
run of zeros. A page of zeros is a valid and common output.
5. **Score only the speaker's turns.** Ignore interviewer questions, prompts, and back-channels
except as context for deciding whether something was spontaneous (relevant to C14).
6. **Do not compute a total.** Report per-criterion scores only.
7. **Score each criterion independently**, with one exception stated under C2.

## Procedure
Work in two passes.

**Pass 1 — Evidence extraction.** Read the statement once. Then, for each criterion in order 1–19,
scan the whole statement and extract every distinct span that instantiates it. A span is distinct if
it concerns different content; the same content repeated later is not a second instance. Record the
spans before deciding on any score.

**Pass 2 — Scoring.** For each criterion, assign 0, 1, or 2 using the general anchors below, adjusted
by any criterion-specific rule.

- **0 — Absent.** No qualifying span.
- **1 — Present.** At least one clear qualifying span, but the criterion is not a pervasive or
pronounced feature of the account.
- **2 — Strongly present.** The criterion is instantiated repeatedly, or a single instance is
markedly elaborated. For the three whole-statement criteria (C1, C2, C3) the distinction is one of
degree across the account, not of count.
Borderline cases score **down**, not up. If you are weighing 1 against 2, assign 1. If you are
weighing 0 against 1, assign 0 unless the span is unambiguous.

## The 19 criteria

### General characteristics — judged over the statement as a whole
**C1. Logical structure.**
Do the parts of the account fit together without contradicting one another? Judge internal coherence
only: whether the segments are mutually consistent and comprehensible as one account. Do not judge
external plausibility.
*Score 2* if the account is coherent throughout with no internal contradictions.
*Score 1* if broadly coherent with a minor unresolved inconsistency.
*Score 0* if segments contradict each other or the account cannot be assembled into a comprehensible
whole.
*Rule:* Peculiar details, unusual events, and unexpected complications do **not** reduce this score.
Neither does non-chronological telling — that is C2.

**C2. Unstructured production.**
Is the information delivered in a way that is not neatly ordered — digressions, spontaneous shifts of
topic, jumps in time, elements of the same episode surfacing at several separated points rather than
all at once?
*Score 2* if the account is markedly non-linear with information about the same episode distributed
across the telling.
*Score 1* if there are some digressions or shifts against a mostly ordered account.
*Score 0* if the account proceeds in tidy, linear, chronological order.
*Dependency rule:* C2 requires that the account is logically coherent. **If C1 = 0, set C2 = 0.**
*Rule:* Disorder here means order of presentation, not incoherence. An account can be thoroughly
disordered in sequence and still score 2 on C1.

**C3. Quantity of details.**
How specific and rich are the descriptions of persons, objects, actions, locations, and times?
*Score 2* for an account densely specified across several of these dimensions.
*Score 1* for some specific description against a generally sparse account.
*Score 0* for an account pitched almost entirely at the level of summary or generic description.
*Rule:* Count each distinct detail once. Repetitions of a detail already given do not add.
*Rule:* This is a judgment of density relative to what an account of this event could plausibly
contain, not a raw word count. Say so in your justification if the statement is very short.

### Specific contents
**C4. Contextual embedding.**
Is the event anchored in place and time, and connected to the surrounding fabric of the speaker's life — daily routines, what preceded or followed, where in the day or week it sat, other activities going on around it?
Both anchoring and connection matter: bare coordinates ("it was in the kitchen, around three") are weaker than an event located within an ongoing course of activity.

**C5. Descriptions of interactions.**
Are there chains of action and reaction between at least two people — one party acts, the other responds, the first responds to that?
*Rule:* A single one-directional action is not an interaction. The reciprocal structure is what counts.
*Rule:* Reported speech may be part of an interaction chain here and separately scored under C6. Overlap is expected and is not double-counting.

**C6. Reproduction of conversation.**
Is speech reported in something like its original form — direct speech, rendered as uttered — rather than summarised as content?
*Stronger indicator:* wording idiosyncratic to the speaker, or vocabulary the narrator would not ordinarily use.
*Rule:* "He asked me to leave" is a summary and does not qualify. "He said, get out" qualifies.

**C7. Unexpected complications during the incident.**
Was there an unplanned interruption, a difficulty, or something that went wrong or did not proceed as intended during the event?
*Rule:* The complication must fall within the incident, not before or after it.

### Peculiarities of content
**C8. Unusual details.**
Details of persons, objects, or events that are atypical or distinctive, yet realistic and meaningful in this context.
*Rule:* The detail must remain plausible. Bizarre or impossible content does not qualify.
*Rule:* Distinguish from C9: an unusual detail is atypical; a superfluous detail is peripheral. A detail can be both, in which case score it under both.

**C9. Superfluous details.**
Peripheral information connected to the situation but contributing nothing to the substance of what is being reported — descriptive matter the account does not need.
*Rule:* Test by deletion. If removing it leaves the substance of the account intact, it is superfluous.

**C10. Accurately reported details misunderstood.**
Does the speaker describe something accurately at the level of perception while showing they did not grasp its meaning or significance — reporting what was seen or heard but interpreting it wrongly, or not interpreting it at all?
*Rule:* This criterion originates in children's accounts and is rare in adult statements. Score 0 unless the misapprehension is explicit in the text. Do not infer misunderstanding from what an adult would be expected to know.

**C11. Related external associations.**
Does the speaker refer to an event, conversation, or relationship outside the incident itself but connected to it in subject matter — for instance a prior exchange with the same person on a related theme?
`[[CONFIGURE]]` Specify what counts as topically related for this corpus: [[EVENT_DOMAIN]].

**C12. Accounts of subjective mental state.**
Does the speaker report their own feelings, thoughts, or bodily sensations as experienced during the event?
*Rule:* Feelings held at the time of telling do not qualify. Physical sensations count. Evaluative commentary about the event made from the present vantage point does not.

**C13. Attribution of the other party's mental state.**
Does the speaker ascribe thoughts, feelings, intentions, or motives to another person involved, concerning that person's state during the incident?
`[[CONFIGURE]]` Specify which participant is relevant for this corpus: [[CBCA_PARTICIPANT]].
*Rule:* Describing another person's visible behaviour is not attribution of mental state. An inferred inner state is required.

### Motivation-related contents
**C14. Spontaneous corrections.**
Does the speaker, without being prompted, correct, retract, qualify, or add to something said earlier in the account?
*Rule:* A correction produced in response to a direct question or challenge does not qualify. Consult interviewer turns only to establish that nothing elicited it.

**C15. Admitting lack of memory.**
Does the speaker state that they do not remember, do not know, or are unsure about some aspect of the incident?
*Rule:* The gap must concern the incident. Distinguish from C16: this is a declared gap; C16 is doubt about material actually given.

**C16. Raising doubts about one's own testimony.**
Does the speaker volunteer that some part of what they have said may be wrong, or anticipate that it will sound implausible to the listener?
*Rule:* Asserting one's own truthfulness does not qualify — it is the opposite move.

**C17. Self-deprecation.**
Does the speaker describe their own conduct in relation to the incident as wrong, foolish, blameworthy, or otherwise unfavourable?
*Rule:* Must concern the speaker's own behaviour connected to the incident. General negative self-description does not qualify.

**C18. Pardoning the other party.**
Does the speaker excuse, minimise, or decline to attribute blame to the other person at a point where blaming them was available?
*Rule:* Also score the omission where the account presents an obvious occasion for blame and none is offered, provided that occasion is explicit in the text.

### Offence-specific elements
**C19. Details characteristic of the event type.**
Are there elements consistent with what is professionally known about how this kind of event actually unfolds, particularly where those elements run counter to lay expectation?
`[[CONFIGURE]]` Insert event-specific guidance here: [[EVENT_DOMAIN]]. If none is supplied, return `null` for C19.

## Output format
Return a single JSON object and nothing else. No preamble, no markdown fences, no closing commentary.

{
    "statement_id": "[[STATEMENT_ID]]",
    "speaker_word_count": 0,
    "codeable": true,
    "codeability_note": "",
    "criteria": [
        {
            "id": 1,
            "name": "Logical structure",
            "score": 0,
            "instance_count": 0,
            "evidence": [],
            "justification": "",
            "confidence": "high"
        }
    ]
}

Field rules:
- `criteria` must contain 19 objects in order, with IDs `1` through `19`.
- Each object has: `id`, `name`, `score`, `instance_count`, `evidence`, `justification`, `confidence`.
- Use scores `0`, `1`, or `2`; use `null` only for C19 without event-domain guidance.
- Scores `1` or `2` require verbatim evidence. Score `0` requires `evidence: []`.
- C1-C3 use `instance_count: 0`; C4-C19 count distinct evidence spans, maximum six.
- Example with evidence: `"score": 1, "instance_count": 1, "evidence": ["I woke at 5 am"]`.
- Example without evidence: `"score": 0, "instance_count": 0, "evidence": []`.
- `justification` is at most 3 words; `confidence` is `high`, `medium`, or `low`.
- If `codeable` is `false`, explain why in `codeability_note` and set all 19 scores to `null`.

## Calibration examples
`[[CONFIGURE]]` Insert corpus-specific examples here:
[[CALIBRATION_EXAMPLES]]

Statement to analyse:
[[STATEMENT_TEXT]]
'''
        return (
                prompt.replace("[[EVENT_DOMAIN]]", CBCA_EVENT_DOMAIN or "not configured")
                .replace("[[CBCA_PARTICIPANT]]", CBCA_PARTICIPANT or "not configured")
                .replace("[[CALIBRATION_EXAMPLES]]", CBCA_CALIBRATION_EXAMPLES or "None configured.")
                .replace("[[STATEMENT_ID]]", str(statement_id))
                .replace("[[STATEMENT_TEXT]]", statement_text)
        )

_ANNOTATION_PROMPT_BUILDERS = {
    "reality_monitoring": _annotation_prompt_reality_monitoring,
    "specificity": _annotation_prompt_specificity,
    "cbca": _annotation_prompt_cbca,
}


def annotation_prompt(annotation_type: str, statement_text: str, statement_id=None) -> str:
    """Return the user prompt associated with the requested annotation type."""
    builder = _ANNOTATION_PROMPT_BUILDERS.get(annotation_type)
    if builder is None:
        supported = ", ".join(sorted(_ANNOTATION_PROMPT_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported annotation_type '{annotation_type}'. Supported values: {supported}."
        )
    if annotation_type == "cbca":
        return builder(statement_text, statement_id)
    return builder(statement_text)

# ---------------------------------------------------------------------------
# CSV output (wide format matching the human all_sessions.csv structure)
# ---------------------------------------------------------------------------

def _build_columns(annotation_type):
    cols = [
        "session_id", "statement_id", "original_text", "original_label", "annotation_id", "annotation_number_per_statement"
    ]
    if annotation_type == "cbca":
        cols.extend(["speaker_word_count", "codeable", "codeability_note"])
        for criterion_id in range(1, 20):
            cols.extend([
                f"C{criterion_id}_score",
                f"C{criterion_id}_instance_count",
                f"C{criterion_id}_evidence",
                f"C{criterion_id}_justification",
                f"C{criterion_id}_confidence",
                f"C{criterion_id}_alpha",
            ])
    else:
        cols.extend(["annotation_score", "annotation_type", "rationale"])
    cols.extend([
        "llm_architecture", "temperature", "started_at", "received_at",
        "annotation_duration_ms", "annotation_prompt",
    ])
    return cols


def init_results_csv(architecture, annotation_type, timestamp, subdir=None):
    """Create an empty results CSV and return its path."""
    out_dir = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{architecture}_{annotation_type}_{timestamp}.csv")
    pd.DataFrame(columns=_build_columns(annotation_type)).to_csv(path, index=False)
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
        "llm_architecture": getattr(sequence, "llm_architecture", ""),
        "temperature": round(float(getattr(sequence, "temperature", 0.0)), 4),
        "started_at": getattr(sequence, "started_at", getattr(sequence, "session_start", "")),
        "received_at": getattr(sequence, "received_at", getattr(sequence, "session_end", "")),
        "annotation_duration_ms": getattr(
            sequence,
            "annotation_duration_ms",
            getattr(sequence, "total_duration_ms", ""),
        ),
        "annotation_prompt": getattr(sequence, "annotation_prompt", ""),
    }

    payload = getattr(sequence, "annotation_payload", {})
    if "criteria" in payload:
        row.update({
            "speaker_word_count": payload.get("speaker_word_count", ""),
            "codeable": payload.get("codeable", ""),
            "codeability_note": payload.get("codeability_note", ""),
        })
        for criterion in payload.get("criteria", []):
            criterion_id = criterion["id"]
            prefix = f"C{criterion_id}"
            row.update({
                f"{prefix}_score": criterion.get("score", ""),
                f"{prefix}_instance_count": criterion.get("instance_count", ""),
                f"{prefix}_evidence": json.dumps(criterion.get("evidence", []), ensure_ascii=False),
                f"{prefix}_justification": criterion.get("justification", ""),
                f"{prefix}_confidence": criterion.get("confidence", ""),
            })
        columns = _build_columns("cbca")
    else:
        row.update({
            "annotation_score": payload.get("annotation_score", getattr(sequence, "annotation_score", "")),
            "annotation_type": payload.get("annotation_type", getattr(sequence, "annotation_type", "")),
            "rationale": payload.get("rationale", getattr(sequence, "rationale", "")),
        })
        columns = _build_columns("annotation")

    pd.DataFrame([{column: row.get(column, "") for column in columns}]).to_csv(
        path, mode="a", index=False, header=False
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
        for criterion_index in range(19):
            criterion_id = criterion_index + 1
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
                    alphas[f"C{criterion_index + 1}"] = (
                        None if pd.isna(alpha) else round(float(alpha), 2)
                    )
                except ValueError:
                    alphas[f"C{criterion_index + 1}"] = None
            else:
                alphas[f"C{criterion_index + 1}"] = None
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
