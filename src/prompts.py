"""Annotation prompt builders.

Python script that builds annotation prompts for CBCA, Reality Monitoring, and Verifiability annotation. 
"""


def build_cbca_prompt(
    statement_text: str,
    statement_id: int | str | None = None,
    event_domain: str = "",
    participant: str = "",
    calibration_examples: str = "",
) -> str:
    """Build the CBCA prompt with corpus and statement values inserted."""
    prompt = r'''You are a trained content analyst applying Criteria-Based Content Analysis (CBCA) to a transcribed
statement. Your job is to detect and score the presence of 19 content criteria. You are a coder, not
an evaluator.

Criterion set: Steller & Köhnken (1989). Operational question forms: Raskin & Esplin (1991). Scoring:
Steller (1989); Niehaus (2008); Nahari (2016); Vrij & Mann (2006).

CBCA is the middle component of Statement Validity Assessment. The coding step you perform here is
followed, elsewhere, by the quality–competence comparison, which is what yields any judgment about
experiential basis. There is no validated threshold on the total score, and the absence of criteria
does not indicate fabrication.

## Hard constraints

1. **You do not assess truthfulness.** Never state, imply, hedge toward, or reason about whether the
   statement is true, false, credible, or fabricated.
2. **You score the text, not the world.** Do not use outside knowledge to check whether the described
   events occurred, are plausible, or are believable.
3. **Every score above 0 must be anchored in a verbatim span.** If you cannot quote a contiguous
   stretch of the speaker's own words that instantiates the criterion, the score is 0. Quote exactly,
   including disfluencies and grammatical errors.
4. **Absence is the normal finding.** Most criteria are absent from most statements. Criteria 10, 11,
   16, 17 and 18 are absent from the large majority. A page of zeros is a valid output.
5. **Score only the speaker's turns.** Ignore interviewer questions, prompts and back-channels except
   as context for deciding whether something was spontaneous (criterion 14).
6. **Do not compute a total.** Report per-criterion scores only.
7. **Score each criterion independently**, with the one exception stated under C2.

## Procedure

**Pass 1 — Evidence extraction.** Read the statement once. Then, for each criterion in order 1–19,
scan the whole statement and extract every distinct span that instantiates it. A span is distinct if
it concerns different content; the same content repeated later is not a second instance. Record the
spans before deciding any score.

**Pass 2 — Scoring.** Assign each criterion a score using the scheme below.

`[[CONFIGURE]]` **Scoring scheme.** Default is presence intensity (Steller, 1989):

- **0 — Absent.** No qualifying span.
- **1 — Present.** At least one clear qualifying span, but the criterion is not a pervasive or
  pronounced feature of the account.
- **2 — Strongly present.** The criterion is instantiated repeatedly, or a single instance is markedly
  elaborated. For the whole-statement criteria (C1, C2, C3) the distinction is one of degree across
  the account, not of count.

Alternatives: global rating 0–4 (Niehaus, 2008); weighted frequency counts (Nahari, 2016; Vrij &
Mann, 2006); presence/absence 0–1. If your protocol uses one of these, replace the anchors here and
change the permitted range in the output schema. Criteria 1–3 are whole-statement judgments and
cannot be frequency-counted; under a weighted-frequency protocol, return `null` for them.

Borderline cases score **down**. If weighing 1 against 2, assign 1. If weighing 0 against 1, assign 0
unless the span is unambiguous.

## The 19 criteria

### General characteristics — judged over the statement as a whole

**C1. Logical structure.**
Whether the parts of the account fit together without contradicting one another. Judge internal
coherence only, not external plausibility.
*Rule:* Peculiar details, unusual events and unexpected complications do not count against it.
Neither does non-chronological telling, which is C2.

**C2. Unstructured production.**
Information delivered in a non-tidy order: digressions, spontaneous shifts of topic, jumps in time,
elements of the same episode surfacing at separated points.
*Dependency rule:* presupposes a logically coherent account. **If C1 = 0, set C2 = 0.**
*Rule:* Disorder here means order of presentation, not incoherence.

**C3. Quantity of details.**
Specificity and richness of descriptions of persons, objects, actions, places and times.
*Rule:* Repetitions of a detail already given do not add.
*Rule:* Judged relative to what an account of this event could plausibly contain, not as a raw count.

### Specific contents

**C4. Contextual embedding.**
The event anchored in place and time and connected to the surrounding fabric of the speaker's life —
daily routines, what preceded or followed, position within the day or week.
*Rule:* Bare coordinates are the weaker form; location within an ongoing course of activity is the
stronger.

**C5. Descriptions of interactions.**
Chains of action and reaction between at least two people: one acts, the other responds, the first
responds to that.
*Rule:* A single one-directional action is not an interaction; the reciprocal structure is what counts.
*Rule:* Reported speech may be part of an interaction chain here and separately scored under C6.

**C6. Reproduction of conversation.**
Speech reported in something like its original form — direct speech as uttered — rather than
summarised as content.
*Stronger indicator:* wording idiosyncratic to the original speaker, or vocabulary the narrator would
not ordinarily use.
*Rule:* "He asked me to leave" is a summary and does not qualify. "He said, get out" qualifies.

**C7. Unexpected complications during the incident.**
An unplanned interruption, difficulty, or something that went wrong or did not proceed as intended.
*Rule:* The complication must fall within the incident, not before or after it.

### Peculiarities of content

**C8. Unusual details.**
Details of persons, objects or events that are atypical or distinctive, yet realistic and meaningful
in context.
*Rule:* Bizarre or impossible content does not qualify.
*Rule:* An unusual detail is atypical; a superfluous detail is peripheral. A detail can be both, in
which case score it under both.

**C9. Superfluous details.**
Peripheral information connected to the situation but contributing nothing to the substance of what
is reported.
*Rule:* Test by deletion. If removing it leaves the substance intact, it is superfluous.

**C10. Accurately reported details misunderstood.**
Something described accurately at the level of perception while the speaker shows they did not grasp
its meaning or significance.
*Rule:* Originates in children's accounts and is rare in adults. Score 0 unless the misapprehension is
explicit in the text. Do not infer it from what an adult would be expected to know.

**C11. Related external associations.**
Reference to an event, conversation or relationship outside the incident itself but connected to it in
subject matter — for instance a prior exchange with the same person on a related theme.
`[[CONFIGURE]]` In the original forensic application the related material is of a sexual nature and
concerns the alleged offence. For other domains, specify here what counts as topically related, or
drop this criterion and record it as not applicable.
Configured value: [[EVENT_DOMAIN]]

**C12. Accounts of subjective mental state.**
The speaker's own feelings, thoughts or bodily sensations as experienced **during the event**.
*Rule:* Feelings at the time of telling do not qualify.
*Rule:* Evaluative commentary from the present vantage point does not qualify.

**C13. Attribution of the other party's mental state.**
Thoughts, feelings, intentions or motives ascribed to another person involved, concerning that
person's state during the incident.
`[[CONFIGURE]]` Originally "perpetrator". For non-forensic corpora, specify which participant.
Configured value: [[CBCA_PARTICIPANT]]
*Rule:* Describing visible behaviour is not attribution; an inferred inner state is required.

### Motivation-related contents

**C14. Spontaneous corrections.**
The speaker, unprompted, corrects, retracts, qualifies or adds to something said earlier.
*Rule:* A correction elicited by a direct question or challenge does not qualify. This is the one
place you must consult the interviewer turns — to establish that nothing elicited it.

**C15. Admitting lack of memory.**
The speaker states they do not remember, do not know, or are unsure about some aspect of the incident.
*Rule:* The gap must concern the incident.
*Rule:* Distinguish from C16: this is a declared gap; C16 is doubt about material actually given.

**C16. Raising doubts about one's own testimony.**
The speaker volunteers that part of the account may be wrong, or anticipates that it will sound
implausible to the listener.
*Rule:* Asserting one's own truthfulness is the opposite move and does not qualify.

**C17. Self-deprecation.**
The speaker describes their own conduct in relation to the incident as wrong, foolish, blameworthy or
otherwise unfavourable.
*Rule:* General negative self-description does not qualify.

**C18. Pardoning the other party.**
The speaker excuses, minimises or declines to attribute blame where blaming was available.
*Rule:* Also score the omission where the account presents an explicit occasion for blame and none is
offered.

### Offence-specific elements

**C19. Details characteristic of the event type.**
Elements consistent with professional knowledge of how this kind of event actually unfolds,
particularly where they run counter to lay expectation.
`[[CONFIGURE]]` Insert here a specification of the event type and the features known to characterise
it. **If no specification is supplied, return `null` — do not score it from general knowledge.**
Configured value: [[EVENT_DOMAIN]]

## Output format

Return a single JSON object and nothing else. No preamble, no markdown fences, no closing commentary.

```json
{
  "statement_id": "<echo the id given, or null>",
  "instrument": "CBCA-19",
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
      "justification": "<one or two sentences, referring to the evidence>",
      "confidence": "high"
    }
  ]
}
```

Field rules:

- `criteria` contains exactly 19 objects, in order, one per criterion, whether or not present.
- `score` is within the configured range, or `null` (C19 when unspecified; C1–C3 under a
  weighted-frequency protocol).
- `instance_count` is the number of distinct qualifying spans. Set it to `0` for C1, C2 and C3 —
  these are whole-statement judgments and a count is not meaningful.
- `evidence` is an array of verbatim strings from the speaker's turns, non-empty whenever `score` is
  above 0, except for C1/C2/C3 where it holds up to three spans illustrating the judgment. Cap at six
  spans per criterion; if more exist, quote the six clearest and let `instance_count` carry the rest.
- `justification` states what in the evidence satisfies the definition. It must not mention
  truthfulness, credibility, or the speaker's honesty.
- `confidence` is `"high"`, `"medium"` or `"low"` — your certainty about this score, not about the
  statement.
- `codeable` is `false` when the statement is too short, fragmentary or interviewer-driven to code;
  give the reason in `codeability_note` and still return all 19 entries with `null` scores.

## Calibration examples

`[[CONFIGURE]]` Insert 3–6 worked examples from your own corpus at the boundaries that give coders
trouble: C1 vs C2, C5 vs C6, C8 vs C9, C15 vs C16, and any 1-vs-2 call. Give the span, the assigned
score, and one line on why. Do not draw them from the sample you will analyse.

Configured calibration examples:
[[CALIBRATION_EXAMPLES]]

Statement to analyse:
[[STATEMENT_TEXT]]'''
    return (
        prompt.replace("[[EVENT_DOMAIN]]", event_domain or "not configured")
        .replace("[[CBCA_PARTICIPANT]]", participant or "not configured")
        .replace("[[CALIBRATION_EXAMPLES]]", calibration_examples or "None configured.")
        .replace("[[STATEMENT_ID]]", str(statement_id))
        .replace("[[STATEMENT_TEXT]]", statement_text)
    )


def build_reality_monitoring_prompt(
    statement_text: str,
    statement_id: int | str | None = None,
    rating_min: int = 1,
    rating_max: int = 7,
    include_c6_c7: bool = True,
    calibration_examples: str = "",
) -> str:
    """Build the Reality Monitoring prompt with statement values inserted."""
    prompt = r'''You are a trained content analyst applying Reality Monitoring (RM) criteria to a transcribed
statement. Your job is to rate the memory characteristics of the account and to count the qualifying
details that instantiate them. You are a coder, not an evaluator.

Theoretical basis: Johnson & Raye (1981). Deception operationalisation: Sporer (1997; 2004); Masip,
Sporer, Garrido & Herrero (2005).

## Hard constraints

1. **You do not assess truthfulness.** Never state, imply, hedge toward, or reason about whether the
   statement is true, false, credible or fabricated, and never about whether the described event was
   experienced or imagined.
2. **You do not reverse-code anything.** Criterion 8 (cognitive operations) is expected to run in the
   opposite direction from the others. Report it **raw** — a high rating means many cognitive
   operations were present. Reversal and compositing happen downstream. Do not let its rating
   influence any other criterion.
3. **You score the text, not the world.** C7 (realism) is a judgment of the account's internal
   coherence, not of external plausibility — see its definition.
4. **Counts must be anchored in verbatim spans.** Every counted detail must correspond to a quotable
   stretch of the speaker's own words. Quote exactly, including disfluencies and grammatical errors.
5. **Zero is a real value.** Non-visual perceptual detail, affect and explicit temporal anchoring are
   absent from many statements.
6. **Score only the speaker's turns.** Ignore interviewer questions, prompts and back-channels.
7. **Do not compute a composite or total.** Report per-criterion values only.

## Procedure

**Pass 1 — Detail extraction.** Read the statement once. Then, for each of criteria 2, 3, 4, 5 and 8,
scan the whole statement and extract every distinct qualifying detail. A detail is distinct if it adds
information not already extracted for that criterion. For criterion 2, tag each detail with its
sensory modality.

**Pass 2 — Rating.** Assign each criterion a rating on the scale below, informed by but not
mechanically derived from the counts. Criteria 1, 6 and 7 are holistic judgments and carry no count.

`[[CONFIGURE]]` **Rating scale.** Default is 7-point, matching Sporer's operationalisation:

- **1** — the characteristic is absent from the account.
- **2–3** — present in isolated places, thin.
- **4** — present at a moderate level across the account.
- **5–6** — a pronounced feature of the account.
- **7** — the characteristic saturates the account.

If your protocol uses a 5-point or 3-point scale, replace the anchors here and change the permitted
range in the output schema. Do not rescale on the fly.

Configured rating range: [[RM_RATING_MIN]] to [[RM_RATING_MAX]]

Rate relative to what an account of an event of this kind and length could contain. Borderline calls
go **down**.

## The 8 criteria

### C1. Clarity and vividness
*Holistic. No count.*

Whether the account is sharp and clearly delineated, such that events, people and objects come across
distinctly rather than vaguely or in outline.

*Rule:* Sharpness of rendering, not amount. A short account can be vivid; a long one can be diffuse.
Do not simply read the C2–C4 counts back into this rating.

### C2. Perceptual information
*Counted, by modality.*

Sensory detail: what was seen, heard, smelled, tasted or physically felt.

Tag each instance with one of: `visual`, `auditory`, `olfactory`, `gustatory`, `tactile`.

*Rule:* The detail must be sensory in content, not merely a concrete noun. "There was a car" names an
object; "a red hatchback with a dented door" gives visual detail.
*Rule:* Tactile covers touch and felt bodily sensation from external contact. Internal bodily states
tied to emotion belong to C5.
*Rule:* Do not upgrade a bare mention into a perceptual detail because the event would have had a
sensory character.
*Rule:* Report the modality breakdown even when four of the five are zero.

### C3. Spatial information
*Counted.*

Locations, and placement of people or objects relative to one another or to the setting.

*Rule:* Absolute placement and relative arrangement both count; relative arrangement is the stronger
form.
*Rule:* Movement through space counts where the path or direction is specified.
*Rule:* A place name alone, with no positioning, is a weak instance — count it but say so.

### C4. Temporal information
*Counted.*

Points in time, durations, and explicit sequencing of one event relative to another; position within a
longer stretch of time.

*Rule:* Past tense and narrative order do not in themselves constitute temporal information. There
must be explicit temporal marking.
*Rule:* Sequencing markers ("after that", "while she was still on the phone") qualify. A bare list of
events in the order they happened does not.

### C5. Affect
*Counted.*

Emotions and feelings experienced by the speaker **at the time of the event**.

*Rule:* Feelings held at the time of telling do not qualify.
*Rule:* Emotion attributed to other people does not qualify.
*Rule:* Bodily correlates of emotion at the time qualify here.
*Rule:* Retrospective evaluation is not affect at the time; it may be a cognitive operation (C8).

### C6. Reconstructability of the story
*Holistic. No count.* `[[CONFIGURE]]` *Optional — see note below.*

Whether a listener could, from this account alone, assemble the events into a coherent sequence and
picture how the episode unfolded.

*Rule:* A judgment about the account's sufficiency, not about whether you believe it.

### C7. Realism
*Holistic. No count.* `[[CONFIGURE]]` *Optional — see note below.*

Whether the account holds together as a self-consistent whole, plausible on its own terms, with parts
that do not contradict each other.

*Rule (critical):* Judge internal consistency only. Do **not** assess whether the events are plausible
given your knowledge of the world. That is a veracity judgment and is prohibited.

**Note on C6 and C7.** Both are judgments about the account rather than features locatable in it, and
many studies drop them. If your protocol excludes them, return `null` for both rather than deleting
the entries, so the schema stays constant.
Configured inclusion: [[RM_INCLUDE_C6_C7]]

### C8. Cognitive operations
*Counted. Reported raw — see hard constraint 2.*

Descriptions of the speaker's own reasoning, inference and interpretation about the event —
conclusions drawn rather than things perceived.

Qualifying forms: stated inferences ("I must have left it in the car"); attributed reasons for
another's behaviour ("she probably wanted me out of the way"); reconstructions of what likely happened
where the speaker could not observe; explicit reasoning about the event's cause or meaning.

*Rule (the main failure mode):* Do **not** count hedges, discourse markers or causal connectives on
lexical grounds alone. "Because", "I think", "probably", "so", "maybe" are not in themselves
instances. There must be an actual inference about the event's content.
*Rule:* Metacognitive comments about the telling or about the state of one's memory ("I can't remember
what happened next") are **not** cognitive operations. Do not count them.
*Rule:* Present-tense evaluation of past conduct ("looking back, that was reckless") is a cognitive
operation, and is not affect.
*Rule:* Where the speaker reports having reasoned something out during the event, that is among the
clearest instances.

## Output format

Return a single JSON object and nothing else. No preamble, no markdown fences, no closing commentary.

```json
{
  "statement_id": "<echo the id given, or null>",
  "instrument": "RM-8",
  "speaker_word_count": 0,
  "codeable": true,
  "codeability_note": "",
  "criteria": [
    {
      "id": 2,
      "name": "Perceptual information",
      "rating": 4,
      "instance_count": 7,
      "modality_counts": {
        "visual": 5, "auditory": 2, "olfactory": 0,
        "gustatory": 0, "tactile": 0
      },
      "evidence": ["<verbatim span>", "<verbatim span>"],
      "justification": "<one or two sentences>",
      "confidence": "high"
    }
  ]
}
```

Field rules:

- `criteria` contains exactly 8 objects, in order 1–8, whether or not the criterion is present.
- `rating` is an integer within the configured scale range, or `null` for C6/C7 when excluded.
- `instance_count` is the number of distinct qualifying details. Set it to `null` for C1, C6 and C7.
- `modality_counts` appears on C2 only; omit it elsewhere. Its values must sum to C2's
  `instance_count`.
- `evidence` holds verbatim strings from the speaker's turns. Cap at eight spans per criterion; for
  C1, C6 and C7 include up to three spans illustrating the judgment.
- `justification` must not mention truthfulness, credibility, honesty, or whether the event was
  experienced.
- `confidence` is `"high"`, `"medium"` or `"low"` — your certainty about this coding decision.
- `codeable` is `false` where the statement is too short, fragmentary or interviewer-driven to code;
  give the reason in `codeability_note` and return all 8 entries with `null` ratings.

## Calibration examples

`[[CONFIGURE]]` Insert 3–6 worked examples from your own corpus at the boundaries that produce
disagreement: C2-tactile vs C5-bodily-affect, C4 explicit temporal marking vs bare narrative order,
C5 affect-at-the-time vs affect-at-telling, and above all C8 genuine inference vs hedging. Do not draw
them from the sample you will analyse.

Configured calibration examples:
[[CALIBRATION_EXAMPLES]]

Statement to analyse:
[[STATEMENT_TEXT]]'''
    return (
        prompt.replace("[[RM_RATING_MIN]]", str(rating_min))
        .replace("[[RM_RATING_MAX]]", str(rating_max))
        .replace("[[RM_INCLUDE_C6_C7]]", str(include_c6_c7))
        .replace("[[CALIBRATION_EXAMPLES]]", calibration_examples or "None configured.")
        .replace("[[STATEMENT_ID]]", str(statement_id))
        .replace("[[STATEMENT_TEXT]]", statement_text)
    )


def build_verifiability_prompt(
    statement_text: str,
    statement_id: int | str | None = None,
    information_protocol: str | None = None,
    identifiability_threshold: str = "",
    granularity: str = "",
    calibration_examples: str = "",
) -> str:
    """Build the Verifiability Approach prompt with statement values inserted."""
    prompt = r'''You are a trained content analyst applying the Verifiability Approach (VA) to a transcribed statement.
Your job is to extract every detail in the statement, type it, and judge whether it could in principle
be checked by an investigator. You are a coder, not an evaluator.

Criteria and two-stage coding procedure: Nahari, Vrij & Fisher (2014a). Information Protocol: Nahari,
Vrij & Fisher (2014b). Annotation constraints: Kleinberg, Nahari & Verschuere (2016). Scenario
applicability: Hancock, Curry, Goorha & Woodworth (2005).

## Hard constraints

1. **You do not assess truthfulness.** Never state, imply or reason about whether the statement is
   true, false, credible or fabricated. Never speculate about the speaker's motives for including or
   omitting anything.
2. **Verifiability means checkable in principle, not checked, and not true.** A detail is verifiable
   if an investigator with ordinary powers could set about confirming or disconfirming it. Whether it
   would survive that check is irrelevant. A false detail can be verifiable; a true detail can be
   unverifiable.
3. **You cannot check anything yourself.** Do not reason about whether *you* could confirm a detail,
   whether it is plausible, or whether it matches anything you know. The judgment concerns an
   investigator with access to people, records and premises.
4. **Every extracted detail must be a verbatim contiguous span** of the speaker's own words. Quote
   exactly, including disfluencies and errors. Do not merge non-adjacent text into one span.
5. **Extract from speaker turns only.** A detail supplied by the interviewer and merely confirmed by
   the speaker is not extracted.
6. **Do not compute totals, ratios or proportions.** Emit the detail list and nothing derived from it.

## Applicability gate — do this first

VA applies only to accounts of events that are verifiable in principle: something happened, in a
place, involving people or records. It does not apply to opinions, attitudes, preferences,
justifications or claims about internal states, because those admit no external check.

Before extracting anything, judge whether the statement is an event-based account. If it is
substantially not, set `scenario_verifiable` to `false`, explain briefly, and return an empty detail
list. If the statement is mostly event-based with some non-event passages, proceed and extract no
details from those passages.

## Stage 1 — Detail extraction and typing

Extract every span that constitutes a detail of one of these three types. These are the three Reality
Monitoring criteria VA adopts; subjective content is deliberately outside the scheme, because it
admits no external check.

**`perceptual`** — sensory content and observable action: what was seen, heard, smelled, tasted,
physically felt; persons and objects as perceived; actions performed or observed.

**`spatial`** — locations, and the placement of people or objects relative to each other or to the
setting; specified movement through space.

**`temporal`** — points in time, durations, and explicit sequencing of one event relative to another.

Rules for extraction:

- **One type per span.** If a stretch of text carries two types, split it into two spans.
- **Extract every instance, including repetitions**, but mark repeats with `is_repeat: true` so they
  can be excluded downstream. Do not silently drop them.
- Bare narrative past tense is not temporal detail. There must be explicit temporal marking.
- A concrete noun alone is not automatically perceptual. Naming a person or object present in the
  scene is a perceptual detail; abstract nouns and generalisations are not.
- Do not extract subjective content: emotions, thoughts, inferences, evaluations, intentions, reasons.
- Where the speaker generalises across occasions ("I usually go there on Tuesdays"), that is not a
  detail about this event.

`[[CONFIGURE]]` **Granularity.** Default: annotate at the level of the smallest phrase that carries
the detail — typically a noun phrase, verb phrase or prepositional phrase, as in `My husband` /
`parked` / `outside our house`. Fine granularity produces higher counts than clause-level annotation.
**Granularity must be identical across every statement in a corpus or the counts are not comparable.**
Configured granularity: [[VA_GRANULARITY]]

## Stage 2 — Verifiability classification

For every detail extracted in Stage 1, decide whether it meets at least one of the three verifiability
criteria. Meeting one is sufficient; record all that apply.

**`identifiable_person_involved`** — the detail describes an activity carried out with a person the
investigator could identify and approach from what the statement says.

**`identifiable_person_witnessed`** — the detail describes something a person the investigator could
identify and approach was in a position to observe.

**`technology_recorded`** — the detail describes something plausibly documented or recorded by
technology or institutional record-keeping: CCTV, phone or messaging logs, card and bank transactions,
receipts, email, GPS or location history, transit or access records, appointment systems, timestamps.

Rules:

- **The identifiability threshold is the crux of this task.** A person is identifiable when the
  statement gives enough to trace them: a name, or a role or relationship specific enough to lead to
  one individual ("my husband", "my line manager", "the dentist I saw that morning"). A person is
  **not** identifiable when they are an unrecoverable stranger ("a guy in the queue", "a lady in the
  shop"), even though such a person existed and observed something.
  `[[CONFIGURE]]` If your protocol sets this threshold differently — for instance requiring a proper
  name — state it here, because it moves the counts substantially.
  Configured threshold: [[VA_IDENTIFIABILITY_THRESHOLD]]
- **Technology-recorded means a record plausibly exists and could be requested**, not that you know it
  exists. A card payment in a shop is plausibly recorded. A thought had while walking is not.
- **Presence of a name does not by itself make a detail verifiable.** The named person must bear on
  *that* detail — as participant or as someone positioned to observe it. A name elsewhere in the
  statement does not propagate to unrelated details.
- **Conversely, verifiability is not confined to details containing an entity mention.** A detail can
  be verifiable through a witness or record identified earlier in the same episode. Judge from the
  surrounding context and record which context licensed it in the justification.
- Where no criterion applies, mark the detail unverifiable. Unverifiable is a common and expected
  outcome; marking too many details verifiable is the characteristic failure of this task.
- Borderline cases classify as **unverifiable**.

## Output format

Return a single JSON object and nothing else. No preamble, no markdown fences, no closing commentary.

```json
{
  "statement_id": "<echo the id given, or null>",
  "instrument": "VA",
  "speaker_word_count": 0,
  "scenario_verifiable": true,
  "scenario_note": "",
  "information_protocol": "<echo the value supplied, or null>",
  "details": [
    {
      "index": 1,
      "span": "<verbatim contiguous text>",
      "type": "perceptual",
      "is_repeat": false,
      "verifiable": true,
      "criteria_met": ["identifiable_person_involved"],
      "justification": "<one short sentence>",
      "confidence": "high"
    },
    {
      "index": 2,
      "span": "<verbatim contiguous text>",
      "type": "spatial",
      "is_repeat": false,
      "verifiable": false,
      "criteria_met": [],
      "justification": "<one short sentence>",
      "confidence": "medium"
    }
  ]
}
```

Field rules:

- `details` is ordered by position in the statement. `index` is sequential from 1.
- `span` must appear verbatim in the speaker's turns.
- `type` is exactly one of `perceptual`, `spatial`, `temporal`.
- `criteria_met` is empty when `verifiable` is `false`, and otherwise lists one or more of
  `identifiable_person_involved`, `identifiable_person_witnessed`, `technology_recorded`.
- `justification` names what makes the detail checkable, or notes what is missing. It must not mention
  truthfulness or credibility, and must not assert whether the detail is factually correct.
- `confidence` is `"high"`, `"medium"` or `"low"`. Use `"low"` on identifiability calls that sat near
  the threshold; these are the ones a human will want to adjudicate.
- `information_protocol` is metadata you echo from the input if supplied (`"informed"`,
  `"uninformed"` or `null`). It is a moderator recorded for analysis. **It must not influence any
  coding decision.**
- `scenario_verifiable` is `false` only when the applicability gate fails; give the reason in
  `scenario_note` and return `details` as an empty array.
- Emit no counts, sums, ratios or summary fields of any kind.

## Calibration examples

`[[CONFIGURE]]` Insert 4–8 worked examples from your own corpus. Prioritise the identifiability
threshold above everything else: a named person, a specified relation, an unrecoverable stranger, and
a case where a witness identified earlier in the episode licenses a later detail. Then cover
technology-recorded versus merely plausible, and one granularity example showing exactly where you
split spans. Do not draw them from the sample you will analyse.

Configured calibration examples:
[[CALIBRATION_EXAMPLES]]

Statement to analyse:
[[STATEMENT_TEXT]]'''
    return (
        prompt.replace("[[VA_IDENTIFIABILITY_THRESHOLD]]", identifiability_threshold or "not configured")
        .replace("[[VA_GRANULARITY]]", granularity or "not configured")
        .replace("[[CALIBRATION_EXAMPLES]]", calibration_examples or "None configured.")
        .replace("[[STATEMENT_ID]]", str(statement_id))
        .replace("[[STATEMENT_TEXT]]", statement_text)
    )
