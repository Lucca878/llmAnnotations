from dataclasses import dataclass, field

@dataclass
class AnnotationRecord:
    """One structured LLM annotation for one statement."""

    session_id: str
    statement_id: int
    original_text: str
    original_label: str
    annotation_id: str
    annotation_number_per_statement: int
    annotation_score: int
    annotation_type: str
    rationale: str
    llm_architecture: str
    temperature: float
    started_at: str
    received_at: str
    annotation_duration_ms: int
    annotation_prompt: str
    annotation_payload: dict = field(default_factory=dict)
