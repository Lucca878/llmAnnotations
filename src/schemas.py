"""OpenRouter JSON schemas for supported annotation instruments."""

CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}


def cbca_schema() -> dict:
    """Schema for the CBCA-19 response."""
    criterion_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 1, "maximum": 19},
            "name": {"type": "string"},
            "score": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 2}, {"type": "null"}]},
            "instance_count": {"type": "integer", "minimum": 0},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "justification": {"type": "string"},
            "confidence": CONFIDENCE,
        },
        "required": ["id", "name", "score", "instance_count", "evidence", "justification", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "statement_id": {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]},
            "instrument": {"type": "string", "enum": ["CBCA-19"]},
            "speaker_word_count": {"type": "integer", "minimum": 0},
            "codeable": {"type": "boolean"},
            "codeability_note": {"type": "string"},
            "criteria": {"type": "array", "minItems": 19, "maxItems": 19, "items": criterion_schema},
        },
        "required": [
            "statement_id",
            "instrument",
            "speaker_word_count",
            "codeable",
            "codeability_note",
            "criteria",
        ],
        "additionalProperties": False,
    }


def reality_monitoring_schema() -> dict:
    """Schema for the RM-8 response."""
    modality_counts = {
        "type": "object",
        "properties": {
            "visual": {"type": "integer", "minimum": 0},
            "auditory": {"type": "integer", "minimum": 0},
            "olfactory": {"type": "integer", "minimum": 0},
            "gustatory": {"type": "integer", "minimum": 0},
            "tactile": {"type": "integer", "minimum": 0},
        },
        "required": ["visual", "auditory", "olfactory", "gustatory", "tactile"],
        "additionalProperties": False,
    }
    criterion_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 1, "maximum": 8},
            "name": {"type": "string"},
            "rating": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 7}, {"type": "null"}]},
            "instance_count": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
            "modality_counts": {"anyOf": [modality_counts, {"type": "null"}]},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "justification": {"type": "string"},
            "confidence": CONFIDENCE,
        },
        "required": ["id", "name", "rating", "instance_count", "evidence", "justification", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "statement_id": {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]},
            "instrument": {"type": "string", "enum": ["RM-8"]},
            "speaker_word_count": {"type": "integer", "minimum": 0},
            "codeable": {"type": "boolean"},
            "codeability_note": {"type": "string"},
            "criteria": {"type": "array", "minItems": 8, "maxItems": 8, "items": criterion_schema},
        },
        "required": [
            "statement_id",
            "instrument",
            "speaker_word_count",
            "codeable",
            "codeability_note",
            "criteria",
        ],
        "additionalProperties": False,
    }


def verifiability_schema() -> dict:
    """Schema for the Verifiability Approach response."""
    detail_schema = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "minimum": 1},
            "span": {"type": "string"},
            "type": {"type": "string", "enum": ["perceptual", "spatial", "temporal"]},
            "is_repeat": {"type": "boolean"},
            "verifiable": {"type": "boolean"},
            "criteria_met": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "identifiable_person_involved",
                        "identifiable_person_witnessed",
                        "technology_recorded",
                    ],
                },
            },
            "justification": {"type": "string"},
            "confidence": CONFIDENCE,
        },
        "required": [
            "index",
            "span",
            "type",
            "is_repeat",
            "verifiable",
            "criteria_met",
            "justification",
            "confidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "statement_id": {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]},
            "instrument": {"type": "string", "enum": ["VA"]},
            "speaker_word_count": {"type": "integer", "minimum": 0},
            "scenario_verifiable": {"type": "boolean"},
            "scenario_note": {"type": "string"},
            "information_protocol": {
                "anyOf": [
                    {"type": "string", "enum": ["informed", "uninformed"]},
                    {"type": "null"},
                ]
            },
            "details": {"type": "array", "items": detail_schema},
        },
        "required": [
            "statement_id",
            "instrument",
            "speaker_word_count",
            "scenario_verifiable",
            "scenario_note",
            "information_protocol",
            "details",
        ],
        "additionalProperties": False,
    }


def response_schema(annotation_type: str) -> dict:
    """Return the schema for one supported annotation type."""
    schemas = {
        "cbca": cbca_schema,
        "reality_monitoring": reality_monitoring_schema,
        "verifiability": verifiability_schema,
    }
    try:
        return schemas[annotation_type]()
    except KeyError as exc:
        supported = ", ".join(sorted(schemas))
        raise ValueError(
            f"Unsupported annotation type '{annotation_type}'. Supported values: {supported}."
        ) from exc
