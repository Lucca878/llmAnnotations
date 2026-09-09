import os
import json
from openrouter import OpenRouter
from config import (
    LLM_ARCHITECTURE,
)
# from utility import validate_annotation_response

def _build_client() -> OpenRouter:
    """Return an OpenRouter SDK client."""
    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def _get_model_id(architecture: str) -> str:
    if architecture != LLM_ARCHITECTURE["name"]:
        raise ValueError(f"Unknown architecture '{architecture}'. Expected '{LLM_ARCHITECTURE['name']}'.")
    return LLM_ARCHITECTURE["model"]


def _response_schema():
    criterion_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 1, "maximum": 19},
            "name": {"type": "string"},
            "score": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 2}, {"type": "null"}]},
            "instance_count": {"type": "integer", "minimum": 0},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "justification": {
                "type": "string",
                "pattern": r"^$|^\S+(\s+\S+){0,2}$",
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["id", "name", "score", "instance_count", "evidence", "justification", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "statement_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "speaker_word_count": {"type": "integer", "minimum": 0},
            "codeable": {"type": "boolean"},
            "codeability_note": {"type": "string"},
            "criteria": {"type": "array", "minItems": 19, "maxItems": 19, "items": criterion_schema},
        },
        "required": ["statement_id", "speaker_word_count", "codeable", "codeability_note", "criteria"],
        "additionalProperties": False,
    }


def call_llm(
    architecture: str,
    prompt: str,
    temperature: float,
    max_tokens: int = None,
) -> dict:
    """Call OpenRouter with the complete annotation prompt."""
    client = _build_client()
    model_id = _get_model_id(architecture)

    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "annotation_response",
                "strict": True,
                "schema": _response_schema(),
            },
        },
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.send(**kwargs)
            content = response.choices[0].message.content

            # Keep a tiny normalization layer in case SDK returns segmented content.
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

            if not isinstance(content, str):
                raise ValueError("OpenRouter returned non-text content for structured output.")

            structured = json.loads(content)

            # return validate_annotation_response(content)
            return structured
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break

    raise RuntimeError(
        f"Failed to get structured output after {max_attempts} attempts. Last error: {last_error}"
    ) from last_error
