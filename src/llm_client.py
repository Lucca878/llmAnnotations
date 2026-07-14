import os
import json
from openrouter import OpenRouter
from config import (
    LLM_ARCHITECTURE,
    ANNOTATION_TYPE,
    ANNOTATION_SCORE_MIN,
    ANNOTATION_SCORE_MAX,
)
# from utility import validate_annotation_response

def _build_client() -> OpenRouter:
    """Return an OpenRouter SDK client."""
    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def _get_model_id(architecture: str) -> str:
    if architecture != LLM_ARCHITECTURE["name"]:
        raise ValueError(f"Unknown architecture '{architecture}'. Expected '{LLM_ARCHITECTURE['name']}'.")
    return LLM_ARCHITECTURE["model"]


def call_llm(
    architecture: str,
    prompt: str,
    temperature: float,
    system_prompt: str,
    max_tokens: int = None,
) -> dict:
    """Call OpenRouter and return validated structured annotation output."""
    client = _build_client()
    model_id = _get_model_id(architecture)

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "annotation_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "annotation_type": {
                            "type": "string",
                            "enum": [ANNOTATION_TYPE],
                        },
                        "annotation_score": {
                            "type": "integer",
                            "minimum": ANNOTATION_SCORE_MIN,
                            "maximum": ANNOTATION_SCORE_MAX,
                        },
                        "rationale": {
                            "type": "string",
                        },
                    },
                    "required": ["annotation_type", "annotation_score", "rationale"],
                    "additionalProperties": False,
                },
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
