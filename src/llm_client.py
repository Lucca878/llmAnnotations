import json
import os

from openrouter import OpenRouter

from config import LLM_ARCHITECTURE
from schemas import response_schema


def _build_client() -> OpenRouter:
    """Return an OpenRouter SDK client."""
    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def _get_model_id(architecture: str) -> str:
    if architecture != LLM_ARCHITECTURE["name"]:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Expected '{LLM_ARCHITECTURE['name']}'."
        )
    return LLM_ARCHITECTURE["model"]


def call_llm(
    architecture: str,
    annotation_type: str,
    prompt: str,
    temperature: float,
    max_tokens: int = None,
) -> dict:
    """Call OpenRouter and return structured output for one annotation type."""
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
                "name": f"{annotation_type}_response",
                "strict": True,
                "schema": response_schema(annotation_type),
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

            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

            if not isinstance(content, str):
                raise ValueError("OpenRouter returned non-text content for structured output.")

            return json.loads(content)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break

    raise RuntimeError(
        f"Failed to get structured output after {max_attempts} attempts. Last error: {last_error}"
    ) from last_error
