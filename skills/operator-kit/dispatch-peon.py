"""
Peon dispatcher — synchronous interrupt resolver for GLM-5.1 Handlers.

A Handler calls this mid-generation when it hits a blocking subtask that would
interrupt its writing flow. This script calls a lower-tier model (GLM-Flash),
validates the output, and returns a clean result. The Handler continues writing.

Usage (called by Handler via subprocess or direct import):
  from skills.operator_kit.dispatch_peon import call_peon, PeonError

  result = call_peon(task={
      "type": "citation_lookup",
      "content": "What is the source for 'ICP match rate improves by 23% with enrichment'?",
      "lesson_id": "01-math-foundations/01-linear-algebra",
  })

  result = call_peon(task={
      "type": "format_validation",
      "content": "## Learning Objectives\n- Understand...\n## The Concept\n...",
      "lesson_id": "...",
  })

Supported task types:
  citation_lookup     — find or flag a GTM claim that needs a source
  format_validation   — check heading structure and front-matter
  text_extraction     — pull learning objectives from a full lesson
  metadata_generation — generate tags, difficulty rating, strand labels

Environment:
  ZHIPUAI_API_KEY must be set.

CLI flags:
  --smoke-test    Make one real Flash call; assert non-empty response. Use before any full run.
  --task-type     Task type to smoke-test (default: citation_lookup)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ── Model registry ────────────────────────────────────────────────────────────
# Change model names here; no other file needs updating.

MODEL_REGISTRY = {
    "default":            "GLM-5-Turbo",
    "citation_lookup":    "GLM-5-Turbo",
    "format_validation":  "GLM-5-Turbo",
    "text_extraction":    "GLM-5-Turbo",
    "metadata_generation":"GLM-5-Turbo",
    "diagram_generation": "GLM-5-Turbo",
}

ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"

# ── Task schemas ──────────────────────────────────────────────────────────────
# Allowlisted fields per task type. Any extra field in the task dict is rejected.

TASK_SCHEMA = {
    "citation_lookup": {
        "required": {"type", "content", "lesson_id"},
        "optional": {"max_chars"},
    },
    "format_validation": {
        "required": {"type", "content", "lesson_id"},
        "optional": {},
    },
    "text_extraction": {
        "required": {"type", "content", "lesson_id"},
        "optional": {"extract_field"},
    },
    "metadata_generation": {
        "required": {"type", "content", "lesson_id"},
        "optional": {"strand"},
    },
    "diagram_generation": {
        "required": {"type", "content", "lesson_id"},
        "optional": {"format", "width", "height"},
    },
}

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "citation_lookup": """You are a GTM curriculum citation assistant.
Given a claim from a lesson, identify a plausible source (URL + publication name + what it supports).
If no credible source can be identified, respond exactly: [CITATION NEEDED — claim: <claim>]
Output: one line. Either a citation or the [CITATION NEEDED] marker. Nothing else.""",

    "format_validation": """You are a curriculum format auditor.
Check if the provided lesson content has: correct heading levels (##, ###), a ## Learning Objectives section,
a ## Sources section, and no orphaned headings. Respond with:
PASS — if all checks pass
FAIL: <comma-separated list of issues> — if any check fails
Output: one line only.""",

    "text_extraction": """You are a curriculum content extractor.
Extract the learning objectives from the provided lesson content.
Output: a bullet list of objectives, one per line, starting with "-". Nothing else.""",

    "metadata_generation": """You are a curriculum metadata tagger.
Given lesson content, output a JSON object with:
  "tags": [list of 3-5 topic slugs, kebab-case],
  "difficulty": "beginner" | "intermediate" | "advanced",
  "strand": "ai-engineering" | "gtm-application" | "both"
Output: valid JSON only, no prose before or after.""",

    "diagram_generation": """You are a technical diagram creator for an AI/GTM curriculum.
Given a description of a concept that needs a visual, produce a clean SVG diagram that a student can read inline in a markdown lesson.
Rules:
- Output raw SVG only — no markdown fences, no prose before or after
- viewBox="0 0 800 500" unless the description specifies different dimensions
- Use flat colors, no gradients. Label every node clearly.
- Font: sans-serif, minimum 13px
- If the concept is a flow/pipeline: left-to-right arrows
- If the concept is a hierarchy: top-down tree
- If the concept is a comparison: side-by-side columns
- Fallback marker if concept is too vague: [DIAGRAM PENDING — description: <description>]
Output: raw SVG string starting with <svg or the fallback marker. Nothing else.""",
}

# ── Error class ───────────────────────────────────────────────────────────────

class PeonError(Exception):
    """Raised when peon fails after max retries. Handler should use fallback."""
    pass

# ── Validation ────────────────────────────────────────────────────────────────

def validate_task(task: dict) -> str:
    """Validate task dict against allowlist. Returns task type. Raises ValueError on bad input."""
    task_type = task.get("type")
    if not task_type:
        raise ValueError("task missing 'type' field")
    if task_type not in TASK_SCHEMA:
        raise ValueError(f"unknown task type: {task_type!r}. Valid: {list(TASK_SCHEMA)}")

    schema = TASK_SCHEMA[task_type]
    allowed = schema["required"] | schema["optional"]
    extra = set(task.keys()) - allowed
    if extra:
        raise ValueError(f"task dict has disallowed fields: {extra}. Only allowed: {allowed}")

    missing = schema["required"] - set(task.keys())
    if missing:
        raise ValueError(f"task dict missing required fields: {missing}")

    # Validate output_path is within OUTPUT_ROOT if present (injection prevention)
    if "output_path" in task:
        output_root = Path("stages").resolve()
        try:
            resolved = Path(task["output_path"]).resolve()
            if not str(resolved).startswith(str(output_root)):
                raise ValueError(f"output_path escapes OUTPUT_ROOT: {resolved}")
        except Exception as e:
            raise ValueError(f"invalid output_path: {e}")

    return task_type


def validate_output(task_type: str, result: str) -> str:
    """Validate peon output meets minimum structure. Returns cleaned result."""
    result = result.strip()
    if not result:
        raise ValueError("empty output")

    if task_type == "metadata_generation":
        # Must be valid JSON with required keys
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            # Strip markdown fences if model wrapped in ```json
            import re
            result = re.sub(r"^```json\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
            data = json.loads(result)
        required_keys = {"tags", "difficulty", "strand"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"metadata JSON missing keys: {missing}")
        return json.dumps(data, ensure_ascii=False)

    if task_type == "format_validation":
        if not (result.startswith("PASS") or result.startswith("FAIL")):
            raise ValueError(f"format_validation output must start with PASS or FAIL, got: {result[:50]}")

    if task_type == "diagram_generation":
        if not (result.startswith("<svg") or result.startswith("[DIAGRAM PENDING")):
            # Strip markdown fences if model wrapped SVG in ```svg
            import re
            result = re.sub(r"^```(?:svg|xml)?\s*", "", result)
            result = re.sub(r"\s*```$", "", result).strip()
            if not (result.startswith("<svg") or result.startswith("[DIAGRAM PENDING")):
                raise ValueError(f"diagram_generation output must be SVG or fallback marker, got: {result[:80]}")

    if len(result) < 5:
        raise ValueError(f"output suspiciously short ({len(result)} chars)")

    return result


# ── Core call ─────────────────────────────────────────────────────────────────

def call_peon(task: dict, max_retries: int = 2) -> str:
    """
    Call a peon model to resolve a blocking subtask.

    Args:
        task: dict with 'type', 'content', 'lesson_id', and optional type-specific fields
        max_retries: attempts before raising PeonError

    Returns:
        Validated string result

    Raises:
        PeonError: if all retries fail
        ValueError: if task dict is invalid (caller bug, not peon bug)
    """
    task_type = validate_task(task)  # raises ValueError on bad input — caller's problem

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        raise PeonError("ZHIPUAI_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url=ZAI_BASE_URL)
    model  = MODEL_REGISTRY.get(task_type, MODEL_REGISTRY["default"])

    system_prompt = SYSTEM_PROMPTS[task_type]
    user_prompt   = task["content"]

    last_error = None
    for attempt in range(1, max_retries + 2):  # max_retries + 1 attempts total
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=500,
                stream=False,
                timeout=30,
            )
            raw = response.choices[0].message.content or ""
            return validate_output(task_type, raw)

        except RateLimitError as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  [PEON-429] {task_type} attempt {attempt}: rate limited, waiting {wait}s")
            time.sleep(wait)

        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            if attempt > max_retries:
                break
            print(f"  [PEON-INVALID] {task_type} attempt {attempt}: {e} — retrying")
            time.sleep(1)

        except Exception as e:
            last_error = e
            if attempt > max_retries:
                break
            print(f"  [PEON-ERR] {task_type} attempt {attempt}: {e} — retrying")
            time.sleep(1)

    raise PeonError(f"{task_type} failed after {max_retries + 1} attempts: {last_error}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def smoke_test(task_type: str) -> None:
    """Make one real API call and assert a non-empty validated response."""
    print(f"Smoke test: task_type={task_type}")

    sample_content = {
        "citation_lookup":     "ICP match rate improves by 23% with AI-driven enrichment.",
        "format_validation":   "## Learning Objectives\n- Apply X\n## The Concept\nY\n## Sources\n- Z",
        "text_extraction":     "## Learning Objectives\n- Apply linear algebra\n- Understand tensors\n## The Concept\n...",
        "metadata_generation": "## Learning Objectives\n- Apply embeddings in GTM pipelines\n## The Concept\n...",
        "diagram_generation":  "A pipeline diagram showing: raw data → feature engineering → model training → inference → GTM action. Left to right flow, 5 nodes.",
    }

    task = {
        "type":      task_type,
        "content":   sample_content[task_type],
        "lesson_id": "smoke-test/00-test",
    }

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        print("ERROR: ZHIPUAI_API_KEY not set. Use run.ps1 or export manually.")
        sys.exit(1)

    try:
        result = call_peon(task)
        print(f"PASS — response ({len(result)} chars):")
        print(result[:300])
        print("\nSmoke test passed.")
    except PeonError as e:
        print(f"FAIL — PeonError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"FAIL — invalid task: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Peon dispatcher — interrupt resolver for GLM-5.1 Handlers")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Make one real Flash call and assert non-empty response")
    parser.add_argument("--task-type",   default="citation_lookup",
                        choices=list(TASK_SCHEMA.keys()),
                        help="Task type to smoke-test (default: citation_lookup)")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args.task_type)
    else:
        parser.print_help()
        print("\nThis script is imported by Handlers. Use --smoke-test to validate the endpoint.")


if __name__ == "__main__":
    main()
