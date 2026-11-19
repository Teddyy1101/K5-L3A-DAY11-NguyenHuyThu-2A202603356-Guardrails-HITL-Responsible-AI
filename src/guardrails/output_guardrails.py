"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


_FIELD_ASSIGNMENT = r"""["']?\s*[:=]\s*"""
_CREDENTIAL_VALUE = (
    r"""("(?:[^"\\\r\n]|\\[^\r\n])+"|'(?:[^'\\\r\n]|\\[^\r\n])+'"""
    r"""|\x60[^\x60\r\n]+\x60|\[REDACTED\]|(?!["'\x60])[^\s,;}\]]+)"""
)
_CREDENTIAL_FIELDS = {"password", "api_key", "db_host"}

_REDACTION_PATTERNS = {
    "password": (
        r"(\b(?:(?:(?:admin|current|new|old)[ _-]*)?password|mật\s+khẩu)\b"
        rf"(?:{_FIELD_ASSIGNMENT}"
        r"""|\s+(?:is|là)\s+(?=\S*[0-9!@#$%^&*]|["'\x60])))"""
        + _CREDENTIAL_VALUE
    ),
    "email": r"[\w.%+-]+@(?:[\w-]+\.)+[a-z]{2,}\b",
    "api_key": (
        rf"(\bapi[\s_-]*key\b{_FIELD_ASSIGNMENT}){_CREDENTIAL_VALUE}"
        r"|(?<![\w-])sk-[a-z0-9_-]+"
    ),
    "db_host": (
        rf"(\b(?:db|database)[\s_-]*host\b{_FIELD_ASSIGNMENT}){_CREDENTIAL_VALUE}"
        r"|(?<![\w.-])(?:[a-z0-9-]+\.)+internal(?::[0-9]+)?(?![\w-]|\.[\w-])"
    ),
    "phone": r"(?<![\w+])(?:0[0-9]{9,10}|\+84[0-9]{9,10})(?!\w)",
    "national_id": (
        r"(?<![\w$€£¥])(?<![0-9][.,])(?:[0-9]{9}|[0-9]{12})(?!\w)"
        r"(?![.,][0-9]|\s*(?:vnd\b|vnđ\b|đ\b|₫|%))"
    ),
}

if DEMO_SECRETS:
    _DEMO_VALUES = "|".join(
        re.escape(secret) for secret in sorted(DEMO_SECRETS, key=len, reverse=True)
    )
    _REDACTION_PATTERNS["protected_value"] = rf"(?<!\w)(?:{_DEMO_VALUES})(?!\w)"


def _redact_credential(match: re.Match[str]) -> str:
    prefix, value = match.groups()
    if prefix is None:
        return "[REDACTED]"
    quote = value[0] if value[0] in {"\"", "'", "\x60"} else ""
    return f"{prefix}{quote}[REDACTED]{quote}"


def content_filter(response: str) -> dict:
    """Redact PII and protected values in model response text.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response

    for name, pattern in _REDACTION_PATTERNS.items():
        replacement = _redact_credential if name in _CREDENTIAL_FIELDS else "[REDACTED]"
        redacted, count = re.subn(pattern, replacement, redacted, flags=re.IGNORECASE)
        if count:
            issues.append(f"{name}: {count} found")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Output guardrail callback
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Applies content_filter() to the user-visible response.
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract the user-visible text across response parts."""
        content = getattr(llm_response, "content", None)
        if content is None:
            return ""
        return "".join(
            part.text
            for part in (content.parts or [])
            if getattr(part, "text", None) and not getattr(part, "thought", False)
        )

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        result = content_filter(response_text)
        if result["safe"]:
            return llm_response

        content = llm_response.content
        parts = [types.Part.from_text(text=result["redacted"])]
        parts.extend(part for part in (content.parts or []) if part.text is None)
        llm_response.content = content.model_copy(update={"parts": parts})
        self.redacted_count += 1
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
