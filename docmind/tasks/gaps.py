"""Gap analysis / architecture critique for a senior tech-management lens."""
from __future__ import annotations

from docmind.store.vectordb import Hit
from docmind.tasks.context import format_context

_RUBRIC = """Evaluate the documented process/architecture against this rubric:
1. Ownership & RACI — is it clear who owns each step/component?
2. Failure modes & resilience — what happens when a step/service fails? retries, fallbacks, DR?
3. Single points of failure — components or people without redundancy.
4. Scalability — where does this break under 10x load or scope growth?
5. Security & compliance — authn/z, data handling, secrets, audit, PII.
6. Observability — logging, metrics, tracing, alerting; can issues be detected?
7. Dependencies & integration — external systems, coupling, versioning.
8. Documentation gaps — what is referenced but not defined, or missing entirely?
9. Process bottlenecks & manual steps — toil, approvals, hand-offs."""

SYSTEM_GAPS = (
    "You are a seasoned VP of Engineering reviewing internal process and "
    "architecture documentation to give a fast, senior-level read. Work ONLY "
    "from the provided context excerpts; where the context is silent on a rubric "
    "dimension, call that out explicitly as a documentation gap rather than "
    "assuming. Cite source tags for concrete findings.\n\n" + _RUBRIC + "\n\n"
    "Output format:\n"
    "## Summary (3-5 sentences)\n"
    "## Gaps & Risks (grouped by rubric area; each: finding, why it matters, "
    "severity High/Med/Low, source or 'not documented')\n"
    "## Recommended next steps (prioritised)"
)

SYSTEM_CRITIQUE = (
    "You are a principal architect giving a critical design review. Work ONLY "
    "from the provided context excerpts. Assess correctness, robustness, "
    "scalability, security, and operability of the described architecture, cite "
    "source tags, and be direct about weaknesses. Where the context lacks detail "
    "needed to judge a dimension, say what you'd need to see.\n\n"
    "Output: ## Overview  ## Strengths  ## Concerns (with severity)  "
    "## Open questions  ## Recommendations"
)


def build_gaps_prompt(hits: list[Hit], scope: str | None) -> tuple[str, str]:
    context = format_context(hits)
    scope_line = f"Focus scope: {scope}\n\n" if scope else ""
    user = f"{scope_line}Context excerpts:\n\n{context}"
    return SYSTEM_GAPS, user


def build_critique_prompt(hits: list[Hit], scope: str | None) -> tuple[str, str]:
    context = format_context(hits)
    scope_line = f"Focus scope: {scope}\n\n" if scope else ""
    user = f"{scope_line}Context excerpts:\n\n{context}"
    return SYSTEM_CRITIQUE, user
