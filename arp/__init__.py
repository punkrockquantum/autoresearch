"""
arp — the Agentic Research Platform.

A layer on top of the autoresearch experiment loop that turns "hack train.py and
eyeball val_bpb" into a general, subject-driven research engine:

  * probabilistic search (Thompson sampling over hypotheses) that lands on
    deterministic verdicts (anytime-valid confidence sequences + an exhaustive,
    enumerable confirmation grid),
  * a capability memory that evolves per subject, so the platform gets better at
    the kinds of questions it is asked,
  * human intervention by prompt: second-guess any result and the engine
    re-opens it and keeps testing until the thesis is proven or refuted,
  * web search agents that score findings for business impact,
  * subject suggestions mined from prior prompts, chats and findings.

Everything here is standard library only, so the platform runs without a GPU and
without installing anything beyond what autoresearch already needs.
"""

__version__ = "0.1.0"

from arp.models import (  # noqa: F401  (re-exported for convenience)
    Challenge,
    Finding,
    Hypothesis,
    Lead,
    Prompt,
    Subject,
    Trial,
    TrialSpec,
    TrialResult,
    Verdict,
)

__all__ = [
    "__version__",
    "Challenge",
    "Finding",
    "Hypothesis",
    "Lead",
    "Prompt",
    "Subject",
    "Trial",
    "TrialSpec",
    "TrialResult",
    "Verdict",
]
