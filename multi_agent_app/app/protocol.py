"""Shared parsing of the reviewer's verdict.

Real models rarely answer with a bare 'APPROVED: ...'. They write '**APPROVED**', add leading
whitespace, or open with 'Verdict: APPROVED'. Treating those as rejections caused false
deadlocks in live mode, so the orchestrator and the detector both use this one parser.
"""
import re

_VERDICT = re.compile(r"\b(APPROVED|REJECTED)\b")
_MARKUP = re.compile(r"[*_`#>]")


def parse_verdict(text: str):
    """Return 'APPROVED', 'REJECTED' or None (no verdict found).

    The first verdict word in the reply wins, so 'REJECTED - cannot be APPROVED yet' is a
    rejection. Matching is case-sensitive on purpose: prose like 'once approved' is not a verdict.
    """
    m = _VERDICT.search(_MARKUP.sub("", text or ""))
    return m.group(1) if m else None


def is_strict_verdict(text: str) -> bool:
    """True when the reply follows the exact protocol: starts with APPROVED or REJECTED."""
    return _MARKUP.sub("", text or "").strip().upper().startswith(("APPROVED", "REJECTED"))
