import re
from dataclasses import dataclass

RISKY_RE = re.compile(
    r"(prescrib|dose|how much should i take|diagnos|urgent|severe pain|medical advice|what should i take)",
    re.I,
)

@dataclass
class SafetyDecision:
    allow: bool
    message: str | None = None

REFUSAL = (
    "I can’t provide medical advice such as diagnosing conditions or recommending doses. "
    "For personal guidance, please speak to your pharmacist or GP. "
    "Here is general information that may help below."
)

def guard(query: str) -> SafetyDecision:
    if RISKY_RE.search(query or ""):
        return SafetyDecision(allow=False, message=REFUSAL)
    return SafetyDecision(allow=True)
