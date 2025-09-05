import re

COST_KWS = re.compile(r"(cost|price|copay|co-?pay|out[- ]?of[- ]?pocket|concession|safety net)", re.I)
COVERAGE_KWS = re.compile(r"(pbs|authority|restriction|criteria|listed|reimbursement)", re.I)

def route(query: str) -> str:
    if COST_KWS.search(query or ""):
        return "cost"
    if COVERAGE_KWS.search(query or ""):
        return "coverage"
    return "general"
