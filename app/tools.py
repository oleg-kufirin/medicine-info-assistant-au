import yaml
from pydantic import BaseModel
from typing import Literal

class CostInput(BaseModel):
    patient_type: Literal["general","concession"] = "general"
    quantity: int = 1
    repeats: int = 0
    safety_net_progress: float = 0.0   # 0..1
    under_copay_flag: bool = True

def load_pricing_config(path: str = "data/pricing_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def estimate_cost(args: CostInput, cfg: dict) -> dict:
    pbs = cfg.get("pbs", {})
    gen = float(pbs.get("general_copay", 31.60))
    con = float(pbs.get("concession_copay", 7.70))
    per = con if args.patient_type == "concession" else gen
    # MVP: ignore under_copay / brand premiums; refine later.
    scripts = 1 + max(args.repeats, 0)
    total = per * scripts
    return {
        "per_script": round(per, 2),
        "scripts_total": scripts,
        "repeats_total_cost": round(total, 2),
        "notes": "Indicative figures only; confirm on official PBS sites."
    }
