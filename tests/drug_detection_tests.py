"""

Drug name extraction tests for DrugDetectionAgent.
The script exits non-zero if any query returns an unexpected extracted name.

"""
from __future__ import annotations

import os, sys, time

from pathlib import Path
from typing import Dict, List

# Ensure imports work when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in map(str, sys.path):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "app"))
    
from app.drug_detection_agent import DrugDetectionAgent

Case = Dict[str, object]

AGENT = DrugDetectionAgent()

CASES_10_RANDOM: List[Case] = [
    # 10 mixed queries without overlapping the prompt examples
    {"query": "Does cetirizine make you sleepy?", "expected": ["cetirizine"], "note": "Generic antihistamine"},
    {"query": "Is Voltaren (diclofenac) gel safe for daily use?", "expected": ["Voltaren", "diclofenac"], "note": "Brand + generic; include both"},
    {"query": "Can I take omeprazole and famotidine together?", "expected": ["omeprazole", "famotidine"], "note": "Two names; extract both"},
    {"query": "Is Crestor the same as rosuvastatin?", "expected": ["Crestor", "rosuvastatin"], "note": "Brand + generic; include both"},
    {"query": "Does melatonin affect REM sleep?", "expected": ["melatonin"], "note": "Active ingredient (supplement)"},
    {"query": "Are antihistamines safe during pregnancy?", "expected": [], "note": "Class only"},
    {"query": "What do decongestants do to blood vessels?", "expected": [], "note": "Class only"},
    {"query": "Do vitamins help with energy?", "expected": [], "note": "Category only"},
    {"query": "Are sleeping pills habit-forming?", "expected": [], "note": "Category only"},
    {"query": "Does Pepto-Bismol contain bismuth subsalicylate?", "expected": ["Pepto-Bismol", "bismuth subsalicylate"], "note": "Brand and ingredient"},
]

CASES_20_POSITIVE: List[Case] = [
    # 20 queries with explicit drug/ingredient names (brand or generic)
    {"query": "Is ibuprofen safe on an empty stomach?", "expected": ["ibuprofen"], "note": "Generic name present"},
    {"query": "How long does amoxicillin take to work?", "expected": ["amoxicillin"], "note": "Generic antibiotic"},
    {"query": "Can I drink alcohol with metformin?", "expected": ["metformin"], "note": "Diabetes med"},
    {"query": "Does sertraline cause weight gain?", "expected": ["sertraline"], "note": "SSRI generic"},
    {"query": "Is Panadol (paracetamol) safe for headaches?", "expected": ["Panadol", "paracetamol"], "note": "Brand + generic; include both"},
    {"query": "Can I split a 50 mg tramadol tablet?", "expected": ["tramadol"], "note": "Opioid analgesic"},
    {"query": "What is the max daily dose of naproxen?", "expected": ["naproxen"], "note": "NSAID generic"},
    {"query": "Does omeprazole interact with clopidogrel?", "expected": ["omeprazole", "clopidogrel"], "note": "Two names"},
    {"query": "Is Crestor the same as rosuvastatin?", "expected": ["Crestor", "rosuvastatin"], "note": "Brand + generic; include both"},
    {"query": "How quickly does Xanax work?", "expected": ["Xanax"], "note": "Brand name"},
    {"query": "Is it safe to take Claritin with pseudoephedrine?", "expected": ["Claritin", "pseudoephedrine"], "note": "Two names"},
    {"query": "Can I combine Advil and Tylenol?", "expected": ["Advil", "Tylenol"], "note": "Two brands"},
    {"query": "Does doxycycline make you sensitive to sunlight?", "expected": ["doxycycline"], "note": "Generic antibiotic"},
    {"query": "What are side effects of prednisone?", "expected": ["prednisone"], "note": "Steroid generic"},
    {"query": "Is Augmentin safe for sinus infection?", "expected": ["Augmentin"], "note": "Brand name"},
    {"query": "Can I take insulin glargine in the morning?", "expected": ["insulin glargine"], "note": "Biologic name"},
    {"query": "Does warfarin interact with cranberry juice?", "expected": ["warfarin"], "note": "Anticoagulant generic"},
    {"query": "Can you overdose on loperamide?", "expected": ["loperamide"], "note": "Antidiarrheal generic"},
    {"query": "Does atorvastatin cause muscle pain?", "expected": ["atorvastatin"], "note": "Statin generic"},
    {"query": "Is melatonin safe for teenagers?", "expected": ["melatonin"], "note": "Active ingredient (supplement)"},
]

CASES_20_NEGATIVE: List[Case] = [
    # 20 queries without explicit names (only categories/classes or none)
    {"query": "Do painkillers cause stomach ulcers?", "expected": [], "note": "Category only"},
    {"query": "Are antibiotics effective against viruses?", "expected": [], "note": "Category only"},
    {"query": "What does an antihistamine do?", "expected": [], "note": "Class only"},
    {"query": "Can cough syrups make you drowsy?", "expected": [], "note": "Category only"},
    {"query": "Is it safe to mix different vitamins?", "expected": [], "note": "Category only"},
    {"query": "Do antidepressants affect weight?", "expected": [], "note": "Class only"},
    {"query": "What is the best pain reliever for back pain?", "expected": [], "note": "Category only"},
    {"query": "How do decongestants work?", "expected": [], "note": "Class only"},
    {"query": "Should I take probiotics after medicine?", "expected": [], "note": "No explicit name"},
    {"query": "Are sleeping pills addictive?", "expected": [], "note": "Category only"},
    {"query": "Do blood pressure tablets cause cough?", "expected": [], "note": "Category only"},
    {"query": "How fast do anti-inflammatory drugs work?", "expected": [], "note": "Class only"},
    {"query": "What is the difference between syrup and suspension?", "expected": [], "note": "Dosage forms"},
    {"query": "Can over-the-counter cold medicine help with flu?", "expected": [], "note": "Category only"},
    {"query": "Are laxatives safe for long-term use?", "expected": [], "note": "Category only"},
    {"query": "Should I avoid alcohol while on antibiotics?", "expected": [], "note": "Category only"},
    {"query": "Is it safe to combine a painkiller and an antihistamine?", "expected": [], "note": "Categories"},
    {"query": "Why do some medications cause dry mouth?", "expected": [], "note": "Generic term 'medications'"},
    {"query": "Who discovered the printing press?", "expected": [], "note": "Not medical"},
    {"query": "What is the shelf life of tablets?", "expected": [], "note": "Dosage form"},
]

CASES = CASES_20_POSITIVE

def detection_available() -> bool:
    """Heuristic: if the agent cannot create a chain (e.g., missing API key or prompt), skip tests."""
    try:
        chain = AGENT._get_chain()  # type: ignore[attr-defined]
        return chain is not None
    except Exception:
        return False


def main() -> int:
    if not detection_available():
        print("All cases skipped because detection is unavailable (missing API key or prompt?).")
        return 0

    failures: List[Dict[str, object]] = []
    total_inference_time = 0.0

    for idx, case in enumerate(CASES):
        query = case["query"]  # type: ignore[index]
        expected = case["expected"]  # type: ignore[index]

        start_time = time.perf_counter()
        actual = AGENT.extract_drug_names(str(query))
        elapsed = time.perf_counter() - start_time
        total_inference_time += elapsed

        if actual != expected:
            failures.append({
                "index": idx,
                "query": query,
                "note": case.get("note"),
                "expected": expected,
                "actual": actual,
            })
            print(f"{idx:02d} [FAIL] {query}")
            print(f"  expected='{expected}' actual='{actual}'")
        else:
            print(f"{idx:02d} [OK] names={actual} :: {query}")

    total = len(CASES)
    average_inference_time = total_inference_time / total if total else 0.0
    print("\nSummary:")
    print(f"Ran {total} cases -> passed={total - len(failures)} failed={len(failures)}")
    print(f"Total inference time: {total_inference_time:.4f}s")
    print(f"Average inference time per request: {average_inference_time:.4f}s")

    if failures:
        print("Failures:")
        for failure in failures:
            print(f" - {failure['index']:02d}: {failure['query']} :: note={failure.get('note')}")
            print(f"   expected='{failure['expected']}' actual='{failure['actual']}'")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
