import csv, json
from app import chains, safety

def run_eval(path="eval/qa_seed.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        total = 0
        pass_cnt = 0
        for row in rdr:
            total += 1
            q = row["query"]
            exp = row["expected_contains"]
            should_cite = row["should_cite_url"] == "1"
            should_refuse = row["should_refuse"] == "1"

            dec = safety.guard(q)
            if should_refuse and dec.allow:
                print("[FAIL] expected refusal:", q)
                continue
            if should_refuse and not dec.allow:
                pass_cnt += 1
                continue

            passages = chains.retrieve(q)
            ans = chains.synthesize_answer(q, passages)
            ok = True
            if exp and exp.lower() not in json.dumps(ans).lower():
                ok = False
                print("[MISS exp] ", q)
            if should_cite and not ans.get("citations"):
                ok = False
                print("[MISS cite] ", q)
            if ok:
                pass_cnt += 1
        print(f"Score: {pass_cnt}/{total} = {pass_cnt/total:.1%}")

if __name__ == "__main__":
    run_eval()
