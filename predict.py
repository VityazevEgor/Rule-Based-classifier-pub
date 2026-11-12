import argparse
import csv
import json
from pathlib import Path

from classifier import classify_case, load_rules

ROOT = Path(__file__).resolve().parent
FIELDS = ["case_id", "label", "matched_rules", "fallback_ops", "explanation"]


def case_ids(split):
    names = ["dev", "test"] if split == "all" else [split]
    ids = []
    for n in names:
        with open(ROOT / "data" / f"labels_{n}.csv", encoding="utf-8", newline="") as f:
            ids += [r["case_id"] for r in csv.DictReader(f)]
    return sorted(ids)


def predict(split, rules_path, out_path=None):
    ruleset = load_rules(rules_path)
    out_path = out_path or ROOT / "results" / f"predictions_{split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for cid in case_ids(split):
            case = json.loads((ROOT / "data" / "input" / f"{cid}.json").read_text(encoding="utf-8"))
            res = classify_case(case, ruleset)
            w.writerow({"case_id": cid, "label": res["label"],
                        "matched_rules": ";".join(res["matched_rules"]),
                        "fallback_ops": res["fallback_ops"],
                        "explanation": res["explanation"].replace("\n", " ")})
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], required=True)
    ap.add_argument("--rules", default=str(ROOT / "rules" / "rules.yaml"))
    a = ap.parse_args()
    print("written", predict(a.split, a.rules).name)
