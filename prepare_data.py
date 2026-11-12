import os
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("DRIFTBENCH_DIR", ROOT.parent / "DriftBench-Offline-Pub"))
PREP = CORPUS / "dataset"
RAW = PREP / "raw"
SPLITS = PREP / "splits" / "cases.json"
LABELS_MAIN = CORPUS / "annotated_dataset" / "annotation_main_labeled.csv"
LABELS_SECOND = CORPUS / "annotated_dataset" / "annotation_second_labeled.csv"

ALLOWED_META = ("resource_kind", "resource_name", "namespace", "environment")
FORBIDDEN_KEYS = {"mutation_operators", "operator_params", "operator_split",
                   "case_type", "pair_id", "app", "chart"}
INPUT_KEYS = {"case_id", "resource_kind", "resource_name", "namespace",
              "environment", "diff", "desired"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_forbidden(obj, found=None):
    return sorted(set(obj) & FORBIDDEN_KEYS)


def main():
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    dev, test = list(splits["dev"]), list(splits["test"])
    if len(dev) != 70 or len(test) != 252:
        sys.exit(f"ERROR: split sizes {len(dev)}/{len(test)} != 70/252")
    if set(dev) & set(test):
        sys.exit("ERROR: dev/test overlap")

    case_dirs = sorted(p for p in RAW.iterdir() if p.is_dir())
    ids = [p.name for p in case_dirs]
    if len(ids) != 322:
        sys.exit(f"ERROR: {len(ids)} cases != 322")
    if set(ids) != set(dev) | set(test):
        sys.exit("ERROR: raw cases do not match splits")

    labels = {}
    with open(LABELS_MAIN, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["case_id"]] = int(row["label"])
    for cid in ids:
        if labels.get(cid) not in (1, 2, 3, 4):
            sys.exit(f"ERROR: bad/missing label for {cid}")

    out = ROOT / "data" / "input"
    out.mkdir(parents=True, exist_ok=True)
    for p in case_dirs:
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        rec = {"case_id": p.name}
        for k in ALLOWED_META:
            rec[k] = meta[k]
        rec["diff"] = json.loads((p / "diff.json").read_text(encoding="utf-8"))
        rec["desired"] = yaml.safe_load((p / "desired.yaml").read_text(encoding="utf-8"))
        bad = find_forbidden(rec)
        if bad or set(rec) != INPUT_KEYS:
            sys.exit(f"ERROR: forbidden/unexpected keys in {p.name}: {bad}")
        (out / f"{p.name}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")

    for name, lst in (("dev", dev), ("test", test)):
        with open(ROOT / "data" / f"labels_{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["case_id", "label"])
            for cid in sorted(lst):
                w.writerow([cid, labels[cid]])

    manifest = {
        "date": datetime.date.today().isoformat(),
        "n_cases": len(ids),
        "n_dev": len(dev),
        "n_test": len(test),
        "sha256": {
            "annotation_main_labeled.csv": sha256(LABELS_MAIN),
            "annotation_second_labeled.csv": sha256(LABELS_SECOND),
        },
    }
    (ROOT / "data" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("prepared", len(ids), "cases; dev", len(dev), "test", len(test))


if __name__ == "__main__":
    main()
