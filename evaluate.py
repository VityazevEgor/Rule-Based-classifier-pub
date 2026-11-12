import os
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("DRIFTBENCH_DIR", ROOT.parent / "DriftBench-Offline-Pub"))
INDEX_CSV = CORPUS / "dataset" / "index.csv"
SECOND_CSV = CORPUS / "annotated_dataset" / "annotation_second_labeled.csv"
CLASSES = [1, 2, 3, 4]
N_BOOT = 2000
SEED = 42


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def macro_f1(y, p):


    return float(f1_score(y, p, average="macro", zero_division=0))


def kappa(y, p, weights=None):
    if len(set(y)) < 2 and len(set(p)) < 2 and set(y) == set(p):
        return 1.0
    v = cohen_kappa_score(y, p, labels=CLASSES, weights=weights)
    return float(v)


def basic_metrics(y, p):
    y, p = list(y), list(p)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, p, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y, p, labels=CLASSES)
    n = len(y)
    return {
        "n": n,
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": macro_f1(y, p),
        "per_class": {str(c): {"precision": float(prec[i]), "recall": float(rec[i]),
                               "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion": cm.tolist(),
        "kappa": kappa(y, p),
        "kappa_linear": kappa(y, p, "linear"),
        "under_rate": float(np.mean([a < b for a, b in zip(p, y)])),
        "over_rate": float(np.mean([a > b for a, b in zip(p, y)])),
        "recall_class4": float(rec[3]) if sup[3] > 0 else None,
    }


def bootstrap_ci(y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.array(y), np.array(p)
    n = len(y)
    acc, f1s, kap = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        acc.append(accuracy_score(yy, pp))
        f1s.append(macro_f1(yy, pp))
        kap.append(kappa(list(yy), list(pp), "linear"))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy": ci(acc), "macro_f1": ci(f1s), "kappa_linear": ci(kap)}


def is_held_out(value):
    v = (value or "").strip()
    if v.startswith("["):
        items = json.loads(v)
        return any("held" in str(x) for x in items)
    return "held" in v


def rel(a, b):
    return "=" if a == b else ("<" if a < b else ">")


def slice_metrics(ids, gold, pred, index):
    slices = {}

    def add(name, group_fn):
        groups = {}
        for c in ids:
            groups.setdefault(group_fn(index[c]), []).append(c)
        slices[name] = {}
        for g, cs in sorted(groups.items()):
            y = [gold[c] for c in cs]
            p = [pred[c] for c in cs]
            slices[name][str(g)] = {"n": len(cs), "accuracy": float(accuracy_score(y, p)),
                                    "macro_f1": macro_f1(y, p)}
    add("operator_split", lambda r: "held_out" if is_held_out(r["operator_split"]) else "seen")
    add("case_type", lambda r: r["case_type"])
    add("environment", lambda r: r["environment"])
    return slices


def pair_consistency(ids, gold, pred, index):
    pairs = {}
    for c in ids:
        r = index[c]
        if r["case_type"] == "paired" and r["pair_id"]:
            pairs.setdefault(r["pair_id"], {})[r["environment"]] = c
    total = same = 0
    for members in pairs.values():
        if "dev" in members and "prod" in members:
            d, pr = members["dev"], members["prod"]
            total += 1
            same += rel(gold[d], gold[pr]) == rel(pred[d], pred[pr])
    return {"n_pairs": total, "consistent": same,
            "rate": (same / total) if total else None}


def vs_expert2(ids, gold, pred):
    e2 = {r["case_id"]: int(r["label"]) for r in read_csv(SECOND_CSV)}
    common = [c for c in ids if c in e2]
    if len(common) < 2:
        return {"n": len(common)}
    y1 = [gold[c] for c in common]
    y2 = [e2[c] for c in common]
    p = [pred[c] for c in common]
    return {"n": len(common),
            "kappa_clf_vs_expert2": kappa(p, y2),
            "kappa_expert1_vs_expert2": kappa(y1, y2),
            "kappa_clf_vs_expert1": kappa(p, y1)}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(split, rules_path, pred_path=None, out_path=None):
    pred_path = pred_path or ROOT / "results" / f"predictions_{split}.csv"
    preds = read_csv(pred_path)
    gold = {r["case_id"]: int(r["label"]) for r in read_csv(ROOT / "data" / f"labels_{split}.csv")}
    ids = sorted(gold)
    pred = {r["case_id"]: int(r["label"]) for r in preds}
    if set(pred) != set(gold):
        raise SystemExit("predictions do not cover the split")
    y = [gold[c] for c in ids]
    p = [pred[c] for c in ids]

    m = basic_metrics(y, p)
    m["ci95"] = bootstrap_ci(y, p)
    index = {r["case_id"]: r for r in read_csv(INDEX_CSV)}
    m["slices"] = slice_metrics(ids, gold, pred, index)
    m["pair_consistency"] = pair_consistency(ids, gold, pred, index)
    m["vs_expert2"] = vs_expert2(ids, gold, pred)

    fb_cases = sum(1 for r in preds if int(r["fallback_ops"]) > 0)
    fb_ops = sum(int(r["fallback_ops"]) for r in preds)
    n_ops = sum(len(json.loads((ROOT / "data" / "input" / f"{c}.json").read_text(encoding="utf-8"))["diff"])
                for c in ids)
    m["fallback_case_rate"] = fb_cases / len(preds)
    m["fallback_op_rate"] = fb_ops / n_ops if n_ops else 0.0
    m["n_ops"] = n_ops
    m["rules_sha256"] = file_sha(rules_path)

    out_path = out_path or ROOT / "results" / f"metrics_{split}.json"
    Path(out_path).write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--rules", default=str(ROOT / "rules" / "rules.yaml"))
    a = ap.parse_args()
    res = evaluate(a.split, a.rules)
    print(f"n={res['n']} acc={res['accuracy']:.3f} macroF1={res['macro_f1']:.3f} "
          f"kappa_lin={res['kappa_linear']:.3f} under={res['under_rate']:.3f} over={res['over_rate']:.3f} "
          f"fallback_cases={res['fallback_case_rate']:.3f}")
