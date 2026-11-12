import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classifier import CATEGORY_CLASS, classify_case, load_rules
from classifier.engine import category_to_class, expand_op, parse_image
from classifier.quantity import parse_quantity
import evaluate as ev
import predict as pr

RULES = ROOT / "rules" / "rules.yaml"


@pytest.mark.parametrize("s,expected", [
    ("100m", Fraction(1, 10)), ("1", 1), ("1.5", Fraction(3, 2)),
    ("128Mi", 128 * 2**20), ("2Gi", 2 * 2**30), ("1k", 1000), ("1Ki", 1024),
    ("1e3", 1000), ("500m", Fraction(1, 2)), (3, 3), ("0", 0),
])
def test_quantity_values(s, expected):
    assert parse_quantity(s) == expected


def test_quantity_comparison_not_string():
    assert parse_quantity("1Gi") > parse_quantity("512Mi")
    assert parse_quantity("1000m") == parse_quantity("1")
    assert "9" > "10"
    assert parse_quantity("9") < parse_quantity("10")


@pytest.mark.parametrize("bad", ["abc", "", "1Zi", None, True, [1]])
def test_quantity_invalid(bad):
    assert parse_quantity(bad) is None


@pytest.mark.parametrize("cat,dev,prod", [
    ("security_direct", 4, 4), ("security_weakening", 3, 3),
    ("availability", 2, 3), ("behavior", 2, 2), ("cosmetic", 1, 1),
])
def test_category_table(cat, dev, prod):
    assert category_to_class(cat, "dev") == dev
    assert category_to_class(cat, "prod") == prod
    assert CATEGORY_CLASS[cat] == (dev, prod)


def case(kind, env, diff, desired=None):
    return {"case_id": "T", "resource_kind": kind, "resource_name": "x",
            "namespace": "ns", "environment": env, "diff": diff,
            "desired": desired or {}}


def test_aggregation_max_and_fallback():
    rs = load_rules(RULES)
    d = [{"op": "add", "path": "/metadata/labels/owner", "old": None, "new": "a"},
         {"op": "replace", "path": "/spec/template/spec/hostPID", "old": False, "new": True}]
    res = classify_case(case("Deployment", "dev", d), rs)
    assert res["label"] == 4
    d2 = [{"op": "add", "path": "/metadata/labels/owner", "old": None, "new": "a"},
          {"op": "replace", "path": "/spec/replicas", "old": 1, "new": 3}]
    assert classify_case(case("Deployment", "dev", d2), rs)["label"] == 2
    d3 = [{"op": "replace", "path": "/spec/unknownField", "old": 1, "new": 2}]
    res3 = classify_case(case("Deployment", "dev", d3), rs)
    assert res3["label"] == 2 and res3["fallback_ops"] == 1


def test_environment_effect_only_on_availability():
    rs = load_rules(RULES)
    rep0 = [{"op": "replace", "path": "/spec/replicas", "old": 1, "new": 0}]
    assert classify_case(case("Deployment", "dev", rep0), rs)["label"] == 2
    assert classify_case(case("Deployment", "prod", rep0), rs)["label"] == 3
    priv = [{"op": "replace", "path": "/spec/template/spec/containers/0/securityContext/privileged",
             "old": False, "new": True}]
    assert classify_case(case("Deployment", "dev", priv), rs)["label"] == 4
    assert classify_case(case("Deployment", "prod", priv), rs)["label"] == 4


def test_direction_matters():
    rs = load_rules(RULES)
    add_drop = [{"op": "add", "path": "/spec/template/spec/containers/0/securityContext/capabilities",
                 "old": None, "new": {"drop": ["ALL"]}}]
    assert classify_case(case("Deployment", "prod", add_drop), rs)["label"] == 2
    rem_rule = [{"op": "remove", "path": "/rules/1", "old": {"resources": ["events"], "verbs": ["create"]},
                 "new": None}]
    assert classify_case(case("Role", "prod", rem_rule), rs)["label"] == 3


def test_expand_and_image():
    ops = expand_op({"op": "add", "path": "/data", "old": None, "new": {"a": {"b": 1}}})
    assert [o["path"] for o in ops] == ["/data", "/data/a", "/data/a/b"]
    img = parse_image("registry.k8s.io/x/y:v1@sha256:ab")
    assert img["registry"] == "registry.k8s.io" and img["tag"] == "v1" and img["digest"] == "sha256:ab"
    assert parse_image("nginx")["tag"] == "latest"


def test_predict_deterministic(tmp_path):
    a = pr.predict("dev", RULES, tmp_path / "a.csv")
    b = pr.predict("dev", RULES, tmp_path / "b.csv")
    assert a.read_bytes() == b.read_bytes()


def test_no_forbidden_keys_in_input():
    forbidden = {"mutation_operators", "operator_params", "operator_split",
                 "case_type", "pair_id", "app", "chart"}
    allowed = {"case_id", "resource_kind", "resource_name", "namespace",
               "environment", "diff", "desired"}
    files = list((ROOT / "data" / "input").glob("CASE_*.json"))
    assert len(files) == 322
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        assert set(rec) == allowed
        assert not (set(rec) & forbidden)


def test_splits_disjoint_and_labelled():
    import csv
    dev = {r["case_id"] for r in csv.DictReader(open(ROOT / "data" / "labels_dev.csv"))}
    test = {r["case_id"] for r in csv.DictReader(open(ROOT / "data" / "labels_test.csv"))}
    assert len(dev) == 70 and len(test) == 252 and not dev & test


def test_metrics_against_sklearn():
    y = [1, 2, 2, 3, 4, 4, 3, 2, 1, 4, 3, 2]
    p = [1, 2, 3, 3, 4, 3, 3, 1, 1, 4, 4, 2]
    m = ev.basic_metrics(y, p)
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, p))
    assert m["macro_f1"] == pytest.approx(skm.f1_score(y, p, average="macro"))
    assert m["kappa"] == pytest.approx(skm.cohen_kappa_score(y, p))
    assert m["kappa_linear"] == pytest.approx(skm.cohen_kappa_score(y, p, weights="linear"))
    assert m["confusion"] == skm.confusion_matrix(y, p, labels=[1, 2, 3, 4]).tolist()
    pc = skm.precision_recall_fscore_support(y, p, labels=[1, 2, 3, 4], zero_division=0)
    for i, c in enumerate([1, 2, 3, 4]):
        assert m["per_class"][str(c)]["precision"] == pytest.approx(pc[0][i])
        assert m["per_class"][str(c)]["recall"] == pytest.approx(pc[1][i])
        assert m["per_class"][str(c)]["f1"] == pytest.approx(pc[2][i])
    assert m["under_rate"] == pytest.approx(sum(b < a for a, b in zip(y, p)) / len(y))
    assert m["over_rate"] == pytest.approx(sum(b > a for a, b in zip(y, p)) / len(y))
    assert m["recall_class4"] == pytest.approx(2 / 3)


def test_bootstrap_reproducible_and_brackets():
    y = [1, 2, 3, 4] * 10
    p = [1, 2, 3, 3] * 10
    a, b = ev.bootstrap_ci(y, p, n_boot=200), ev.bootstrap_ci(y, p, n_boot=200)
    assert a == b
    lo, hi = a["accuracy"]
    assert lo <= 0.75 <= hi


def test_pair_consistency_toy():
    idx = {"A": {"case_type": "paired", "pair_id": "P1", "environment": "dev"},
           "B": {"case_type": "paired", "pair_id": "P1", "environment": "prod"},
           "C": {"case_type": "paired", "pair_id": "P2", "environment": "dev"},
           "D": {"case_type": "paired", "pair_id": "P2", "environment": "prod"},
           "E": {"case_type": "single", "pair_id": "", "environment": "dev"}}
    gold = {"A": 2, "B": 3, "C": 3, "D": 3, "E": 1}
    pred = {"A": 2, "B": 3, "C": 3, "D": 4, "E": 1}
    r = ev.pair_consistency(list(gold), gold, pred, idx)
    assert r == {"n_pairs": 2, "consistent": 1, "rate": 0.5}
