import json
import re
from fractions import Fraction
from pathlib import Path

import yaml

from .quantity import parse_quantity

CATEGORY_ORDER = ["security_direct", "security_weakening", "availability",
                  "behavior", "cosmetic"]

CATEGORY_CLASS = {
    "security_direct": (4, 4),
    "security_weakening": (3, 3),
    "availability": (2, 3),
    "behavior": (2, 2),
    "cosmetic": (1, 1),
}
FALLBACK_CLASS = 2
FALLBACK_ID = "FALLBACK"
VALID_OPS = {"add", "remove", "replace"}


def category_to_class(category, environment):
    dev_cls, prod_cls = CATEGORY_CLASS[category]
    return prod_cls if environment == "prod" else dev_cls


def unescape(seg):
    return seg.replace("~1", "/").replace("~0", "~")


def escape(seg):
    return str(seg).replace("~", "~0").replace("/", "~1")


def split_pointer(path):
    return [unescape(s) for s in path.split("/")[1:]] if path else []


def get_at(doc, segs):
    cur = doc
    for s in segs:
        if isinstance(cur, dict):
            if s not in cur:
                return None
            cur = cur[s]
        elif isinstance(cur, list):
            try:
                cur = cur[int(s)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


_MISSING = object()


def _walk(path, old, new, out, op_root, depth=0):
    if old == new and old is not _MISSING:
        return
    if old is _MISSING or new is _MISSING:
        present = new if old is _MISSING else old
        out.append({"op": "add" if old is _MISSING else "remove", "path": path,
                    "old": None if old is _MISSING else old,
                    "new": None if new is _MISSING else new, "root": False})
        if isinstance(present, dict):
            for k, v in present.items():
                _walk(f"{path}/{escape(k)}",
                      _MISSING if old is _MISSING else v,
                      v if old is _MISSING else _MISSING, out, op_root, depth + 1)
        elif isinstance(present, list):
            for i, v in enumerate(present):
                _walk(f"{path}/{i}",
                      _MISSING if old is _MISSING else v,
                      v if old is _MISSING else _MISSING, out, op_root, depth + 1)
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for k in list(old) + [k for k in new if k not in old]:
            _walk(f"{path}/{escape(k)}", old.get(k, _MISSING), new.get(k, _MISSING),
                  out, op_root, depth + 1)
    elif isinstance(old, list) and isinstance(new, list):
        out.append({"op": "replace", "path": path, "old": old, "new": new, "root": False})
        for i in range(max(len(old), len(new))):
            _walk(f"{path}/{i}", old[i] if i < len(old) else _MISSING,
                  new[i] if i < len(new) else _MISSING, out, op_root, depth + 1)
    else:
        out.append({"op": "replace", "path": path, "old": old, "new": new, "root": False})
        if isinstance(new, (dict, list)) and not isinstance(old, (dict, list)):
            for k, v in (new.items() if isinstance(new, dict) else enumerate(new)):
                _walk(f"{path}/{escape(k)}", _MISSING, v, out, op_root, depth + 1)


def expand_op(op):
    base = {"op": op["op"], "path": op["path"], "old": op.get("old"),
            "new": op.get("new"), "root": True}
    out = [base]
    o, n = op.get("old"), op.get("new")
    kind = op["op"]
    if kind == "add":
        o = _MISSING
    elif kind == "remove":
        n = _MISSING
    if isinstance(o, (dict, list)) or isinstance(n, (dict, list)):
        tmp = []
        if o is _MISSING:
            for k, v in (n.items() if isinstance(n, dict) else enumerate(n)):
                _walk(f"{op['path']}/{escape(k)}", _MISSING, v, tmp, op)
        elif n is _MISSING:
            for k, v in (o.items() if isinstance(o, dict) else enumerate(o)):
                _walk(f"{op['path']}/{escape(k)}", v, _MISSING, tmp, op)
        else:
            _walk(op["path"], o, n, tmp, op)
            tmp = [t for t in tmp if t["path"] != op["path"]]
        out.extend(tmp)
    return out


def parse_image(ref):
    if not isinstance(ref, str) or not ref:
        return None
    digest = None
    if "@" in ref:
        ref, digest = ref.split("@", 1)
    first, _, rest = ref.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, name = first, rest
    else:
        registry, name = "docker.io", ref
    tag = None
    last = name.rsplit("/", 1)[-1]
    if ":" in last:
        name, tag = name.rsplit(":", 1)
    if tag is None and digest is None:
        tag = "latest"
    return {"registry": registry, "repo": name, "tag": tag, "digest": digest}


def _num(v):
    q = parse_quantity(v)
    return q


def _field(ctx, name):
    head, _, rest = name.partition(".")
    cur = ctx.get(head)
    if rest:
        for part in rest.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
    return cur


def _as_list(v):
    return v if isinstance(v, list) else [v]


def eval_cond(cond, ctx, params):
    val = _field(ctx, cond["field"])
    res = _eval_cond_value(cond, val, ctx, params)
    return (not res) if cond.get("negate") else res


def _eval_cond_value(cond, val, ctx, params):
    if "eq" in cond:
        return val == cond["eq"] and type(val) is type(cond["eq"])
    if "ne" in cond:
        return not (val == cond["ne"] and type(val) is type(cond["ne"]))
    if "in" in cond:
        return val in cond["in"] if isinstance(val, (str, int, float, bool)) else False
    if "not_in" in cond:
        return val not in cond["not_in"]
    if "regex" in cond:
        return isinstance(val, str) and re.search(cond["regex"], val) is not None
    if "not_regex" in cond:
        return isinstance(val, str) and re.search(cond["not_regex"], val) is None
    if "contains_any" in cond:
        if val is None:
            return False
        targets = set(map(str, cond["contains_any"]))
        return any(str(x) in targets for x in _as_list(val))
    if "any_regex" in cond:
        if val is None:
            return False
        return any(isinstance(x, str) and re.search(cond["any_regex"], x)
                   for x in _as_list(val))
    if "is_null" in cond:
        return (val is None) == bool(cond["is_null"])
    if "is_empty" in cond:
        return (val in (None, "", [], {})) == bool(cond["is_empty"])
    if "eq_field" in cond:
        return val == _field(ctx, cond["eq_field"])
    if "ne_field" in cond:
        return val != _field(ctx, cond["ne_field"])
    for key, fn in (("gt", lambda a, b: a > b), ("ge", lambda a, b: a >= b),
                    ("lt", lambda a, b: a < b), ("le", lambda a, b: a <= b)):
        if key in cond:
            a, b = _num(val), _num(cond[key])
            return a is not None and b is not None and fn(a, b)
    if "qty_decreased" in cond or "qty_increased" in cond:
        a, b = _num(ctx.get("old")), _num(ctx.get("new"))
        if a is None or b is None:
            return False
        want_dec = "qty_decreased" in cond
        ok = b < a if want_dec else b > a
        return ok if cond.get("qty_decreased", cond.get("qty_increased")) else not ok
    if "qty_ratio_le" in cond:
        a, b = _num(ctx.get("old")), _num(ctx.get("new"))
        return a is not None and b is not None and a > 0 and b / a <= Fraction(str(cond["qty_ratio_le"]))
    if "registry_untrusted" in cond:
        trusted = params[cond["registry_untrusted"]]
        return isinstance(val, str) and val not in trusted
    raise ValueError(f"unknown condition: {cond}")


def _expand_macros(s, macros):
    for k in sorted(macros, key=len, reverse=True):
        s = s.replace("{" + k + "}", macros[k])
    return s


def load_rules(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    macros = spec.get("macros", {})

    for _ in range(3):
        macros = {k: _expand_macros(v, macros) for k, v in macros.items()}
    groups = spec.get("kind_groups", {})
    params = spec.get("params", {})
    rules, seen = [], set()
    for r in spec["rules"]:
        if r["id"] in seen:
            raise ValueError(f"duplicate rule id {r['id']}")
        seen.add(r["id"])
        if r["category"] not in CATEGORY_CLASS:
            raise ValueError(f"bad category in {r['id']}")
        kinds = None
        if r.get("kinds"):
            kinds = set()
            for k in r["kinds"]:
                kinds.update(groups.get(k, [k]))
        ops = set(r.get("op", VALID_OPS))
        if not ops <= VALID_OPS:
            raise ValueError(f"bad op in {r['id']}")
        rules.append({
            "id": r["id"], "description": r["description"], "category": r["category"],
            "kinds": kinds, "ops": ops,
            "path": re.compile(_expand_macros(r["path"], macros)) if r.get("path") else None,
            "where": r.get("where", []),
        })
    rules.sort(key=lambda x: CATEGORY_ORDER.index(x["category"]))
    return {"rules": rules, "params": params}


def _rule_matches(rule, ctx, params):
    if rule["kinds"] is not None and ctx["kind"] not in rule["kinds"]:
        return False
    if ctx["op"] not in rule["ops"]:
        return False
    if rule["path"] is not None and not rule["path"].fullmatch(ctx["path"]):
        return False
    return all(eval_cond(c, ctx, params) for c in rule["where"])


def classify_operation(op, case, ruleset):
    best = None
    desired = case.get("desired")
    for p in expand_op(op):
        segs = split_pointer(p["path"])
        ctx = {
            "op": p["op"], "path": p["path"], "old": p["old"], "new": p["new"],
            "key": segs[-1] if segs else "",
            "kind": case["resource_kind"], "resource_name": case["resource_name"],
            "environment": case["environment"],
            "desired_parent": get_at(desired, segs[:-1]),
            "old_image": parse_image(p["old"]), "new_image": parse_image(p["new"]),
        }
        hit = next((r for r in ruleset["rules"]
                    if _rule_matches(r, ctx, ruleset["params"])), None)
        if hit is None:
            continue
        cls = category_to_class(hit["category"], case["environment"])
        cand = (cls, hit["id"], hit["description"], hit["category"], p["path"])
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        best = (FALLBACK_CLASS, FALLBACK_ID, "no rule matched", "fallback", op["path"])
    return best


def classify_case(case, ruleset):
    results = [classify_operation(op, case, ruleset) for op in case["diff"]]
    if not results:
        return {"label": FALLBACK_CLASS, "matched_rules": [], "fallback_ops": 0,
                "explanation": "empty diff"}
    top = max(results, key=lambda r: (r[0], r[1] != FALLBACK_ID))
    matched = sorted({r[1] for r in results if r[1] != FALLBACK_ID})
    fb = sum(1 for r in results if r[1] == FALLBACK_ID)
    expl = f"{top[1]}: {top[2]} [{top[4]}]"[:300]
    return {"label": top[0], "matched_rules": matched, "fallback_ops": fb,
            "explanation": expl, "n_ops": len(results)}
