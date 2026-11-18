#!/usr/bin/env python3
"""Перебор весов по всем признакам сразу, на обеих мишенях. Плюс мутации."""
import pickle, math, random, itertools, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

D = pickle.load((Path(__file__).resolve().parent / "occ.pkl").open("rb"))
occ = D["occ"]
FEATS = ["repeat", "spread", "fresh", "rare", "glob"]


def parse_time(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def build(field, split):
    train = defaultdict(lambda: {"n": 0, "last": "", "projects": set(), "type": ""})
    test = defaultdict(lambda: {"n": 0, "projects": set()})
    for o in occ:
        if not o["day"]:
            continue
        k = o[field]
        if o["day"] < split:
            r = train[k]
            r["n"] += 1
            r["projects"].add(o["project"])
            r["type"] = o["type"]
            if o["at"] > r["last"]:
                r["last"] = o["at"]
        else:
            r = test[k]
            r["n"] += 1
            r["projects"].add(o["project"])
    return train, test


def features(train):
    newest = max((r["last"] for r in train.values() if r["last"]), default="")
    nb = parse_time(newest)
    out = {}
    for k, r in train.items():
        repeat = min(r["n"], 10) / 10.0
        spread = min(len(r["projects"]), 3) / 3.0
        fresh = 0.0
        na = parse_time(r["last"])
        if na and nb:
            fresh = max(0.0, 1.0 - (nb - na).days / 30.0)
        rare = 1.0 - repeat
        glob = 1.0 if r["type"] in ("preference", "external_resource") else 0.0
        out[k] = {"repeat": repeat, "spread": spread, "fresh": fresh,
                  "rare": rare, "glob": glob}
    return out


def auc(scores, target, keys):
    pos = [scores[k] for k in keys if target[k]]
    neg = [scores[k] for k in keys if not target[k]]
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def thirds(scores, target, keys):
    """Верхняя треть против нижней, как в ADR 0002. Ничьи режем по рангу."""
    order = sorted(keys, key=lambda k: scores[k])
    n = len(order)
    t = max(1, n // 3)
    low, high = order[:t], order[-t:]
    ph = sum(target[k] for k in high) / len(high)
    pl = sum(target[k] for k in low) / len(low)
    ratio = (ph / pl) if pl else float("inf") if ph else float("nan")
    return ph, pl, ratio, sum(target[k] for k in high), len(high), sum(target[k] for k in low), len(low)


def simplex(feats, step=10):
    """Все веса кратные 1/step, сумма 1."""
    n = len(feats)
    for combo in itertools.product(range(step + 1), repeat=n - 1):
        rest = step - sum(combo)
        if rest < 0:
            continue
        yield tuple(x / step for x in combo) + (rest / step,)


def score_all(F, w, feats, keys):
    return {k: sum(w[i] * F[k][feats[i]] for i in range(len(feats))) for k in keys}


def targets(train, test):
    t1, t2 = {}, {}
    for k, r in train.items():
        te = test.get(k)
        t1[k] = 1 if te else 0
        t2[k] = 1 if te and (te["projects"] - r["projects"]) else 0
    return t1, t2


def run(field, split, feats, label, topn=12, extra_targets=None):
    train, test = build(field, split)
    F = features(train)
    keys = list(train)
    t1, t2 = targets(train, test)
    tgts = {"T1 повтор": t1, "T2 перенос": t2}
    if extra_targets:
        tgts.update(extra_targets(train, test, keys))

    rows = []
    for w in simplex(feats):
        s = score_all(F, w, feats, keys)
        row = {"w": w}
        for name, t in tgts.items():
            row[name] = (auc(s, t, keys), thirds(s, t, keys)[2])
        rows.append(row)
    print("=" * 100)
    print("%s | граница %s | узлов %d | T1=%d T2=%d" % (label, split, len(keys), sum(t1.values()), sum(t2.values())))
    print("признаки: %s" % ", ".join(feats))
    for name in tgts:
        rows.sort(key=lambda r: -r[name][0])
        print("\n  лучшие по AUC на мишени «%s»:" % name)
        print("   " + " ".join("%6s" % f for f in feats) + " |" + "".join("  AUC(%s)  лифт" % n[:2] for n in tgts))
        seen = set()
        shown = 0
        for r in rows:
            if shown >= topn:
                break
            sig = tuple(round(x, 2) for x in r["w"])
            if sig in seen:
                continue
            seen.add(sig)
            shown += 1
            line = "   " + " ".join("%6.1f" % x for x in r["w"]) + " |"
            for n2 in tgts:
                a, l = r[n2]
                line += "  %6.3f %6s" % (a, ("%.2f" % l) if l == l and l != float("inf") else "inf")
            print(line)
    return rows, F, keys, tgts, train, test


if __name__ == "__main__":
    run("pkey", "2026-08-09", FEATS, "ПОРТАБЕЛЬНАЯ ИДЕНТИЧНОСТЬ", topn=10)
