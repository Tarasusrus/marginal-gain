#!/usr/bin/env python3
"""Вторая мишень и её связь с первой. Ничего не пишем, только считаем."""
import pickle, math, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

D = pickle.load((Path(__file__).resolve().parent / "occ.pkl").open("rb"))
occ = D["occ"]
SPLITS = ["2026-08-05", "2026-08-09", "2026-08-11", "2026-08-15", "2026-08-19"]


def parse_time(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def build(field, split):
    """train: признаки по первой половине. test: что было во второй."""
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


def targets(train, test):
    """t1: повторился. t2: всплыл в проекте, которого у него раньше не было."""
    t1, t2 = {}, {}
    for k, r in train.items():
        te = test.get(k)
        t1[k] = 1 if te else 0
        t2[k] = 1 if te and (te["projects"] - r["projects"]) else 0
    return t1, t2


def phi(a, b, keys):
    n11 = sum(1 for k in keys if a[k] and b[k])
    n10 = sum(1 for k in keys if a[k] and not b[k])
    n01 = sum(1 for k in keys if not a[k] and b[k])
    n00 = sum(1 for k in keys if not a[k] and not b[k])
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / den if den else float("nan"), (n11, n10, n01, n00)


for field, label in (("pkey", "portable (путь целиком)"), ("lkey", "loose (имя файла)")):
    print("=" * 78)
    print("ИДЕНТИЧНОСТЬ: %s" % label)
    for split in SPLITS:
        train, test = build(field, split)
        t1, t2 = targets(train, test)
        keys = list(train)
        n = len(keys)
        s1, s2 = sum(t1.values()), sum(t2.values())
        p, (n11, n10, n01, n00) = phi(t1, t2, keys)
        # условная мишень: среди повторившихся - переехал ли
        rep = [k for k in keys if t1[k]]
        cond = sum(t2[k] for k in rep)
        print("  %s | узлов до границы %3d, после %3d | T1=%3d (%.1f%%) | T2=%3d (%.1f%%) | phi=%.3f | T2 среди T1: %d из %d"
              % (split, n, len(test), s1, 100.0 * s1 / n, s2, 100.0 * s2 / n, p, cond, len(rep)))
        if split == "2026-08-09":
            print("      таблица сопряжённости: T1&T2=%d  T1&!T2=%d  !T1&T2=%d  !T1&!T2=%d"
                  % (n11, n10, n01, n00))
    print()
