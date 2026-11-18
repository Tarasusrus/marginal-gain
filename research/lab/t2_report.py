#!/usr/bin/env python3
"""Сводка: базисы, перебор, устойчивость по границам, мутации."""
import pickle, math, random, itertools
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import t2_grid as G

FEATS = ["repeat", "spread", "fresh", "rare", "glob"]
SPLITS = ["2026-08-05", "2026-08-09", "2026-08-11", "2026-08-15", "2026-08-19"]
CUR = {"repeat": 0.5, "spread": 0.2, "fresh": 0.3, "rare": 0.0, "glob": 0.0}


def wvec(d, feats=FEATS):
    return tuple(d.get(f, 0.0) for f in feats)


def thirds_full(scores, target, keys):
    return G.thirds(scores, target, keys)


def boot_auc(scores, target, keys, n=2000, seed=7):
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [rnd.choice(keys) for _ in keys]
        pos = [scores[k] for k in s if target[k]]
        neg = [scores[k] for k in s if not target[k]]
        if not pos or not neg:
            continue
        tot = 0.0
        for p in pos:
            for q in neg:
                tot += 1.0 if p > q else (0.5 if p == q else 0.0)
        vals.append(tot / (len(pos) * len(neg)))
    vals.sort()
    if not vals:
        return (float("nan"),) * 2
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def line(name, w, F, keys, tgts):
    s = G.score_all(F, w, FEATS, keys)
    out = "  %-34s" % name
    for tn, t in tgts.items():
        a = G.auc(s, t, keys)
        ph, pl, r, hh, nh, hl, nl = thirds_full(s, t, keys)
        rs = "%.1f" % r if r == r and r != float("inf") else ("inf" if r == float("inf") else "n/a")
        out += " | AUC %.3f  верх %2d/%d низ %2d/%d лифт %5s" % (a, hh, nh, hl, nl, rs)
    print(out)


def section(field, split, label):
    train, test = G.build(field, split)
    F = G.features(train)
    keys = list(train)
    t1, t2 = G.targets(train, test)
    tgts = {"T1": t1, "T2": t2}
    print("\n" + "=" * 118)
    print("%s | граница %s | узлов %d | T1=%d (%.1f%%) T2=%d (%.1f%%)"
          % (label, split, len(keys), sum(t1.values()), 100.0 * sum(t1.values()) / len(keys),
             sum(t2.values()), 100.0 * sum(t2.values()) / len(keys)))
    print("  %-34s | %-46s | %-46s" % ("набор весов", "T1 повторится", "T2 переедет в новый проект"))
    line("текущая мера 0.5/0.2/0.3", wvec(CUR), F, keys, tgts)
    for f in FEATS:
        line("только %s" % f, wvec({f: 1.0}), F, keys, tgts)
    line("равные веса 4 признака", wvec({"repeat": .25, "spread": .25, "fresh": .25, "rare": .25}), F, keys, tgts)
    # перебор
    best = {}
    for w in G.simplex(FEATS):
        s = G.score_all(F, w, FEATS, keys)
        for tn, t in tgts.items():
            a = G.auc(s, t, keys)
            if a == a and (tn not in best or a > best[tn][0]):
                best[tn] = (a, w)
    for tn, (a, w) in best.items():
        line("перебор, лучший по %s" % tn, w, F, keys, tgts)
    # доверительный интервал на лучшем по T1 и на текущей
    for nm, w in (("текущая", wvec(CUR)), ("лучший по T1", best["T1"][1]), ("лучший по T2", best["T2"][1])):
        s = G.score_all(F, w, FEATS, keys)
        lo1, hi1 = boot_auc(s, t1, keys)
        lo2, hi2 = boot_auc(s, t2, keys)
        print("    ДИ95 AUC %-14s: T1 [%.3f, %.3f]  T2 [%.3f, %.3f]" % (nm, lo1, hi1, lo2, hi2))
    return F, keys, tgts, best, train, test


F, keys, tgts, best, train, test = section("pkey", "2026-08-09", "ПОРТАБЕЛЬНАЯ ИДЕНТИЧНОСТЬ (путь целиком)")
section("lkey", "2026-08-09", "СВОБОДНАЯ ИДЕНТИЧНОСТЬ (имя файла)")
section("key", "2026-08-09", "ШТАТНАЯ ИДЕНТИЧНОСТЬ fact_key (T2 тут невыразима)")

print("\n" + "=" * 118)
print("УСТОЙЧИВОСТЬ ПО ГРАНИЦАМ, AUC (portable key)")
print("  %-26s %s" % ("набор", " ".join("%-24s" % s for s in SPLITS)))
cands = [("текущая 0.5/0.2/0.3", wvec(CUR)),
         ("только fresh", wvec({"fresh": 1.0})),
         ("только repeat", wvec({"repeat": 1.0})),
         ("только rare", wvec({"rare": 1.0})),
         ("0.2 rep /0.8 fresh", wvec({"repeat": .2, "fresh": .8})),
         ("0.2/0.0/0.7/0.0/0.1", wvec({"repeat": .2, "fresh": .7, "glob": .1}))]
for nm, w in cands:
    row = "  %-26s" % nm
    for split in SPLITS:
        tr, te = G.build("pkey", split)
        Fx = G.features(tr)
        ks = list(tr)
        a1, a2 = G.targets(tr, te)
        s = G.score_all(Fx, w, FEATS, ks)
        row += " T1 %.3f/T2 %.3f    " % (G.auc(s, a1, ks), G.auc(s, a2, ks))
    print(row)
