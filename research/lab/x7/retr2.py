#!/usr/bin/env python3
"""Разбор дефекта, который поймал потолок: голова распределения.
Плюс тот же замер без узлов-предпочтений и AUC без них же."""
import math, re
from collections import defaultdict, Counter
import numpy as np
import feat, formulas
from feat import auc, NAMES

FLD = "pkey"
T0, T1 = "2026-08-15", "2026-08-26"
WORD = re.compile(r"[a-zA-Zа-яА-Я0-9_./-]{3,}")


def tk(s):
    return [w.lower() for w in WORD.findall(s or "")]


keys, X, agg = feat.build(FLD, T0)
y = feat.labels(FLD, keys, T0, T1)
border = max((formulas._pt(r["last"]) for r in agg.values() if r["last"]), default=None)
A = {k: formulas.ages(agg[k], border) for k in keys}
cur = np.array([formulas.f_current(agg[k], A[k]) for k in keys])

ep_facts = defaultdict(set)
for o in feat.OCC:
    if T0 <= o["day"] < T1:
        ep_facts[o["ep"]].add(o[FLD])

print("=" * 96)
print("ДЕФЕКТ, КОТОРЫЙ ПОЙМАЛ ПОТОЛОК: концентрация в голове")
pairs = Counter()
for ei, s in ep_facts.items():
    for k in s:
        if k in agg:
            pairs[k] += 1
tot = sum(pairs.values())
print("  пар (эпизод, уже известный узел) в окне проверки: %d, различных узлов %d" % (tot, len(pairs)))
print("  верхние узлы по числу эпизодов, в которых они всплыли:")
for k, c in pairs.most_common(8):
    print("    %-4d эпизодов  %-6s %s" % (c, agg[k]["type"][:6], str(k)[:74]))
top5 = sum(c for _, c in pairs.most_common(5))
print("  доля пар, приходящаяся на 5 самых частых узлов: %.1f%% (%d из %d)" % (100 * top5 / tot, top5, tot))
print("  все они типа preference: %s" % ({agg[k]["type"] for k, _ in pairs.most_common(5)}))
print("  Отсюда: статический top-10 попадает в 0.746 полноты, не глядя на запрос вовсе.")
print("  Потолок на перемешанной релевантности даёт ровно то же число — сигнал у подачи нулевой.")

# ---------------------------------------------------------------- без предпочтений
print("\n" + "=" * 96)
print("ТО ЖЕ БЕЗ УЗЛОВ-ПРЕДПОЧТЕНИЙ (они и есть голова)")
mask = np.array([agg[k]["type"] != "preference" for k in keys])
keys2 = [k for k, m in zip(keys, mask) if m]
idx2 = {k: i for i, k in enumerate(keys2)}
y2 = y[mask]
print("  узлов %d, положительных %d (%.1f%%)" % (len(keys2), int(y2.sum()), 100 * y2.mean()))
print("\n  AUC одиночных признаков без предпочтений (топ-10 по отклонению от 0.5):")
rows = sorted(((abs(auc(X[mask][:, j], y2) - 0.5), auc(X[mask][:, j], y2), nm)
               for j, nm in enumerate(NAMES)), reverse=True)[:10]
for _, a, nm in rows:
    print("    %-16s %.3f" % (nm, a))
print("\n  формулы без предпочтений:")
for name, src, fn in formulas.FORMULAS:
    v = np.array([fn(agg[k], A[k]) for k in keys2])
    print("    %-26s AUC %.3f" % (name, auc(v, y2)))

# ---------------------------------------------------------------- подача без предпочтений
print("\n" + "=" * 96)
print("ПОДАЧА БЕЗ ПРЕДПОЧТЕНИЙ: есть ли запросо-зависимый сигнал в остатке")
df = defaultdict(int)
for k in keys2:
    for t in set(tk(agg[k]["content"])):
        df[t] += 1
N = len(keys2)
idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
ktoks = {k: set(tk(agg[k]["content"])) for k in keys2}
eps = {e["i"]: e for e in feat.EPS}
static = {
    "текущая мера": np.array([formulas.f_current(agg[k], A[k]) for k in keys2]),
    "свежесть (LRU)": np.array([formulas.f_lru(agg[k], A[k]) for k in keys2]),
    "ACT-R d=0.5": np.array([formulas._actr(0.5)(agg[k], A[k]) for k in keys2]),
}
rows = []
for ei, rel in sorted(ep_facts.items()):
    rel = {k for k in rel if k in idx2}
    if not rel:
        continue
    rt = set(tk(eps[ei]["req"]))
    ov = np.array([sum(idf.get(t, 0.0) for t in rt & ktoks[k]) / math.sqrt(1 + len(ktoks[k]))
                   for k in keys2])
    r = {"rel": {idx2[k] for k in rel}, "n": len(rel)}
    r["совпадение (IDF)"] = ov
    for nm, s in static.items():
        r[nm] = s
        sn = (s - s.min()) / (s.max() - s.min() + 1e-12)
        r["%s x совпадение" % nm] = sn * (ov / (ov.max() + 1e-12))
    r["случайный"] = np.random.RandomState(ei).rand(len(keys2))
    rows.append(r)
print("  эпизодов с хотя бы одним известным непредпочтением: %d, пар %d"
      % (len(rows), sum(r["n"] for r in rows)))
names = ["случайный", "текущая мера", "свежесть (LRU)", "ACT-R d=0.5", "совпадение (IDF)",
         "текущая мера x совпадение", "свежесть (LRU) x совпадение", "ACT-R d=0.5 x совпадение"]


def rec(rws, nm, k):
    a = 0.0
    for r in rws:
        o = np.argsort(-r[nm], kind="mergesort")[:k]
        a += sum(1 for j in o if j in r["rel"]) / r["n"]
    return a / len(rws)


print("\n  %-30s %-10s %-10s %-10s" % ("ранжирование", "recall@5", "recall@10", "recall@25"))
for nm in names:
    print("  %-30s %-10.3f %-10.3f %-10.3f" % (nm, rec(rows, nm, 5), rec(rows, nm, 10), rec(rows, nm, 25)))

print("\n  ПОТОЛОК: релевантность перемешана между эпизодами, 300 посевов, max recall@10 по %d способам"
      % len(names))
best = []
rels = [(r["rel"], r["n"]) for r in rows]
for sd in range(300):
    perm = np.random.RandomState(sd).permutation(len(rows))
    b = 0.0
    for nm in names:
        a = 0.0
        for i, r in enumerate(rows):
            rl, n = rels[perm[i]]
            o = np.argsort(-r[nm], kind="mergesort")[:10]
            a += sum(1 for j in o if j in rl) / n
        b = max(b, a / len(rows))
    best.append(b)
best.sort()
print("  медиана %.3f | 95-й %.3f | максимум %.3f" % (best[len(best) // 2], best[int(0.95 * len(best))], best[-1]))
print("  лучшая настоящая: %s recall@10 %.3f"
      % (max(names, key=lambda n: rec(rows, n, 10)), max(rec(rows, n, 10) for n in names)))
