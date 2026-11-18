#!/usr/bin/env python3
"""Одна величина или три.

Подача (что дать агенту сейчас) зависит от текущего запроса, продление и
отставка — нет. Проверяем, разные ли это порядки: ранжируем один и тот же
набор узлов статической мерой и запросо-зависимым совпадением (IDF-веса,
формула BM25-типа, ничего не обучается) и меряем полноту на будущих эпизодах.
"""
import math, pickle
from collections import defaultdict
import numpy as np
import feat, formulas
from feat import auc

FLD = "pkey"
T0, T1 = "2026-08-15", "2026-08-26"


def idf_table(agg):
    df = defaultdict(int)
    for k, r in agg.items():
        for t in set(feat_toks(r["content"])):
            df[t] += 1
    N = len(agg)
    return {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}


import re
WORD = re.compile(r"[a-zA-Zа-яА-Я0-9_./-]{3,}")


def feat_toks(s):
    return [w.lower() for w in WORD.findall(s or "")]


def main():
    keys, X, agg = feat.build(FLD, T0)
    border = max((formulas._pt(r["last"]) for r in agg.values() if r["last"]), default=None)
    A = {k: formulas.ages(agg[k], border) for k in keys}
    static = {
        "текущая мера": np.array([formulas.f_current(agg[k], A[k]) for k in keys]),
        "свежесть (LRU)": np.array([formulas.f_lru(agg[k], A[k]) for k in keys]),
        "ACT-R d=0.5": np.array([formulas._actr(0.5)(agg[k], A[k]) for k in keys]),
    }
    idf = idf_table(agg)
    ktoks = {k: set(feat_toks(agg[k]["content"])) for k in keys}
    kidx = {k: i for i, k in enumerate(keys)}
    pool = set(keys)

    # эпизоды окна проверки и факты, которые в них реально всплыли
    ep_facts = defaultdict(set)
    for o in feat.OCC:
        if T0 <= o["day"] < T1:
            ep_facts[o["ep"]].add(o[FLD])
    eps = {e["i"]: e for e in feat.EPS}

    def overlap(req_toks, k):
        s = 0.0
        for t in req_toks & ktoks[k]:
            s += idf.get(t, 0.0)
        return s / math.sqrt(1.0 + len(ktoks[k]))

    ranked = []
    for ei, rel in sorted(ep_facts.items()):
        rel = rel & pool
        if not rel:
            continue
        e = eps[ei]
        rt = set(feat_toks(e["req"]))
        ov = np.array([overlap(rt, k) for k in keys])
        row = {"n_rel": len(rel), "rel": {kidx[k] for k in rel}}
        row["совпадение с запросом (IDF)"] = ov
        for nm, s in static.items():
            row[nm] = s
            # произведение рангов: статика * совпадение (обе нормированы в [0,1])
            sn = (s - s.min()) / (s.max() - s.min() + 1e-12)
            on = ov / (ov.max() + 1e-12)
            row["%s x совпадение" % nm] = sn * on
        row["случайный"] = np.random.RandomState(ei).rand(len(keys))
        ranked.append(row)

    print("=" * 96)
    print("ПОДАЧА: ранжирование %d узлов под КОНКРЕТНЫЙ запрос будущего эпизода." % len(keys))
    print("Эпизодов в окне [%s, %s) с хотя бы одним уже известным фактом: %d" % (T0, T1, len(ranked)))
    tot_rel = sum(r["n_rel"] for r in ranked)
    print("Всего пар (эпизод, релевантный узел): %d" % tot_rel)
    allep = len([e for e in ep_facts])
    print("Эпизодов с фактами всего %d; у %d все факты новые, они выброшены." % (allep, allep - len(ranked)))

    names = ["случайный", "текущая мера", "свежесть (LRU)", "ACT-R d=0.5",
             "совпадение с запросом (IDF)",
             "текущая мера x совпадение", "свежесть (LRU) x совпадение", "ACT-R d=0.5 x совпадение"]
    print("\n  %-32s %-9s %-9s %-9s %-9s" % ("ранжирование", "recall@5", "recall@10", "recall@25", "MRR"))
    out = {}
    for nm in names:
        r5 = r10 = r25 = mrr = 0.0
        for row in ranked:
            s = row[nm]
            order = np.argsort(-s, kind="mergesort")
            pos = [i for i, j in enumerate(order) if j in row["rel"]]
            r5 += sum(1 for p in pos if p < 5) / row["n_rel"]
            r10 += sum(1 for p in pos if p < 10) / row["n_rel"]
            r25 += sum(1 for p in pos if p < 25) / row["n_rel"]
            mrr += 1.0 / (pos[0] + 1) if pos else 0.0
        n = len(ranked)
        out[nm] = (r5 / n, r10 / n, r25 / n, mrr / n)
        print("  %-32s %-9.3f %-9.3f %-9.3f %-9.3f" % (nm, r5 / n, r10 / n, r25 / n, mrr / n))

    # потолок отбора: перемешиваем соответствие эпизод->релевантные
    print("\n  ПОТОЛОК ОТБОРА: релевантность перемешана между эпизодами, 200 посевов,")
    print("  максимум recall@10 по %d способам ранжирования." % len(names))
    rels = [r["rel"] for r in ranked]
    nrel = [r["n_rel"] for r in ranked]
    best = []
    for sd in range(200):
        rng = np.random.RandomState(sd)
        perm = rng.permutation(len(ranked))
        b = 0.0
        for nm in names:
            acc = 0.0
            for i, row in enumerate(ranked):
                rl, nr = rels[perm[i]], nrel[perm[i]]
                order = np.argsort(-row[nm], kind="mergesort")[:10]
                acc += sum(1 for j in order if j in rl) / nr
            b = max(b, acc / len(ranked))
        best.append(b)
    best.sort()
    print("  медиана %.3f | 95-й %.3f | максимум %.3f" % (best[len(best) // 2], best[int(0.95 * len(best))], best[-1]))
    gain = out["свежесть (LRU) x совпадение"][1] - out["свежесть (LRU)"][1]
    print("  выигрыш «свежесть x совпадение» над чистой свежестью по recall@10: %+0.3f" % gain)
    print("  потолок сверх медианы шума: %.3f" % (best[len(best) // 2]))

    # бутстрап по эпизодам для главной пары
    print("\n  ПАРНЫЙ БУТСТРАП по эпизодам (recall@10):")
    for a, b in (("совпадение с запросом (IDF)", "свежесть (LRU)"),
                 ("свежесть (LRU) x совпадение", "свежесть (LRU)"),
                 ("свежесть (LRU) x совпадение", "совпадение с запросом (IDF)")):
        d = []
        rng = np.random.RandomState(9)
        for _ in range(2000):
            idx = rng.randint(0, len(ranked), len(ranked))
            va = vb = 0.0
            for i in idx:
                row = ranked[i]
                for nm, acc in ((a, "a"), (b, "b")):
                    order = np.argsort(-row[nm], kind="mergesort")[:10]
                    v = sum(1 for j in order if j in row["rel"]) / row["n_rel"]
                    if acc == "a":
                        va += v
                    else:
                        vb += v
            d.append((va - vb) / len(idx))
        d.sort()
        print("    %-30s против %-30s %+0.3f ДИ95 [%+0.3f, %+0.3f] %s"
              % (a, b, sum(d) / len(d), d[50], d[-50],
                 "значимо" if (d[50] > 0 or d[-50] < 0) else "НЕ значимо"))


if __name__ == "__main__":
    main()
