#!/usr/bin/env python3
"""Решающие числа: связь мишеней по всей решётке весов, условная мишень,
мутации (в том числе потолок отбора на перемешанных метках), перенос весов
между границами, цена признаков."""
import pickle, math, random, time, sys
from collections import defaultdict
from pathlib import Path
import t2_grid as G

FEATS = ["repeat", "spread", "fresh", "rare", "glob"]
CUR = (0.5, 0.2, 0.3, 0.0, 0.0)
SPLITS = ["2026-08-05", "2026-08-09", "2026-08-11", "2026-08-15", "2026-08-19"]


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def wfmt(w):
    return "/".join("%.1f" % x for x in w)


def grid_all(F, keys, targets):
    """Для каждого набора весов — AUC по каждой мишени."""
    ws, res = [], {t: [] for t in targets}
    for w in G.simplex(FEATS):
        s = G.score_all(F, w, FEATS, keys)
        ws.append(w)
        for tn, t in targets.items():
            res[tn].append(G.auc(s, t, keys))
    return ws, res


def block(field, split, label):
    train, test = G.build(field, split)
    F = G.features(train)
    keys = list(train)
    t1, t2 = G.targets(train, test)

    print("\n" + "=" * 100)
    print("%s | граница %s | узлов %d | T1=%d T2=%d" % (label, split, len(keys), sum(t1.values()), sum(t2.values())))

    # --- мутации ---
    rnd = random.Random(11)
    # M1: мишень тождественна частоте, известной на обучении
    m1 = {k: 1 if train[k]["n"] > 1 else 0 for k in keys}
    # M2: перемешанные метки T1
    vals = [t1[k] for k in keys]
    rnd.shuffle(vals)
    m2 = dict(zip(keys, vals))
    # M3: условная — среди повторившихся, переехал ли
    rep_keys = [k for k in keys if t1[k]]
    cond = {k: t2[k] for k in rep_keys}

    tg = {"T1": t1, "T2": t2, "M1 частота": m1, "M2 шум": m2}
    ws, res = grid_all(F, keys, tg)

    print("\n  связь мишеней по всей решётке (%d наборов весов):" % len(ws))
    for a, b in (("T1", "T2"), ("T1", "M1 частота"), ("T1", "M2 шум")):
        print("    AUC(%s) vs AUC(%s): Spearman %.3f, Pearson %.3f"
              % (a, b, spearman(res[a], res[b]), pearson(res[a], res[b])))

    print("\n  %-22s %-8s %-8s %-8s %-8s" % ("набор весов", "AUC T1", "AUC T2", "M1", "M2"))
    def row(name, w):
        i = min(range(len(ws)), key=lambda j: sum(abs(ws[j][t] - w[t]) for t in range(len(w))))
        print("  %-22s %-8.3f %-8.3f %-8.3f %-8.3f  %s"
              % (name, res["T1"][i], res["T2"][i], res["M1 частота"][i], res["M2 шум"][i], wfmt(ws[i])))
    row("текущая", CUR)
    for f in FEATS:
        w = tuple(1.0 if g == f else 0.0 for g in FEATS)
        row("только " + f, w)
    for tn in ("T1", "T2", "M1 частота", "M2 шум"):
        i = max(range(len(ws)), key=lambda j: (res[tn][j] if res[tn][j] == res[tn][j] else -1))
        print("  %-22s %-8.3f %-8.3f %-8.3f %-8.3f  %s   <- лучший по %s"
              % ("перебор→" + tn, res["T1"][i], res["T2"][i], res["M1 частота"][i], res["M2 шум"][i], wfmt(ws[i]), tn))

    print("\n  ПРОВЕРКА НА ПУСТОТУ:")
    i1 = max(range(len(ws)), key=lambda j: res["M1 частота"][j])
    print("    M1 (мишень = «n>1 на обучении», тождественна частоте): лучший набор %s, AUC %.3f."
          % (wfmt(ws[i1]), res["M1 частота"][i1]))
    print("      перебор обязан выродиться в чистый repeat и упереться в потолок — веса выше это и показывают.")
    i2 = max(range(len(ws)), key=lambda j: res["M2 шум"][j])
    print("    M2 (метки T1 перемешаны): лучший из %d наборов даёт AUC %.3f при истинном 0.5."
          % (len(ws), res["M2 шум"][i2]))
    print("      это ПОТОЛОК ОТБОРА: любой выигрыш перебора над текущей мерой меньше (%.3f-0.5=%.3f) — шум отбора."
          % (res["M2 шум"][i2], res["M2 шум"][i2] - 0.5))

    # условная мишень
    if rep_keys:
        print("\n  УСЛОВНАЯ МИШЕНЬ T2|T1 (только среди повторившихся, n=%d, из них переехали %d):"
              % (len(rep_keys), sum(cond.values())))
        for f in FEATS:
            w = tuple(1.0 if g == f else 0.0 for g in FEATS)
            s = G.score_all(F, w, FEATS, rep_keys)
            print("    только %-7s AUC %.3f" % (f, G.auc(s, cond, rep_keys)))
        s = G.score_all(F, CUR, FEATS, rep_keys)
        print("    текущая     AUC %.3f" % G.auc(s, cond, rep_keys))
    return ws, res, F, keys, t1, t2


ws, res, F, keys, t1, t2 = block("pkey", "2026-08-09", "ПОРТАБЕЛЬНАЯ ИДЕНТИЧНОСТЬ")
block("lkey", "2026-08-09", "СВОБОДНАЯ ИДЕНТИЧНОСТЬ")

# --- честный перенос весов: выбираем на одной границе, меряем на другой ---
print("\n" + "=" * 100)
print("ПЕРЕНОС ВЕСОВ МЕЖДУ ГРАНИЦАМИ (выбор на A, замер на B) — portable key")
pick = {}
for split in SPLITS:
    tr, te = G.build("pkey", split)
    Fx, ks = G.features(tr), None
    ks = list(tr)
    a1, a2 = G.targets(tr, te)
    best = None
    for w in G.simplex(FEATS):
        s = G.score_all(Fx, w, FEATS, ks)
        a = G.auc(s, a1, ks)
        if a == a and (best is None or a > best[0]):
            best = (a, w)
    pick[split] = best[1]
    print("  выбор на %s: %s (AUC на своей границе %.3f)" % (split, wfmt(best[1]), best[0]))

print("\n  %-20s %s" % ("веса", " ".join("%-14s" % s for s in SPLITS)))
cands = [("текущая", CUR), ("только fresh", (0, 0, 1, 0, 0)), ("только repeat", (1, 0, 0, 0, 0))]
cands += [("выбран на %s" % s, pick[s]) for s in SPLITS]
seen = set()
for nm, w in cands:
    if w in seen:
        continue
    seen.add(w)
    row = "  %-20s" % (nm + " " + wfmt(w))
    for split in SPLITS:
        tr, te = G.build("pkey", split)
        Fx = G.features(tr)
        ks = list(tr)
        a1, a2 = G.targets(tr, te)
        s = G.score_all(Fx, w, FEATS, ks)
        row += " %.3f/%.3f  " % (G.auc(s, a1, ks), G.auc(s, a2, ks))
    print(row + "   (T1/T2)")

# --- цена признаков ---
print("\n" + "=" * 100)
print("ЦЕНА ПРИЗНАКОВ")
D = pickle.load((Path(__file__).resolve().parent / "occ.pkl").open("rb"))
occ = D["occ"]
print("  разбор архива штатным парсером: %.2f c на %d файлов / %d эпизодов / %d вхождений"
      % (D["parse_seconds"], D["files"], D["episodes"], len(occ)))

t0 = time.time()
for _ in range(20):
    agg = defaultdict(lambda: {"n": 0, "last": "", "projects": set(), "type": ""})
    for o in occ:
        r = agg[o["pkey"]]
        r["n"] += 1
        r["projects"].add(o["project"])
        r["type"] = o["type"]
        if o["at"] > r["last"]:
            r["last"] = o["at"]
agg_t = (time.time() - t0) / 20
t0 = time.time()
for _ in range(20):
    Fx = G.features(agg)
feat_t = (time.time() - t0) / 20
ks = list(agg)
t0 = time.time()
for _ in range(200):
    G.score_all(Fx, CUR, FEATS, ks)
sc_t = (time.time() - t0) / 200
print("  свёртка вхождений в узлы (n, projects, last):  %.4f c на весь архив = %.1f мкс/вхождение"
      % (agg_t, 1e6 * agg_t / len(occ)))
print("  вычисление четырёх признаков из свёртки:       %.4f c на %d узлов = %.1f мкс/узел"
      % (feat_t, len(ks), 1e6 * feat_t / len(ks)))
print("  свёртка одного набора весов в оценку:          %.4f c на %d узлов = %.1f мкс/узел"
      % (sc_t, len(ks), 1e6 * sc_t / len(ks)))
t0 = time.time()
cnt = 0
for w in G.simplex(FEATS):
    cnt += 1
print("  один проход решётки (%d наборов) без AUC: %.4f c" % (cnt, time.time() - t0))
