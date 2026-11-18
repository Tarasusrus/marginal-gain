#!/usr/bin/env python3
"""Потолок отбора на перемешанных метках (много посевов), парный бутстрап
разности AUC, и сколько данных нужно третьей мишени."""
import math, random
import t2_grid as G

FEATS = ["repeat", "spread", "fresh", "rare", "glob"]
CUR = (0.5, 0.2, 0.3, 0.0, 0.0)
FRESH = (0.1, 0.0, 0.9, 0.0, 0.0)
SPLIT = "2026-08-09"

WS = list(G.simplex(FEATS))


def auc_fast(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def ceiling(field, label, n_shuf=40, seed=3):
    train, test = G.build(field, SPLIT)
    F = G.features(train)
    keys = list(train)
    t1, t2 = G.targets(train, test)
    # предподсчёт оценок для всех наборов весов
    S = [G.score_all(F, w, FEATS, keys) for w in WS]
    rnd = random.Random(seed)
    labels = [t1[k] for k in keys]
    maxes = []
    for _ in range(n_shuf):
        lab = labels[:]
        rnd.shuffle(lab)
        idx_pos = [i for i, v in enumerate(lab) if v]
        idx_neg = [i for i, v in enumerate(lab) if not v]
        best = 0.0
        for s in S:
            vals = [s[k] for k in keys]
            a = auc_fast([vals[i] for i in idx_pos], [vals[i] for i in idx_neg])
            best = max(best, a)
        maxes.append(best)
    maxes.sort()
    real_cur = G.auc(G.score_all(F, CUR, FEATS, keys), t1, keys)
    real_best = max(G.auc(s, t1, keys) for s in S)
    print("\n%s (узлов %d, T1=%d)" % (label, len(keys), sum(t1.values())))
    print("  потолок отбора на шуме: max по %d наборам, %d посевов" % (len(WS), n_shuf))
    print("    медиана %.3f | 95-й процентиль %.3f | максимум %.3f  (истина 0.5)"
          % (maxes[len(maxes) // 2], maxes[int(0.95 * len(maxes))], maxes[-1]))
    print("  реально: текущая AUC %.3f, лучший перебором %.3f, выигрыш перебора %.3f"
          % (real_cur, real_best, real_best - real_cur))
    print("  выигрыш перебора против шумового: %.3f против %.3f сверх 0.5 -> %s"
          % (real_best - real_cur, maxes[len(maxes) // 2] - 0.5,
             "НЕ отличим от шума отбора" if (real_best - real_cur) <= (maxes[len(maxes) // 2] - 0.5)
             else "больше шума отбора"))
    return F, keys, t1, t2


def paired_boot(F, keys, target, wa, wb, n=4000, seed=5):
    """Парный бутстрап: разность AUC(wa)-AUC(wb) на одних и тех же выборках."""
    sa = G.score_all(F, wa, FEATS, keys)
    sb = G.score_all(F, wb, FEATS, keys)
    rnd = random.Random(seed)
    diffs = []
    for _ in range(n):
        smp = [rnd.choice(keys) for _ in keys]
        pos = [k for k in smp if target[k]]
        neg = [k for k in smp if not target[k]]
        if not pos or not neg:
            continue
        da = auc_fast([sa[k] for k in pos], [sa[k] for k in neg])
        db = auc_fast([sb[k] for k in pos], [sb[k] for k in neg])
        diffs.append(da - db)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    frac = sum(1 for d in diffs if d > 0) / len(diffs)
    return sum(diffs) / len(diffs), lo, hi, frac


F, keys, t1, t2 = ceiling("pkey", "ПОРТАБЕЛЬНАЯ ИДЕНТИЧНОСТЬ")
ceiling("lkey", "СВОБОДНАЯ ИДЕНТИЧНОСТЬ")

print("\nПАРНЫЙ БУТСТРАП: свежая мера 0.1/0/0.9 против текущей 0.5/0.2/0.3 (portable, %s)" % SPLIT)
for nm, t in (("T1", t1), ("T2", t2)):
    m, lo, hi, frac = paired_boot(F, keys, t, FRESH, CUR)
    print("  %s: разность AUC %+0.3f, ДИ95 [%+0.3f, %+0.3f], доля выборок с выигрышем %.0f%% -> %s"
          % (nm, m, lo, hi, 100 * frac, "значимо" if lo > 0 else "НЕ значимо"))

# --- сколько данных нужно третьей мишени ---
print("\nТРЕТЬЯ МИШЕНЬ: сколько подставленных подсказок нужно, чтобы отличить полезную от бесполезной")
Z_A, Z_B = 1.96, 0.84


def n_per_group(p0, p1):
    pb = (p0 + p1) / 2
    num = (Z_A * math.sqrt(2 * pb * (1 - pb)) + Z_B * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / (p1 - p0) ** 2)


print("  доля «помогло» в верхней группе против нижней, alpha 0.05, мощность 0.80:")
print("  %-14s %-14s %-12s %s" % ("низ", "верх", "на группу", "всего подсказок"))
for p0, p1 in ((0.10, 0.40), (0.20, 0.50), (0.20, 0.40), (0.30, 0.50), (0.30, 0.45), (0.30, 0.40)):
    n = n_per_group(p0, p1)
    print("  %-14.0f%% %-13.0f%% %-12d %d" % (100 * p0, 100 * p1, n, 2 * n))


def n_auc(a, se_target=0.05, ratio=0.3):
    """Hanley-McNeil: сколько положительных нужно, чтобы SE(AUC) была не хуже se_target."""
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    for np_ in range(5, 100000):
        nn = np_ / ratio
        var = (a * (1 - a) + (np_ - 1) * (q1 - a * a) + (nn - 1) * (q2 - a * a)) / (np_ * nn)
        if math.sqrt(var) <= se_target:
            return np_, int(nn)
    return None, None


print("\n  через AUC (Hanley-McNeil), доля «помогло» 30%%, нужна SE(AUC)<=0.05:")
for a in (0.65, 0.70, 0.75):
    np_, nn = n_auc(a)
    print("    чтобы измерить AUC ~%.2f: положительных %d, отрицательных %d, всего %d" % (a, np_, nn, np_ + nn))
print("\n  текущий темп: 1652 эпизода за 68 дней = %.1f эпизода в день." % (1652 / 68))
print("  подсказка срабатывает не на каждом эпизоде: доля неизвестна, журнал подсказок пуст.")
