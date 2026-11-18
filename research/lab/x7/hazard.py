#!/usr/bin/env python3
"""Эмпирическая опасность повторного обращения: описательная статистика, не модель.
Плюс несимметричный порог по Элкану на измеренных долях."""
import numpy as np
from collections import defaultdict
import feat, formulas
from feat import NAMES, auc

FLD = "pkey"
T0, T1 = "2026-08-15", "2026-08-26"

keys, X, agg = feat.build(FLD, T0)
y = feat.labels(FLD, keys, T0, T1)
border = max((formulas._pt(r["last"]) for r in agg.values() if r["last"]), default=None)
A = {k: formulas.ages(agg[k], border) for k in keys}
dlast = np.array([A[k][0] for k in keys])
n = np.array([agg[k]["n"] for k in keys])

print("=" * 96)
print("ЭМПИРИЧЕСКАЯ ОПАСНОСТЬ ПОВТОРНОГО ОБРАЩЕНИЯ")
print("P(факт всплывёт в [%s, %s) | давность последнего обращения)" % (T0, T1))
print("  Это замер долей по корзинам, а не подобранная кривая.")
bins = [(0, 1), (1, 2), (2, 4), (4, 7), (7, 14), (14, 30), (30, 1e9)]
print("\n  %-16s %-8s %-8s %-18s" % ("давность, дней", "узлов", "всплыли", "доля (ДИ95 Уилсона)"))


def wilson(k, m, z=1.96):
    if m == 0:
        return (float("nan"), float("nan"))
    p = k / m
    d = 1 + z * z / m
    c = p + z * z / (2 * m)
    s = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    return ((c - s) / d, (c + s) / d)


for lo, hi in bins:
    m = (dlast >= lo) & (dlast < hi)
    if m.sum() == 0:
        continue
    k = int(y[m].sum())
    l, h = wilson(k, int(m.sum()))
    print("  %-16s %-8d %-8d %.3f [%.3f, %.3f]"
          % ("%g..%g" % (lo, hi if hi < 1e8 else 60), m.sum(), k, y[m].mean(), l, h))

print("\n  ТО ЖЕ ПО ЧИСЛУ ПРЕДЫДУЩИХ ОБРАЩЕНИЙ (частота, не давность)")
print("  %-16s %-8s %-8s %-18s" % ("n обращений", "узлов", "всплыли", "доля (ДИ95)"))
for lo, hi in [(1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]:
    m = (n >= lo) & (n < hi)
    if m.sum() == 0:
        continue
    k = int(y[m].sum())
    l, h = wilson(k, int(m.sum()))
    print("  %-16s %-8d %-8d %.3f [%.3f, %.3f]" % ("%g..%g" % (lo, hi if hi < 1e8 else 99), m.sum(), k, y[m].mean(), l, h))

print("\n  СОВМЕСТНО (строки — давность, столбцы — число обращений). В клетке: доля / узлов")
rb = [(0, 2), (2, 7), (7, 30), (30, 1e9)]
nb = [(1, 2), (2, 4), (4, 1e9)]
print("  %-14s %s" % ("давность \\ n", " ".join("%-16s" % ("n=%g..%g" % (a, b if b < 1e8 else 99)) for a, b in nb)))
for lo, hi in rb:
    line = "  %-14s" % ("%g..%g" % (lo, hi if hi < 1e8 else 60))
    for a, b in nb:
        m = (dlast >= lo) & (dlast < hi) & (n >= a) & (n < b)
        line += " %-16s" % ("%.2f / %d" % (y[m].mean(), m.sum()) if m.sum() else "-")
    print(line)

print("\n" + "=" * 96)
print("НЕСИММЕТРИЧНЫЙ ПОРОГ (Elkan 2001, Теорема 1)")
print("  Цена лишнего удержания: место и токены в бюджете подачи -> C_fp.")
print("  Цена лишнего выброса: безвозвратная потеря -> C_fn.")
print("  Оптимальный порог по вероятности: p* = C_fp / (C_fp + C_fn).")
print("\n  %-10s %-10s %-38s" % ("C_fn/C_fp", "p*", "какой доле повтора соответствует"))
for ratio in (1, 3, 5, 10, 20, 50):
    p = 1.0 / (1.0 + ratio)
    print("  %-10s %-10.4f %s" % ("%d:1" % ratio, p,
                                  "держим всё, где шанс повтора выше %.1f%%" % (100 * p)))
print("\n  Наблюдаемые доли повтора по корзинам давности (см. выше) дают, при каком пороге")
print("  какая корзина ещё проходит. Пример при C_fn/C_fp = 10:1, p* = 0.0909:")
for lo, hi in bins:
    m = (dlast >= lo) & (dlast < hi)
    if m.sum() == 0:
        continue
    keep = "ДЕРЖИМ" if y[m].mean() >= 1.0 / 11 else "в отставку"
    print("    давность %-8s доля %.3f -> %s" % ("%g..%g" % (lo, hi if hi < 1e8 else 60), y[m].mean(), keep))

print("\n  ВАЖНО: доли выше измерены на 29 положительных из 386. ДИ Уилсона по корзинам")
print("  перекрываются почти везде, кроме крайних. Порог, выведенный из них, тоже с ДИ.")

print("\n" + "=" * 96)
print("ЧТО СТОИТ УДЕРЖАНИЕ В ТОКЕНАХ (для C_fp)")
lens = np.array([len(agg[k]["content"]) for k in keys])
print("  длина факта в символах: медиана %d, среднее %.0f, сумма по всем %d узлам %d"
      % (np.median(lens), lens.mean(), len(keys), lens.sum()))
print("  ~ %d токенов на факт (4 символа на токен), весь набор ~%d токенов"
      % (np.median(lens) / 4, lens.sum() / 4))
ordr = np.argsort(-np.array([formulas.f_current(agg[k], A[k]) for k in keys]))
cum = np.cumsum(lens[ordr]) / 4
for budget in (500, 1000, 2000, 5000):
    m = cum <= budget
    got = y[ordr][m].sum()
    print("  бюджет %5d токенов -> влезает %3d фактов, из них всплывут %2d из %d (%.0f%% полноты)"
          % (budget, m.sum(), int(got), int(y.sum()), 100 * got / y.sum()))
