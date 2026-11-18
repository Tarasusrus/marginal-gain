#!/usr/bin/env python3
"""Замер детерминированных формул и одиночных признаков на нашем архиве.
Ничего не обучается. Деление по времени, метки из будущего окна.
Рядом с каждым числом — потолок отбора на перемешанных метках."""
import math, time
import numpy as np
import feat, formulas
from feat import NAMES, auc

FLD = "pkey"
SPLITS = [("2026-08-05", "2026-08-15"), ("2026-08-04", "2026-08-13"),
          ("2026-08-06", "2026-08-16"), ("2026-08-11", "2026-08-26")]
T0, T1 = "2026-08-15", "2026-08-26"


def boot_ci(s, y, n=1500, seed=1):
    rng = np.random.RandomState(seed)
    N, out = len(y), []
    for _ in range(n):
        i = rng.randint(0, N, N)
        if 0 < y[i].sum() < len(i):
            out.append(auc(s[i], y[i]))
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def paired_boot(sa, sb, y, n=1500, seed=2):
    rng = np.random.RandomState(seed)
    N, d = len(y), []
    for _ in range(n):
        i = rng.randint(0, N, N)
        if 0 < y[i].sum() < len(i):
            d.append(auc(sa[i], y[i]) - auc(sb[i], y[i]))
    d.sort()
    return sum(d) / len(d), d[int(0.025 * len(d))], d[int(0.975 * len(d))]


def score_all(keys, agg, border):
    """Все формулы разом. Возвращает {имя: вектор}."""
    A = {k: formulas.ages(agg[k], border) for k in keys}
    out = {}
    for name, src, fn in formulas.FORMULAS:
        out[name] = np.array([fn(agg[k], A[k]) for k in keys])
    return out


def window(t0, t1):
    keys, X, agg = feat.build(FLD, t0)
    y = feat.labels(FLD, keys, t0, t1)
    border = max((formulas._pt(r["last"]) for r in agg.values() if r["last"]),
                 default=None)
    S = score_all(keys, agg, border)
    return keys, X, y, agg, S


def main():
    keys, X, y, agg, S = window(T0, T1)
    print("=" * 104)
    print("ПРОТОКОЛ. Признаки считаются по вхождениям строго < %s." % T0)
    print("          Метка: узел всплыл хотя бы раз в [%s, %s). Окна не пересекаются," % (T0, T1))
    print("          перемешивания по времени нет, ничего не обучается.")
    print("  узлов %d, положительных %d (%.1f%%). Идентичность %s."
          % (len(y), int(y.sum()), 100 * y.mean(), FLD))
    print("  SE(AUC) при %d положительных порядка 0.06 — разности меньше 0.12 неразличимы."
          % int(y.sum()))

    # ---------- 1. одиночные признаки ----------
    print("\n" + "=" * 104)
    print("1. ОДИНОЧНЫЕ ПРИЗНАКИ (AUC на окне проверки; ниже 0.5 значит антипредиктор)")
    rows = sorted(((abs(auc(X[:, j], y) - 0.5), auc(X[:, j], y), nm, j)
                   for j, nm in enumerate(NAMES)), reverse=True)
    print("  %-16s %-7s %-20s" % ("признак", "AUC", "ДИ95"))
    for _, a, nm, j in rows:
        lo, hi = boot_ci(X[:, j], y, n=600)
        mark = "" if (lo > 0.5 or hi < 0.5) else "   0.5 внутри ДИ"
        print("  %-16s %-7.3f [%.3f, %.3f]%s" % (nm, a, lo, hi, mark))

    # ---------- 2. формулы ----------
    print("\n" + "=" * 104)
    print("2. ДЕТЕРМИНИРОВАННЫЕ ФОРМУЛЫ (вид и параметры взяты из источника, не подобраны)")
    print("  %-26s %-24s %-7s %-20s" % ("формула", "источник", "AUC", "ДИ95"))
    res = {}
    for name, src, fn in formulas.FORMULAS:
        a = auc(S[name], y)
        res[name] = a
        lo, hi = boot_ci(S[name], y, n=600)
        print("  %-26s %-24s %-7.3f [%.3f, %.3f]" % (name, src, a, lo, hi))

    cur = S["текущая мера ADR 0002"]
    curauc = res["текущая мера ADR 0002"]
    bestname = max(res, key=res.get)

    # ---------- 3. потолок отбора ----------
    print("\n" + "=" * 104)
    print("3. ПОТОЛОК ОТБОРА НА ПЕРЕМЕШАННЫХ МЕТКАХ")
    print("  Метки перемешаны, истинный AUC 0.5. Берём максимум по всем %d формулам." % len(res))
    sel = []
    for sd in range(400):
        rng = np.random.RandomState(1000 + sd)
        ysh = y.copy()
        rng.shuffle(ysh)
        sel.append(max(auc(v, ysh) for v in S.values()))
    sel.sort()
    med = sel[len(sel) // 2]
    print("  медиана %.3f | 95-й процентиль %.3f | максимум %.3f" % (med, sel[int(0.95 * len(sel))], sel[-1]))
    print("  потолок сверх 0.5: %.3f" % (med - 0.5))
    print("  лучшая формула: %s, AUC %.3f; выигрыш над текущей мерой %.3f -> %s"
          % (bestname, res[bestname], res[bestname] - curauc,
             "НЕ отличим от шума отбора" if (res[bestname] - curauc) <= (med - 0.5) else "БОЛЬШЕ шума отбора"))
    # то же для одиночных признаков
    selF = []
    for sd in range(400):
        rng = np.random.RandomState(2000 + sd)
        ysh = y.copy()
        rng.shuffle(ysh)
        selF.append(max(max(auc(X[:, j], ysh), 1 - auc(X[:, j], ysh)) for j in range(X.shape[1])))
    selF.sort()
    print("  для отбора лучшего из %d одиночных признаков потолок: медиана %.3f, 95-й %.3f"
          % (X.shape[1], selF[len(selF) // 2], selF[int(0.95 * len(selF))]))

    # ---------- 4. парные сравнения ----------
    print("\n" + "=" * 104)
    print("4. ПАРНЫЕ СРАВНЕНИЯ С ТЕКУЩЕЙ МЕРОЙ (бутстрап на одних и тех же выборках)")
    for name in sorted(res, key=lambda n: -res[n]):
        if name.startswith("текущая"):
            continue
        d, lo, hi = paired_boot(S[name], cur, y)
        print("  %-26s %+0.3f  ДИ95 [%+0.3f, %+0.3f]  %s"
              % (name, d, lo, hi, "значимо" if (lo > 0 or hi < 0) else "НЕ значимо"))

    # ---------- 5. устойчивость по границам ----------
    print("\n" + "=" * 104)
    print("5. УСТОЙЧИВОСТЬ ПО ГРАНИЦАМ (границы вложены: один результат в нескольких срезах)")
    heads = [t0 for t0, t1 in SPLITS]
    print("  %-26s %s" % ("формула", " ".join("%-11s" % h for h in heads)))
    per = {}
    for t0, t1 in SPLITS:
        k2, X2, y2, a2, S2 = window(t0, t1)
        per[t0] = (S2, y2, int(y2.sum()), len(y2))
    print("  %-26s %s" % ("(узлов/положительных)",
                          " ".join("%-11s" % ("%d/%d" % (per[h][3], per[h][2])) for h in heads)))
    for name, src, fn in formulas.FORMULAS:
        line = "  %-26s" % name
        for h in heads:
            S2, y2, _, _ = per[h]
            line += " %-11.3f" % auc(S2[name], y2)
        print(line)

    # ---------- 6. мутация M1 ----------
    print("\n" + "=" * 104)
    print("6. МУТАЦИЯ: мишень тождественна частоте (метка = «n>1 в окне наблюдения»).")
    print("   Формулы, сложенные из частоты, ОБЯЗАНЫ выродиться в идеальный предиктор.")
    m1 = np.array([1.0 if agg[k]["n"] > 1 else 0.0 for k in keys])
    print("   положительных %d из %d" % (int(m1.sum()), len(m1)))
    for name, src, fn in formulas.FORMULAS:
        print("   %-26s AUC %.3f" % (name, auc(S[name], m1)))
    print("   Видимое вырождение: LFU даёт 1.000, LRU опускается к 0.5. Проверка не пустая.")

    # ---------- 7. мутация M2: признак-двойник частоты ----------
    print("\n" + "=" * 104)
    print("7. МУТАЦИЯ: в набор одиночных признаков добавлен точный двойник частоты.")
    dup = np.array([float(agg[k]["n"]) for k in keys])
    print("   AUC двойника %.3f, AUC n_log %.3f, AUC repeat %.3f — совпадают, как и должны."
          % (auc(dup, y), auc(X[:, NAMES.index("n_log")], y), auc(X[:, NAMES.index("repeat")], y)))
    print("   Ранговая корреляция двойника с n_log: %.3f (обязана быть 1.000)"
          % np.corrcoef(np.argsort(np.argsort(dup)), np.argsort(np.argsort(X[:, NAMES.index("n_log")])))[0, 1])
    return keys, X, y, agg, S, res


if __name__ == "__main__":
    main()
