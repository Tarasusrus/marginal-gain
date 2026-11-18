#!/usr/bin/env python3
"""Детерминированные формулы сочетания признаков из литературы.
Ничего не обучается: у каждой формулы вид задан заранее, параметры взяты
из первоисточника, не подобраны по нашим данным. Мы их только меряем.
"""
import math
from datetime import datetime

EPS = 1e-3   # чтобы возраст 0 не давал бесконечность


def ages(rec, border):
    """Возрасты всех вхождений узла в днях к границе окна."""
    b = border
    out = []
    for t in rec["times"]:
        a = _pt(t)
        if a and b:
            out.append(max(EPS, (b - a).total_seconds() / 86400.0))
    return sorted(out)


def _pt(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


# ------------------------------------------------------------------ формулы
# Каждая: (имя, источник, функция(rec, ages) -> число)

def f_current(r, a):
    """Наша текущая мера, ADR 0002."""
    return (0.5 * min(r["n"], 10) / 10.0
            + 0.2 * min(len(r["proj"]), 3) / 3.0
            + 0.3 * max(0.0, 1.0 - a[0] / 30.0))


def f_lru(r, a):
    """LRU: только давность последнего обращения."""
    return -a[0]


def f_lfu(r, a):
    """LFU: только число обращений."""
    return r["n"]


def _actr(d):
    def f(r, a):
        # Anderson & Schooler 1991, base-level activation B = ln sum t_j^-d
        return math.log(sum(x ** (-d) for x in a))
    return f


def _lrfu(lam):
    def f(r, a):
        # Lee et al. 2001, Combined Recency and Frequency: sum (1/2)^(lam*t)
        return sum(0.5 ** (lam * x) for x in a)
    return f


def f_hyper(r, a):
    """Hyperbolic caching, Blankstein et al. ATC'17: частота попаданий за время жизни."""
    return r["n"] / (a[-1] + 1.0)


def f_ebbing(r, a):
    """MemoryBank (Zhong 2023/AAAI'24): R = exp(-t/S), S=1 при первом упоминании,
    +1 при каждом повторном, t сбрасывается. Формула и параметры из источника."""
    S = float(r["n"])          # 1 при первом упоминании, +1 за каждое повторное
    return math.exp(-a[0] / S)  # t = дней с последнего упоминания


def f_ga(r, a):
    """Generative Agents (Park 2023): score = a_rec*recency + a_imp*importance + a_rel*relevance,
    все alpha = 1, recency = 0.995^час. Без модели-судьи и без эмбеддингов остаётся
    ровно затухание 0.995^час — то есть чистая свежесть."""
    return 0.995 ** (a[0] * 24.0)


def f_gdsf(r, a):
    """GDSF (Cherkasova 1998): частота * цена / размер. Цена=1, размер=длина в символах.
    Это единственная формула в списке, которая знает про бюджет в токенах."""
    return r["n"] / max(1.0, len(r["content"]) / 4.0)


def f_actr_gdsf(r, a):
    """ACT-R, поделённая на размер: ценность на токен."""
    return math.exp(_actr(0.5)(r, a)) / max(1.0, len(r["content"]) / 4.0)


def f_budgetmem(r, a):
    """BudgetMem (2511.04919), ЧАСТИЧНАЯ реализация: из шести слагаемых с опубликованными
    весами мы умеем посчитать три (плотность чисел 0.15, маркеры-инструкции 0.1,
    средний вес токенов 0.2). Плотность именованных сущностей требует spaCy,
    позиционный признак к нашим фактам неприменим. Помечено как частичное."""
    c = r["content"]
    numd = sum(1 for ch in c if ch.isdigit()) / max(1, len(c))
    disc = 1.0 if r["directive"] else 0.0
    tfidf = min(1.0, len(set(c.split())) / max(1.0, len(c.split()) + 1e-9))
    return 0.15 * numd + 0.1 * disc + 0.2 * tfidf


def f_spread(r, a):
    return len(r["proj"])


def f_actr_spread(r, a):
    """ACT-R + охват проектов, единственное сочетание, заданное нами, а не источником."""
    return _actr(0.5)(r, a) + 0.5 * math.log(len(r["proj"]) + 1)


FORMULAS = [
    ("текущая мера ADR 0002", "наш ADR 0002", f_current),
    ("LRU (только свежесть)", "Denning 1968", f_lru),
    ("LFU (только частота)", "классика", f_lfu),
    ("охват проектов", "наш признак", f_spread),
    ("ACT-R base-level d=0.5", "Anderson&Schooler 1991", _actr(0.5)),
    ("ACT-R base-level d=0.3", "Anderson&Schooler 1991", _actr(0.3)),
    ("ACT-R base-level d=0.7", "Anderson&Schooler 1991", _actr(0.7)),
    ("LRFU CRF lam=0.01", "Lee et al. 2001", _lrfu(0.01)),
    ("LRFU CRF lam=0.1", "Lee et al. 2001", _lrfu(0.1)),
    ("LRFU CRF lam=0.5", "Lee et al. 2001", _lrfu(0.5)),
    ("Hyperbolic n/возраст", "Blankstein 2017", f_hyper),
    ("Эббингауз exp(-t/S)", "MemoryBank, Zhong 2024", f_ebbing),
    ("Gen.Agents без судьи", "Park 2023", f_ga),
    ("GDSF n/размер", "Cherkasova 1998", f_gdsf),
    ("BudgetMem (частично)", "arXiv 2511.04919", f_budgetmem),
    ("ACT-R / размер", "ACT-R + GDSF", f_actr_gdsf),
    ("ACT-R + охват", "наше сочетание", f_actr_spread),
]
