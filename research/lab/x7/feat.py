#!/usr/bin/env python3
"""Признаки узла факта, посчитанные по окну наблюдения. Всё детерминированно,
без эмбеддингов и без модели-судьи."""
import pickle, math, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
# Признаки, перенесённые в боевой код, стенд берёт оттуда: своя копия формулы
# означала бы, что замер относится к копии, а не к продукту.
sys.path.append(str(HERE.parents[2]))
from domain import features
D = pickle.load((HERE / "occ7.pkl").open("rb"))
OCC = [o for o in D["occ"] if o["day"]]
EPS = D["eps"]

CODE_EXT = {"py", "ts", "tsx", "js", "go", "rs", "java", "kt", "sh", "sql", "c", "h", "cpp"}
DOC_EXT = {"md", "txt", "rst", "json", "yaml", "yml", "toml"}


def pt(s):
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None


NAMES = [
    # A. частота и давность — то, из чего сложены LRU/LFU
    "n_log", "repeat", "fresh30", "recip_last", "d_last",
    # B. структура во времени и по проектам
    "spread", "sess_log", "days_log", "age", "life", "rate", "burst", "iat_cv",
    # C. тип факта
    "t_pref", "t_url", "t_file", "t_obst",
    # D. конкретность сущности
    "clen_log", "digit_frac", "has_hex", "path_depth", "ext_code", "ext_doc",
    # E. происхождение
    "directive", "prefreq", "frac_done", "frac_err", "ep_facts_log", "req_len_log",
    # F. контекст проекта
    "proj_act", "proj_fresh", "proj_share",
    # G. новизна и вытеснение
    "novel7", "contra", "solo",
]


def build(field, upto, horizon_days=30.0):
    """Свёртка вхождений строго ДО дня upto в узлы + признаки."""
    agg = defaultdict(lambda: {"n": 0, "proj": set(), "sess": set(), "days": set(),
                               "times": [], "first": "", "last": "", "type": "",
                               "directive": 0, "prefreq": 0, "done": 0, "err": 0,
                               "epf": [], "reql": [], "content": "", "tasks": set()})
    proj_occ = defaultdict(int)
    proj_last = defaultdict(str)
    for o in OCC:
        if o["day"] >= upto:
            continue
        k = o[field]
        r = agg[k]
        r["n"] += 1
        r["proj"].add(o["project"])
        r["sess"].add(o["session"])
        r["days"].add(o["day"])
        r["times"].append(o["at"])
        r["type"] = o["type"]
        r["content"] = o["content"]
        if not r["first"] or o["at"] < r["first"]:
            r["first"] = o["at"]
        if o["at"] > r["last"]:
            r["last"] = o["at"]
        r["directive"] = max(r["directive"], o["directive"])
        r["prefreq"] = max(r["prefreq"], o["prefreq"])
        r["done"] += 1 if o["outcome"] == "done" else 0
        r["err"] += 1 if o["ep_err"] else 0
        r["epf"].append(o["ep_facts"])
        r["reql"].append(o["req_len"])
        if "ради задачи: " in o["content"]:
            r["tasks"].add(o["content"].split("ради задачи: ")[-1][:80])
        proj_occ[o["project"]] += 1
        if o["at"] > proj_last[o["project"]]:
            proj_last[o["project"]] = o["at"]
        r["pj"] = o["project"]

    border = max((r["last"] for r in agg.values() if r["last"]), default="")
    nb = pt(border)
    keys = sorted(agg)
    X = np.zeros((len(keys), len(NAMES)))
    for i, k in enumerate(keys):
        r = agg[k]
        na, nf = pt(r["last"]), pt(r["first"])
        d_last = (nb - na).total_seconds() / 86400.0 if (na and nb) else 0.0
        age = (nb - nf).total_seconds() / 86400.0 if (nf and nb) else 0.0
        life = (na - nf).total_seconds() / 86400.0 if (na and nf) else 0.0
        ts = sorted(pt(t) for t in r["times"] if pt(t))
        gaps = [(ts[j + 1] - ts[j]).total_seconds() / 86400.0 for j in range(len(ts) - 1)]
        if len(gaps) >= 2:
            m = sum(gaps) / len(gaps)
            sd = math.sqrt(sum((g - m) ** 2 for g in gaps) / len(gaps))
            iat_cv = sd / m if m > 0 else 0.0
        else:
            iat_cv = 0.0
        c = r["content"]
        path = ""
        if "правился файл " in c:
            path = c.split("правился файл ")[-1].split(" ради")[0]
        ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
        pj = r.get("pj", "unknown")
        pl = pt(proj_last.get(pj, ""))
        pfresh = max(0.0, 1.0 - ((nb - pl).total_seconds() / 86400.0) / horizon_days) if (pl and nb) else 0.0
        row = {
            "n_log": features.n_log(r),
            "repeat": min(r["n"], 10) / 10.0,
            "fresh30": max(0.0, 1.0 - d_last / horizon_days),
            "recip_last": 1.0 / (1.0 + d_last),
            "d_last": min(d_last, 60.0) / 60.0,
            "spread": min(len(r["proj"]), 3) / 3.0,
            "sess_log": math.log1p(len(r["sess"])),
            "days_log": math.log1p(len(r["days"])),
            "age": min(age, 60.0) / 60.0,
            "life": min(life, 60.0) / 60.0,
            "rate": r["n"] / (age + 1.0),
            "burst": r["n"] / (life + 1.0),
            "iat_cv": min(iat_cv, 3.0) / 3.0,
            "t_pref": 1.0 if r["type"] == "preference" else 0.0,
            "t_url": 1.0 if r["type"] == "external_resource" else 0.0,
            "t_file": 1.0 if "правился файл" in c else 0.0,
            "t_obst": 1.0 if "упирались в препятствие" in c else 0.0,
            "clen_log": math.log1p(len(c)) / 8.0,
            "digit_frac": sum(1 for ch in c if ch.isdigit()) / max(1, len(c)),
            "has_hex": 1.0 if any(len(w) >= 16 and all(ch in "0123456789abcdef" for ch in w)
                                  for w in c.split()) else 0.0,
            "path_depth": min(path.count("/"), 12) / 12.0,
            "ext_code": 1.0 if ext in CODE_EXT else 0.0,
            "ext_doc": 1.0 if ext in DOC_EXT else 0.0,
            "directive": float(r["directive"]),
            "prefreq": float(r["prefreq"]),
            "frac_done": r["done"] / r["n"],
            "frac_err": r["err"] / r["n"],
            "ep_facts_log": math.log1p(sum(r["epf"]) / len(r["epf"])) / 3.0,
            "req_len_log": math.log1p(sum(r["reql"]) / len(r["reql"])) / 7.0,
            "proj_act": math.log1p(proj_occ.get(pj, 0)) / 7.0,
            "proj_fresh": pfresh,
            "proj_share": r["n"] / max(1, proj_occ.get(pj, 1)),
            "novel7": 1.0 if age <= 7.0 else 0.0,
            "contra": min(len(r["tasks"]), 5) / 5.0,
            "solo": 1.0 if r["n"] == 1 else 0.0,
        }
        for j, nm in enumerate(NAMES):
            X[i, j] = row[nm]
    return keys, X, agg


def labels(field, keys, lo, hi):
    """Метка: узел всплыл хотя бы раз в окне [lo, hi)."""
    seen = set(o[field] for o in OCC if lo <= o["day"] < hi)
    return np.array([1.0 if k in seen else 0.0 for k in keys])


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
