#!/usr/bin/env python3
"""Описательная часть: что вообще может переехать, при трёх определениях идентичности."""
import pickle
from collections import defaultdict
from pathlib import Path

D = pickle.load((Path(__file__).resolve().parent / "occ.pkl").open("rb"))
occ = D["occ"]

print("эпизодов %d, вхождений фактов %d, разбор %.1f c" % (D["episodes"], len(occ), D["parse_seconds"]))
print()

for name, field in (("fact_key (как в holdout)", "key"),
                    ("portable (проект выброшен, путь целиком)", "pkey"),
                    ("loose (файл сведён к имени)", "lkey")):
    byk = defaultdict(lambda: {"n": 0, "projects": set(), "days": set()})
    for o in occ:
        r = byk[o[field]]
        r["n"] += 1
        r["projects"].add(o["project"])
        r["days"].add(o["day"])
    multi = sum(1 for r in byk.values() if len(r["projects"]) > 1)
    rep = sum(1 for r in byk.values() if r["n"] > 1)
    print("%-42s узлов %4d | встретились >1 раз %4d | в >1 проекте %4d (%.1f%%)"
          % (name, len(byk), rep, multi, 100.0 * multi / max(1, len(byk))))

print()
print("разбивка по типам факта (portable key):")
byt = defaultdict(lambda: {"occ": 0, "keys": set(), "multi": set()})
proj = defaultdict(set)
for o in occ:
    proj[o["pkey"]].add(o["project"])
for o in occ:
    r = byt[o["type"]]
    r["occ"] += 1
    r["keys"].add(o["pkey"])
    if len(proj[o["pkey"]]) > 1:
        r["multi"].add(o["pkey"])
for t, r in sorted(byt.items(), key=lambda x: -x[1]["occ"]):
    print("  %-20s вхождений %4d, узлов %4d, из них в >1 проекте %3d"
          % (t, r["occ"], len(r["keys"]), len(r["multi"])))

print()
print("охват проектов по fact_key (штатная идентичность):")
pk = defaultdict(set)
for o in occ:
    pk[o["key"]].add(o["project"])
dist = defaultdict(int)
for k, ps in pk.items():
    dist[len(ps)] += 1
for k in sorted(dist):
    print("  проектов %d: узлов %d" % (k, dist[k]))

print()
days = sorted({o["day"] for o in occ if o["day"]})
print("диапазон дней: %s .. %s" % (days[0], days[-1]))
