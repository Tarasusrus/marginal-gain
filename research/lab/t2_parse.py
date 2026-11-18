#!/usr/bin/env python3
"""Одноразовый: разбираем архив штатным парсером и складываем вхождения фактов в pickle."""
import pickle, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "repo"
sys.path.insert(0, str(REPO))
from pipeline import understand as u

OUT = Path(__file__).resolve().parent / "occ.pkl"


def portable_key(fact_type, subject, scope, content):
    """Идентичность факта БЕЗ проекта: то, что теоретически может переехать."""
    if fact_type == "external_resource":
        return ("url", content.split()[-1])
    if fact_type == "preference":
        return ("pref", subject)
    if "правился файл " in content:
        path = content.split("правился файл ")[-1].split(" ради")[0]
        return ("file", path)
    if "упирались в препятствие: " in content:
        return ("obst", content.split("упирались в препятствие: ")[-1][:120])
    return ("other", content[:60])


def loose_key(fact_type, subject, scope, content):
    """То же, но файл сводится к имени файла, а не полному пути."""
    k = portable_key(fact_type, subject, scope, content)
    if k[0] == "file":
        return ("base", k[1].rsplit("/", 1)[-1])
    return k


def main():
    t0 = time.time()
    files = sorted(u.TRANSCRIPTS.rglob("*.jsonl"))
    occ = []
    n_ep = 0
    for path in files:
        for ep in u.episodes_from_file(path):
            n_ep += 1
            project = Path(ep["cwd"]).name if ep["cwd"] else "unknown"
            day = (ep["ended_at"] or "")[:10]
            for fact in u.facts_of(ep):
                occ.append({
                    "key": u.fact_key(*fact),
                    "pkey": portable_key(*fact),
                    "lkey": loose_key(*fact),
                    "type": fact[0],
                    "project": project,
                    "day": day,
                    "at": ep["ended_at"] or "",
                    "session": ep["session_id"],
                })
    dt = time.time() - t0
    pickle.dump({"occ": occ, "files": len(files), "episodes": n_ep, "parse_seconds": dt},
                OUT.open("wb"))
    print("файлов %d, эпизодов %d, вхождений фактов %d, секунд %.1f"
          % (len(files), n_ep, len(occ), dt))


if __name__ == "__main__":
    main()
