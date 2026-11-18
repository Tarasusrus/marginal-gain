#!/usr/bin/env python3
"""Богатый разбор архива: вхождения фактов + сами эпизоды.
Ничего не пишем ни в репозиторий, ни в xmemory. Только читаем транскрипты.
"""
import pickle, sys, time, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "repo"
sys.path.insert(0, str(REPO))
from pipeline import understand as u

OUT = HERE / "occ7.pkl"

# Прямое указание запомнить. Детерминированные строки, не модель.
DIRECTIVE = re.compile(
    r"(запомни|не забуд|всегда |никогда |обязательно |на будущее|"
    r"впредь|запиши себе|учти на|правило:|инвариант)", re.I)
# Просьба-предпочтение (те же темы, что в PREF_TOPICS, но как признак эпизода)
PREFREQ = re.compile(
    r"(кратк|короче|не выдумыв|провер|по одному|тире|дефис|сначала план|согласу)", re.I)
HEX32 = re.compile(r"\b[0-9a-f]{16,}\b")
NUM = re.compile(r"\d")


def portable_key(fact_type, subject, scope, content):
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
    k = portable_key(fact_type, subject, scope, content)
    if k[0] == "file":
        return ("base", k[1].rsplit("/", 1)[-1])
    return k


WORD = re.compile(r"[a-zA-Zа-яА-Я0-9_./-]{3,}")


def toks(s):
    return set(w.lower() for w in WORD.findall(s or ""))


def main():
    t0 = time.time()
    files = sorted(u.TRANSCRIPTS.rglob("*.jsonl"))
    occ, eps = [], []
    ep_i = 0
    for path in files:
        for ep in u.episodes_from_file(path):
            project = Path(ep["cwd"]).name if ep["cwd"] else "unknown"
            day = (ep["ended_at"] or "")[:10]
            req = " ".join(ep["request"].split())
            outcome = u.outcome_of(ep)
            eps.append({
                "i": ep_i, "project": project, "day": day,
                "at": ep["ended_at"] or "", "started": ep["started_at"] or "",
                "session": ep["session_id"], "req": req[:600],
                "req_toks": toks(req[:600]), "outcome": outcome,
                "n_files": len(ep["files"]), "n_cmds": len(ep["commands"]),
                "n_err": len(ep["errors"]), "branch": ep["branch"],
            })
            facts = u.facts_of(ep)
            for fact in facts:
                ftype, subject, scope, content = fact
                occ.append({
                    "key": u.fact_key(*fact),
                    "pkey": portable_key(*fact),
                    "lkey": loose_key(*fact),
                    "type": ftype, "subject": subject, "scope": scope,
                    "content": content,
                    "toks": toks(content),
                    "project": project, "day": day, "at": ep["ended_at"] or "",
                    "session": ep["session_id"], "ep": ep_i,
                    "outcome": outcome,
                    "directive": 1 if DIRECTIVE.search(req) else 0,
                    "prefreq": 1 if PREFREQ.search(req) else 0,
                    "req_len": len(req),
                    "ep_facts": len(facts),
                    "ep_files": len(ep["files"]),
                    "ep_err": len(ep["errors"]),
                    "hex": 1 if HEX32.search(content) else 0,
                    "digits": sum(1 for c in content if c.isdigit()),
                    "clen": len(content),
                    "slash": content.count("/"),
                })
            ep_i += 1
    dt = time.time() - t0
    pickle.dump({"occ": occ, "eps": eps, "files": len(files),
                 "episodes": ep_i, "parse_seconds": dt}, OUT.open("wb"))
    print("файлов %d, эпизодов %d, вхождений фактов %d, секунд %.2f"
          % (len(files), ep_i, len(occ), dt))
    days = sorted({o["day"] for o in occ if o["day"]})
    print("дни %s .. %s" % (days[0], days[-1]))
    print("прямых указаний запомнить в эпизодах: %d из %d вхождений"
          % (sum(o["directive"] for o in occ), len(occ)))


if __name__ == "__main__":
    main()
