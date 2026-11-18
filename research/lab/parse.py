import sys, json, pickle
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, "/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/repo")
from pipeline import understand as u
from archive.transcripts import TRANSCRIPTS

files = sorted(TRANSCRIPTS.rglob("*.jsonl"))
eps = []
for p in files:
    try:
        for ep in u.episodes_from_file(p):
            fs = u.facts_of(ep)
            eps.append({
                "session": ep["session_id"],
                "num": ep["number"],
                "started": ep["started_at"],
                "ended": ep["ended_at"],
                "project": Path(ep["cwd"]).name if ep["cwd"] else "unknown",
                "outcome": u.outcome_of(ep),
                "files": [f for f in ep["files"] if not any(b in f for b in u.NOT_CODE)],
                "n_errors": len(ep["errors"]),
                "facts": [(f, u.fact_key(*f)) for f in fs],
            })
    except Exception as e:
        print("skip", p, e, file=sys.stderr)

days = Counter((e["ended"] or "")[:10] for e in eps if e["ended"])
print("файлов", len(files), "эпизодов", len(eps))
print("дней", len(days), "диапазон", min(days), "..", max(days))
tot=0
for d in sorted(days):
    tot+=days[d]
print("эпизодов с датой", tot)
# кумулятив -> медианная дата
acc=0
for d in sorted(days):
    acc+=days[d]
    if acc >= tot/2:
        print("медианная граница:", d, "(до неё %d, после %d)"%(acc-days[d], tot-(acc-days[d])))
        break
allfacts=[k for e in eps for _,k in e["facts"]]
print("фактов всего", len(allfacts), "различных", len(set(allfacts)))
print("проектов", len(set(e["project"] for e in eps)))
c=Counter(k[0] for k in allfacts); print("по видам ключа", dict(c))
c2=Counter(f[0] for e in eps for f,_ in e["facts"]); print("по fact_type", dict(c2))
print("distinct subject", len(set(f[1] for e in eps for f,_ in e["facts"])))
nf=Counter(len(set(k for _,k in e["facts"])) for e in eps)
print("фактов в эпизоде (распределение)", dict(sorted(nf.items())))
pickle.dump(eps, open("/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/lab/eps.pkl","wb"))
