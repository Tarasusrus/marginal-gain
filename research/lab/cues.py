import pickle, itertools, sys
from collections import Counter, defaultdict
from datetime import datetime
B="/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/lab/"
eps = pickle.load(open(B+"eps.pkl","rb"))
def pt(s):
    try: return datetime.fromisoformat((s or "").replace("Z","+00:00"))
    except Exception: return None
for e in eps: e["t"]=pt(e["ended"])
eps.sort(key=lambda e:(e["t"] or datetime.min.replace(tzinfo=None).astimezone()))

def keys(e): return sorted(set(k for _,k in e["facts"]))
def up(a,b): return (a,b) if a<b else (b,a)

edges=defaultdict(Counter)   # cue -> Counter[(a,b)]

# 1 same_episode
for e in eps:
    ks=keys(e)
    for a,b in itertools.combinations(ks,2): edges["same_episode"][up(a,b)]+=1

# 2 same_file : факты, ссылающиеся на один и тот же путь файла
path2keys=defaultdict(set)
for e in eps:
    for f,k in e["facts"]:
        if k[0]=="file": path2keys[k[2]].add(k)
multi=[p for p,ks in path2keys.items() if len(ks)>1]
print("различных путей файлов:", len(path2keys), "| путей, дающих >1 узла:", len(multi))
for p in multi:
    for a,b in itertools.combinations(sorted(path2keys[p]),2): edges["same_file"][up(a,b)]+=1

# 2b same_file широкая трактовка: путь упоминается в контенте любого факта эпизода
# (т.е. связываем факт-о-файле со всеми фактами эпизода, где этот файл правился)
edges2b=Counter()
for e in eps:
    ks=keys(e)
    fks=[k for k in ks if k[0]=="file"]
    for a in fks:
        for b in ks:
            if a!=b: edges2b[up(a,b)]+=1

# 3 same_project
proj2keys=defaultdict(set)
for e in eps:
    for _,k in e["facts"]:
        if k[0]!="pref": proj2keys[e["project"]].add(k)
sp=0; sp_pairs=set()
for p,ks in proj2keys.items():
    n=len(ks); sp+=n*(n-1)//2
    if n<=400:
        for a,b in itertools.combinations(sorted(ks),2): sp_pairs.add(up(a,b))
print("same_project: сумма пар по проектам =", sp, "| различных пар =", len(sp_pairs))
print("топ проектов по числу узлов:", sorted(((len(v),k) for k,v in proj2keys.items()), reverse=True)[:8])

# 4 adjacent_in_time: соседние эпизоды одной сессии
bysess=defaultdict(list)
for e in eps: bysess[e["session"]].append(e)
for s,lst in bysess.items():
    lst.sort(key=lambda e:e["num"])
    for x,y in zip(lst,lst[1:]):
        for a in keys(x):
            for b in keys(y):
                if a!=b: edges["adjacent_in_time"][up(a,b)]+=1
# 4b окно 30 минут, независимо от сессии
e4b=Counter()
seq=[e for e in eps if e["t"]]
for i,x in enumerate(seq):
    for y in seq[i+1:]:
        if (y["t"]-x["t"]).total_seconds()>1800: break
        for a in keys(x):
            for b in keys(y):
                if a!=b: e4b[up(a,b)]+=1

# 5 error_then_fix: эпизод с ошибкой -> следующий эпизод того же проекта с outcome done
byproj=defaultdict(list)
for e in eps:
    if e["t"]: byproj[e["project"]].append(e)
etf=Counter(); n_err_eps=0; n_matched=0
for p,lst in byproj.items():
    lst.sort(key=lambda e:e["t"])
    for i,x in enumerate(lst):
        if not x["n_errors"]: continue
        obst=[k for _,k in x["facts"] if k[0]=="other"]
        if not obst: continue
        n_err_eps+=1
        for y in lst[i+1:]:
            if (y["t"]-x["t"]).total_seconds()>86400: break
            if y["outcome"]!="done": continue
            tgt=[k for _,k in y["facts"] if k[0]=="file"]
            if not tgt: continue
            n_matched+=1
            for a in obst:
                for b in set(tgt): etf[up(a,b)]+=1
            break
print("error_then_fix: эпизодов-источников с фактом-препятствием =", n_err_eps, "| нашли починку =", n_matched)
edges["error_then_fix"]=etf

print()
print("%-20s %10s %10s %10s" % ("повод","рёбер","доля","макс.вес"))
tot=0
allsets={"same_episode":set(edges["same_episode"]), "same_file":set(edges["same_file"]),
         "same_project":sp_pairs, "adjacent_in_time":set(edges["adjacent_in_time"]),
         "error_then_fix":set(etf)}
for c in ["same_episode","same_file","same_project","adjacent_in_time","error_then_fix"]:
    tot+=len(allsets[c])
for c in ["same_episode","same_file","same_project","adjacent_in_time","error_then_fix"]:
    cnt=edges[c] if c!="same_project" else Counter({p:1 for p in sp_pairs})
    print("%-20s %10d %9.1f%% %10d" % (c, len(allsets[c]), 100*len(allsets[c])/tot, max(cnt.values()) if cnt else 0))
print("ИТОГО карточек Association:", tot)
print()
print("same_file широкая трактовка (файл-факт со всем эпизодом):", len(edges2b), "рёбер")
print("adjacent окно 30 мин без учёта сессии:", len(e4b), "рёбер")
print()
# пересечения
import itertools as it
for a,b in it.combinations(["same_episode","same_file","same_project","adjacent_in_time","error_then_fix"],2):
    inter=allsets[a]&allsets[b]
    if inter: print("пересечение %s ∩ %s = %d (%.0f%% от %s)"%(a,b,len(inter),100*len(inter)/max(1,len(allsets[a])),a))
pickle.dump({"edges":{k:dict(v) for k,v in edges.items()},"same_project":sp_pairs,"e2b":dict(edges2b)}, open(B+"edges.pkl","wb"))
