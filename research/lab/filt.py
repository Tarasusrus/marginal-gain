import pickle, itertools, random
from collections import Counter, defaultdict
B="/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/lab/"
eps=pickle.load(open(B+"eps.pkl","rb"))
SPLIT="2026-08-11"
tr=[e for e in eps if (e["ended"] or "")[:10]<SPLIT]; te=[e for e in eps if (e["ended"] or "")[:10]>=SPLIT]
def keys(e): return sorted(set(k for _,k in e["facts"]))
def up(a,b): return (a,b) if a<b else (b,a)
nodes=sorted({k for e in tr for _,k in e["facts"]}); N=len(nodes); S=set(nodes)
pop=Counter(k for e in tr for _,k in e["facts"]); mx=max(pop.values())
node_proj=defaultdict(set)
for e in tr:
    for _,k in e["facts"]: node_proj[k].add(e["project"])
E=Counter()
for e in tr:
    ks=keys(e)
    if len(ks)>8: ks=sorted(ks,key=lambda k:-pop[k])[:8]
    for a,b in itertools.combinations(sorted(ks),2): E[up(a,b)]+=1
A=defaultdict(dict)
for (a,b),w in E.items(): A[a][b]=w; A[b][a]=w
truth=set()
for e in te:
    ks=[k for k in keys(e) if k in S]
    for a,b in itertools.combinations(ks,2): truth.add(up(a,b))
Q=[]
for a,b in truth: Q+=[(a,b),(b,a)]
def ev(pool_fn,name):
    rk=[]; oob=0
    for q,g in Q:
        pool=[k for k in pool_fn(q) if k!=q]
        if g not in pool:
            oob+=1; rk.append(N); continue
        sc={k:A.get(q,{}).get(k,0)+0.01*pop[k]/mx for k in pool}
        gs=sc[g]; better=sum(1 for v in sc.values() if v>gs); ties=sum(1 for v in sc.values() if v==gs)
        rk.append(better+(ties+1)/2)
    print("%-34s пул~%5.0f  MRR %.4f  H@5 %5.1f%%  H@10 %5.1f%%  вне пула %d"%(
        name, sum(len(pool_fn(q)) for q,_ in Q)/len(Q),
        sum(1/r for r in rk)/len(rk), 100*sum(r<=5 for r in rk)/len(rk), 100*sum(r<=10 for r in rk)/len(rk), oob))
ev(lambda q: nodes, "без фильтра (весь архив)")
ev(lambda q: [k for k in nodes if node_proj[k] & node_proj[q]], "фильтр same_project")
ev(lambda q: [k for k in nodes if (node_proj[k]&node_proj[q]) or k in A.get(q,{})], "фильтр same_project + соседи")
