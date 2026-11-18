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
truth=set()
for e in te:
    ks=[k for k in keys(e) if k in S]
    for a,b in itertools.combinations(ks,2): truth.add(up(a,b))
Q=[]
for a,b in truth: Q+= [(a,b),(b,a)]
def run(cap,eps_by_pop=True):
    E=Counter()
    for e in tr:
        ks=keys(e)
        if len(ks)>cap: ks=sorted(ks,key=lambda k:-pop[k])[:cap] if eps_by_pop else ks[:cap]
        for a,b in itertools.combinations(sorted(ks),2): E[up(a,b)]+=1
    A=defaultdict(dict)
    for (a,b),w in E.items(): A[a][b]=w; A[b][a]=w
    rk=[]
    for q,g in Q:
        sc={k: A.get(q,{}).get(k,0)+0.01*pop[k]/mx for k in nodes if k!=q}
        gs=sc.get(g,0); better=sum(1 for v in sc.values() if v>gs); ties=sum(1 for v in sc.values() if v==gs)
        rk.append(better+(ties+1)/2)
    return len(E), sum(1/r for r in rk)/len(rk), sum(r<=5 for r in rk)/len(rk), sum(r<=10 for r in rk)/len(rk)
print("%-8s %7s %8s %8s %8s"%("потолок","рёбер","MRR","H@5","H@10"))
for cap in (4,6,8,10,16,99):
    ne,m,h5,h10=run(cap); print("%-8d %7d %8.4f %7.1f%% %7.1f%%"%(cap,ne,m,100*h5,100*h10))
