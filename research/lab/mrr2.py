import pickle, itertools, random, sys, math
from collections import Counter, defaultdict
from datetime import datetime
B="/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/lab/"
eps=pickle.load(open(B+"eps.pkl","rb"))
def pt(s):
    try: return datetime.fromisoformat((s or "").replace("Z","+00:00"))
    except Exception: return None
for e in eps: e["t"]=pt(e["ended"])
SPLIT="2026-08-11"
tr=[e for e in eps if (e["ended"] or "")[:10] < SPLIT]
te=[e for e in eps if (e["ended"] or "")[:10] >= SPLIT]
def keys(e): return sorted(set(k for _,k in e["facts"]))
def up(a,b): return (a,b) if a<b else (b,a)
E=Counter()
for e in tr:
    for a,b in itertools.combinations(keys(e),2): E[up(a,b)]+=1
Eadj=Counter()
bs=defaultdict(list)
for e in tr: bs[e["session"]].append(e)
for s,l in bs.items():
    l.sort(key=lambda x:x["num"])
    for x,y in zip(l,l[1:]):
        for a in keys(x):
            for b in keys(y):
                if a!=b: Eadj[up(a,b)]+=1
def adjm(EE):
    A=defaultdict(dict)
    for (a,b),w in EE.items(): A[a][b]=w; A[b][a]=w
    return A
A=adjm(E); Aa=adjm(Eadj)
train_nodes=sorted({k for e in tr for _,k in e["facts"]}); N=len(train_nodes)
pop=Counter(k for e in tr for _,k in e["facts"]); T=sum(pop.values())
truth=set()
NI=set(train_nodes)
for e in te:
    ks=[k for k in keys(e) if k in NI]
    for a,b in itertools.combinations(ks,2): truth.add(up(a,b))
queries=[(a,b) for (a,b) in truth for a,b in ((a,b),(b,a))]
queries=[]
for (a,b) in truth: queries.append((a,b)); queries.append((b,a))

def ranks(scorefn):
    out=[]
    for q,g in queries:
        sc=scorefn(q); gs=sc.get(g,0.0); nz=len(sc)
        better=sum(1 for k,v in sc.items() if v>gs and k!=q)
        if gs>0: ties=sum(1 for k,v in sc.items() if v==gs and k!=q)
        else: ties=(N-1-nz)+sum(1 for k,v in sc.items() if v==0 and k!=q)
        out.append(better+(ties+1)/2.0)
    return out

def rep(name, rk):
    mrr=sum(1/r for r in rk)/len(rk)
    return name,mrr,sum(r<=1.5 for r in rk)/len(rk),sum(r<=5 for r in rk)/len(rk),sum(r<=10 for r in rk)/len(rk)

variants={}
variants["same_episode w>=1"]=ranks(lambda q: dict(A.get(q,{})))
variants["same_episode w>=2"]=ranks(lambda q: {k:v for k,v in A.get(q,{}).items() if v>=2})
variants["same_episode PMI"]=ranks(lambda q: {k: math.log((v*T)/(pop[q]*pop[k]+1e-9)+1e-12) for k,v in A.get(q,{}).items()})
variants["эп + 0.01*популярность"]=ranks(lambda q: {k: A.get(q,{}).get(k,0)+0.01*pop[k]/max(pop.values()) for k in train_nodes if k!=q})
variants["популярность"]=ranks(lambda q: {k:pop[k] for k in train_nodes if k!=q})
rnd=random.Random(11)
variants["случайный ранг"]=ranks(lambda q: {k:rnd.random() for k in train_nodes if k!=q})

print("запросов:", len(queries))
print("%-28s %7s %8s %8s %8s"%("вариант","MRR","H@1","H@5","H@10"))
for n,rk in variants.items():
    _,m,h1,h5,h10=rep(n,rk); print("%-28s %7.4f %7.1f%% %7.1f%% %7.1f%%"%(n,m,100*h1,100*h5,100*h10))

# бутстрап разницы MRR: same_episode - популярность
a=variants["same_episode w>=1"]; b=variants["популярность"]; c=variants["случайный ранг"]
r=random.Random(3); n=len(a)
def boot(x,y,trials=5000):
    d=[]
    for _ in range(trials):
        idx=[r.randrange(n) for _ in range(n)]
        d.append(sum(1/x[i] for i in idx)/n - sum(1/y[i] for i in idx)/n)
    d.sort(); return d[int(0.025*trials)], d[int(0.975*trials)], sum(1 for v in d if v<=0)/trials
lo,hi,p=boot(a,b); print("\nMRR(same_episode) - MRR(популярность): 95%% ДИ [%.4f, %.4f], доля бутстрапов <=0: %.3f"%(lo,hi,p))
lo,hi,p=boot(a,c); print("MRR(same_episode) - MRR(случайный):     95%% ДИ [%.4f, %.4f], доля бутстрапов <=0: %.3f"%(lo,hi,p))
# то же для Hits@10
def boot_h(x,y,K=10,trials=5000):
    d=[]
    for _ in range(trials):
        idx=[r.randrange(n) for _ in range(n)]
        d.append(sum(x[i]<=K for i in idx)/n - sum(y[i]<=K for i in idx)/n)
    d.sort(); return d[int(0.025*trials)], d[int(0.975*trials)], sum(1 for v in d if v<=0)/trials
lo,hi,p=boot_h(a,b); print("Hits@10(same_episode) - Hits@10(популярность): 95%% ДИ [%.3f, %.3f], доля <=0: %.3f"%(lo,hi,p))
