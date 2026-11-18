import pickle, itertools, random, sys
from collections import Counter, defaultdict
from datetime import datetime
B="/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/lab/"
eps=pickle.load(open(B+"eps.pkl","rb"))
def pt(s):
    try: return datetime.fromisoformat((s or "").replace("Z","+00:00"))
    except Exception: return None
for e in eps: e["t"]=pt(e["ended"])
SPLIT=sys.argv[1] if len(sys.argv)>1 else "2026-08-11"
tr=[e for e in eps if (e["ended"] or "")[:10] < SPLIT]
te=[e for e in eps if (e["ended"] or "")[:10] >= SPLIT]
def keys(e): return sorted(set(k for _,k in e["facts"]))
def up(a,b): return (a,b) if a<b else (b,a)

def build(eplist, cue):
    E=Counter()
    if cue=="same_episode":
        for e in eplist:
            for a,b in itertools.combinations(keys(e),2): E[up(a,b)]+=1
    elif cue=="same_project":
        p2=defaultdict(set)
        for e in eplist:
            for _,k in e["facts"]:
                if k[0]!="pref": p2[e["project"]].add(k)
        for p,ks in p2.items():
            for a,b in itertools.combinations(sorted(ks),2): E[up(a,b)]+=1
    elif cue=="adjacent_in_time":
        bs=defaultdict(list)
        for e in eplist: bs[e["session"]].append(e)
        for s,l in bs.items():
            l.sort(key=lambda e:e["num"])
            for x,y in zip(l,l[1:]):
                for a in keys(x):
                    for b in keys(y):
                        if a!=b: E[up(a,b)]+=1
    elif cue=="same_file":
        p2=defaultdict(set)
        for e in eplist:
            for _,k in e["facts"]:
                if k[0]=="file": p2[k[2]].add(k)
        for p,ks in p2.items():
            for a,b in itertools.combinations(sorted(ks),2): E[up(a,b)]+=1
    elif cue=="error_then_fix":
        bp=defaultdict(list)
        for e in eplist:
            if e["t"]: bp[e["project"]].append(e)
        for p,l in bp.items():
            l.sort(key=lambda e:e["t"])
            for i,x in enumerate(l):
                if not x["n_errors"]: continue
                ob=[k for _,k in x["facts"] if k[0]=="other"]
                if not ob: continue
                for y in l[i+1:]:
                    if (y["t"]-x["t"]).total_seconds()>86400: break
                    if y["outcome"]!="done": continue
                    tg=set(k for _,k in y["facts"] if k[0]=="file")
                    if not tg: continue
                    for a in ob:
                        for b in tg: E[up(a,b)]+=1
                    break
    return E

def adj(E):
    A=defaultdict(dict)
    for (a,b),w in E.items(): A[a][b]=w; A[b][a]=w
    return A

# узлы, известные до границы
train_nodes=sorted({k for e in tr for _,k in e["facts"]})
NI={k:i for i,k in enumerate(train_nodes)}
N=len(train_nodes)
# частота в train (популярность = базовая линия)
pop=Counter(k for e in tr for _,k in e["facts"])

# истина: со-встречаемость в одном эпизоде ПОСЛЕ границы, оба конца известны до границы
truth=set()
for e in te:
    ks=[k for k in keys(e) if k in NI]
    for a,b in itertools.combinations(ks,2): truth.add(up(a,b))
print("граница %s | эпизодов до %d, после %d" % (SPLIT,len(tr),len(te)))
print("узлов до границы: %d | пар-истин после границы: %d" % (N,len(truth)))
raw=set()
for e in te:
    for a,b in itertools.combinations(keys(e),2): raw.add(up(a,b))
print("всего пар после границы: %d, из них обоими концами известны: %d (%.0f%%)"%(len(raw),len(truth),100*len(truth)/max(1,len(raw))))
print()

def evaluate(scorefn, name, trials=1):
    rr=[];h1=h5=h10=0;n=0
    for (a,b) in truth:
        for q,g in ((a,b),(b,a)):
            sc=scorefn(q)
            gs=sc.get(g,0.0)
            better=sum(1 for k,v in sc.items() if v>gs and k!=q)
            # узлы без ребра имеют 0
            nz=len(sc)
            zero_cnt = N-1-nz if g in sc else N-1-nz
            if gs>0:
                ties=sum(1 for k,v in sc.items() if v==gs and k!=q)
            else:
                ties=(N-1-nz)+sum(1 for k,v in sc.items() if v==0 and k!=q)
            rank=better+(ties+1)/2.0
            rr.append(1.0/rank); n+=1
            h1+= rank<=1.5; h5+= rank<=5; h10+= rank<=10
    return (name, n, sum(rr)/n, h1/n, h5/n, h10/n)

rows=[]
graphs={}
for cue in ["same_episode","adjacent_in_time","same_project","same_file","error_then_fix"]:
    E=build(tr,cue); A=adj(E); graphs[cue]=E
    cov=sum(1 for (a,b) in truth if b in A.get(a,{}))
    rows.append(evaluate(lambda q,A=A: dict(A.get(q,{})), cue)+ (len(E), cov))

# комбинация выбранных
Ec=Counter()
for cue,w in (("same_episode",1.0),("adjacent_in_time",0.5),("error_then_fix",2.0)):
    for k,v in graphs[cue].items(): Ec[k]+=w*v
Ac=adj(Ec)
cov=sum(1 for (a,b) in truth if b in Ac.get(a,{}))
rows.append(evaluate(lambda q: dict(Ac.get(q,{})), "КОМБО(эп+сосед+ошибка)")+(len(Ec),cov))

# базовые линии
rows.append(evaluate(lambda q: {k:pop[k] for k in train_nodes if k!=q}, "базис: популярность")+(0,0))
rnd=random.Random(7)
rows.append(evaluate(lambda q: {k:rnd.random() for k in train_nodes if k!=q}, "базис: случайный ранг")+(0,0))

# МУТАЦИЯ: случайный граф с тем же числом рёбер и тем же распределением степеней
def rewire(E, seed=1):
    r=random.Random(seed)
    stubs=[]
    for (a,b),w in E.items(): stubs+= [a,b]
    r.shuffle(stubs)
    out=Counter(); ws=list(E.values()); r.shuffle(ws)
    it=iter(range(len(ws)))
    for i in range(len(ws)):
        a=stubs[2*i]; b=stubs[2*i+1]
        if a==b: continue
        out[up(a,b)]+=ws[i]
    return out
for cue in ["same_episode","adjacent_in_time"]:
    Em=rewire(graphs[cue]); Am=adj(Em)
    cov=sum(1 for (a,b) in truth if b in Am.get(a,{}))
    rows.append(evaluate(lambda q,Am=Am: dict(Am.get(q,{})), "МУТАЦИЯ "+cue)+(len(Em),cov))

print("%-28s %6s %7s %8s %8s %8s %8s %6s"%("правило","запр.","MRR","Hits@1","Hits@5","Hits@10","рёбер","покр."))
for name,n,mrr,h1,h5,h10,ne,cov in rows:
    print("%-28s %6d %7.4f %7.1f%% %7.1f%% %7.1f%% %8d %5d"%(name,n,mrr,100*h1,100*h5,100*h10,ne,cov))
