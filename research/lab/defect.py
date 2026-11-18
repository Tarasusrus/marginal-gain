import pickle, itertools
from collections import Counter, defaultdict
B="/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/lab/"
eps=pickle.load(open(B+"eps.pkl","rb"))
sub=Counter(); trip=Counter(); pairs=Counter()
for e in eps:
    for f,k in e["facts"]:
        t,s,sc,c=f
        sub[s]+=1; trip[(t,s,sc)]+=1
        pairs[(t,s)]+=1
print("фактов всего:", sum(sub.values()))
print("различных subject:", len(sub))
print("различных (fact_type, subject, scope):", len(trip))
print("различных локальных fact_key (то, чем меряется мера):", len({k for e in eps for _,k in e["facts"]}))
ps=[(t,s) for (t,s,sc) in trip if t=="project_state"]
n_ps=sum(v for (t,s,sc),v in trip.items() if t=="project_state")
print("project_state: фактов %d -> узлов по subject %d"%(n_ps,len(set(ps))))
print("топ subject по числу схлопнутых фактов:", sub.most_common(8))
# сколько фактов о файлах в самом крупном проекте
byproj=Counter()
for e in eps:
    for f,k in e["facts"]:
        if k[0]=="file": byproj[f[1]]+=1
print("файловых фактов по проектам, топ:", byproj.most_common(6), "| всего", sum(byproj.values()))
# same_episode после схлопывания: сколько эпизодов дадут >=2 РАЗНЫХ subject-узла
cnt=Counter()
for e in eps:
    ks={(f[0],f[1],f[2]) for f,_ in e["facts"]}
    cnt[len(ks)]+=1
print("узлов-по-subject в эпизоде:", dict(sorted(cnt.items())))
E=set()
for e in eps:
    ks=sorted({(f[0],f[1],f[2]) for f,_ in e["facts"]})
    for a,b in itertools.combinations(ks,2): E.add((a,b))
print("same_episode рёбер ПОСЛЕ схлопывания:", len(E), "(до схлопывания было 2086)")
