import pickle, itertools
from collections import Counter
B="/private/tmp/claude-501/-Users-tarasmalinovskij--throne-workspaces-intents-1009913849e049edac5e8ead1ad62134/707fdfb8-ee1d-45d8-9789-b2f676921e37/scratchpad/lab/"
eps=pickle.load(open(B+"eps.pkl","rb"))
def keys(e): return sorted(set(k for _,k in e["facts"]))
sizes=sorted((len(keys(e)) for e in eps), reverse=True)
tot=sum(n*(n-1)//2 for n in sizes)
print("всего пар same_episode (с повторами):", tot)
acc=0
for i,n in enumerate(sizes[:12]):
    acc+=n*(n-1)//2
    print("  эпизод #%d: %2d фактов -> %3d пар, накоплено %d (%.0f%%)"%(i+1,n,n*(n-1)//2,acc,100*acc/tot))
for cap in (4,6,8,10,16):
    s=sum(min(n,cap)*(min(n,cap)-1)//2 for n in sizes)
    E=set()
    for e in eps:
        ks=keys(e)[:cap]
        for a,b in itertools.combinations(ks,2): E.add((a,b) if a<b else (b,a))
    print("потолок %2d фактов на эпизод -> пар %5d, различных рёбер %5d"%(cap,s,len(E)))
