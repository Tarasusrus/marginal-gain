#!/usr/bin/env python3
"""Классические политики кэша как базисы на нашем потоке фактов.

Поток обращений = вхождения фактов в порядке времени. Промах = факт всплыл,
а его в памяти нет. Это ровно та задача, которую решает наша мера: держать
ограниченное число фактов так, чтобы нужный оказался на месте.

Политики: LRU, LFU, ARC (Megiddo&Modha FAST'03), W-TinyLFU (Einziger TOS'17),
наша мера как политика вытеснения, Belady (оракул, потолок), random (пол).
"""
import pickle, random, math
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = pickle.load((HERE / "occ7.pkl").open("rb"))
OCC = sorted(D["occ"], key=lambda o: (o["at"], o["ep"]))


def stream(field="pkey"):
    return [o[field] for o in OCC if o["day"]]


# ---------------------------------------------------------------- политики
class LRU:
    name = "LRU"

    def __init__(self, c):
        self.c, self.d = c, OrderedDict()

    def access(self, k):
        hit = k in self.d
        if hit:
            self.d.move_to_end(k)
        else:
            self.d[k] = 1
            if len(self.d) > self.c:
                self.d.popitem(last=False)
        return hit


class LFU:
    name = "LFU"

    def __init__(self, c):
        self.c, self.d, self.f, self.t = c, {}, defaultdict(int), 0

    def access(self, k):
        self.t += 1
        self.f[k] += 1
        hit = k in self.d
        if not hit:
            if len(self.d) >= self.c:
                # выбрасываем наименее частый, ничьи по давности
                victim = min(self.d, key=lambda x: (self.f[x], self.d[x]))
                del self.d[victim]
            self.d[k] = self.t
        else:
            self.d[k] = self.t
        return hit


class RAND:
    name = "random"

    def __init__(self, c, seed=1):
        self.c, self.d, self.r = c, [], random.Random(seed)

    def access(self, k):
        hit = k in self.d
        if not hit:
            if len(self.d) >= self.c:
                self.d.pop(self.r.randrange(len(self.d)))
            self.d.append(k)
        return hit


class ARC:
    """Adaptive Replacement Cache, Megiddo & Modha, FAST 2003."""
    name = "ARC"

    def __init__(self, c):
        self.c = c
        self.p = 0
        self.t1, self.t2 = OrderedDict(), OrderedDict()
        self.b1, self.b2 = OrderedDict(), OrderedDict()

    def _replace(self, k):
        if self.t1 and (len(self.t1) > self.p or (k in self.b2 and len(self.t1) == self.p)):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = 1
        elif self.t2:
            old, _ = self.t2.popitem(last=False)
            self.b2[old] = 1

    def access(self, k):
        c = self.c
        if k in self.t1:
            del self.t1[k]
            self.t2[k] = 1
            return True
        if k in self.t2:
            self.t2.move_to_end(k)
            return True
        if k in self.b1:
            d = 1 if len(self.b1) >= len(self.b2) else len(self.b2) // max(1, len(self.b1))
            self.p = min(c, self.p + d)
            self._replace(k)
            del self.b1[k]
            self.t2[k] = 1
            return False
        if k in self.b2:
            d = 1 if len(self.b2) >= len(self.b1) else len(self.b1) // max(1, len(self.b2))
            self.p = max(0, self.p - d)
            self._replace(k)
            del self.b2[k]
            self.t2[k] = 1
            return False
        # полный промах
        if len(self.t1) + len(self.b1) >= c:
            if len(self.t1) < c:
                self.b1.popitem(last=False)
                self._replace(k)
            else:
                self.t1.popitem(last=False)
        else:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= c:
                if total >= 2 * c and self.b2:
                    self.b2.popitem(last=False)
                self._replace(k)
        self.t1[k] = 1
        return False


class CMS:
    """Count-Min Sketch со старением: половинит счётчики каждые sample обращений."""

    def __init__(self, width, depth=4, sample=1000, seed=7):
        self.w, self.d, self.sample = width, depth, sample
        self.t = [[0] * width for _ in range(depth)]
        r = random.Random(seed)
        self.salt = [r.randrange(1 << 30) for _ in range(depth)]
        self.n = 0

    def add(self, k):
        for i in range(self.d):
            self.t[i][hash((self.salt[i], k)) % self.w] += 1
        self.n += 1
        if self.n >= self.sample:
            self.n = 0
            for row in self.t:
                for j in range(self.w):
                    row[j] >>= 1

    def est(self, k):
        return min(self.t[i][hash((self.salt[i], k)) % self.w] for i in range(self.d))


class WTinyLFU:
    """W-TinyLFU: окно LRU 1%, основной SLRU, допуск по частотному эскизу."""
    name = "W-TinyLFU"

    def __init__(self, c, window_frac=0.01, protected_frac=0.8):
        self.cw = max(1, int(round(c * window_frac)))
        self.cm = max(1, c - self.cw)
        self.cp = max(1, int(self.cm * protected_frac))
        self.window = OrderedDict()
        self.probation = OrderedDict()
        self.protected = OrderedDict()
        self.sk = CMS(width=max(64, 8 * c), sample=10 * c)

    def access(self, k):
        self.sk.add(k)
        if k in self.window:
            self.window.move_to_end(k)
            return True
        if k in self.protected:
            self.protected.move_to_end(k)
            return True
        if k in self.probation:
            del self.probation[k]
            self.protected[k] = 1
            if len(self.protected) > self.cp:
                dem, _ = self.protected.popitem(last=False)
                self.probation[dem] = 1
                self._trim_main()
            return True
        # промах: кладём в окно
        self.window[k] = 1
        if len(self.window) > self.cw:
            cand, _ = self.window.popitem(last=False)
            self._admit(cand)
        return False

    def _trim_main(self):
        while len(self.probation) + len(self.protected) > self.cm and self.probation:
            self.probation.popitem(last=False)

    def _admit(self, cand):
        if len(self.probation) + len(self.protected) < self.cm:
            self.probation[cand] = 1
            return
        if not self.probation:
            self.probation[cand] = 1
            self._trim_main()
            return
        victim = next(iter(self.probation))
        # допуск: кандидат вытесняет жертву только если чаще её
        if self.sk.est(cand) > self.sk.est(victim):
            del self.probation[victim]
            self.probation[cand] = 1


class Score:
    """Наша мера как политика вытеснения: считается онлайн по истории."""
    name = "наша мера 0.5/0.2/0.3"
    W = (0.5, 0.2, 0.3)

    def __init__(self, c):
        self.c = c
        self.d = {}
        self.n = defaultdict(int)
        self.proj = defaultdict(set)
        self.last = {}
        self.t = 0

    def _score(self, k):
        rep = min(self.n[k], 10) / 10.0
        spr = min(len(self.proj[k]), 3) / 3.0
        # свежесть в шагах потока: 30 дней ~ доля архива; используем ранг времени
        fresh = max(0.0, 1.0 - (self.t - self.last[k]) / float(self.horizon))
        w = self.W
        return w[0] * rep + w[1] * spr + w[2] * fresh

    def access(self, k, project=None):
        self.t += 1
        self.n[k] += 1
        if project:
            self.proj[k].add(project)
        self.last[k] = self.t
        hit = k in self.d
        if not hit:
            if len(self.d) >= self.c:
                victim = min(self.d, key=self._score)
                del self.d[victim]
            self.d[k] = 1
        return hit


class ScoreFresh(Score):
    name = "мера 0.1/0/0.9 (свежесть)"
    W = (0.1, 0.0, 0.9)


class ScoreRepeat(Score):
    name = "мера 1/0/0 (=LFU)"
    W = (1.0, 0.0, 0.0)


def belady(seq, c, warm=0):
    """Оракул Belady: выбрасываем то, что понадобится позже всех. Потолок."""
    nxt = defaultdict(deque)
    for i, k in enumerate(seq):
        nxt[k].append(i)
    cache, hits, tot = set(), 0, 0
    for i, k in enumerate(seq):
        nxt[k].popleft()
        if i >= warm:
            tot += 1
        if k in cache:
            if i >= warm:
                hits += 1
            continue
        if len(cache) >= c:
            worst, wpos = None, -1
            for x in cache:
                pos = nxt[x][0] if nxt[x] else float("inf")
                if pos > wpos:
                    worst, wpos = x, pos
            cache.discard(worst)
        cache.add(k)
    return hits / tot if tot else float("nan")


def run(seq, projects, policy_cls, c, horizon=None, warm=0):
    p = policy_cls(c)
    if isinstance(p, Score):
        p.horizon = horizon or (len(seq) * 30.0 / 68.0)
    hits = tot = 0
    for i, k in enumerate(seq):
        if isinstance(p, Score):
            h = p.access(k, projects[i])
        else:
            h = p.access(k)
        if i >= warm:
            tot += 1
            hits += 1 if h else 0
    return hits / tot if tot else float("nan")


def main():
    for field, label in (("pkey", "portable"), ("key", "штатный fact_key")):
        occ = [o for o in OCC if o["day"]]
        seq = [o[field] for o in occ]
        proj = [o["project"] for o in occ]
        uniq = len(set(seq))
        inf_hit = (len(seq) - uniq) / len(seq)
        # тёплая часть: вхождения после границы
        warm = sum(1 for o in occ if o["day"] < "2026-08-09")
        print("=" * 92)
        print("ИДЕНТИЧНОСТЬ %s | обращений %d, различных %d | потолок бесконечного кэша %.3f"
              % (label, len(seq), uniq, inf_hit))
        print("  (промах неизбежен на первом обращении к каждому факту: %d из %d)" % (uniq, len(seq)))
        sizes = [10, 25, 50, 100, 200, 400]
        head = "  %-26s" % "политика" + "".join("%9s" % ("C=%d" % c) for c in sizes)
        print(head)
        rows = [LRU, LFU, ARC, WTinyLFU, Score, ScoreFresh, ScoreRepeat, RAND]
        for cls in rows:
            line = "  %-26s" % cls.name
            for c in sizes:
                line += "%9.3f" % run(seq, proj, cls, c, warm=warm)
            print(line)
        line = "  %-26s" % "Belady (оракул, потолок)"
        for c in sizes:
            line += "%9.3f" % belady(seq, c, warm=warm)
        print(line)
        print("  (доля попаданий считается на хвосте после 2026-08-09, кэш прогрет головой)")
        print()


if __name__ == "__main__":
    main()
