from math import ceil

input()
a = sorted(list(map(int, input().split(" "))))
indexer = ceil(len(a) / 2)
p, q = sorted(a[indexer:]), list(sorted(a[:indexer], reverse=True))

print(p, q)
print(sum(p) + sum(q))
print(" ".join(p), " ".join(q))
