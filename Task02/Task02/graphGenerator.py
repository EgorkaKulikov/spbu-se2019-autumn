from random import *

outfile = open("input.txt", "w")
v = 5000
print(v, file=outfile)
edges = set()
root = randint(0, v - 1)
for i in range(v):
    if (i != root):
        j = randint(0, v - 1)
        while (i == j or (min(i, j), max(i, j)) in edges):
            j = randint(0, v - 1)
        w = randint(1, 100000)
        if (j < i):
            tmp = i
            i = j
            j = tmp
        edges.add((i, j))
        print(*[i, j, w], file=outfile)

e = 1000000 - v + 1

print("NYA!")

for i in range(e):
    if (i % 100000 == 0):
        print("NYA!" + str(i))
    i = randint(0, v - 1)
    j = randint(0, v - 1)
    c = 0
    w = randint(1, 100000)
    while (i == j or (min(i, j), max(i, j)) in edges):
        j = randint(0, v - 1)
        if (c > 1000):
            i = randint(0, v - 1)
    if (j < i):
        tmp = i
        i = j
        j = tmp
    edges.add((i, j))
    print(*[i, j, w], file=outfile)
outfile.close()
