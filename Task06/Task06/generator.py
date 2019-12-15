import random

sz = 50000000
outfile = open("input.txt", "w")

print(sz, file=outfile, end=" ")

for i in range(sz):
    print(random.randint(-1000000000, 1000000000), file=outfile, end=" ")

print("", file=outfile)
outfile.close()
