#!/usr/bin/python
import sys
from collections import defaultdict


count = defaultdict(int)

for line in sys.stdin:
    ws = line.strip().split()
    for w in ws:
        count[w] += 1

type_count = defaultdict(int)
num_types = 0
for k, v in count.iteritems():
    type_count[v] += 1
    num_types += 1

sys.stdout.write('total number of word types: {}\n\n'.format(num_types))
for i in range(1, 20):
    if not i in type_count:
        continue
    num_types -= type_count[i]
    sys.stdout.write('# types with freq > {}:\t{}\n'.format(i, num_types))
