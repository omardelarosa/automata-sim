import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import json
import re

DIRS_GLOB = "renderings/midi/wolfram_eca_8bit_seed_*bit/test_results.json"

files = glob(DIRS_GLOB)

d_by_file = {}

for f in files:
    with open(f, "r") as json_file:
        d = json.load(json_file)
        d_by_file[f] = d

pairings = []

for k in d_by_file:
    f = k
    r = d_by_file[k]["success_rate"]
    num_bits = 0
    matches = re.findall(r"_seed_(\d+)bit", f)
    if matches:
        num_bits = int(matches[0])
    pairings.append((num_bits, r))
    # print(num_bits, r, f)

pairings.sort(key=lambda x: x[0])

print(pairings)

x = []
y = []

for p in pairings:
    x.append(p[0])
    y.append(p[1])

plt.plot(x, y, "bo")
plt.ylabel("percentage correctly classified")
plt.xlabel("number of activated bits in seed")
plt.show()
