import json
import numpy as np
import glob

prefix = "./renderings/midi/kernel_space_3x1/t_1581277553_"
f_patt = "{prefix}*.json".format(prefix=prefix)

files = glob.glob(f_patt)

for f in files:
    print("file: ", f.replace(prefix, ""))
