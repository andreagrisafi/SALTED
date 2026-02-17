import os
import sys
import math
import numpy as np
from ase.io import read
import copy
import time

from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range

inp = ParseConfig().parse_input()

species = inp.system.species
df_basis = inp.qm.dfbasis

cp2k_basis_filename = sys.argv[1]

with open(cp2k_basis_filename, "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i]
    if line.split()[0] in species and "ri" in line:
        print(line.split()[0])
        index = i
        with open(line.split()[0] + "-" + df_basis, "w") as f:
            while lines[index][0] != "#" and index + 1 < len(lines): # # is a separator except for the end of the file
                f.write(lines[index])
                index+=1
            if index+1==len(lines):
                f.write(lines[index])
