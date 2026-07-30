import os
import sys
import math
import numpy as np
from ase.io import read
import copy
import time
import os.path as osp

from salted.sys_utils import ParseConfig

inp = ParseConfig().parse_input()

species = inp.system.species
df_basis = inp.qm.dfbasis

cp2k_basis_filename = sys.argv[1]
cp2k_input_filename = sys.argv[2]
potential_filepath = sys.argv[3]

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

# Generate directory for saving basis set info 
bdir = osp.join(inp.salted.saltedpath, "basis")
if not osp.exists(bdir):
    os.mkdir(bdir)

pseudocharge = {}

rloc = {}

with open(cp2k_input_filename, "r") as f:
    lines = f.readlines()

pseudo_name = {}
subsys_section = False

for i in range(len(lines)):
    line = lines[i]
    
    if "SUBSYS" in line:
        subsys_section = True
    
    if "&KIND" in line and subsys_section:
        spe = lines[i].split()[1]
        pseudo_name[spe] = lines[i+2].split()[1]

with open(potential_filepath, "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i]
    if len(line.split()) != 0:
        if line.split()[0] in species:
            spe = line.split()[0]
            if line.split()[1] == pseudo_name[spe]:
                pseudocharge[spe] = 0
                e_cfg = lines[i+1].split()
                for x in e_cfg:
                    pseudocharge[spe] += float(x)
            
                rloc[spe] = float(lines[i+2].split()[0])
i = 0
for spe in inp.system.species:
    if i == 0:
        with open(osp.join(bdir,f"pseudocharge.txt"), "w") as f:
            f.write(str(pseudocharge[spe])+"\n")
        with open(osp.join(bdir,f"rloc.txt"), "w") as f:
            f.write(str(rloc[spe])+"\n")
    else:
        with open(osp.join(bdir,f"pseudocharge.txt"), "a") as f:
            f.write(str(pseudocharge[spe])+"\n")
        with open(osp.join(bdir,f"rloc.txt"), "a") as f:
            f.write(str(rloc[spe])+"\n")
    i+=1
