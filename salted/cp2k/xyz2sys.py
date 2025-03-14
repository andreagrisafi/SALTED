import os
import sys
import argparse

from ase.io import read

from salted.sys_utils import ParseConfig

inp = ParseConfig().parse_input()

filename = inp.system.filename
path2qm = inp.qm.path2qm 
periodic = inp.qm.periodic 

xyz = read(filename, ":")
ndata = len(xyz)

if periodic=="0D":
    PERIODIC = "None"
elif periodic=="2D":
    PERIODIC = "XY"
elif periodic=="3D":
    PERIODIC = "XYZ"

for iconf in range(ndata):
    dirpath = os.path.join(path2qm, "conf_"+str(iconf+1))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    symbol = xyz[iconf].get_chemical_symbols()
    coords = xyz[iconf].get_positions()
    natoms = len(coords)
    f = open(os.path.join(path2qm, f"conf_{iconf+1}", "coords.sys"), "w")
    print("&COORD",file=f)
    for iat in range(natoms):
        print(symbol[iat],coords[iat,0],coords[iat,1],coords[iat,2],file=f)
    print("&END COORD",file=f)
    f.close()
    cell = xyz[iconf].get_cell()
    f = open(os.path.join(path2qm, f"conf_{iconf+1}", "cell.sys"), "w")
    print("&CELL",file=f)
    print("PERIODIC "+str(PERIODIC),file=f)
    print("A",cell[0,0],cell[0,1],cell[0,2],file=f)
    print("B",cell[1,0],cell[1,1],cell[1,2],file=f)
    print("C",cell[2,0],cell[2,1],cell[2,2],file=f)
    print("&END CELL",file=f)
    f.close()
