import os
import sys
from ase.io import read

sys.path.insert(0, './')
import inp

xyz = read(inp.filename,":")
ndata = len(xyz)

if inp.periodic=="0D":
    PERIODIC = "None"
elif inp.periodic=="2D":
    PERIODIC = "XY"
elif inp.periodic=="3D":
    PERIODIC = "XYZ"

for iconf in range(ndata):
    dirpath = os.path.join(inp.path2qm, "conf_"+str(iconf+1))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    symbol = xyz[iconf].get_chemical_symbols()
    coords = xyz[iconf].get_positions()
    natoms = len(coords)
    f = open(inp.path2qm+"/conf_"+str(iconf+1)+"/coords.sys","w")
    print("&COORD",file=f)
    for iat in range(natoms):
        print(symbol[iat],coords[iat,0],coords[iat,1],coords[iat,2],file=f)
    print("&END COORD",file=f)
    f.close()
    cell = xyz[iconf].get_cell()
    f = open(inp.path2qm+"/conf_"+str(iconf+1)+"/cell.sys","w")
    print("&CELL",file=f)
    print("PERIODIC "+str(PERIODIC),file=f)
    print("A",cell[0,0],cell[0,1],cell[0,2],file=f)
    print("B",cell[1,0],cell[1,1],cell[1,2],file=f)
    print("C",cell[2,0],cell[2,1],cell[2,2],file=f)
    print("&END CELL",file=f)
    f.close()
