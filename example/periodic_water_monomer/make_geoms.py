import inp
import os
from ase.io import read, write

if not os.path.exists(inp.path2qm):
    os.mkdir(inp.path2qm)
if not os.path.exists(inp.path2qm+'data'):
    os.mkdir(inp.path2qm+'data')
dirpath = inp.path2qm+'data/geoms/'
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

xyz_file = read(inp.filename,":")
n = len(xyz_file)
for i in range(n):
    write(dirpath+str(i+1)+'.in',xyz_file[i])
