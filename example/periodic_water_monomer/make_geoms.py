import inp
import os
from ase.io import read, write
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-pr", "--predict", action='store_true', help="Prepare geometries for a true prediction")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
predict = args.predict

if predict:
    fname = inp.predict_filename
    datadir = inp.predict_data
else:
    fname = inp.filename
    datadir = "data/"

if not os.path.exists(inp.path2qm):
    os.mkdir(inp.path2qm)
if not os.path.exists(inp.path2qm+datadir):
    os.mkdir(inp.path2qm+datadir)
dirpath = inp.path2qm+datadir+'geoms/'
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

xyz_file = read(fname,":")
n = len(xyz_file)
for i in range(n):
    write(dirpath+str(i+1)+'.in',xyz_file[i])
