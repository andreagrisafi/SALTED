import os
import argparse
import os.path as osp

from ase.io import read, write

import inp

def add_command_line_arguments_contraction():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pr", "--predict", action='store_true', help="Prepare geometries for a true prediction")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction()
predict = args.predict

fname = inp.filename
if predict:
    datadir = inp.predict_data
else:
    datadir = "data/"

dirpath = osp.join(inp.path2qm, datadir, "geoms")
if not osp.exists(dirpath):
    os.makedirs(dirpath)

xyz_file = read(fname,":")
n = len(xyz_file)
for i in range(n):
    write(osp.join(dirpath, f"{i+1}.in"), xyz_file[i])
