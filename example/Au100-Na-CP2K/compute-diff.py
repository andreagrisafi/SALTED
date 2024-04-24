import os
import sys
import numpy as np

from salted.sys_utils import ParseConfig
inp = ParseConfig().parse_input()

dirpath = os.path.join(inp.qm.path2qm, "coefficients-response")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

for i in range(40):
    c1 = np.load("/scratchbeta/grisafia/Au-fcc100-223/runs/coefficients/coefficients_conf"+str(i)+".npy")
    c2 = np.load(inp.qm.path2qm+"coefficients/coefficients_conf"+str(i)+".npy")
    c = c2-c1
    np.save(inp.qm.path2qm+"/coefficients-response/coefficients_conf"+str(i)+".npy",c)
