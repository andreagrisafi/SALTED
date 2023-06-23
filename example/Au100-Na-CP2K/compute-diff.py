import os
import sys
import numpy as np

sys.path.insert(0, './')
import inp

dirpath = os.path.join(inp.path2qm, "coefficients-response")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

for i in range(40):
    c1 = np.load("/scratchbeta/grisafia/Au-fcc100-223/runs/coefficients/coefficients_conf"+str(i)+".npy")
    c2 = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(i)+".npy")
    c = c2-c1
    np.save(inp.path2qm+"/coefficients-response/coefficients_conf"+str(i)+".npy",c)
