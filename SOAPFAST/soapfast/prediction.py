#!/usr/bin/env python

from utils import parsing,regression_utils,sagpr_utils
import scipy.linalg
import sys
import numpy as np
from ase.io import read

###############################################################################################################################

def main():

    # This is a wrapper that calls python scripts to do SA-GPR with pre-built L-SOAP kernels.
    
    # Parse input arguments
    args = parsing.add_command_line_arguments_predict("SA-GPR prediction")
    [rank,wfile,kernels,ofile,threshold,asymmetric] = parsing.set_variable_values_predict(args)
    
    if (args.spherical == False):
    
        # Read-in kernels
        print("Loading kernel matrices...")
    
        kernel = []
        for k in range(len(kernels)):
            kr = np.load(kernels[k])
            if (kernel == []):
                ns = len(kr)
                nt = len(kr[0])
            else:
                # Check that we have this right
                if (ns != len(kr)):
                    print("The dimensions of these kernels do not agree with each other!")
                    sys.exit(0)
            kernel.append(kr)
    
        print("...Kernels loaded.")

        # Initialize variables describing how we get a full tensor from its spherical components
        if (asymmetric == ''):
            [sph_comp,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = sagpr_utils.get_spherical_tensor_components(['1.0 ' * 3**rank for i in range(nt)],rank,threshold)
        else:
            if (len(asymmetric) != 2):
                print("ERROR: with asymmetric option, two arguments must be given!")
                sys.exit(0)
            if (rank <= 1):
                print("ERROR: scalar properties cannot be asymmetric!")
                sys.exit(0)
            elif (rank == 2):
                tens = ' '.join(np.concatenate(np.array(read(asymmetric[0],':')[0].info[asymmetric[1]]).astype(str)))
            else:
                tens = ' '.join(np.array(read(asymmetric[0],':')[0].info[asymmetric[1]]).astype(str))
            [sph_comp,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = sagpr_utils.get_spherical_tensor_components([tens for i in range(nt)],rank,threshold)
    
        outvec = []
        for l in range(len(degen)):
            # Find a prediction for each spherical component
            lval = keep_list[l][-1]
            str_rank = ''.join(map(str,keep_list[l][1:]))
            if (str_rank == ''):
                str_rank = ''.join(map(str,keep_list[l]))
            outvec.append(sagpr_utils.do_prediction_spherical(kernel[l],rank_str=str_rank,weightfile=wfile,outfile=ofile))
    
        ns = int(len(outvec[0]) / degen[0])
        predcart = regression_utils.convert_spherical_to_cartesian(outvec,degen,ns,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list)
        corrfile = open(ofile + "_cartesian.txt",'w')
        for i in range(ns):
            print(' '.join(str(e) for e in list(np.split(predcart,ns)[i])), file=corrfile)
        corrfile.close()
    
    else:
    
        # Read-in kernels
        print("Loading kernel matrices...")
    
        kr = np.load(kernels)
        kernel = kr
    
        print("...Kernels loaded.")
    
        # Do the prediction for this spherical component
        sagpr_utils.do_prediction_spherical(kernel,rank_str=str(rank),weightfile=wfile,outfile=ofile)

if __name__=="__main__":
    main()
