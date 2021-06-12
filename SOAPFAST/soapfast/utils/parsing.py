#!/usr/bin/python
import argparse
import sys
import numpy as np
import ase
from packaging import version
from ase.io import read
from ase.data import atomic_numbers,chemical_symbols

# ASE 3.20 stopped interpreting vectors with 9 values as 3x3 tensors,
# so we need to be able to distinguish them
ASE_LOWER_3_20 = version.parse(ase.__version__) < version.parse("3.20")

###############################################################################################################################

def add_command_line_arguments_learn(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r",   "--rank",                type=str,   required=True,                              help="Rank of tensor to learn")
    parser.add_argument("-reg",  "--regularization",     type=float, required=True,  nargs='+',                  help="Lambda values list for KRR calculation")
    parser.add_argument("-ftr", "--ftrain",              type=float, default=1.0,                                help="Fraction of data points used for testing")
    parser.add_argument("-f",   "--features",            type=str,   required=True,                              help="File containing atomic coordinates")
    parser.add_argument("-p",   "--property",            type=str,   required=True,                              help="Property to be learned")
    parser.add_argument("-k",   "--kernel",              type=str,   required=False,  nargs='+',                 help="Files containing kernels")
    parser.add_argument("-sel", "--select",              type=str,   default=[],     nargs='+',                  help="Select maximum training partition")
    parser.add_argument("-rdm", "--random",              type=int,   default=0,                                  help="Number of random training points")
    parser.add_argument("-w",    "--weights",            type=str,   default='weights',                          help="File prefix to print out weights")
    parser.add_argument("-perat","--peratom",                        action='store_true',                        help="Call for scaling the properties by the number of atoms")
    parser.add_argument("-pr",   "--prediction",                     action='store_true',                        help="Carry out prediction?")
    parser.add_argument("-sf",   "--sparsify",           type=str,   required=False,  nargs='+',                 help="Kernels for sparsification")
    parser.add_argument("-sp",   "--spherical",                      action='store_true',                        help="Regression on a single spherical component")
    parser.add_argument("-m",    "--mode",               type=str,   choices=['solve','pinv'], default='solve',  help="Mode to use for inversion of kernel matrices")
    parser.add_argument("-t",    "--threshold",          type=float, default=1e-8,                               help="Threshold value for spherical component zeroing")
    parser.add_argument("-c",    "--center",             type=str,   default='',                                 help="Species to be used for property extraction ")
    parser.add_argument("-j",    "--jitter",             type=str,   required=False,  nargs='+',                 help="Jitter term for sparse regression")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_learn(args):

    ftr = args.ftrain
    if (int(args.rank) == 0):
        args.spherical = True
    # Get regularization
    reg = args.regularization
    if args.spherical:
        reg = reg[0]
        rank = args.rank
        int_rank = int(rank[-1])
    else:
        rank = int(args.rank)
        int_rank = rank

    # Read in features
    ftrs = read(args.features,':')

    # Either we have supplied kernels for carrying out the regression, or sparsification kernels, but not both (or neither).
    kernels = args.kernel
    sparsify = args.sparsify
    nat = [ftrs[i].get_global_number_of_atoms() for i in range(len(ftrs))]

    # Read in tensor data for training the model
    if (not args.spherical):
        if args.center != '':
            if int_rank == 0:
                tens = [ str(frame_prop) for fr in ftrs for frame_prop in fr.arrays[args.property][np.where(fr.numbers==atomic_numbers[args.center])[0]] ]
            else:
                tens = [' '.join(frame_prop.astype(str))  for fr in ftrs for frame_prop in fr.arrays[args.property][np.where(fr.numbers==atomic_numbers[args.center])[0]]]
            nat = [1 for i in range(len(tens))]
        elif args.peratom:
            if int_rank == 0:
                tens = [str(ftrs[i].info[args.property]/nat[i]) for i in range(len(ftrs))]
            elif ASE_LOWER_3_20 and int_rank == 2:
                tens = [' '.join((np.concatenate(ftrs[i].info[args.property])/nat[i]).astype(str)) for i in range(len(ftrs))]
            else:
                tens = [' '.join((np.array(ftrs[i].info[args.property])/nat[i]).astype(str)) for i in range(len(ftrs))]
        else:
            if int_rank == 0:
                tens = [str(ftrs[i].info[args.property]) for i in range(len(ftrs))]
            elif ASE_LOWER_3_20 and int_rank == 2:
                tens = [' '.join(np.concatenate(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
            else:
                tens = [' '.join(np.array(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]


        if (kernels == None and sparsify == None):
            print("Either regular kernels or sparsification kernels must be specified!")
            sys.exit(0)
        if (kernels != None and sparsify != None):
            print("Either regular kernels or sparsification kernels must be specified (not both)!")
            sys.exit(0)
    else:
        if args.center != '':
            if int_rank == 0:
                tens = [ str(frame_prop) for fr in ftrs for frame_prop in fr.arrays[args.property][np.where(fr.numbers==atomic_numbers[args.center])[0]] ]
            else:
                tens = [' '.join(frame_prop.astype(str).reshape(2*int_rank + 1))  for fr in ftrs for frame_prop in fr.arrays[args.property][np.where(fr.numbers==atomic_numbers[args.center])[0]]]
            nat = [1 for i in range(len(tens))]
        elif args.peratom:
            if int_rank == 0:
                tens = [str(ftrs[i].info[args.property]/nat[i]) for i in range(len(ftrs))]
            else:
                tens = [' '.join((np.array(ftrs[i].info[args.property].reshape(2*int_rank + 1))/nat[i]).astype(str)) for i in range(len(ftrs))]
        else:
            if int_rank == 0:
                tens = [str(ftrs[i].info[args.property]) for i in range(len(ftrs))]
            else:
                tens = [' '.join((np.array(ftrs[i].info[args.property].reshape(2*int_rank + 1))).astype(str)) for i in range(len(ftrs))]


        if kernels != None:
            kernels = kernels[0]
        if (kernels == None and sparsify == None):
            print("Either regular kernels or sparsification kernels must be specified!")
            sys.exit(0)
        if (kernels != None and sparsify != None):
            print("Either regular kernels or sparsification kernels must be specified (not both)!")
            sys.exit(0)

    # If a selection is given for the training set, read it in
    sel = args.select
    if (len(sel)==2):
        sel = [int(sel[0]),int(sel[1])]
    elif (len(sel)==1):
        # Read in an input file giving a training set
        sel = sel
    elif (len(sel) > 2):
        print("ERROR: too many arguments given to selection!")
        sys.exit(0)

    rdm = args.random

    if ((rdm == 0) & (len(sel)==0)):
        sel = [0,-1]

    jitter = args.jitter
    if ((jitter != None) and (sparsify == None)):
        print("NOTE: jitter term is not used without sparse kernels")
    if (jitter == None and isinstance(reg,list)):
        jitter = [None for i in reg]
    else:
        if (isinstance(reg,list)):
            if (len(jitter)<len(reg)):
                print("ERROR: as many jitter terms as regularizations must be included!")
                sys.exit(0)

    if (args.spherical):
        jitter = [jitter]

    if ((sparsify != None) and (jitter == None) and (args.mode=='solve')):
        print("NOTE: with environmental sparsification, a jitter term is recommended if the solve mode is used")

    return [reg,ftr,tens,kernels,sel,rdm,rank,nat,args.peratom,args.prediction,args.weights,sparsify,args.mode,args.threshold,jitter]
###############################################################################################################################
def add_command_line_arguments_PS(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f",  "--fname",         type=str,   required=True,                            help="Filename")
    parser.add_argument("-n",  "--nmax",          type=int,   default=8,                                help="Number of radial functions")
    parser.add_argument("-l",  "--lmax",          type=int,   default=6,                                help="Number of angular functions")
    parser.add_argument("-rc", "--rcut",          type=float, default=4.0,                              help="Environment cutoff")
    parser.add_argument("-sg", "--sigma",         type=float, default=0.3,                              help="Gaussian width")
    parser.add_argument("-c",  "--centres",       type=str,   default = '', nargs='+',                  help="List of centres")
    parser.add_argument("-s",  "--species",       type=str,   default = '', nargs='+',                  help="List of species")
    parser.add_argument("-cw", "--cweight",       type=float, default=1.0,                              help="Central atom weight")
    parser.add_argument("-lm", "--lambdaval",     type=int,   default=0,                                help="Spherical tensor order")
    parser.add_argument("-p",  "--periodic",                  action='store_true',                      help="Is the system periodic?")
    parser.add_argument("-nc", "--ncut",          type=int,   default=-1,                               help="Dimensionality cutoff")
    parser.add_argument("-i",  "--initial",       type=int,   default=-1,                               help="Initial column for spherical component sparsification")
    parser.add_argument("-sf", "--sparsefile",    type=str,   default='',                               help="File with sparsification parameters")
    parser.add_argument("-ns", "--nsubset",       type=int,   default=-1,                               help="Take this many configurations and sparsify on them")
    parser.add_argument("-sm", "--subsetmode",    type=str,   choices=['seq','random'], default='seq',  help="Method of choosing subset for sparsification")
    parser.add_argument("-o",  "--outfile",       type=str,   default='',                               help="Output file for power spectrum")
    parser.add_argument("-a",  "--atomic",        type=str,                 nargs='*',                  help="Atomic power spectrum")
    parser.add_argument("-rs", "--radialscaling", type=float, nargs='+',                                help="Options for radial scaling (c, r0, m)")
    parser.add_argument("-ul", "--uselist",                   action='store_true',                      help="Use list of allowed features in PS calculation (may be quicker)")
    parser.add_argument("-sl", "--slice",         type=int,   default=-1,    nargs='+',                 help="Choose a slice of the input frames to calculate the power spectrum")
    parser.add_argument("-im", "--imag",                      action='store_true',                      help="Get imaginary power spectrum for building SO(3) kernel")
    parser.add_argument("-nn", "--nonorm",                    action='store_true',                      help="Do not normalize power spectrum")

    args = parser.parse_args()
    return args

###############################################################################################################################
def set_variable_values_PS(args):

    # SOAP PARAMETERS
    nmax = args.nmax              # number of radial functions
    lmax = args.lmax              # number of angular functions
    rc = args.rcut                # environment cutoff
    sg = args.sigma               # Gaussian width
    if args.centres != '':
        cen = args.centres
    else:
        cen = []
    if args.species != '':
        spec = args.species
    else:
        spec = []
    cw = args.cweight             # central atom weight
    lam = args.lambdaval          # spherical tensor order
    periodic = args.periodic      # True for periodic systems
    ncut = args.ncut              # dimensionality cutoff
    sparsefile = args.sparsefile
    fname = args.fname
    frames = read(fname,':')
    outfile = args.outfile

    sparse_options = [sparsefile]
    if sparsefile != '':
        # Here we will read in a file containing sparsification details.
        sparse_fps   = np.load(sparsefile + "_fps.npy")
        sparse_options.append(sparse_fps)
        sparse_Amatr = np.load(sparsefile + "_Amat.npy")
        sparse_options.append(sparse_Amatr)

    atomic = [False,None]
    if (args.atomic != None):
        atomic = [True,args.atomic]

    # If we are doing radial scaling, read in the appropriate variables
    radialscale = args.radialscaling
    if radialscale != None:
        if (len(radialscale) != 3):
            print("ERROR: three arguments (c,r0,m) must be given for radial scaling!")
            sys.exit(0)
        radial_c  = radialscale[0]
        radial_r0 = radialscale[1]
        radial_m  = radialscale[2]
    else:
        radial_c  = 1.0
        radial_r0 = 0.0
        radial_m  = 0.0

    all_radial = [radial_c,radial_r0,radial_m]

    nsubset = args.nsubset
    submode = args.subsetmode

    # Decide whether or not to take a subset of the data for sparsification
    subset = ['NO',None]
    if (nsubset > -1):
        if sparsefile != '':
            print("ERROR: subset option is not compatible with supplying a sparsification filename!")
            sys.exit(0)
        if ncut == -1:
            print("ERROR: ncut must be specified for use with subset option!")
            sys.exit(0)
        if (submode == 'seq'):
            print("Taking the first %i of the coordinates."%nsubset)
            subset = ['SEQ',nsubset]
        elif (submode == 'random'):
            print("Shuffling coordinates and taking %i of them."%nsubset)
            subset = ['RANDOM',nsubset]

    if (args.slice==-1):
        xyz_slice = []
    else:
        if (len(args.slice)!=2):
            print("ERROR: incorrect number of elements provided to slice!")
            sys.exit(0)
        xyz_slice = [args.slice[0],args.slice[1]]

    return [nmax,lmax,rc,sg,cen,spec,cw,lam,periodic,ncut,sparsefile,frames,subset,sparse_options,outfile,args.initial,atomic,all_radial,not args.uselist,xyz_slice,args.imag,args.nonorm]

#########################################################################

def add_command_line_arguments_kernel(parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-z",   "--zeta",      type=int, default=1,                 help="Kernel exponentiation")
    parser.add_argument("-ps",  "--power",     type=str, required=True,  nargs='+', help="Power spectrum file(s)")
    parser.add_argument("-ps0", "--power0",    type=str, default='',     nargs='+', help="lambda=0 power spectrum file(s) for zeta > 1")
    parser.add_argument("-o",   "--output",    type=str, required=True,             help="Output file name")
    parser.add_argument("-s",   "--scaling",   type=str, required=False, nargs='+', help="Scaling file names")
    args = parser.parse_args()
    return args

###############################################################################################################################
def set_variable_values_kernel(args):

    # SOAP PARAMETERS
    zeta = args.zeta                   # Kernel exponentiation

    power = args.power
    if (len(power)>1):
        PS = [np.load(power[0]),np.load(power[1])]
        use_hermiticity = False
    else:
        PS = [np.load(power[0]),np.load(power[0])]
        use_hermiticity = True

    npoints = [len(PS[0]),len(PS[1])]

    # Check that these power spectra can be combined
    if (len(np.shape(PS[0])) == 3):
        degen = 1
        lam = 0
        featsize = [len(PS[0][0,0]),len(PS[1][0,0])]
    else:
        degen = len(PS[0][0,0])
        lam = (degen-1) / 2
        if ((zeta > 1) and (args.power0 == None)):
            print("ERROR: lambda=0 power spectrum must be specified for zeta > 1!")
            sys.exit(0)
        for i in range(2):
            if (len(PS[i][0,0]) != 2*lam+1):
                print("ERROR: power spectrum number " + str(i+1) + " has the wrong lambda value!")
                sys.exit(0)
        featsize = [len(PS[0][0,0,0]),len(PS[1][0,0,0])]

    if (featsize[0] != featsize[1]):
        print("The two feature sizes are different!")
        sys.exit(0)

    # Read in L=0 power spectra
    PS0 = [None,None]
    if (args.power0 != ''):
        if (len(args.power0) > 1):
            PS0 = [np.load(args.power0[0]),np.load(args.power0[1])]
        else:
            PS0 = [np.load(args.power0[0]),np.load(args.power0[0])]
        for i in range(2):
            if (len(PS0[i]) != npoints[i]):
                print("ERROR: lambda=0 power spectrum number " + str(i+1) + " should have the same number of points as lambda=" + str(lam) + "!")
                sys.exit(0)

    # Read in scaling data if appropriate
    scaling = args.scaling
    if scaling == None:
        scaling = ['NONE','NONE']
    elif len(scaling)==1:
        scaling.append(scaling[0])
    scale = []
    for i in range(len(scaling)):
        if scaling[i] == 'NONE':
            scale.append(np.array([1 for j in range(npoints[i])]))
        else:
            scale.append(np.load(scaling[i]))

    return [PS,scale,PS0,zeta,use_hermiticity]

###############################################################################################################################

def add_command_line_arguments_predict(parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-w",   "--weights",    type=str,   default='weights',          help="File prefix to read in weights")
    parser.add_argument("-r",   "--rank",       type=str,   required=True,              help="Rank of tensor to learn")
    parser.add_argument("-k",   "--kernel",     type=str,   required=True, nargs='+',   help="Files containing kernels")
    parser.add_argument("-o",   "--ofile",      type=str,   default='prediction',       help="Output file for predictions")
    parser.add_argument("-sp",  "--spherical",              action='store_true',        help="Prediction of a single spherical component")
    parser.add_argument("-t",   "--threshold",  type=float, default=1e-8,               help="Threshold value for spherical component zeroing")
    parser.add_argument("-as",  "--asymmetric", type=str,   default='',    nargs='+',   help="If applicable, example of asymmetric tensor")
    args = parser.parse_args()
    return args

###############################################################################################################################

def set_variable_values_predict(args):

    if (int(args.rank) == 0):
        args.spherical = True

    if args.spherical:
        rank = args.rank
    else:
        rank = int(args.rank)

    wfile = args.weights
    kernels = args.kernel
    if (args.spherical):
        kernels = kernels[0]
    ofile = args.ofile

    return [rank,wfile,kernels,ofile,args.threshold,args.asymmetric]
###############################################################################################################################
