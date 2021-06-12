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
    parser.add_argument("-ele", "--electro",                    action='store_true',                    help="Use for electrostatic representations")
    parser.add_argument("-sew", "--sigewald", type=float,  default=1.0,                                 help="Gaussian width for ewald splitting")
    parser.add_argument("-srad", "--sradial",                    action='store_true',                   help="Use for single radial channel representations")
    parser.add_argument("-rad", "--radsize", type=int,  default=50,                                 help="Dimension of the Gauss-Legendre grid needed for the numerical radial integration of the direct-space potential")
    parser.add_argument("-leb", "--lebsize", type=int,  default=146,                                 help="Dimension of the Lebedev grid needed for the numerical angular integration of the direct-space potential. Choose among [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030 (army grade), 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810]")

    args = parser.parse_args()
    return args

###############################################################################################################################
def set_variable_values_PS(args):

    # SOAP PARAMETERS
    rad = args.radsize
    leb = args.lebsize
    srad = args.sradial
    ele = args.electro
    sew = args.sigewald               # Gaussian width
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

    return [nmax,lmax,rc,sg,cen,spec,cw,lam,periodic,ncut,sparsefile,frames,subset,sparse_options,outfile,args.initial,atomic,all_radial,not args.uselist,xyz_slice,args.imag,args.nonorm,ele,sew,srad,rad,leb]

#########################################################################
