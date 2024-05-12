# Given a CP2K output file using an automatically generated RI HFX basis, this script produces a 
#CP2K basis set file containing a fully decontracted version of the same basis, for each element

#usage:
#python uncontract_ri_basis.py cp2k_output_file new_basis_set_file

import sys

#reading arguments
cp2k_output = sys.argv[1]
basis_file = sys.argv[2]

#defining some helper functions
def get_l_from_string(string):
    if "s" in string:
        return 0
    if "p" in string:
        return 1
    if "d" in string:
        return 2
    if "f" in string:
        return 3
    if "g" in string:
        return 4
    if "h" in string:
        return 5
    if "i" in string:
        return 6
    if "j" in string:
        return 7
    if "k" in string:
        return 8
    return "not a valid l quantum number"

def get_set_exponents(kind, iset, lines):
    correct_kind = False
    correct_basis = False
    correct_set = False

    exps = []
    momenta = []
    for line in lines:
        if " Atomic kind: " in line:
            if line.split()[3] == kind:
                correct_kind = True
            else:
                correct_kind = False

        if "Basis Set                 " in line:
            if "RI HFX " in line:
                correct_basis = True
                if not "RI-AUX" in line:
                    print("Warning: the script is meant to be used with automatically generated RI basis sets")
            else:
                correct_basis = False

        if len(line) == 1:
            correct_set = False

        if correct_kind and correct_basis and len(line) > 1:
            if line.split()[0] == str(iset):
                correct_set = True
                momenta.append(get_l_from_string(line.split()[2]))

        if correct_kind and correct_basis and correct_set:
            exps.append(line.split()[-2])

        if "  Atomic covalent radius " in line:
            break

    #remove values appearing multiple times before return
    exps = [float(e) for e in dict.fromkeys(exps).keys()]
    momenta = [l for l in dict.fromkeys(momenta).keys()]

    return exps, momenta

def get_kinds(lines):
    kinds = []
    for line in output_lines:
        if " Atomic kind: " in line:
            kinds.append(line.split()[3])
    return kinds

def get_sets(lines):
    nsets = []
    for i, line in enumerate(output_lines):
        if "  Number of orbital shell sets:   " in line:
            if "RI HFX Basis Set" in output_lines[i-2]:
                nsets.append(int(line.split()[-1]))
    return nsets
    

#reading the CP2K output file
with open(cp2k_output, "r") as myfile:
    output_lines = myfile.readlines()

#parse the number of atomic kinds
kinds = get_kinds(output_lines)
nkinds = len(kinds)

#parse the number of sets per kind
nsets = get_sets(output_lines)

#printing to new basis set file
with open(basis_file, "w") as myfile:
    
    for ikind, kind in enumerate(kinds):

        #get total number of basis functions for this kind
        nfunc = 0
        nset = nsets[ikind]
        for iset in range(1, nset + 1):
            exps, momenta = get_set_exponents(kind, iset, output_lines)
            nfunc += len(exps)*len(momenta)

        myfile.write("{} RI_AUTO_uncontracted\n".format(kind))
        myfile.write("   {}\n".format(nfunc))

        for iset in range(1, nset+1):
            exps, momenta = get_set_exponents(kind, iset, output_lines)
            nfunc += len(exps)*len(momenta)
            for l in momenta:
                for exp in exps:
                    myfile.write(" 1 {} {} 1 1\n".format(l,l))
                    myfile.write("   {:11.6f}  1.0\n".format(exp))

        myfile.write("\n")

