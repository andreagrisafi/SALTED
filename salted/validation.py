import os
import sys
import time
import os.path as osp

import numpy as np
from scipy import special

#from sympy.parsing import mathematica
#from sympy import symbols
#from sympy import lambdify

from salted import basis
from salted.sys_utils import read_system, get_atom_idx, get_conf_range

def build():

    sys.path.insert(0, './')
    import inp

    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank = 0
        size = 1

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    bohr2angs = 0.529177210670

    if inp.field:
        vdir = f"validations_{inp.saltedname}_field"
        rdir = f"regrdir_{inp.saltedname}_field"
        kdir = f"kernels_{inp.saltedname}_field"
    else:
        vdir = f"validations_{inp.saltedname}"
        rdir = f"regrdir_{inp.saltedname}"
        kdir = f"kernels_{inp.saltedname}"

    # read basis

    # number of sparse environments
    M = inp.Menv
    eigcut = inp.eigcut
    reg = inp.regul
    zeta = inp.z

    for iconf in range(ndata):
        # Define relevant species
        excluded_species = []
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            if spe not in species:
                excluded_species.append(spe)
        excluded_species = set(excluded_species)
        for spe in excluded_species:
            atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))

    # recompute number of atoms
    natoms_total = 0
    natoms_list = []
    natoms = np.zeros(ndata,int)
    for iconf in range(ndata):
        natoms[iconf] = 0
        for spe in species:
            natoms[iconf] += natom_dict[(iconf,spe)]
        natoms_total += natoms[iconf]
        natoms_list.append(natoms[iconf])
    natmax = max(natoms_list)

    # define test set
    trainrangetot = np.loadtxt(osp.join(
        inp.saltedpath, rdir, f"training_set_N{inp.Ntrain}.txt"
    ), int)
    ntrain = round(inp.trainfrac*len(trainrangetot))
    testrange = np.setdiff1d(list(range(ndata)),trainrangetot)

    # Distribute structures to tasks
    ntest = len(testrange)
    if inp.parallel:
        testrange = get_conf_range(rank,size,ntest,testrange)
        testrange = comm.scatter(testrange,root=0)
        print(f"Task {rank+1} handles the following structures: {testrange}", flush=True)

    reg_log10_intstr = str(int(np.log10(reg)))

    # load regression weights
    weights = np.load(osp.join(
        inp.saltedpath, rdir, f"M{M}_zeta{zeta}", f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
    ))

    dirpath = os.path.join(inp.saltedpath, vdir, f"M{M}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}")
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    if size > 1: comm.Barrier()


    if inp.qmcode=="cp2k":
        # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
        if rank == 0: print("Reading auxiliary basis info...")
        alphas = {}
        sigmas = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                avals = np.loadtxt(osp.join(f"{spe}-{inp.dfbasis}-alphas-L{l}.dat"))
                if nmax[(spe,l)]==1:
                    alphas[(spe,l,0)] = float(avals)
                    sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
                else:
                    for n in range(nmax[(spe,l)]):
                        alphas[(spe,l,n)] = avals[n]
                        sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

        # compute integrals of basis functions (needed to a posteriori correction of the charge)
        charge_integrals = {}
        dipole_integrals = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                charge_integrals_temp = np.zeros(nmax[(spe,l)])
                dipole_integrals_temp = np.zeros(nmax[(spe,l)])
                for n in range(nmax[(spe,l)]):
                    inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                    charge_radint = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
                    dipole_radint = 2**float(1.0+float(l)/2.0) * sigmas[(spe,l,n)]**(4+l) * special.gamma(2.0+float(l)/2.0)
                    charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
                    dipole_integrals[(spe,l,n)] = dipole_radint * np.sqrt(4.0*np.pi/3.0) / np.sqrt(inner)

    # If total charge density is asked, read in the GTH pseudo-charge and return a radial numpy function
    #if inp.totcharge:
    #    pseudof = open(inp.pseudochargefile,"r")
    #    pseudochargedensity = mathematica.mathematica(pseudof.readlines()[0],{'Erf[x]':'erf(x)'})
    #    pseudof.close()
    #    rpseudo = symbols('r')
    #    pseudochargedensity = lambdify(rpseudo, pseudochargedensity, modules=['numpy'])
    #    pseudochargedensity = np.vectorize(pseudochargedensity)
    #    nn = 100000
    #    dr = 5.0/nn
    #    pseudocharge = 0.0
    #    for ir in range(1,nn):
    #        r = ir*dr
    #        pseudocharge += r**2*pseudochargedensity(r)
    #    pseudocharge *= 4*np.pi*dr
    #    print("Integrated pseudo-charge =", pseudocharge)

    # Load spherical averages if required
    if inp.average:
        av_coefs = {}
        for spe in inp.species:
            av_coefs[spe] = np.load(f"averages_{spe}.npy")

    # compute error over test set

    efname = osp.join(
        inp.saltedpath, vdir, f"M{M}_zeta{zeta}",
        f"N{ntrain}_reg{reg_log10_intstr}", f"errors.dat"
    )
    if rank == 0 and os.path.exists(efname):
        os.remove(efname)
    if inp.qmcode == "cp2k":
        dfname = osp.join(
            inp.saltedpath, vdir, f"M{M}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}", f"dipoles.dat"
        )
        qfname = osp.join(
            inp.saltedpath, vdir, f"M{M}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}", f"charges.dat"
        )
        if rank == 0 and os.path.exists(dfname): os.remove(dfname)
        if rank == 0 and os.path.exists(qfname): os.remove(qfname)
    if inp.parallel: comm.Barrier()
    efile = open(efname,"a")
    if inp.qmcode=="cp2k":
        dfile = open(dfname,"a")
        qfile = open(qfname,"a")

    error_density = 0
    variance = 0
    for iconf in testrange:

        # load reference
        ref_coefs = np.load(osp.join(
            inp.saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"
        ))
        overl = np.load(osp.join(
            inp.saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
        ))
        ref_projs = np.dot(overl,ref_coefs)
        Tsize = len(ref_coefs)

        # compute predictions per channel
        C = {}
        ispe = {}
        isize = 0
        iii = 0
        for spe in species:
            ispe[spe] = 0
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    psi_nm = np.load(osp.join(
                        inp.saltedpath, kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}",
                        f"psi-nm_conf{iconf}.npy"
                    ))
                    Mcut = psi_nm.shape[1]
                    C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                    isize += Mcut
                    iii += 1

        # fill vector of predictions
        pred_coefs = np.zeros(Tsize)
        if inp.average:
            Av_coeffs = np.zeros(Tsize)
        i = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                    if inp.average and l==0:
                        Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1
            ispe[spe] += 1

        # add back spherical averages if required
        if inp.average:
            pred_coefs += Av_coeffs

        if inp.qmcode=="cp2k":

            from ase.io import read
            xyzfile = read(inp.filename, ":")
            geom = xyzfile[iconf]
            geom.wrap()
            coords = geom.get_positions()/bohr2angs
            all_symbols = xyzfile[iconf].get_chemical_symbols()
            all_natoms = int(len(all_symbols))

            # compute integral of predicted density
            iaux = 0
            nele = 0.0
            rho_int = 0.0
            ref_rho_int = 0.0
            for iat in range(natoms[iconf]):
                spe = atomic_symbols[iconf][iat]
                nele += inp.pseudocharge
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        if l==0:
                            rho_int += charge_integrals[(spe,l,n)] * pred_coefs[iaux]
                            ref_rho_int += charge_integrals[(spe,l,n)] * ref_coefs[iaux]
                        iaux += 2*l+1

            # compute charge and dipole
            iaux = 0
            charge = 0.0
            dipole = 0.0
            ref_dipole = 0.0
            ref_charge = 0.0
            for iat in range(all_natoms):
                spe = all_symbols[iat]
                if spe in species:
                    if inp.average:
                        dipole += inp.pseudocharge * coords[iat,2]
                        ref_dipole += inp.pseudocharge * coords[iat,2]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            for im in range(2*l+1):
                                if l==0 and im==0:
                                    # rescale spherical coefficients to conserve the electronic charge
                                    if inp.average:
                                        pred_coefs[iaux] *= nele/rho_int
                                    else:
                                       if n==nmax[(spe,l)]-1:
                                           pred_coefs[iaux] -= rho_int/(charge_integrals[(spe,l,n)]*natoms[iconf])
                                    charge += pred_coefs[iaux] * charge_integrals[(spe,l,n)]
                                    dipole -= pred_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                                    ref_charge += ref_coefs[iaux] * charge_integrals[(spe,l,n)]
                                    ref_dipole -= ref_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                                if l==1 and im==1:
                                    dipole -= pred_coefs[iaux] * dipole_integrals[(spe,l,n)]
                                    ref_dipole -= ref_coefs[iaux] * dipole_integrals[(spe,l,n)]
                                iaux += 1
            print(iconf+1,ref_dipole,dipole,file=dfile)
            print(iconf+1,ref_charge,rho_int,file=qfile)

        # save predicted coefficients
        np.save(osp.join(
            inp.saltedpath, vdir, f"M{M}_zeta{zeta}",
            f"N{ntrain}_reg{reg_log10_intstr}", f"prediction_conf{iconf}.npy"
        ), pred_coefs)

        # save reference coefficients
        np.savetxt(osp.join(
            inp.saltedpath, vdir, f"M{M}_zeta{zeta}",
            f"N{ntrain}_reg{reg_log10_intstr}", f"RI-COEFFS-{iconf+1}.dat"
        ), ref_coefs)
        # save predicted coefficients
        np.savetxt(osp.join(
            inp.saltedpath, vdir, f"M{M}_zeta{zeta}",
            f"N{ntrain}_reg{reg_log10_intstr}", f"COEFFS-{iconf+1}.dat"
        ), pred_coefs)

        # compute predicted density projections <phi|rho>
        pred_projs = np.dot(overl,pred_coefs)

        # compute error
        error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
        error_density += error
        if inp.average:
            ref_projs -= np.dot(overl,Av_coeffs)
            ref_coefs -= Av_coeffs
        var = np.dot(ref_coefs,ref_projs)
        variance += var
        print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=efile)
        print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)
        #print(iconf+1, ":", "rho integral =", rho_int, "normalized rho integral =", charge, "ref_dipole =", ref_dipole, "dipole =",dipole, ", error =", np.sqrt(error/var)*100, "% RMSE", flush=True)

        # UNCOMMENT TO CHECK PREDICTIONS OF <phi|rho-rho_0>
        # -------------------------------------------------
    #    pred_projs = np.dot(overl,pred_coefs-Av_coeffs)
    #    av_projs = np.dot(overl,Av_coeffs)
    #    iaux = 0
    #    for iat in range(natoms[iconf]):
    #        spe = atomic_symbols[iconf][iat]
    #        for l in range(lmax[spe]+1):
    #            for n in range(nmax[(spe,l)]):
    #                for im in range(2*l+1):
    #                    if l==4 and im==0:
    #                        print(pred_projs[iaux],ref_projs[iaux])
    #                    iaux += 1

    efile.close()
    if inp.qmcode == "cp2k":
        dfile.close()
        qfile.close()

    if inp.parallel:
        error_density = comm.allreduce(error_density)
        variance = comm.allreduce(variance)
        if rank == 0:
            errs = np.loadtxt(efname)
            np.savetxt(efname, errs[errs[:,0].argsort()])
            if inp.qmcode == "cp2k":
                dips = np.loadtxt(dfname)
                np.savetxt(dfname, dips[dips[:,0].argsort()])
                qs = np.loadtxt(qfname)
                np.savetxt(qfname, qs[qs[:,0].argsort()])
    if rank == 0:
        print(f"\nsummary: {(100*np.sqrt(error_density/variance)):.3e} % RMSE", flush=True)


if __name__ == "__main__":
    build()
