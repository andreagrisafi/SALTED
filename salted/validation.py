import os
import sys
import time
import os.path as osp

import numpy as np
from scipy import special
from scipy import sparse

#from sympy.parsing import mathematica
#from sympy import symbols
#from sympy import lambdify

from salted import basis
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range

def build():

    inp = ParseConfig().parse_input()
    (saltedname, saltedpath, saltedtype,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    if parallel:
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

    vdir = f"validations_{saltedname}"
    rdir = f"regrdir_{saltedname}"
    fdir = f"rkhs-vectors_{saltedname}"

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
        saltedpath, rdir, f"training_set_N{Ntrain}.txt"
    ), int)
    ntrain = round(trainfrac*len(trainrangetot))
    testrange = np.setdiff1d(list(range(ndata)),trainrangetot)

    # Distribute structures to tasks
    ntest = len(testrange)
    if parallel:
        testrange = get_conf_range(rank,size,ntest,testrange)
        testrange = comm.scatter(testrange,root=0)
        print(f"Task {rank+1} handles the following structures: {testrange}", flush=True)

    reg_log10_intstr = str(int(np.log10(regul)))

    # load regression weights
    weights = np.load(osp.join(
        saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
    ))

    dirpath = os.path.join(saltedpath, vdir, f"M{Menv}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}")
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
    if size > 1: comm.Barrier()


    if qmcode=="cp2k" and saltedtype=="density":
        bdir = osp.join(saltedpath,"basis")
        # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
        if rank == 0: print("Reading auxiliary basis info...")
        alphas = {}
        sigmas = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                avals = np.loadtxt(osp.join(bdir,f"{spe}-{dfbasis}-alphas-L{l}.dat"))
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
    #if inp.qm.totcharge:
    #    pseudof = open(inp.qm.pseudochargefile,"r")
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
    if average:
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    # compute error over test set

    efname = osp.join(
        saltedpath, vdir, f"M{Menv}_zeta{zeta}",
        f"N{ntrain}_reg{reg_log10_intstr}", f"errors.dat"
    )
    if rank == 0 and os.path.exists(efname):
        os.remove(efname)
    if qmcode=="cp2k" and saltedtype=="density":
        dfname = osp.join(
            saltedpath, vdir, f"M{Menv}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}", f"dipoles.dat"
        )
        qfname = osp.join(
            saltedpath, vdir, f"M{Menv}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}", f"charges.dat"
        )
        if rank == 0 and os.path.exists(dfname): os.remove(dfname)
        if rank == 0 and os.path.exists(qfname): os.remove(qfname)
    if parallel: comm.Barrier()
    efile = open(efname,"a")
    if qmcode=="cp2k" and saltedtype=="density":
        dfile = open(dfname,"a")
        qfile = open(qfname,"a")

    error_density = 0
    variance = 0
    for iconf in testrange:

        if saltedtype=="density":

            # load reference
            ref_coefs = np.load(osp.join(
                saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"
            ))
            overl = np.load(osp.join(
                saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
            ))
            ref_projs = np.dot(overl,ref_coefs)
            Tsize = len(ref_coefs)

            psivec = sparse.load_npz(osp.join(
                saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
            ))
            psi = psivec.toarray()

            pred_coefs = np.dot(psi,weights)

            # Compute vector of isotropic average coefficients
            if average:
                Av_coeffs = np.zeros(Tsize)
                i = 0
                for iat in range(natoms[iconf]):
                    spe = atomic_symbols[iconf][iat]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if l==0:
                                Av_coeffs[i] = av_coefs[spe][n]
                            i += 2*l+1
                # add back spherical averages if required
                pred_coefs += Av_coeffs

        if saltedtype=="density-response":

            overl = np.load(osp.join(
                saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
            ))

            cart = ["x","y","z"]
            
            ref_coefs = {}
            ref_projs = {}
            pred_coefs = {}
            pred_projs = {}
            for icart in cart:

                # load reference
                ref_coefs[icart] = np.load(osp.join(
                    saltedpath, "coefficients", f"{icart}/coefficients_conf{iconf}.npy"
                ))
                ref_projs[icart] = np.dot(overl,ref_coefs[icart])
                Tsize = len(ref_coefs[icart])

                psivec = sparse.load_npz(osp.join(
                    saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz"
                ))
                psi = psivec.toarray()

                pred_coefs[icart] = np.dot(psi,weights)
                
                # compute predicted density projections <phi|rho>
                pred_projs[icart] = np.dot(overl,pred_coefs[icart])


        if qmcode=="cp2k" and saltedtype=="density":

            from ase.io import read
            xyzfile = read(filename, ":")
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
                nele += inp.qm.pseudocharge
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
                    if average:
                        dipole += inp.qm.pseudocharge * coords[iat,2]
                        ref_dipole += inp.qm.pseudocharge * coords[iat,2]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            for im in range(2*l+1):
                                if l==0 and im==0:
                                    # rescale spherical coefficients to conserve the electronic charge
                                    if average:
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

        ## save reference coefficients
        #np.savetxt(osp.join(
        #    saltedpath, vdir, f"M{Menv}_zeta{zeta}",
        #    f"N{ntrain}_reg{reg_log10_intstr}", f"RI-COEFFS-{iconf+1}.dat"
        #), ref_coefs)
        # save predicted coefficients

        if saltedtype=="density":        

            np.savetxt(osp.join(
                saltedpath, vdir, f"M{Menv}_zeta{zeta}",
                f"N{ntrain}_reg{reg_log10_intstr}", f"COEFFS-{iconf+1}.dat"
            ), pred_coefs)

            # compute predicted density projections <phi|rho>
            pred_projs = np.dot(overl,pred_coefs)

            # compute error
            error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
            error_density += error
            if average:
                ref_projs -= np.dot(overl,Av_coeffs)
                ref_coefs -= Av_coeffs
            var = np.dot(ref_coefs,ref_projs)
            variance += var
            print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=efile)
            print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)

        elif saltedtype=="density-response":

            error = 0
            var = 0
            for icart in cart:
                np.savetxt(osp.join(
                    saltedpath, vdir, f"M{Menv}_zeta{zeta}",
                    f"N{ntrain}_reg{reg_log10_intstr}", f"COEFFS-{icart}_{iconf+1}.dat"
                ), pred_coefs[icart])

                # compute error
                error += np.dot(pred_coefs[icart]-ref_coefs[icart],pred_projs[icart]-ref_projs[icart])
                var += np.dot(ref_coefs[icart],ref_projs[icart])

            error_density += error
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
    if qmcode == "cp2k" and saltedtype=="density":
        dfile.close()
        qfile.close()

    if parallel:
        error_density = comm.allreduce(error_density)
        variance = comm.allreduce(variance)
        if rank == 0:
            errs = np.loadtxt(efname)
            np.savetxt(efname, errs[errs[:,0].argsort()])
            if qmcode == "cp2k" and saltedtype=="density":
                dips = np.loadtxt(dfname)
                np.savetxt(dfname, dips[dips[:,0].argsort()])
                qs = np.loadtxt(qfname)
                np.savetxt(qfname, qs[qs[:,0].argsort()])
    if rank == 0:
        print(f"\n % RMSE: {(100*np.sqrt(error_density/variance)):.3e}", flush=True)


if __name__ == "__main__":
    build()
