import sys,os
import numpy as np
import scipy.special as sc

def setup_orthomatrix(nmax,rc):
    """Compute orthogonalization matrix"""

    sigma = np.zeros(nmax,float)
    for i in range(nmax):
        sigma[i] = max(np.sqrt(float(i)),1.0)*(rc)/float(nmax)

    overlap = np.zeros((nmax,nmax),float)
    for n1 in range(nmax):
        for n2 in range(nmax):
            overlap[n1,n2] = (0.5/(sigma[n1])**2 + 0.5/(sigma[n2])**2)**(-0.5*(3.0 +n1 +n2)) \
                             /(sigma[n1]**n1 * sigma[n2]**n2)*\
                              sc.gamma(0.5*(3.0 + n1 + n2))/ (  \
                    (sigma[n1]*sigma[n2])**1.5 * np.sqrt(sc.gamma(1.5+n1)*sc.gamma(1.5+n2)) )

    eigenvalues, unitary = np.linalg.eig(overlap)
    sqrteigen = np.sqrt(eigenvalues)
    diagoverlap = np.diag(sqrteigen)
    newoverlap = np.dot(np.conj(unitary),np.dot(diagoverlap,unitary.T))
    orthomatrix = np.linalg.inv(newoverlap)

    return [orthomatrix,sigma]


def radint_efield(nmax,sigma):
    """Compute external field contribution to local potential"""

    # compute radial integrals int_0^\infty dr r^3 R_n(r)
    radint = np.zeros(nmax)
    for n in range(nmax):
        inner = 0.5*sc.gamma(n+1.5)*(sigma[n]**2)**(n+1.5)
        radint[n] = 2**float(1.0+float(n)/2.0) * sigma[n]**(4+n) * sc.gamma(2.0+float(n)/2.0) / np.sqrt(inner)

    return radint

def get_efield_sph(nmax,rc):
    """Compute the SPH components (l=1, m=0) of a uniform and constant field aligned along Z"""

    [orthomatrix,sigma] = setup_orthomatrix(nmax,rc)

    radint = radint_efield(nmax,sigma)

    orthoradint = np.dot(orthomatrix,radint)

    efield_coef = np.zeros(nmax,complex)
    for n in range(nmax):
        efield_coef[n] = complex(np.sqrt(4.0*np.pi/3.0)*orthoradint[n],0.0)

    return efield_coef 
