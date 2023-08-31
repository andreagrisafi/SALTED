import sys
import math
import numpy as np
from scipy import special

def cartesian_to_spherical_transformation(l):
        """Compute Cartesian to spherical transformation matrices sorting the spherical components as {-l,..,0,..,+l} 
        while sorting the Cartesian components as shown in the corresponding arrays."""

        if l==0:
            # 1 Cartesian triplet
            cart_idx = [[0,0,0]]
        elif l==1:
            # 3 Cartesian triplets
            cart_idx = [[1,0,0],[0,1,0],[0,0,1]]
        elif l==2:
            # 6 Cartesian triplets
            cart_idx = [[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]]
        elif l==3:
            # 10 Cartesian triplets
            cart_idx = [[3,0,0],[2,1,0],[2,0,1],[1,2,0],[1,1,1],[1,0,2],[0,3,0],[0,2,1],[0,1,2],[0,0,3]]
        elif l==4:
            # 15 Cartesian triplets
            cart_idx = [[4,0,0],[3,1,0],[3,0,1],[2,2,0],[2,1,1],[2,0,2],[1,3,0],[1,2,1],[1,1,2],[1,0,3],[0,4,0],[0,3,1],[0,2,2],[0,1,3],[0,0,4]]
        elif l==5:
            # 21 Cartesian triplets
            cart_idx = [[0,0,5],[2,0,3],[0,2,3],[4,0,1],[0,4,1],[2,2,1],[1,0,4],[0,1,4],[3,0,2],[0,3,2],[1,2,2],[2,1,2],[5,0,0],[0,5,0],[1,4,0],[4,1,0],[3,2,0],[2,3,0],[1,1,3],[3,1,1],[1,3,1]]
        elif l==6:
            # 28 Cartesian triplets
            cart_idx = [[6,0,0],[0,6,0],[0,0,6],[5,0,1],[5,1,0],[0,5,1],[1,5,0],[0,1,5],[1,0,5],[4,0,2],[4,2,0],[0,4,2],[2,4,0],[0,2,4],[2,0,4],[4,1,1],[1,4,1],[1,1,4],[3,1,2],[1,3,2],[1,2,3],[3,2,1],[2,3,1],[2,1,3],[3,3,0],[3,0,3],[0,3,3],[2,2,2]]
        else:
            print("ERROR: Cartesian to spherical transformation not available for l=",l)

        mat = np.zeros((2*l+1,int((l+2)*(l+1)/2)),complex)
        # this implementation follows Eq.15 of SCHLEGEL and FRISH, Inter. J. Quant. Chem., Vol. 54, 83-87 (1995)
        for m in range(l+1):
            itriplet = 0
            for triplet in cart_idx:
                lx = triplet[0]
                ly = triplet[1]
                lz = triplet[2]
                sfact = np.sqrt(math.factorial(l)*math.factorial(2*lx)*math.factorial(2*ly)*math.factorial(2*lz)*math.factorial(l-m)/(math.factorial(lx)*math.factorial(ly)*math.factorial(lz)*math.factorial(2*l)*math.factorial(l+m))) / (math.factorial(l)*2**l)
                j = (lx+ly-m)/2
                if j.is_integer()==True:
                    j = int(j)
                    if j>=0:
                       for ii in range(math.floor((l-m)/2)+1):
                           if j<=ii:
                               afact = special.binom(l,ii)*special.binom(ii,j)*math.factorial(2*l-2*ii)/math.factorial(l-m-2*ii)*(-1.0)**ii
                               for k in range(j+1):
                                   kk = lx-2*k
                                   if m>=kk and kk>=0:
                                      jj = (m-kk)/2
                                      bfact = special.binom(j,k)*special.binom(m,kk)*(-1.0)**(jj)
                                      mat[l+m,itriplet] += afact*bfact
                mat[l+m,itriplet] *= sfact
                mat[l-m,itriplet] = np.conj(mat[l+m,itriplet])
                if m%2!=0:
                    mat[l+m,itriplet] *= -1.0 # TODO convention to be understood
                itriplet += 1

        return[np.asarray(mat), cart_idx]

def complex_to_real_transformation(sizes):
    """Transformation matrix from complex to real spherical harmonics"""

    matrices = []
    for i in range(len(sizes)):
        lval = int((sizes[i]-1)/2)
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range(lval):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[lval][lval] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)

    return matrices

