from ase.io import read
import numpy as np
import spherical 
import quaternionic

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

f = read("water_monomers_1k.xyz",":")

itest = 1 

# shift coordinates about position center 
coords_1 = f[0].get_positions()
coords_2 = f[itest].get_positions()
center_1 = np.mean(coords_1,axis=0)
center_2 = np.mean(coords_2,axis=0)
coords_1 -= center_1
coords_2 -= center_2

# compute OH distance vectors
d1_OH1 = coords_1[1]-coords_1[0]
d1_OH2 = coords_1[2]-coords_1[0]
d2_OH1 = coords_2[1]-coords_2[0]
d2_OH2 = coords_2[2]-coords_2[0]

# Define dipole unit vectors
v1 = (d1_OH1+d1_OH2)/np.linalg.norm(d1_OH1+d1_OH2)
v2 = (d2_OH1+d2_OH2)/np.linalg.norm(d2_OH1+d2_OH2)

# Compute alignement matrix of v2 onto v1 
v = np.cross(v2,v1)
c = np.dot(v2,v1)

vmat = np.zeros((3,3))
vmat[0,1] = -v[2]
vmat[0,2] = v[1]
vmat[1,0] = v[2]
vmat[1,2] = -v[0]
vmat[2,0] = -v[1]
vmat[2,1] = v[0]

vmat2 = np.dot(vmat,vmat)

rmat1 = np.eye(3) + vmat + vmat2 / (1+c)

rot_coords_2 = np.zeros((3,3))

# rotate coordinates
for i in range(3):
    rot_coords_2[i] = np.dot(rmat1,coords_2[i])

# compute HH distance vectors of rotated molecules
d1_HH = coords_1[2]-coords_1[1] 
d2_HH = rot_coords_2[2]-rot_coords_2[1] 

# define unit vectors
v1 = d1_HH/np.linalg.norm(d1_HH)
v2 = d2_HH/np.linalg.norm(d2_HH)

# Compute alignement matrix of v2 onto v1 
v = np.cross(v2,v1)
c = np.dot(v2,v1)

vmat = np.zeros((3,3))
vmat[0,1] = -v[2]
vmat[0,2] = v[1]
vmat[1,0] = v[2]
vmat[1,2] = -v[0]
vmat[2,0] = -v[1]
vmat[2,1] = v[0]

vmat2 = np.dot(vmat,vmat)

rmat2 = np.eye(3) + vmat + vmat2 / (1+c)

# rotate coordinates
for i in range(3):
    coords_2[i] = np.dot(rmat2,rot_coords_2[i])

# check alignement
#print(coords_2-coords_1)

# compute global rotation matrix
rmat = np.dot(rmat2,rmat1)

# compute quaternionic representation of rotation
R = quaternionic.array.from_rotation_matrix(rmat)

# compute Wigner-D matrices up to lmax
lmax = 1
wigner = spherical.Wigner(lmax)
wigner_D = wigner.D(R)

# select Wigner-D matrix for the give L
L = 1
msize = 2*L+1
D = wigner_D[1:].reshape(msize,msize)

# compute complex to real transformation
c2r = complex_to_real_transformation([msize])[0]

# make Wigner-D matrix real
D_real = np.real(np.dot(c2r,np.dot(D,np.conj(c2r.T))))

cvec_1 = np.load("coefficients/coefficients_conf0.npy")[10:13]
cvec_2 = np.load("coefficients/coefficients_conf"+str(itest)+".npy")[10:13]
print(cvec_1)
print(np.dot(D_real,cvec_2))

#path2qm = "/gpfsstore/rech/kln/ulo49cx/water_monomer/"
#
## L=0 covariance test
#
#cvec_1 = np.zeros(3)
#cvec_1[0] = np.load(path2qm+"coefficients/coefficients-x_conf0.npy")[0]
#cvec_1[1] = np.load(path2qm+"coefficients/coefficients-y_conf0.npy")[0]
#cvec_1[2] = np.load(path2qm+"coefficients/coefficients-z_conf0.npy")[0]
#
#cvec_2 = np.zeros(3)
#cvec_2[0] = np.load(path2qm+"coefficients/coefficients-x_conf1.npy")[0]
#cvec_2[1] = np.load(path2qm+"coefficients/coefficients-y_conf1.npy")[0]
#cvec_2[2] = np.load(path2qm+"coefficients/coefficients-z_conf1.npy")[0]
#
#print(cvec_1)
#print(np.dot(rmat,cvec_2))
#
## L=1 covariance test
#
#cvec_1 = np.zeros((3,3))
#cvec_1[0,:] = np.load(path2qm+"coefficients/coefficients-x_conf0.npy")[9:12]
#cvec_1[1,:] = np.load(path2qm+"coefficients/coefficients-y_conf0.npy")[9:12]
#cvec_1[2,:] = np.load(path2qm+"coefficients/coefficients-z_conf0.npy")[9:12]
#
#cvec_2 = np.zeros((3,3))
#cvec_2[0,:] = np.load(path2qm+"coefficients/coefficients-x_conf1.npy")[9:12]
#cvec_2[1,:] = np.load(path2qm+"coefficients/coefficients-y_conf1.npy")[9:12]
#cvec_2[2,:] = np.load(path2qm+"coefficients/coefficients-z_conf1.npy")[9:12]
#
#print(cvec_1)
#print(np.dot(rmat,np.dot(cvec_2,D_real.T)))
