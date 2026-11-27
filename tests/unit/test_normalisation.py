"""descriptor.normalisation: unit mode divides each lambda block by its own
length; rel_power divides lambda > 0 by |q0|, the length of the same atom's
raw lambda = 0 vector, so lambda > 0 features that vanish by site symmetry stay
small near a symmetric site."""

import functools

import numpy as np
import pytest
from ase import Atoms

from salted import sph_utils
from salted.sys_utils import build_featomic_hyper_params
from test_equicomb_density import coeffs, w3j_of

EMPTY = np.empty(0)
EMPTY3 = np.empty((0, 0, 0))


@functools.lru_cache(maxsize=None)
def tables(lam, nang1, nang2):
    llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)
    return llmax, llvec, w3j_of(lam, llmax, llvec), sph_utils.complex_to_real_transformation([2 * lam + 1])[0]


def dcoeffs(rng, natoms, nrad, nang, ngrad):
    """position derivatives of real-density coefficients, laid out as salted_prediction.py passes them"""
    dv = np.zeros((natoms, nrad, nang + 1, 2 * nang + 1, ngrad, 3), complex)
    for l in range(nang + 1):
        c2r = sph_utils.complex_to_real_transformation([2 * l + 1])[0]
        r = rng.standard_normal((natoms, nrad, ngrad, 3, 2 * l + 1)) @ np.conj(c2r)
        dv[:, :, l, :2 * l + 1] = np.moveaxis(r, 4, 2)
    return dv


def random_inputs(seed, same, natoms=3):
    """v1, v2, dv1, dv2 and (nang2, nrad2); rep1 has nang 3, nrad 4"""
    rng = np.random.default_rng(seed)
    nang2, nrad2 = (3, 4) if same else (2, 5)
    v1 = coeffs(rng, natoms, 4, 3)
    dv1 = dcoeffs(rng, natoms, 4, 3, natoms)
    if same:
        v2, dv2 = v1.copy(), dv1.copy()
    else:
        v2, dv2 = coeffs(rng, natoms, nrad2, nang2), dcoeffs(rng, natoms, nrad2, nang2, natoms)
    return rng, v1, v2, dv1, dv2, nang2, nrad2


def norms(v1, v2, dv1, dv2, nang2, nrad2):
    """lam0_norm and grad_lam0_norm of the random inputs"""
    natoms = v1.shape[0]
    llmax0, llvec0, w3j0, _ = tables(0, 3, nang2)
    norm0 = sph_utils.lam0_norm(natoms, 3, nang2, 4, nrad2, v1, v2, w3j0, llmax0, llvec0)
    gnorm0, dot0 = sph_utils.grad_lam0_norm(natoms, natoms, 4, nrad2, v1, v2, dv1, dv2, w3j0, llmax0, llvec0)
    return norm0, gnorm0, dot0


@pytest.mark.parametrize("lam", [0, 1, 2, 3])
@pytest.mark.parametrize("same", [True, False])
def test_rel_power_is_raw_over_lam0_length(lam, same):
    rng, v1, v2, dv1, dv2, nang2, nrad2 = random_inputs(20 + lam, same)
    natoms = v1.shape[0]
    llmax0, llvec0, w3j0, c2r0 = tables(0, 3, nang2)
    q0 = sph_utils.equicombnonorm(natoms, 3, nang2, 4, nrad2, v1, v2, w3j0, llmax0, llvec0, 0, c2r0, 4 * nrad2 * llmax0)
    ref0 = np.linalg.norm(q0.reshape(natoms, -1), axis=1)
    norm0, gnorm0, dot0 = norms(v1, v2, dv1, dv2, nang2, nrad2)
    np.testing.assert_allclose(norm0, ref0, rtol=1e-13)
    np.testing.assert_allclose(gnorm0, ref0, rtol=1e-13)

    llmax, llvec, w3j, c2r = tables(lam, 3, nang2)
    featsize = 4 * nrad2 * llmax
    vfps = rng.permutation(featsize)[: featsize // 3 + 1]
    args = (natoms, 3, nang2, 4, nrad2, v1, v2, w3j, llmax, llvec, lam, c2r, featsize)
    gargs = (natoms, natoms, 3, nang2, 4, nrad2, v1, v2, dv1, dv2, w3j, llmax, llvec, lam, c2r, featsize)
    dense = sph_utils.equicomb_numba(*args, norm0)
    sparse = sph_utils.equicombsparse_numba(*args, len(vfps), vfps, norm0)
    fps = sph_utils.equicombfps(*args, norm0)
    gdense = sph_utils.grad_equicomb_numba(*gargs, gnorm0, dot0)[0]
    gsparse = sph_utils.grad_equicombsparse_numba(*gargs, len(vfps), vfps, gnorm0, dot0)[0]
    if lam == 0:
        # lambda = 0 stays unit-normalised
        assert np.array_equal(dense, sph_utils.equicomb_numba(*args, EMPTY))
        assert np.array_equal(sparse, sph_utils.equicombsparse_numba(*args, len(vfps), vfps, EMPTY))
        assert np.array_equal(fps, sph_utils.equicombfps(*args, EMPTY))
        assert np.array_equal(gdense, sph_utils.grad_equicomb_numba(*gargs, EMPTY, EMPTY3)[0])
        assert np.array_equal(gsparse, sph_utils.grad_equicombsparse_numba(*gargs, len(vfps), vfps, EMPTY, EMPTY3)[0])
        return
    ref = sph_utils.equicombnonorm(*args) / ref0[:, None, None]
    atol = 1e-13 * np.abs(ref).max()
    np.testing.assert_allclose(dense, ref, rtol=0, atol=atol)
    np.testing.assert_allclose(sparse, ref[:, :, vfps], rtol=0, atol=atol)
    np.testing.assert_allclose(fps, ref.transpose(2, 0, 1).reshape(featsize, -1), rtol=0, atol=atol)
    np.testing.assert_allclose(gdense, ref, rtol=0, atol=atol)
    np.testing.assert_allclose(gsparse, ref[:, :, vfps], rtol=0, atol=atol)
    # unit mode differs: the block has unit length
    unit = sph_utils.equicomb_numba(*args, EMPTY)
    np.testing.assert_allclose(np.linalg.norm(unit.reshape(natoms, -1), axis=1), 1.0, rtol=1e-12)


@pytest.mark.parametrize("lam", [0, 2])
def test_atom_without_density(lam):
    """an atom with |q0| = 0 gets zero features and zero gradients"""
    rng, v1, v2, dv1, dv2, nang2, nrad2 = random_inputs(30, True)
    v1[1] = v2[1] = 0.0
    dv1[1] = dv2[1] = 0.0
    natoms = v1.shape[0]
    norm0, gnorm0, dot0 = norms(v1, v2, dv1, dv2, nang2, nrad2)
    assert norm0[1] == 0.0 and gnorm0[1] == 0.0 and np.all(norm0[[0, 2]] > 0)
    llmax, llvec, w3j, c2r = tables(lam, 3, nang2)
    featsize = 4 * nrad2 * llmax
    vfps = np.arange(featsize)
    args = (natoms, 3, nang2, 4, nrad2, v1, v2, w3j, llmax, llvec, lam, c2r, featsize)
    gargs = (natoms, natoms, 3, nang2, 4, nrad2, v1, v2, dv1, dv2, w3j, llmax, llvec, lam, c2r, featsize)
    outs = [sph_utils.equicomb_numba(*args, norm0),
            sph_utils.equicombsparse_numba(*args, featsize, vfps, norm0),
            sph_utils.equicombfps(*args, norm0).T.reshape(natoms, 2 * lam + 1, featsize)]
    for p, grad_p in (sph_utils.grad_equicomb_numba(*gargs, gnorm0, dot0),
                      sph_utils.grad_equicombsparse_numba(*gargs, featsize, vfps, gnorm0, dot0)):
        outs += [p, np.moveaxis(grad_p, 2, 0)]
    for out in outs:
        assert np.all(out[1] == 0.0)
        assert np.all(np.isfinite(out)) and np.abs(out[0]).max() > 0


# CH4 at the origin, Td; the C-H bonds along (+-1, +-1, +-1)
A = 1.09 / np.sqrt(3)
NEIGH = ["C", "H"]
NRAD, NANG = 3, 3


class REP:
    type, rcut, sig, nrad, nang = "rho", 4.0, 0.3, NRAD, NANG


def ch4(d=0.0):
    pos = np.array([[0, 0, 0], [A, A, A], [-A, -A, A], [-A, A, -A], [A, -A, -A]])
    pos[1] += d * np.array([0.3, 0.5, 0.8]) / np.sqrt(0.98)
    return Atoms("CH4", positions=pos)


def v_of(frame):
    omega = sph_utils.get_representation_coeffs(frame, "rho", build_featomic_hyper_params(REP), 0,
                                                NEIGH, NEIGH, NANG, NRAD, len(frame))
    return np.transpose(omega, (1, 3, 0, 2)).copy()


def rel_power(frame, lam, sparse=False):
    """rel_power and unit lam features of every atom of frame"""
    v = v_of(frame)
    nr = len(NEIGH) * NRAD
    llmax0, llvec0, w3j0, _ = tables(0, NANG, NANG)
    norm0 = sph_utils.lam0_norm(len(frame), NANG, NANG, nr, nr, v, v.copy(), w3j0, llmax0, llvec0)
    llmax, llvec, w3j, c2r = tables(lam, NANG, NANG)
    featsize = nr * nr * llmax
    args = (len(frame), NANG, NANG, nr, nr, v, v.copy(), w3j, llmax, llvec, lam, c2r, featsize)
    if sparse:
        vfps = np.arange(featsize)
        return (sph_utils.equicombsparse_numba(*args, featsize, vfps, norm0),
                sph_utils.equicombsparse_numba(*args, featsize, vfps, EMPTY))
    return sph_utils.equicomb_numba(*args, norm0), sph_utils.equicomb_numba(*args, EMPTY)


@pytest.mark.parametrize("lam", [1, 2])
def test_rel_power_gradient(lam):
    """grad kernels in rel_power against central differences of rel_power features"""
    frame = ch4()
    frame.rattle(0.05, seed=3)
    natoms = len(frame)
    nr = len(NEIGH) * NRAD
    omega, domega = sph_utils.get_representation_gradient_coeffs(
        frame, "rho", build_featomic_hyper_params(REP), 0, NEIGH, NEIGH, NANG, NRAD, natoms)
    v = np.transpose(omega, (1, 3, 0, 2)).copy()
    dv = np.transpose(domega.reshape(NANG + 1, natoms, natoms, 3, 2 * NANG + 1, nr), (1, 5, 0, 4, 2, 3)).copy()
    llmax0, llvec0, w3j0, _ = tables(0, NANG, NANG)
    norm0, dot0 = sph_utils.grad_lam0_norm(natoms, natoms, nr, nr, v, v.copy(), dv, dv.copy(), w3j0, llmax0, llvec0)
    llmax, llvec, w3j, c2r = tables(lam, NANG, NANG)
    featsize = nr * nr * llmax
    vfps = np.arange(featsize)[::3]
    gargs = (natoms, natoms, NANG, NANG, nr, nr, v, v.copy(), dv, dv.copy(), w3j, llmax, llvec, lam, c2r, featsize)
    p, grad_p = sph_utils.grad_equicomb_numba(*gargs, norm0, dot0)
    ps, grad_ps = sph_utils.grad_equicombsparse_numba(*gargs, len(vfps), vfps, norm0, dot0)

    h = 1e-5
    fd = np.zeros_like(grad_p)
    for j in range(natoms):
        for x in range(3):
            f = []
            for s in (h, -h):
                moved = frame.copy()
                moved.positions[j, x] += s
                f.append(rel_power(moved, lam)[0])
            fd[j, x] = (f[0] - f[1]) / (2 * h)
    atol = 1e-6 * np.abs(fd).max()
    np.testing.assert_allclose(p, rel_power(frame, lam)[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(grad_p, fd, rtol=0, atol=atol)
    np.testing.assert_allclose(ps, p[:, :, vfps], rtol=0, atol=1e-14)
    np.testing.assert_allclose(grad_ps, fd[..., vfps], rtol=0, atol=atol)


@pytest.mark.parametrize("sparse", [False, True])
def test_ch4_displaced_hydrogen(sparse):
    """carbon lambda 1 and 2 vanish at Td: rel_power goes to zero with the
    displacement of one H, unit mode stays of order 1"""
    for lam in (1, 2):
        rel0, _ = rel_power(ch4(0.0), lam, sparse)
        rel1, unit1 = rel_power(ch4(1e-6), lam, sparse)
        rel2, unit2 = rel_power(ch4(2e-6), lam, sparse)
        assert np.abs(rel0[0]).max() < 1e-12
        assert 1e-9 < np.abs(rel1[0]).max() < 1e-4
        np.testing.assert_allclose(rel2[0], 2 * rel1[0], rtol=0, atol=1e-3 * np.abs(rel1[0]).max())
        assert np.abs(unit1[0]).max() > 1e-2
        np.testing.assert_allclose(unit2[0], unit1[0], rtol=0, atol=1e-3)
        # the hydrogens are not at symmetric sites: rel_power keeps them of order 1
        assert np.abs(rel1[1]).max() > 1e-3
