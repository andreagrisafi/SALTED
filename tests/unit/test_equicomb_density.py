"""equicombsparse_density_numba (norm from density matrices, only the kept features)
must reproduce equicombsparse_full_numba (all features, then select)."""

import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j

from salted import sph_utils


def w3j_of(lam, llmax, llvec):
    """as salted/wigner.py writes them"""
    out = []
    for il in range(llmax):
        l1, l2 = int(llvec[il, 0]), int(llvec[il, 1])
        for imu in range(2 * lam + 1):
            mu = imu - lam
            for im1 in range(2 * l1 + 1):
                m1 = im1 - l1
                m2 = m1 - mu
                if abs(m2) <= l2:
                    out.append(float(wigner_3j(lam, l2, l1, mu, m2, -m1) * (-1.0) ** m1))
    return np.array(out)


def coeffs(rng, natoms, nrad, nang):
    """complex SPH coefficients of a real density, laid out as prediction.py passes them"""
    v = np.zeros((natoms, nrad, nang + 1, 2 * nang + 1), complex)
    for l in range(nang + 1):
        c2r = sph_utils.complex_to_real_transformation([2 * l + 1])[0]
        v[:, :, l, :2 * l + 1] = rng.standard_normal((natoms, nrad, 2 * l + 1)) @ np.conj(c2r)
    return v


@pytest.mark.parametrize("lam", [0, 1, 2, 3])
@pytest.mark.parametrize("same", [True, False])
def test_density_matches_full(lam, same):
    rng = np.random.default_rng(lam)
    natoms, nrad, nang = 3, 4, 3
    llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang, nang)
    w3j = w3j_of(lam, llmax, llvec)
    c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]
    featsize = nrad * nrad * llmax
    vfps = rng.permutation(featsize)[: featsize // 3]
    v1 = coeffs(rng, natoms, nrad, nang)
    v2 = v1.copy() if same else coeffs(rng, natoms, nrad, nang)
    args = (natoms, nang, nang, nrad, nrad, v1, v2, w3j, llmax, llvec, lam, c2r, featsize, len(vfps), vfps)
    ref = sph_utils.equicombsparse_full_numba(*args)
    new = sph_utils.equicombsparse_density_numba(*args)
    assert np.abs(ref).max() > 0
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-12 * np.abs(ref).max())
    np.testing.assert_allclose(sph_utils.equicombsparse_numba(*args), ref, rtol=0, atol=1e-12 * np.abs(ref).max())
