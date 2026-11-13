"""equicombsparse_numba (norm from density matrices, only the kept features) must
reproduce equicomb_numba (all features) restricted to the kept features."""

import os
import subprocess
import sys

import numpy as np
import pytest
from ase.build import bulk
from sympy.physics.wigner import wigner_3j

from salted import sph_utils
from salted.sys_utils import build_featomic_hyper_params


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


def sparse_and_dense(v1, v2, nang1, nang2, lam, rng):
    natoms, nrad1, nrad2 = v1.shape[0], v1.shape[1], v2.shape[1]
    llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)
    w3j = w3j_of(lam, llmax, llvec)
    c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]
    featsize = nrad1 * nrad2 * llmax
    vfps = rng.permutation(featsize)[: max(1, featsize // 3)]
    args = (natoms, nang1, nang2, nrad1, nrad2, v1, v2, w3j, llmax, llvec, lam, c2r, featsize)
    return sph_utils.equicombsparse_numba(*args, len(vfps), vfps), sph_utils.equicomb_numba(*args)[:, :, vfps]


@pytest.mark.parametrize("lam", [0, 1, 2, 3])
@pytest.mark.parametrize("same", [True, False])
def test_sparse_matches_dense(lam, same):
    rng = np.random.default_rng(lam)
    v1 = coeffs(rng, 3, 4, 3)
    v2 = v1.copy() if same else coeffs(rng, 3, 5, 2)
    new, ref = sparse_and_dense(v1, v2, 3, 3 if same else 2, lam, rng)
    assert np.abs(ref).max() > 0
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def nacl_coeffs(frame, nrad, nang):
    class rep:
        type, rcut, sig = "rho", 4.0, 0.3
    rep.nrad, rep.nang = nrad, nang
    omega = sph_utils.get_representation_coeffs(frame, "rho", build_featomic_hyper_params(rep), 0,
                                                ["Cl", "Na"], ["Cl", "Na"], nang, nrad, len(frame))
    return np.transpose(omega, (1, 3, 0, 2)).copy()


@pytest.mark.parametrize("rep2", [(3, 4), (2, 5)])
def test_near_symmetric_site(rep2):
    """lam = 2 features of atoms 1e-4 A away from cubic sites nearly vanish by symmetry"""
    frame = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    frame.rattle(1e-4, seed=5)
    v1 = nacl_coeffs(frame, 3, 4)
    v2 = nacl_coeffs(frame, *rep2)
    new, ref = sparse_and_dense(v1, v2, 4, rep2[1], 2, np.random.default_rng(0))
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-9 * np.abs(ref).max())


def test_atom_without_features():
    """an atom whose coefficients are all zero gets zero features"""
    rng = np.random.default_rng(0)
    v1 = coeffs(rng, 2, 4, 3)
    v1[1] = 0.0
    new, ref = sparse_and_dense(v1, v1.copy(), 3, 3, 2, rng)
    np.testing.assert_allclose(new[0], ref[0], rtol=0, atol=1e-12 * np.abs(ref[0]).max())
    assert np.all(new[1] == 0.0)


def test_atom_with_nan_coefficient():
    """a NaN coefficient gives NaN features for that atom only"""
    rng = np.random.default_rng(1)
    v1 = coeffs(rng, 2, 4, 3)
    v1[1, 2, 3, 5] = np.nan
    new, ref = sparse_and_dense(v1, v1.copy(), 3, 3, 1, rng)
    np.testing.assert_allclose(new[0], ref[0], rtol=0, atol=1e-12 * np.abs(ref[0]).max())
    assert np.all(np.isnan(new[1]))


def test_import_without_writable_numba_cache(tmp_path):
    """sph_utils imports and computes when numba has nowhere to write its cache"""
    blocker = tmp_path / "file"
    blocker.write_text("")
    env = dict(os.environ, NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator",
               NUMBA_CACHE_DIR=str(blocker / "cache"))
    script = ("import warnings, numpy as np\n"
              "with warnings.catch_warnings(record=True) as w:\n"
              "    warnings.simplefilter('always')\n"
              "    from salted import sph_utils\n"
              "assert any('NUMBA_CACHE_DIR' in str(x.message) for x in w)\n"
              "llmax, llvec = sph_utils.get_angular_indexes_symmetric(0, 1, 1)\n"
              "v = np.zeros((1, 1, 2, 3), complex); v[0, 0, 0, 0] = 1.0\n"
              "c2r = sph_utils.complex_to_real_transformation([1])[0]\n"
              "p = sph_utils.equicombsparse_numba(1, 1, 1, 1, 1, v, v.copy(), np.array([1.0, -1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)]),\n"
              "                                   llmax, llvec, 0, c2r, llmax, 1, np.array([0]))\n"
              "assert abs(abs(p[0, 0, 0]) - 1.0) < 1e-12, p\n")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
