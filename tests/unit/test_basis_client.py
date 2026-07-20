"""Unit tests for salted.basis_client.BasisClient."""

import pytest

from salted.basis_client import BasisClient, compare_species_basis_data

BASIS_NAME = "test-basis"
BASIS_DATA = {
    "H": {"lmax": 2, "nmax": [4, 3, 1]},
    "O": {"lmax": 4, "nmax": [8, 7, 6, 3, 1]},
}


@pytest.fixture()
def client(tmp_path):
    """A BasisClient backed by a temp file (does not touch the package data)."""
    return BasisClient(data_fpath=str(tmp_path / "basis_data.yaml"))


def test_write_read_roundtrip(client):
    client.write(BASIS_NAME, BASIS_DATA)
    assert client.read(BASIS_NAME) == BASIS_DATA


def test_write_same_data_twice_is_ok(client):
    client.write(BASIS_NAME, BASIS_DATA)
    client.write(BASIS_NAME, BASIS_DATA)  # identical data: union, no error
    assert client.read(BASIS_NAME) == BASIS_DATA


def test_write_conflicting_data_raises(client):
    client.write(BASIS_NAME, BASIS_DATA)
    conflicting = {"H": {"lmax": 1, "nmax": [2, 1]}}
    with pytest.raises(ValueError):
        client.write(BASIS_NAME, conflicting)


def test_write_conflicting_data_force_overwrite(client):
    client.write(BASIS_NAME, BASIS_DATA)
    new_h = {"H": {"lmax": 1, "nmax": [2, 1]}}
    client.write(BASIS_NAME, new_h, force_overwrite=True)
    data = client.read(BASIS_NAME)
    assert data["H"] == new_h["H"]
    assert data["O"] == BASIS_DATA["O"]  # untouched species survives


def test_read_missing_basis_raises(client):
    with pytest.raises(Exception):
        client.read("no-such-basis")


def test_read_as_old_format(client):
    client.write(BASIS_NAME, BASIS_DATA)
    lmax, nmax = client.read_as_old_format(BASIS_NAME)
    assert lmax == {"H": 2, "O": 4}
    assert nmax[("H", 0)] == 4
    assert nmax[("O", 4)] == 1


def test_data_fpath_points_to_external_file(tmp_path):
    """The public data_fpath arg lets BasisClient load basis from an external file."""
    import yaml

    external = tmp_path / "external_basis_data.yaml"
    external.write_text(yaml.safe_dump({BASIS_NAME: BASIS_DATA}))

    client = BasisClient(data_fpath=str(external))
    assert client.data_fpath == str(external)
    assert client.read(BASIS_NAME) == BASIS_DATA


def test_compare_species_basis_data():
    a = {"lmax": 2, "nmax": [4, 3, 1]}
    assert compare_species_basis_data(a, dict(a))
    assert not compare_species_basis_data(a, {"lmax": 2, "nmax": [4, 3, 2]})
    assert not compare_species_basis_data(a, {"lmax": 1, "nmax": [4, 3]})
