"""Unit tests for salted.sys_utils.ParseConfig."""

import pytest
import yaml

from salted.sys_utils import ParseConfig

MINIMAL_INP = {
    "salted": {"saltedname": "test", "saltedpath": "./"},
    "system": {"filename": "./geoms.xyz", "species": ["H", "O"]},
    "qm": {"path2qm": "./", "qmcode": "aims", "dfbasis": "FHI-aims-light"},
    "descriptor": {
        "rep1": {
            "type": "rho",
            "rcut": 4.0,
            "sig": 0.3,
            "nrad": 8,
            "nang": 6,
            "neighspe": ["H", "O"],
        },
        "rep2": {
            "type": "rho",
            "rcut": 4.0,
            "sig": 0.3,
            "nrad": 8,
            "nang": 6,
            "neighspe": ["H", "O"],
        },
    },
    "gpr": {"z": 2.0, "Menv": 100, "Ntrain": 40},
}


WATER_XYZ = """3
Properties=species:S:1:pos:R:3
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
"""


def write_inp(tmp_path, data):
    (tmp_path / "inp.yaml").write_text(yaml.safe_dump(data))
    # ParseConfig checks that system.filename exists
    (tmp_path / "geoms.xyz").write_text(WATER_XYZ)


def test_parse_minimal_input(tmp_path, monkeypatch):
    write_inp(tmp_path, MINIMAL_INP)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    assert inp.salted.saltedname == "test"
    assert inp.qm.qmcode == "aims"
    assert inp.gpr.Menv == 100


def test_defaults_are_applied(tmp_path, monkeypatch):
    write_inp(tmp_path, MINIMAL_INP)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    assert inp.gpr.regul == pytest.approx(1e-6)
    assert inp.gpr.trainfrac == pytest.approx(1.0)
    assert inp.gpr.trainsel in ("random", "sequential")
    assert inp.descriptor.sparsify.ncut == 0


def test_missing_required_field_raises(tmp_path, monkeypatch):
    broken = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    del broken["gpr"]["Menv"]
    write_inp(tmp_path, broken)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()


def test_invalid_value_raises(tmp_path, monkeypatch):
    broken = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    broken["gpr"]["Ntrain"] = -5
    write_inp(tmp_path, broken)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()


def test_missing_inp_yaml_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()


def test_dfbasis_file_defaults_to_none(tmp_path, monkeypatch):
    write_inp(tmp_path, MINIMAL_INP)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    # optional field: unset -> None
    assert inp.qm.dfbasis_file is None


def test_dfbasis_file_set_resolves_to_path(tmp_path, monkeypatch):
    ext = tmp_path / "basis_data.yaml"
    ext.write_text("FHI-aims-light: {}\n")
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    data["qm"]["dfbasis_file"] = str(ext)
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    assert inp.qm.dfbasis_file == str(ext)


def test_dfbasis_file_nonexistent_path_raises(tmp_path, monkeypatch):
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    data["qm"]["dfbasis_file"] = str(tmp_path / "does_not_exist.yaml")
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()
