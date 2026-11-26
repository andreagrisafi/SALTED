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
    assert inp.qm.numbering_start == 1
    assert inp.qm.predict_coeffs_numbering_start == 1
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


@pytest.mark.parametrize("start", [0, 1])
def test_numbering_start(tmp_path, monkeypatch, start):
    data = {**MINIMAL_INP, "qm": {**MINIMAL_INP["qm"], "numbering_start": start}}
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    qm = ParseConfig().parse_input().qm
    assert qm.numbering_start == start
    assert qm.predict_coeffs_numbering_start == start


@pytest.mark.parametrize("start, override", [(0, 1), (1, 0)])
def test_predicted_coefficients_numbering_override(tmp_path, monkeypatch, start, override):
    qm = {**MINIMAL_INP["qm"], "numbering_start": start, "predict_coeffs_numbering_start": override}
    write_inp(tmp_path, {**MINIMAL_INP, "qm": qm})
    monkeypatch.chdir(tmp_path)
    parsed = ParseConfig().parse_input().qm
    assert parsed.numbering_start == start
    assert parsed.predict_coeffs_numbering_start == override


@pytest.mark.parametrize("key", ["numbering_start", "predict_coeffs_numbering_start"])
@pytest.mark.parametrize("start", [-1, 2, True, 1.0, "0"])
def test_invalid_numbering_start(tmp_path, monkeypatch, start, key):
    data = {**MINIMAL_INP, "qm": {**MINIMAL_INP["qm"], key: start}}
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=key):
        ParseConfig().parse_input()


@pytest.mark.parametrize("filename, response", [
    ("coefficients/coefficients_conf0.npy", False),
    ("overlaps/overlap_conf0.npy", False),
    ("coefficients/z/coefficients_conf0.npy", True),
    ("coefficients/y/coefficients_conf0.npy", True),
])
def test_reference_numbering_guard(tmp_path, monkeypatch, filename, response):
    from salted.sys_utils import check_reference_file_numbering

    data = {**MINIMAL_INP, "salted": {**MINIMAL_INP["salted"]}, "qm": {**MINIMAL_INP["qm"]}}
    if response:
        data["salted"]["saltedtype"] = "density-response"
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    first = tmp_path / filename
    first.parent.mkdir(parents=True)
    first.with_name(first.name.replace("conf0", "conf1")).touch()
    check_reference_file_numbering(inp, tmp_path)
    first.touch()
    with pytest.raises(ValueError, match="numbering_start: 0"):
        check_reference_file_numbering(inp, tmp_path)
    data["qm"]["numbering_start"] = 0
    write_inp(tmp_path, data)
    inp = ParseConfig().parse_input()
    for kind in ("coefficients", "overlaps"):
        components = ("x", "y", "z") if response and kind != "overlaps" else ("",)
        stem = "overlap" if kind == "overlaps" else kind
        for component in components:
            path = tmp_path / kind / component / f"{stem}_conf0.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    check_reference_file_numbering(inp, tmp_path)
    first.unlink()
    with pytest.raises(ValueError, match="numbering_start: 1"):
        check_reference_file_numbering(inp, tmp_path)


@pytest.mark.parametrize("response", [False, True])
@pytest.mark.parametrize("start", [0, 1])
def test_rkhs_numbering_guard(tmp_path, monkeypatch, response, start):
    from salted.sys_utils import check_reference_file_numbering

    data = {**MINIMAL_INP, "qm": {**MINIMAL_INP["qm"], "numbering_start": start}}
    if response:
        data["salted"] = {**data["salted"], "saltedtype": "density-response"}
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    directory = tmp_path / "rkhs-vectors_test" / "M100_zeta2.0"
    directory.mkdir(parents=True)
    suffixes = ("_x", "_y", "_z") if response else ("",)
    for suffix in suffixes:
        (directory / f"psi-nm_conf{start}{suffix}.npz").touch()
    check_reference_file_numbering(inp, tmp_path, check_names=("rkhs",))

    zero = directory / f"psi-nm_conf0{suffixes[-1]}.npz"
    if start == 0:
        zero.unlink()
    else:
        zero.touch()
    with pytest.raises(ValueError, match=zero.name):
        check_reference_file_numbering(inp, tmp_path, check_names=("rkhs",))
