"""Serial-vs-MPI equivalence of the training pipeline.

Bugs in the MPI paths corrupt results silently rather than crash -- job
partitioning and the hand-sliced allreduce in ``hessian_matrix`` have no serial
counterpart. So train the same ``ModelSpec`` twice, once serially and once
under mpirun, and require the results to agree.

``density-response`` earns its own run: ``hessian_matrix`` accumulates over the
three Cartesian components in a branch the density models never reach.

Prediction-side MPI equivalence lives in ``test_prediction.py``.
"""

import numpy as np
import pytest
from conftest import PipelineWorkspace
from models import MPI_MODELS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("model_spec", MPI_MODELS, indirect=True)
def test_regression_matrices_match(serial_run: PipelineWorkspace, mpi_run: PipelineWorkspace):
    for name in ("Avec", "Bmat"):
        a, b = serial_run.regression_array(name), mpi_run.regression_array(name)
        assert a.shape == b.shape
        # summation order differs across ranks -> allow tiny float noise only
        np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10, err_msg=name)


@pytest.mark.parametrize("model_spec", MPI_MODELS, indirect=True)
def test_weights_match(serial_run: PipelineWorkspace, mpi_run: PipelineWorkspace):
    """Relative L2 norm, not element-wise: ``Bmat`` can be ill-conditioned,
    so regression weights can drift even if the regression matrices are identical.
    Change ``ModelSpec.weights_rtol`` for each model if necessary.
    """
    a, b = serial_run.regression_array("weights"), mpi_run.regression_array("weights")
    assert a.shape == b.shape
    deviation = np.linalg.norm(a - b) / np.linalg.norm(a)
    assert deviation < serial_run.weights_rtol, (
        f"{serial_run.name}: serial and MPI weights differ by {deviation:.3e} "
        f"in relative L2 norm, above the {serial_run.weights_rtol:.0e} tolerance"
    )


@pytest.mark.parametrize("model_spec", MPI_MODELS, indirect=True)
def test_validation_rmse_matches(serial_run: PipelineWorkspace, mpi_run: PipelineWorkspace):
    assert serial_run.validation_rmse is not None and mpi_run.validation_rmse is not None
    assert mpi_run.validation_rmse == pytest.approx(serial_run.validation_rmse, rel=1e-6)
