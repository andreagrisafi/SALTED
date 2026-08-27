"""Unit tests for MPI-related helpers in salted.sys_utils (serial code paths)."""

import numpy as np

from salted.sys_utils import detect_mpi, distribute_jobs


def test_detect_mpi_serial():
    comm, size, rank, parallel = detect_mpi()
    # pytest itself is not launched via mpirun
    assert size == 1
    assert rank == 0
    assert parallel is False


def test_distribute_jobs_serial_list_passthrough():
    jobs = [3, 1, 4, 1, 5]
    assert distribute_jobs(None, jobs) == jobs


def test_distribute_jobs_serial_array_passthrough():
    jobs = np.arange(7)
    out = distribute_jobs(None, jobs)
    np.testing.assert_array_equal(out, jobs)


class FakeComm:
    """Minimal stand-in mimicking an MPI communicator for rank 0 of N."""

    def __init__(self, size):
        self._size = size

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return 0

    def scatter(self, chunks, root=0):
        # emulate what rank 0 receives; also validate the full partition
        assert len(chunks) == self._size
        flat = [x for chunk in chunks for x in chunk]
        assert flat == sorted(flat, key=flat.index)  # order preserved
        self.chunks = chunks
        return chunks[0]


def test_distribute_jobs_partitions_all_jobs_exactly_once():
    jobs = list(range(10))
    comm = FakeComm(3)
    mine = distribute_jobs(comm, jobs)
    all_jobs = [x for chunk in comm.chunks for x in chunk]
    assert sorted(all_jobs) == jobs  # nothing lost, nothing duplicated
    assert mine == comm.chunks[0]
    # balanced within 1
    sizes = [len(c) for c in comm.chunks]
    assert max(sizes) - min(sizes) <= 1


def test_distribute_jobs_preserves_type():
    comm = FakeComm(2)
    out = distribute_jobs(comm, np.arange(5))
    assert isinstance(out, np.ndarray)
    comm2 = FakeComm(2)
    out2 = distribute_jobs(comm2, [0, 1, 2, 3, 4])
    assert isinstance(out2, list)
