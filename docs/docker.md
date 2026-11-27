## Included Modules

The container includes the following components:

- Python 3.10 (Bookworm)
- OpenMPI 5.0.2
- SALTED

The **SALTED** installation is imported directly from the local repository, enabling the use of custom modifications without requiring a remote package rebuild.

---

## Building on a Local Client

To build the container locally using **Docker**, run:

```bash
docker build -f Dockerfile -t salted .
```

Alternatively, to build using Podman, run:

```bash
podman build --format docker -f Dockerfile -t salted .
```

This will produce a local container image named `salted`.

## Building for HPC Cluster (Apptainer)
For use on clusters managed by Slurm, the image must be converted into an Apptainer (`.sif`) format.
Ensure **Apptainer** is available on all nodes.

### Workflow

1. Build the container image (using Docker or Podman):
2. 
```bash
  podman build --format docker -t salted:latest .
```

1. Create a tarball of the container image:
```bash
  podman save -o salted.tar salted:latest
```

1. Build the Apptainer image from the tarball[^1]:
```bash
  apptainer build salted.sif docker-archive://$(pwd)/salted.tar
```
[^1]: It is very important to use the `docker-archive` URI scheme to ensure proper handling of the image format. Do not simply pass the tarball path directly to `apptainer build`, as this will lead to errors.

### Running the Container

To execute a command within the Apptainer container:

```bash
  apptainer exec salted.sif [COMMAND]
```

To run a parallel job usig Slurm, use `srun` with the Apptainer image:

```bash
  srun --ntasks=4 --mpi=pmi2 apptainer exec salted.sif [COMMAND]
```

This setup allows seamless integration of SALTED across local development environments and HPC systems, maintaining consistent dependencies and runtime behavior.

### Numba cache

SALTED's numba kernels are compiled on their first call and stored in numba's cache, so later runs load them in a fraction of a second. Inside the image the SALTED installation is read-only, and numba then writes its cache to the user's home directory. If the home directory is not writable inside the container either, numba has nowhere to write: the sparse equicomb kernels then print a warning and are compiled in every run, and the modules that import `salted.cp2k.utils` or `salted.numba_sparse` (for example `salted_prediction` and `hessian_matrix`) fail at import. Point `NUMBA_CACHE_DIR` to a writable folder that persists between runs:

```bash
  apptainer exec --env NUMBA_CACHE_DIR=$HOME/.cache/numba salted.sif [COMMAND]
```
