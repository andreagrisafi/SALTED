## Included Modules

The container includes the following components:

- Python 3.10 (Bookworm)
- Featomic
- OpenMPI 5.0.2
- HDF5 + h5py
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
```bash
  podman build --format docker -t salted:latest .
```

2. Create a tarball of the container image:
```bash
  podman save -o salted.tar salted:latest
```

3. Build the Apptainer image from the tarball[^1]:
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
