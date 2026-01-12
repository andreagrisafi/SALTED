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
  podman build --format docker -f Dockerfile -t salted .
  ```
2. Create a container instance from the image:
```bash
 podman create --name=salted --hostname=salted salted:latest
 ```
3. Export the container filesystem:
```bash
  mkdir salted
  docker export salted | tar -C salted -xf -
 ```
4. Generate a runtime configuration: _(If runc is unavailable, crun can be used instead.)_
```bash
  cd salted
  runc spec --rootless
  cd ..
 ```
5. Build the Apptainer image:
```bash
 apptainer build salted.sif salted
 ```

### Running the Container
To execute a command within the Apptainer container:
```bash
  apptainer exec salted.sif [COMMAND]
 ```
This setup allows seamless integration of SALTED across local development environments and HPC systems, maintaining consistent dependencies and runtime behavior.
