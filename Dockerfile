# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG OPENMPI_VERSION=4.1.8
ARG HDF5_VERSION=1.14.3


# -----------------------------------------------------------------------------
# Build MPI and parallel HDF5
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS native-builder

ARG OPENMPI_VERSION
ARG HDF5_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        gfortran \
        libevent-dev \
        libhwloc-dev \
        libmunge-dev \
        libpmi2-0-dev \
        zlib1g-dev

WORKDIR /tmp/build

# Open MPI
#
# Use the PMIx version bundled with Open MPI instead of independently
# combining Open MPI 4.1 with PMIx 6.
RUN curl -fsSL \
        "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz" \
        | tar -xz \
    && cd "openmpi-${OPENMPI_VERSION}" \
    && ./configure \
        --prefix=/opt/mpi \
        --enable-shared \
        --disable-static \
        --disable-debug \
        --enable-builtin-atomics \
        --disable-mpi-fortran \
        --disable-oshmem \
        --with-slurm \
        --with-pmix=internal \
        --with-hwloc=/usr \
        --with-libevent=/usr \
        --with-zlib=/usr \
        --without-psm \
        --without-psm2 \
    && make -j"$(nproc)" \
    && make install-strip \
    && cd /tmp/build \
    && rm -rf "openmpi-${OPENMPI_VERSION}"

ENV PATH=/opt/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib

# Parallel HDF5
RUN curl -fsSL \
        "https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_14_3/src/hdf5-${HDF5_VERSION}.tar.gz" \
        | tar -xz \
    && cd "hdf5-${HDF5_VERSION}" \
    && CC=/opt/mpi/bin/mpicc ./configure \
        --prefix=/opt/hdf5 \
        --enable-shared \
        --disable-static \
        --enable-parallel \
    && make -j"$(nproc)" \
    && make install-strip \
    && cd /tmp/build \
    && rm -rf "hdf5-${HDF5_VERSION}"

# Create a runtime-only copy without headers, pkg-config files, static
# archives, documentation, or HDF5 developer tools.
RUN mkdir -p /opt/runtime \
    && cp -a /opt/mpi /opt/runtime/mpi \
    && cp -a /opt/hdf5 /opt/runtime/hdf5 \
    && rm -rf \
        /opt/runtime/mpi/include \
        /opt/runtime/mpi/share/man \
        /opt/runtime/mpi/share/doc \
        /opt/runtime/mpi/lib/pkgconfig \
        /opt/runtime/hdf5/include \
        /opt/runtime/hdf5/share \
        /opt/runtime/hdf5/bin \
        /opt/runtime/hdf5/lib/pkgconfig \
    && find /opt/runtime -type f \
        \( -name '*.a' -o -name '*.la' \) \
        -delete


# -----------------------------------------------------------------------------
# Build the Python environment
# -----------------------------------------------------------------------------
FROM native-builder AS python-builder

ENV HDF5_DIR=/opt/hdf5
ENV PATH=/opt/venv/bin:/opt/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib:/opt/hdf5/lib
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade \
        pip \
        setuptools \
        wheel \
        build \
    && python -m pip install \
        featomic \
        numpy \
        cython \
        pkgconfig \
    && MPICC=/opt/mpi/bin/mpicc \
        python -m pip install \
            --no-binary=mpi4py \
            mpi4py \
    && HDF5_DIR=/opt/hdf5 \
       HDF5_MPI=ON \
       CC=/opt/mpi/bin/mpicc \
        python -m pip install \
            --no-build-isolation \
            --no-binary=h5py \
            h5py \
    && python -m pip install \
        meson \
        packaging \
        numba \
        ase \
        scipy \
        pyyaml \
        sympy \
    && python -m pip install \
        --prefer-binary \
        pyscf

# Copy SALTED last so that source-code changes do not invalidate the expensive
# MPI, HDF5, and Python dependency layers.
WORKDIR /src/SALTED
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install . \
    && python -m pip uninstall -y \
        build \
        cython \
        meson \
        pkgconfig \
        wheel \
    && find /opt/venv -type d -name '__pycache__' \
        -prune -exec rm -rf '{}' +


# -----------------------------------------------------------------------------
# Minimal runtime
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libevent-2.1-7 \
        libevent-pthreads-2.1-7 \
        libgfortran5 \
        libhwloc15 \
        libmunge2 \
        libpmi2-0 \
        openssh-client \
        zlib1g

COPY --from=native-builder /opt/runtime/mpi /opt/mpi
COPY --from=native-builder /opt/runtime/hdf5 /opt/hdf5
COPY --from=python-builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:/opt/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib:/opt/hdf5/lib
ENV HDF5_DIR=/opt/hdf5
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /work

CMD ["/bin/bash"]
