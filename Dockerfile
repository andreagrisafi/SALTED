# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG OPENMPI_VERSION=4.1.8


# -----------------------------------------------------------------------------
# Build MPI
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS native-builder

ARG OPENMPI_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libevent-dev \
        libhwloc-dev \
        libmunge-dev \
        libpmi2-0-dev \
        zlib1g-dev

WORKDIR /tmp/build

# Open MPI
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

# Create a runtime-only copy without headers, pkg-config files, static
# archives or documentation.
RUN mkdir -p /opt/runtime \
    && cp -a /opt/mpi /opt/runtime/mpi \
    && rm -rf \
        /opt/runtime/mpi/include \
        /opt/runtime/mpi/share/man \
        /opt/runtime/mpi/share/doc \
        /opt/runtime/mpi/lib/pkgconfig \
    && find /opt/runtime -type f \
        \( -name '*.a' -o -name '*.la' \) \
        -delete


# -----------------------------------------------------------------------------
# Build the Python environment
# -----------------------------------------------------------------------------
ENV PATH=/opt/venv/bin:/opt/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        --prefer-binary \
        pyscf

WORKDIR /src/SALTED
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install . \
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
        libhwloc15 \
        libmunge2 \
        libpmi2-0 \
        openssh-client \
        zlib1g

COPY --from=native-builder /opt/runtime/mpi /opt/mpi
COPY --from=native-builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:/opt/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /work

CMD ["/bin/bash"]