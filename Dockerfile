# syntax=docker/dockerfile:1
FROM python:3.10-bookworm

SHELL ["/bin/bash", "-c"] 
WORKDIR /src/temp

# Build prerequisites
RUN apt-get update && apt-get install -y \
    build-essential \
    wget curl\
    ca-certificates \
    python3-dev \
    python3-pip \
    gfortran \
    ninja-build  \
    libhwloc-dev libpmi2-0-dev \
    libmunge-dev libmunge2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel build

ENV PATH=/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/lib/x86_64-linux-gnu

#Install Featomic
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installRust.sh \
    && chmod 700 installRust.sh \
    && ./installRust.sh -y \
    && source $HOME/.cargo/env \
    && pip install git+https://github.com/metatensor/featomic.git

RUN wget https://github.com/openpmix/openpmix/releases/download/v6.1.0/pmix-6.1.0.tar.gz \
    && tar -xf pmix-6.1.0.tar.gz \
    && cd pmix-6.1.0 \
    && ./configure --prefix=/usr/local/pmix \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf pmix-6.1.0 pmix-6.1.0.tar.gz

RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.bz2 \
    && tar -xf openmpi-4.1.8.tar.bz2 \
    && cd openmpi-4.1.8 \
    && ./configure --prefix=/usr/local \
       --enable-shared --disable-static --disable-debug \
       --enable-builtin-atomics \
       --with-slurm \
       --with-pmix=/usr/local/pmix \
       --with-hwloc \
       --with-libevent \
       --with-zlib \
       --without-psm \
       --without-psm2 \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf openmpi-4.1.8 openmpi-4.1.8.tar.bz2

ENV PATH=/usr/local/bin:/usr/local/pmix/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/local/pmix/lib:/usr/lib/x86_64-linux-gnu

# Build mpi4py from source against this mpicc
RUN MPICC=/usr/local/bin/mpicc pip install --no-binary=mpi4py mpi4py

ENV HDF5_DIR=/usr/local/hdf5
# HDF5 parallel
RUN wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_14_3/src/hdf5-1.14.3.tar.gz \
    && tar -xf hdf5-1.14.3.tar.gz \
    && cd hdf5-1.14.3 \
    && HDF5_MPI="ON" CC=/usr/local/bin/mpicc ./configure --enable-shared --enable-parallel --prefix=/usr/local/hdf5 \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf hdf5-1.14.3 hdf5-1.14.3.tar.gz

# h5py against parallel HDF5 + same MPI
RUN pip install cython \
    && HDF5_DIR=/usr/local/hdf5 HDF5_MPI=ON CC=/usr/local/bin/mpicc \
    pip install --no-build-isolation --no-binary=h5py h5py

# Install Python dependencies
RUN pip install meson packaging numba ase scipy pyyaml sympy \
    && pip install --prefer-binary pyscf

#Install SALTED
COPY . /src/temp/SALTED-master
RUN cd /src/temp/SALTED-master \
    && pip install . \
    && cd .. \
    && rm -rf SALTED-master

RUN rm -R /src/temp

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
