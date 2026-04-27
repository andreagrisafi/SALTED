# syntax=docker/dockerfile:1
FROM python:3.10-bookworm

SHELL ["/bin/bash", "-c"] 
WORKDIR /src/temp

# Build prerequisites, including PMI2 headers/libs
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    python3-dev \
    python3-pip \
    gfortran \
    ninja-build \
    libpmi2-0 \
    libpmi2-0-dev \
    libpmix-dev \
    libevent-dev \
    libhwloc-dev \
    slurm-wlm \
 && rm -rf /var/lib/apt/lists/*

#Install Featomic
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installRust.sh \
    && chmod 700 installRust.sh \
    && ./installRust.sh -y \
    && source $HOME/.cargo/env \
    && pip install git+https://github.com/metatensor/featomic.git

ENV PATH=/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/lib/x86_64-linux-gnu
ENV HDF5_DIR=/usr/local/hdf5

#Create symlinks for PMI2 libs, as Open MPI's configure script expects them in /usr/lib
RUN ln -sf /usr/lib/x86_64-linux-gnu/libpmi2.so /usr/lib/libpmi2.so \
 && ln -sf /usr/lib/x86_64-linux-gnu/libpmi2.a  /usr/lib/libpmi2.a

# Open MPI with PMI2 support
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.bz2 \
    && tar -xf openmpi-4.1.8.tar.bz2 \
    && cd openmpi-4.1.8 \
    && ./configure --prefix=/usr/local --with-slurm --with-pmix --with-libevent=external --with-hwloc=external\
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf openmpi-4.1.8 openmpi-4.1.8.tar.bz2

# Build mpi4py from source against this mpicc
RUN MPICC=/usr/local/bin/mpicc pip install --no-binary=mpi4py mpi4py

# HDF5 parallel
RUN wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_14_3/src/hdf5-1.14.3.tar.gz \
    && tar -xf hdf5-1.14.3.tar.gz \
    && cd hdf5-1.14.3 \
    && CC=/usr/local/bin/mpicc ./configure --enable-shared --enable-parallel --prefix=/usr/local/hdf5 \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf hdf5-1.14.3 hdf5-1.14.3.tar.gz

# h5py against parallel HDF5 + same MPI
RUN HDF5_DIR=/usr/local/hdf5 CC=/usr/local/bin/mpicc HDF5_MPI=ON \
    pip install --no-binary=h5py h5py

# Install Python dependencies
RUN pip install meson packaging numba ase scipy pyyaml \
    && pip install --prefer-binary pyscf

#Install SALTED
COPY . /src/temp/SALTED-master
RUN cd /src/temp/SALTED-master \
    && make \
    && pip install . --no-deps

RUN rm -R /src/temp

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
