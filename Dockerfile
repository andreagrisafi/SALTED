# syntax=docker/dockerfile:1
FROM python:3.10-bookworm

SHELL ["/bin/bash", "-c"] 
WORKDIR /src/temp

#Install Rascaline
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installRust.sh \
    && chmod 700 installRust.sh \
    && ./installRust.sh -y \
    && source $HOME/.cargo/env \
    && pip install git+https://github.com/metatensor/featomic.git
                                                                            
#Install OpenMP
RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.2.tar.bz2 \
    && tar -xf openmpi-5.0.2.tar.bz2 \
    && cd openmpi-5.0.2 \
    && ./configure --prefix=/usr/local/openmpi5/lib \
    && make all install \
    && cd .. \
	&& rm -R openmpi-5.0.2

ENV PATH=/usr/local/openmpi5/lib/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/openmpi5/lib
ENV HDF5_DIR=/usr/local/hdf5

#Install HDF5
RUN wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_14_3/src/hdf5-1.14.3.tar.gz \
    && tar -xf hdf5-1.14.3.tar.gz \
    && cd hdf5-1.14.3 \
    && HDF5_MPI="ON" CC=mpicc ./configure --enable-shared --enable-parallel --prefix=/usr/local/hdf5 \
    && HDF5_DIR=/usr/local/hdf5 make \
    && HDF5_DIR=/usr/local/hdf5 make install \
    && cd .. \
	&& rm -R hdf5-1.14.3

#Install h5py
RUN HDF5_DIR=/usr/local/hdf5 HDF5_MPI="ON" CC=mpicc pip install --no-cache-dir --no-binary=h5py h5py

RUN apt-get update && apt-get install -y \
    ninja-build \
    gfortran \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install meson cython \
    && pip install --prefer-binary pyscf

#Install SALTED
COPY . /src/temp/SALTED-master
RUN cd /src/temp/SALTED-master \
    && make \
    && pip install .

RUN rm -R /src/temp

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
