# syntax=docker/dockerfile:1
FROM python:3.10-bookworm

SHELL ["/bin/bash", "-c"] 
WORKDIR /src/temp

#Install OpenMPI
RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.2.tar.bz2 \
    && tar -xf openmpi-5.0.2.tar.bz2 \
    && cd openmpi-5.0.2 \
    && ./configure --prefix=/usr/local/openmpi5/lib \
    && make all install \
    && cd .. \
	&& rm -R openmpi-5.0.2

ENV PATH=/usr/local/openmpi5/lib/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/openmpi5/lib

RUN pip install cython \
    && pip install --prefer-binary pyscf

#Install SALTED
COPY . /src/temp/SALTED-master
RUN cd /src/temp/SALTED-master \
    && pip install .

RUN rm -R /src/temp

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
