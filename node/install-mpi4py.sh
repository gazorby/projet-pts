#!/usr/bin/env sh

sudo apt-get update && sudo apt-get install python-mpi4py
# The above approach will crash when executed because this installs a copy of openMPI
# which conflicts with the already installed MPICH2 software.
# MPICH2 system is designed to run only one interface and when multiple instances are started, the whole cluster fails

# We still need to the deps of python-mpi4py in order to build mpi4py from source

# Make python pointing to python3
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 2

# Install mpi4py from source to avoid clashes with MPICH2
mpi4py="mpi4py-3.0.3"

sudo wget -N "https://bitbucket.org/mpi4py/mpi4py/downloads/${mpi4py}.tar.gz"

sudo tar -xzf "${mpi4py}.tar.gz"

sudo apt-get update --fix-missing

sudo apt-get install python3-dev

cd "$mpi4py" \
    && sudo python setup.py build --mpicc=/usr/local/mpich2/bin/mpicc \
    && sudo python setup.py install

sudo rm -rf "$mpi4py" "${mpi4py}.tar.gz"