#####################################################
# \file configure_run.sh
# \brief Configuration script for HiFiLES
# \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
#                          Peter Vincent, David Williams (alphabetical by surname).
#         - Current development: Aerospace Computing Laboratory (ACL)
#                                Aero/Astro Department. Stanford University.
# \version 0.1.0
#
# High Fidelity Large Eddy Simulation (HiFiLES) Code.
# Copyright (C) 2014 Aerospace Computing Laboratory (ACL).
#
# HiFiLES is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HiFiLES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HiFiLES.  If not, see <http://www.gnu.org/licenses/>.
#####################################################
# Standard (Helpful) Settings [Should not need to change these]
export HIFILES_HOME=$(pwd)
# ---------------------------------------------------------------
# Basic User-Modifiable Build Settings [Change these as desired]
NODE="CPU"              # CPU or GPU
CODE="RELEASE"            # DEBUG or RELEASE
BLAS="ATLAS"               # ATLAS, STANDARD, ACCLERATE, or NO
PARALLEL="YES"           # YES or NO
TECIO="NO"              # YES or NO
METIS="YES"              # Build & link to the HiFiLES-supplied ParMETIS libraries? YES or NO
# ---------------------------------------------------------------
# Compiler Selections [Change compilers or add full filepaths if needed]
CXX="g++"               # C++ compiler - Typically g++ (default, GNU) or icpc (Intel)
NVCC="nvcc"             # NVidia CUDA compiler
MPICC="mpicxx"          # MPI C compiler
# ---------------------------------------------------------------
# Library & Header File Locations [Change filepaths as needed]
BLAS_LIB="/usr/local/atlas/lib"
BLAS_INCLUDE="/usr/local/atlas/include"

TECIO_LIB="lib/tecio-2008/lib"
TECIO_INCLUDE="lib/tecio-2008/include"

# If building the supplied ParMETIS libraries, need the MPI header location
MPI_INCLUDE="/usr/include/mpich2"       # location of mpi.h

# If NOT building the supplied ParMetis library, location of installed libraries
PARMETIS_LIB="/usr/local/lib"           # location of libparmetis.a
PARMETIS_INCLUDE="/usr/local/include"   # location of parmetis.h

METIS_LIB="/usr/local/lib"              # location of libmetis.a
METIS_INCLUDE="/usr/local/include"      # location of metis.h

# GPU Architechture Selection: -gencode=arch=compute_xx,code=sm_xx (default: 20)
#   compute_10	 Basic features
#   compute_11	 + atomic memory operations on global memory
#   compute_12	 + atomic memory operations on shared memory
#                + vote instructions
#   compute_13	 + double precision floating point support
#   compute_20	 + Fermi support
#   compute_30	 + Kepler support
CUDA_ARCH="20"
CUDA_LIB="/usr/local/cuda-5.0/lib64"
CUDA_INCLUDE="/usr/local/cuda-5.0/include"

# ---------------------------------------------------------------
# Run configure using the chosen options [Should not change this]
if [[ "$NODE" == "GPU" ]]
then
    _GPU=$NVCC
else
    _GPU="NO"
fi
if [[ "$PARALLEL" == "YES" ]]
then
    _MPI=$MPICC
else
    _MPI="NO"
    PARMETIS_LIB="NO"
    PARMETIS_INCLUDE="NO"
fi
if [[ "$TECIO" == "NO" ]]
then
    TECIO_LIB="NO"
    TECIO_INCLUDE="NO"
fi
./configure --prefix=$HIFILES_RUN/.. \
            --with-CXX=$CXX \
            --with-BLAS=$BLAS \
            --with-BLAS-lib=$BLAS_LIB \
            --with-BLAS-include=$BLAS_INCLUDE \
            --with-MPI=$_MPI \
            --with-MPI-include=$MPI_INCLUDE \
            --with-CUDA=$_GPU \
            --with-CUDA-lib=$CUDA_LIB \
            --with-CUDA-include=$CUDA_INCLUDE \
            --with-CUDA-arch=$CUDA_ARCH \
            --with-ParMetis-lib=$PARMETIS_LIB \
            --with-ParMetis-include=$PARMETIS_INCLUDE \
            --with-Metis-lib=$METIS_LIB \
            --with-Metis-include=$METIS_INCLUDE \
            --with-Tecio-lib=$TECIO_LIB \
            --with-Tecio-include=$TECIO_INCLUDE \
            --enable-metis=$METIS \
            --enable-release=$CODE
