#!/usr/bin.bash
#####################################################
# \file configure_run.sh
# \brief Configuration script for HiFiLES
# \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
#                          Peter Vincent, David Williams (alphabetical by surname).
#         - Current development: Aerospace Computing Laboratory (ACL) directed
#                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
# \version 1.0.0
# \date Modified on 3/14/14
#
# HiFiLES (High Fidelity Large Eddy Simulation).
# Copyright (C) 2013 Aerospace Computing Laboratory.
#####################################################
# Standard (Helpful) Settings [Should not need to change these]
HIFILES_HOME=$PWD
HIFILES_RUN=$PWD/bin

# Basic User-Modifiable Build Settings [Change these as desired]
NODE="CPU"
CODE="DEBUG"
BLAS="STANDARD_BLAS"
PARALLEL="MPI"
TECIO="NO"

# Compiler Selection [Add filepaths to executables if needed]
CXX="gcc"
NVCC="nvcc"
MPICC="mpicxx"

# Library Locations [Change filepaths as needed]
BLAS_DIR="/usr/local/atlas"
PARMETIS_LIB="/usr/local/lib"
PARMETIS_INCLUDE="/usr/local/include"
TECIO_LIB="lib/tecio-2008/lib"
TECIO_INCLUDE="lib/tecio-2008/include"
CUDA_LIB="/usr/local/cuda/lib64"
CUDA_INCLUE="/usr/local/cuda/include"

# Run configure using the chosen options [Should not change this]
if [[ "$NODE" == "GPU" ]]
then
    _GPU=$NVCC
else
    _GPU="no"
fi
if [[ "$PARALLEL" == "MPI" ]]
then
    _MPI=$MPICC
else
    _MPI="no"
fi
if [[ "$TECIO" == "NO" ]]
    $TECIO_LIB="no"
    $TECIO_INCLUDE="no"
fi
./configure -prefix=$HIFILES_RUN/.. \
            --with-CXX=$CXX \
            --with-BLAS=$BLAS \
            --with-MPI=$_MPI \
            --with-CUDA=$_GPU \
            --with-CUDA-lib=$CUDA_LIB \
            --with-CUDA-include=$CUDA_INCLUDE \
            --with-ParMetis-lib=$PARMETIS_LIB \
            --with-ParMetis-include=$PARMETIS_INCLUDE \
            --with-Tecio-lib=$TECIO_LIB \
            --with-Tecio-include=$TECIO_INCLUDE
