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
# ---------------------------------------------------------------
# Basic User-Modifiable Build Settings [Change these as desired]
NODE="CPU"              # CPU or GPU
CODE="DEBUG"            # DEBUG or RELEASE
BLAS="ATLAS"            # ATLAS, STANDARD, ACCLERATE, MKL, or NO
PARALLEL="no"           # MPI or NO
TECIO="no"              # YES or NO
# ---------------------------------------------------------------
# Compiler Selections [Change compilers or add full filepaths if needed]
CXX="g++"               # Typically g++ (default) or icpc (Intel)
NVCC="nvcc"             # NVidia CUDA compiler
MPICC="mpicxx"          # MPI compiler
# ---------------------------------------------------------------
# Library Locations [Change filepaths as needed]
BLAS_LIB="/usr/local/atlas/lib"
BLAS_INCLUDE="/usr/local/atlas/include"

PARMETIS_LIB="/usr/local/lib"
PARMETIS_INCLUDE="/usr/local/include"

METIS_LIB="/usr/local/lib"
METIS_INCLUDE="/usr/local/include"

TECIO_LIB="lib/tecio-2008/lib"
TECIO_INCLUDE="lib/tecio-2008/include"

CUDA_LIB="/usr/local/cuda/lib64"
CUDA_INCLUDE="/usr/local/cuda/include"
# ---------------------------------------------------------------
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
    PARMETIS_LIB="no"
    PARMETIS_INCLUDE="no"
fi
if [[ "$TECIO" == "no" ]]
then
    TECIO_LIB="no"
    TECIO_INCLUDE="no"
fi
./configure --prefix=$HIFILES_RUN/.. \
            --with-CXX=$CXX \
            --with-BLAS=$BLAS \
            --with-BLAS-lib=$BLAS_LIB \
            --with-BLAS-include=$BLAS_INCLUDE \
            --with-MPI=$_MPI \
            --with-CUDA=$_GPU \
            --with-CUDA-lib=$CUDA_LIB \
            --with-CUDA-include=$CUDA_INCLUDE \
            --with-ParMetis-lib=$PARMETIS_LIB \
            --with-ParMetis-include=$PARMETIS_INCLUDE \
            --with-Tecio-lib=$TECIO_LIB \
            --with-Tecio-include=$TECIO_INCLUDE
