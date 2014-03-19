#####################################################
# \file Makefile.am
# \brief Primary Makefile for HiFiLES
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
# AutoMake 'conditionals' (booleans) to be substituted at configure time
BUILD_PARMETIS = @BUILD_PARMETIS@
BUILD_MPI =      @BUILD_MPI@
BUILD_GPU =      @BUILD_GPU@

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS}

bin_PROGRAMS = ../bin/HiFiLES

# NOTE: '___bin_' is due to AM substituting the non-variable-name-friendly 
# characters in the given expression above with an underscore
___bin_HiFiLES_SOURCES = ../src/global.cpp \
                  ../src/input.cpp \
                  ../src/flux.cpp \
                  ../src/cubatrue_tet.cpp \
                  ../src/cubature_hexa.cpp \
                  ../src/cubature_quad.cpp \
                  ../src/cubature_tri.cpp \
                  ../src/cubature_1d.cpp \
                  ../src/funcs.cpp \
                  ../src/inters.cpp \
                  ../src/bdy_inters.cpp \
                  ../src/int_inters.cpp \
                  ../src/mpi_inters.cpp \
                  ../src/eles.cpp \
                  ../src/eles_tris.cpp \
                  ../src/eles_quads.cpp \
                  ../src/eles_tets.cpp \
                  ../src/eles_hexas.cpp \
                  ../src/eles_pris.cpp \
                  ../src/output.cpp \
                  ../src/geometry.cpp \
                  ../src/HiFiLES.cpp \

# -------------- GPU --------------
if BUILD_GPU
# Tell AutoMake how to compile CUDA files
# See http://www.clusterchimps.org/autotools.php
# GPU Architechture Selection: -gencode=arch=compute_xx,code=sm_xx
#   compute_10	 Basic features
#   compute_11	 + atomic memory operations on global memory
#   compute_12	 + atomic memory operations on shared memory
#                + vote instructions
#   compute_13	 + double precision floating point support
#   compute_20	 + Fermi support
#   compute_30	 + Kepler support
.cu.o:
	$(NVCC) $(NVCCFLAGS) -gencode=arch=compute_13,code=sm_13 -o $@ -c $<

noinst_LIBRARIES = cudafiles
cudafiles_a_SOURCES = ../src/cuda_kernels.cu
cudafiles_a_LDADD = $(CUDA_LIBS)
cudafiles_a_CFLAGS = $(NVCCFLAGS)
___bin_HiFiLES_LDADD += libcudafiles.a
endif
# ------------ END GPU ------------

# NOTE: these 'if' statements shouldn't be needed here, as it is taken care 
# of in configure.ac (i.e. if not building, TECIO_LD="")  (I think...)
# ParMETIS
___bin_HiFiLES_CXXFLAGS += @PARMETIS_CXX@
___bin_HiFiLES_LDADD += @PARMETIS_LD@

# TECIO
___bin_HiFiLES_CXXFLAGS += @TECIO_CXX@
___bin_HiFiLES_LDADD += @TECIO_LD@

# CUDA
___bin_HiFiLES_CXXFLAGS += @CUDA_CXX@
___bin_HiFiLES_LDFLAGS += @CUDA_LIBS@
___bin_HiFiLES_LDADD += @CUDA_LD@

# BLAS
___bin_HiFiLES_CXXFLAGS += @BLAS_CXX@
___bin_HiFiLES_LDFLAGS += @BLAS_LDFLAGS@
___bin_HiFiLES_LDADD += @BLAS_LD@
