#####################################################
# \file Makefile.am
# \brief Primary Makefile for HiFiLES
# \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
#                          Peter Vincent, David Williams (alphabetical by surname).
#         - Current development: Aerospace Computing Laboratory (ACL) directed
#                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
# \version 1.0.0
# \date Modified on 9/8/14
#
# HiFiLES (High Fidelity Large Eddy Simulation).
# Copyright (C) 2013 Aerospace Computing Laboratory.
#####################################################
# AutoMake 'conditionals' (booleans) to be substituted at configure time
BUILD_PARMETIS = @BUILD_PARMETIS@
BUILD_MPI =      @BUILD_MPI@
BUILD_GPU =      @BUILD_GPU@

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS}
AM_CPPFLAGS = -I../include/

bin_PROGRAMS = ../bin/HiFiLES

___bin_HiFiLES_CXXFLAGS=
___bin_HiFiLES_LDADD=
___bin_HiFiLES_LDFLAGS=

# -------------- GPU --------------
# Tell AutoMake how to compile CUDA (.cu) files
.cu.o:
	nvcc $(NVCCFLAGS) $(CUDA_ARCH) $(NVCC_FLAGS) -Xcudafe "--diag_suppress=subscript_out_of_range" -c -o $@ $<

# NOTE: '___bin_' is due to AM substituting the non-variable-name-friendly 
# characters in the given expression above with an underscore
___bin_HiFiLES_SOURCES = ../src/global.cpp \
                  ../src/input.cpp \
                  ../src/flux.cpp \
                  ../src/source.cpp \
                  ../src/cubature_tet.cpp \
                  ../src/cubature_hexa.cpp \
                  ../src/cubature_quad.cpp \
                  ../src/cubature_tri.cpp \
                  ../src/cubature_1d.cpp \
                  ../src/funcs.cpp \
                  ../src/inters.cpp \
                  ../src/bdy_inters.cpp \
                  ../src/int_inters.cpp \
                  ../src/eles.cpp \
                  ../src/eles_tris.cpp \
                  ../src/eles_quads.cpp \
                  ../src/eles_tets.cpp \
                  ../src/eles_hexas.cpp \
                  ../src/eles_pris.cpp \
                  ../src/output.cpp \
                  ../src/geometry.cpp \
                  ../src/solver.cpp \
                  ../src/mesh.cpp \
                  ../src/matrix_structure.cpp \
                  ../src/vector_structure.cpp \
                  ../src/linear_solvers_structure.cpp \
                  ../src/HiFiLES.cpp

if BUILD_GPU
___bin_HiFiLES_SOURCES += ../src/cuda_kernels.cu
endif

if BUILD_MPI
___bin_HiFiLES_SOURCES += ../src/mpi_inters.cpp
endif

___bin_HiFiLES_CXXFLAGS +=@CXXFLAGS@
___bin_HiFiLES_CXXFLAGS +=@hifiles_externals_INCLUDES@
___bin_HiFiLES_LDADD += @hifiles_externals_LIBS@

# ParMETIS
___bin_HiFiLES_CXXFLAGS += @PARMETIS_CXX@
___bin_HiFiLES_LDADD += @PARMETIS_LD@

# TECIO
___bin_HiFiLES_CXXFLAGS += @TECIO_CXX@
___bin_HiFiLES_LDADD += @TECIO_LD@

# CUDA
___bin_HiFiLES_CXXFLAGS += @CUDA_CXX@
___bin_HiFiLES_LDFLAGS += @CUDA_LDFLAGS@
___bin_HiFiLES_LDADD += @CUDA_LIBS@

# BLAS
___bin_HiFiLES_CXXFLAGS += @BLAS_CXX@
___bin_HiFiLES_LDFLAGS += @BLAS_LDFLAGS@
___bin_HiFiLES_LDADD += @BLAS_LD@

# MPI
___bin_HiFiLES_CXXFLAGS += @MPI_INCLUDE@
