# \file Makefile
# \brief _____________________________
# \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
#                          Peter Vincent, David Williams (alphabetical by surname).
#         - Current development: Aerospace Computing Laboratory (ACL) directed
#                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
# \version 1.0.0
#
# HiFiLES (High Fidelity Large Eddy Simulation).
# Copyright (C) 2013 Aerospace Computing Laboratory.

# Settings

include makefile.in

# Compiler

ifeq ($(COMP),GCC)
CC      = g++
endif

ifeq ($(COMP),INTEL)
CC      = icpc
endif

ifeq ($(PARALLEL),MPI)
CC      = mpicxx
endif

ifeq ($(NODE),GPU)
NVCC	= nvcc
endif

# Pre-processing macros

OPTS    = -D_$(NODE) -D_$(BLAS) 

ifeq ($(PARALLEL),MPI)
	OPTS    += -D_$(PARALLEL) 
endif

# Includes

OPTS    += -I /include 

ifeq ($(NODE),GPU)
	OPTS	+= -I $(CUDA_DIR)/include 
endif

ifeq ($(BLAS),STANDARD_BLAS)
	OPTS	+= -I $(BLAS_DIR)/include
endif

ifeq ($(TECIO),YES)
	OPTS += -I $(TECIO_DIR)/tecsrc
endif

ifeq ($(PARALLEL),MPI)
	OPTS	+= -I /usr/include/mpich2-x86_64/
	OPTS	+= -I /usr/lib64/openmpi/bin/

	OPTS += -I $(PARMETIS_DIR)/include
	OPTS += -I $(PARMETIS_DIR)/metis/include
endif

ifeq ($(CODE),DEBUG)
	OPTS	+= -g 
endif

ifeq ($(CODE),RELEASE)
	ifeq ($(COMP),GCC)
	OPTS	+= -O3 
	endif
	ifeq ($(COMP),INTEL)
	OPTS	+= -xHOST -fast
	endif	
endif

ifeq ($(CODE),RELEASE)
	ifeq ($(NODE),GPU)
	OPTS += -w
	endif
endif

ifeq ($(COMP),INTEL)
	ifeq ($(BLAS),MKL_BLAS)
	OPTS	+= -mkl=parallel
	endif
endif

# Libraries

ifeq ($(BLAS),ACCELERATE_BLAS)
	LIBS	+= -framework Accelerate
endif

ifeq ($(BLAS),STANDARD_BLAS)
	LIBS	+= -L $(BLAS_DIR)/lib -lcblas -latlas
endif

ifeq ($(NODE),GPU)
	LIBS	+= -L $(CUDA_DIR)/lib64 -lcudart -lcublas -lcusparse -lm
endif


# Source

SRC	= ../src/
vpath %.cpp ../src
vpath %.cu ../src
vpath %.h ../include

# Objects

OBJS    = HiFiLES.o geometry.o solver.o output.o eles.o eles_tris.o eles_quads.o eles_hexas.o eles_tets.o eles_pris.o inters.o int_inters.o bdy_inters.o funcs.o flux.o global.o input.o matrix.o cubature_1d.o cubature_tri.o cubature_quad.o cubature_hexa.o cubature_tet.o

ifeq ($(NODE),GPU)
	OBJS	+=  cuda_kernels.o
endif

ifeq ($(PARALLEL),MPI)
	OBJS += mpi_inters.o
	OBJS += $(PARMETIS_DIR)/build/Darwin-x86_64/libparmetis/libparmetis.a $(PARMETIS_DIR)/build/Darwin-x86_64/libmetis/libmetis.a
endif
	
ifeq ($(TECIO),YES)
	OBJS += $(TECIO_DIR)/tecio.a
endif

# Compile

HiFiLES: $(OBJS)
	$(CC) $(OPTS) -o HiFiLES $(OBJS) $(LIBS) 

HiFiLES.o: HiFiLES.cpp geometry.h input.h flux.h error.h
	$(CC) $(OPTS) -c $(SRC)HiFiLES.cpp
	
geometry.o: geometry.cpp geometry.h input.h  error.h
	$(CC) $(OPTS) -c $(SRC)geometry.cpp 

solver.o: solver.cpp solver.h input.h  error.h
	$(CC) $(OPTS) -c $(SRC)solver.cpp 

output.o: output.cpp output.h input.h  error.h
	$(CC) $(OPTS) -c $(SRC)output.cpp 
	
eles.o: eles.cpp eles.h array.h error.h input.h matrix.h error.h
	$(CC) $(OPTS) -c $(SRC)eles.cpp

eles_tris.o: eles_tris.cpp eles_tris.h eles.h funcs.h input.h array.h matrix.h cubature_1d.h error.h
	$(CC) $(OPTS) -c $(SRC)eles_tris.cpp
	
eles_quads.o: eles_quads.cpp eles_quads.h eles.h funcs.h input.h array.h matrix.h error.h
	$(CC) $(OPTS) -c $(SRC)eles_quads.cpp	
	
eles_hexas.o: eles_hexas.cpp eles_hexas.h eles.h funcs.h input.h array.h matrix.h error.h
	$(CC) $(OPTS) -c $(SRC)eles_hexas.cpp
	
eles_tets.o: eles_tets.cpp eles_tets.h eles.h funcs.h input.h array.h matrix.h error.h cubature_tri.h
	$(CC) $(OPTS) -c $(SRC)eles_tets.cpp
	
eles_pris.o: eles_pris.cpp eles_pris.h eles.h funcs.h input.h array.h matrix.h error.h
	$(CC) $(OPTS) -c $(SRC)eles_pris.cpp

inters.o: inters.cpp inters.h flux.h funcs.h input.h error.h
	$(CC) $(OPTS) -c $(SRC)inters.cpp

int_inters.o: int_inters.cpp int_inters.h inters.h flux.h funcs.h input.h error.h
	$(CC) $(OPTS) -c $(SRC)int_inters.cpp

bdy_inters.o: bdy_inters.cpp bdy_inters.h inters.h flux.h funcs.h input.h error.h
	$(CC) $(OPTS) -c $(SRC)bdy_inters.cpp

ifeq ($(PARALLEL),MPI)
mpi_inters.o: mpi_inters.cpp mpi_inters.h inters.h flux.h funcs.h input.h error.h
	$(CC) $(OPTS) -c $(SRC)mpi_inters.cpp
endif

funcs.o: funcs.cpp funcs.h input.h error.h
	$(CC) $(OPTS) -c $(SRC)funcs.cpp

matrix.o: matrix.cpp matrix.h error.h
	$(CC) $(OPTS) -c $(SRC)matrix.cpp

cubature_1d.o: cubature_1d.cpp cubature_1d.h error.h
	$(CC) $(OPTS) -c $(SRC)cubature_1d.cpp

cubature_tri.o: cubature_tri.cpp cubature_tri.h error.h
	$(CC) $(OPTS) -c $(SRC)cubature_tri.cpp

cubature_quad.o: cubature_quad.cpp cubature_quad.h error.h
	$(CC) $(OPTS) -c $(SRC)cubature_quad.cpp

cubature_hexa.o: cubature_hexa.cpp cubature_hexa.h error.h
	$(CC) $(OPTS) -c $(SRC)cubature_hexa.cpp

cubature_tet.o: cubature_tet.cpp cubature_tet.h error.h
	$(CC) $(OPTS) -c $(SRC)cubature_tet.cpp

flux.o: flux.cpp flux.h array.h	error.h input.h error.h
	$(CC) $(OPTS) -c $(SRC)flux.cpp
	
input.o: input.cpp input.h error.h
	$(CC) $(OPTS) -c $(SRC)input.cpp
	
global.o: global.cpp global.h error.h
	$(CC) $(OPTS) -c $(SRC)global.cpp

ifeq ($(NODE),GPU)	
cuda_kernels.o: cuda_kernels.cu cuda_kernels.h 	error.h util.h
	$(NVCC) -arch=sm_20 --ptxas-options=-v $(OPTS) -c $(SRC)cuda_kernels.cu
endif

clean: 
	rm HiFiLES *.o
