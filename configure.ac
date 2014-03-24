#####################################################
# \file configure.ac
# \brief autoconf file for HiFiLES
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
AC_INIT([HiFiLES], [1.0.0], [HiFiLES-dev@lists.stanford.edu],
        [HiFiLES], [http://hifiles.stanford.edu])
AC_PREREQ([2.59])
#AM_INIT_AUTOMAKE([subdir-objects])
AM_INIT_AUTOMAKE([foreign])

# optional bindings
AC_ARG_WITH(CUDA,
    AS_HELP_STRING([--with-CUDA[=ARG]], [Build GPU version with NVidia CUDA compiler, ARG = CUDA compiler]), 
    [with_CUDA=$withval], [with_CUDA="no"])
AC_ARG_WITH(CUDA-lib,
    AS_HELP_STRING([--with-CUDA-lib[=ARG]], [CUDA library directory, ARG = path to libcuda.a]), 
    [with_CUDA_lib=$withval], [with_CUDA_lib="no"])
AC_ARG_WITH(CUDA-include,
    AS_HELP_STRING([--with-CUDA-include[=ARG]], [CUDA include directory, ARG = path to cuda.h]), 
    [with_CUDA_include=$withval], [with_CUDA_include="no"])
AC_ARG_WITH(MPI,
    AS_HELP_STRING([--with-MPI[=ARG]], [Build parallel version with MPI tools, ARG = MPI C++ compiler]), 
    [with_MPI=$withval], [with_MPI="no"])
AC_ARG_WITH(Tecio-lib,
    AS_HELP_STRING([--with-Tecio-lib[=ARG]], [Tecio library directory, ARG = path to libtecio.a]), 
    [with_Tecio_lib=$withval], [with_Tecio_lib="no"])
AC_ARG_WITH(Tecio-include,
    AS_HELP_STRING([--with-Tecio-include[=ARG]], [Tecio include directory, ARG = path to TECIO.h]), 
    [with_Tecio_include=$withval], [with_Tecio_include="no"])
AC_ARG_WITH(ParMetis-lib,
   	AS_HELP_STRING([--with-ParMetis-lib[=ARG]], [ParMetis library directory, ARG = path to libparmetis.a]), 
    [with_ParMetis_lib=$withval], [with_ParMetis_lib="no"])
AC_ARG_WITH(ParMetis-include,
   	AS_HELP_STRING([--with-ParMetis-include[=ARG]], [ParMetis include directory, ARG = path to parmetis.h]), 
    [with_ParMetis_include=$withval], [with_ParMetis_include="no"])
AC_ARG_WITH(ParMetis-version,
   	AS_HELP_STRING([--with-ParMetis-version[=ARG]], [ParMetis major version, ARG = 4 (default) or 5]), 
    [with_ParMetis_version=$withval], [with_ParMetis_version="4"])
AC_ARG_WITH(CXX,
    AS_HELP_STRING([--with-CXX[=ARG]], [Optional: Specify C++ compiler, ARG = C++ compiler]), [with_CXX=$withval], [with_CXX="GCC"])
AC_ARG_WITH(BLAS,
    AS_HELP_STRING([--with-BLAS[=ARG]], [Build using support for a BLAS variant, ARG = BLAS library to use]), 
    [with_BLAS=$withval], [with_BLAS="no"])
AC_ARG_WITH(BLAS-lib,
    AS_HELP_STRING([--with-BLAS-lib[=ARG]], [BLAS library directory, ARG = path to libcblas.a]), 
    [with_BLAS_lib=$withval], [with_BLAS_lib="no"])
AC_ARG_WITH(BLAS-include,
    AS_HELP_STRING([--with-BLAS-include[=ARG]], [BLAS include directory, ARG = path to cblas.h]), 
    [with_BLAS_include=$withval], [with_BLAS_include="no"])
    
########################### CUDA (GPU-Enabled)
# check if we're building GPU or CPU version
# first, check for [a] CUDA library in the given directory
CUDAlib=libcublas.a
CUDAheader=cuda.h
have_CUDA="no"

if test "$with_CUDA" != "no"
then
    AC_CHECK_FILE([$with_CUDA_lib/$CUDAlib],[have_CUDA="yes"],[have_CUDA="no"])
	if test "$have_CUDA" == "no"
	then
		AC_MSG_ERROR([GPU build requested but CUDA library files not found.])
	fi
fi

# next, check for CUDA header
if test "$have_CUDA" != "no"
then
    if test "$with_CUDA_include" != "no"
    then
        AC_CHECK_FILE([$with_CUDA_include/$CUDAinclude],[have_CUDA="yes"],[have_CUDA="no"])
 	    if test "$have_CUDA" == "no"
	    then
	    	AC_MSG_ERROR([GPU build requested but CUDA header file not found.])
	    fi    
	else
	    have_CUDA="no"
	fi
fi

if test "$have_CUDA" != "no"
then
  	AM_CONDITIONAL(BUILD_GPU,true)
	NVCCFLAGS="-D_GPU -I"$with_CUDA_include	
	CUDA_CXX="-D_GPU"
    ARCH=`uname -m`
    if [[ $ARCH == "x86_64" ]];
    then
      SUFFIX="64"
    else
      SUFFIX=""
    fi
	CUDA_LD="-L$with_CUDA_lib -lcudart -lcublas -lcusparse -lm"
    NVCC=$with_CUDA
else
	AM_CONDITIONAL(BUILD_GPU,false)
	CUDA_CXX="-D_CPU"
	CUDA_LD=
fi
AC_SUBST([NVCC])
AC_SUBST([CUDA_LD])
AC_SUBST([CUDA_CXX])
AC_SUBST([NVCCFLAGS])

########################### ParMETIS
# check for lib
ParMETISlib=libparmetis.a
ParMETISheader=parmetis.h
ParMETISversion=
have_ParMetis="no"

if test "$with_ParMetis_lib" != "no" 
then
  	AC_CHECK_FILE([$with_ParMetis_lib/$ParMETISlib],[have_ParMetis="yes"],[have_ParMetis="no"])   
	if test "$have_ParMetis" == "no"
	then
		AC_MSG_ERROR([ParMETIS requested but library file not found.])
	fi
fi

# check for header
if test "$have_ParMetis" != "no"
then
	if test "$with_ParMetis_include" != "no"
	then
		AC_CHECK_FILE([$with_ParMetis_include/$parMETISheader],[have_ParMetis="yes"],[have_ParMetis="no"]) 
		if test "$have_ParMetis" == "no"
		then
			AC_MSG_ERROR([ParMETIS requested but header file not found.])
		fi
	else
		have_ParMetis="no"
	fi  
fi

# ParMETIS version check
if test "$with_ParMetis_version" == "5"
then
	ParMETISversion=-DPARMETIS_5
fi

if test "$have_ParMetis" != "no"
then
  	AM_CONDITIONAL(BUILD_PARMETIS,true)
	PARMETIS_CXX="-I"$with_ParMetis_include $ParMETISversion
	PARMETIS_LD=$with_ParMetis_lib"/"$ParMETISlib
else
	AM_CONDITIONAL(BUILD_PARMETIS,false)
	PARMETIS_CXX=-DNO_PARMETIS
	PARMETIS_LD=
fi
AC_SUBST([PARMETIS_CXX])
AC_SUBST([PARMETIS_LD])

########################### Tecplot
Teciolib=libtecio.a
Tecioheader=TECIO.h
have_Tecio="no"

# check for lib
if test "$with_Tecio_lib" != "no" 
then
  	AC_CHECK_FILE([$with_Tecio_lib/$Teciolib],[have_Tecio="yes"],[have_Tecio="no"]) 
	if test "$have_Tecio" == "no"
	then
		Teciolib=libtecio.a
		AC_CHECK_FILE([$with_Tecio_lib/$Teciolib],[have_Tecio="yes"],[have_Tecio="no"])
	fi
	if test "$have_Tecio" == "no"
	then
		AC_MSG_ERROR([Tecplot linking requested but library file not found.])
	fi
fi

# check for header
if test "$have_Tecio" != "no"
then
	if test "$with_Tecio_include" != "no"
	then
		AC_CHECK_FILE([$with_Tecio_include/$Tecioheader],[have_Tecio="yes"],[have_Tecio="no"]) 
		if test "$have_Tecio" == "no"
		then
			AC_MSG_ERROR([Tecplot linking requested but header file not found.])
		fi
	else
		have_Tecio="no"
	fi  
fi

if test "$have_Tecio" != "no"
then
  	AM_CONDITIONAL(BUILD_TECIO,true)
	TECIO_CXX="-I"$with_Tecio_include
	TECIO_LD=$with_Tecio_lib"/"$Teciolib
else
	AM_CONDITIONAL(BUILD_TECIO,false)
	TECIO_CXX=-DNO_TECIO
	TECIO_LD=
fi
AC_SUBST([TECIO_CXX])
AC_SUBST([TECIO_LD])


########################### MPI

have_MPI="no"
if test "$with_MPI" != "no"
then
	CXX="$with_MPI"	
	have_MPI="yes"
	CXXFLAGS=$CXXFLAGS" -D_MPI"
fi

########################### BLAS
# *** still need to implement library & header-file checks ***
if test "$with_BLAS" == "ACCELERATE_BLAS"
then
    BLAS_CXX=" -flax-vector-conversions -D_ACCELERATE_BLAS"
    BLAS_LDFLAGS=" -framework Accelerate"
    BLAS_LD=
elif test "$with_BLAS" == "STANDARD_BLAS"
then
    AC_CHECK_FILE([$with_BLAS_lib/libcblas.a],[have_BLAS="yes"],[have_BLAS="no"]) 
	if test "$have_BLAS" != "no"
	then
		BLAS_CXX="-I $with_BLAS_include -D_STANDARD_BLAS"
        BLAS_LD="$with_BLAS_lib/libcblas.a"
        BLAS_LDFLAGS=
	fi
	if test "$have_BLAS" == "no"
	then
		AC_MSG_ERROR([BLAS linking requested but library file not found.])
	fi
elif test "$with_BLAS" == "ATLAS_BLAS"
then
    AC_CHECK_FILE([$with_BLAS_lib/libcblas.a],[have_BLAS="yes"],[have_BLAS="no"]) 
	if test "$have_BLAS" != "no"
	then
	    AC_CHECK_FILE([$with_BLAS_lib/libatlas.a],[have_BLAS="yes"],[have_BLAS="no"]) 
	fi
	if test "$have_BLAS" != "no"
	then
       	BLAS_CXX="-I $with_BLAS_include -D_STANDARD_BLAS"
        BLAS_LD="$with_BLAS_lib/libcblas.a $with_BLAS_lib/libatlas.a"
        BLAS_LDFLAGS=
	else
		AC_MSG_ERROR([BLAS linking requested but library file not found.])
	fi
# needs more work here (not terribly familiar with our MKL usage)
elif test "$with_BLAS" == "MKL_BLAS"
then
    if test "$CXX" == "icpc"
    then
        BLAS_CXX="-mkl=parallel -D_MKL_BLAS"
        BLAS_LD=
        BLAS_INCLUDE=
        BLAS_LDFLAGS=
    fi
elif test "$with_BLAS" == "NO_BLAS"
then
    BLAS_CXX="-DNO_BLAS"
    BLAS_LD=
    BLAS_LDFLAGS=
    have_BLAS="no"
elif test "$with_BLAS" == "no"
then
    BLAS_CXX="-DNO_BLAS"
    BLAS_LD=
    BLAS_LDFLAGS=
    have_BLAS="no"
fi

AC_SUBST([BLAS_CXX])
AC_SUBST([BLAS_LDFLAGS])
AC_SUBST([BLAS_LD])

###########################
# AC_CONFIG_HEADERS([config.h])
AC_PROG_CXX
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_RANLIB
AC_OUTPUT(Makefile \
	obj/Makefile)

abs_srcdir=`(cd $srcdir && pwd)`
abs_prefix=`(cd $prefix && pwd)`


AC_MSG_RESULT([

Build Configuration Summary:

    Source code location: ${abs_srcdir}
    Install location: 	  ${abs_prefix}
    Version:              ${VERSION}
    Compiler:             ${CXX}
    Preprocessor flags:   ${CPPFLAGS}
    Compiler flags:       ${CXXFLAGS}
    Linker flags:         ${LDFLAGS}
    BLAS support:         $have_BLAS
    MPI support:          $have_MPI
    ParMetis support:     $have_ParMetis
    TecIO support:        $have_Tecio

])
