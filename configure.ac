#####################################################
# \file configure.ac
# \brief autoconf file for HiFiLES
# \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
#                          Peter Vincent, David Williams (alphabetical by surname).
#         - Current development: Aerospace Computing Laboratory (ACL) directed
#                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
# \version 1.0.0
# \date Created on 3/14/14
#       Modified on 4/1/14
#
# HiFiLES (High Fidelity Large Eddy Simulation).
# Copyright (C) 2013 Aerospace Computing Laboratory.
#####################################################
AC_INIT([HiFiLES], [1.0.0], [HiFiLES-dev@lists.stanford.edu],
        [HiFiLES], [http://hifiles.stanford.edu])
AC_PREREQ([2.59])
# AC_CONFIG_MACRO_DIR([m4])
#AM_INIT_AUTOMAKE([subdir-objects foreign])
AM_INIT_AUTOMAKE([foreign])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES(yes)])

top_dir=`pwd`
AC_SUBST([top_dir])

# --------------------------------------------------------------
# Build up contributed packages that we will build from source
# --------------------------------------------------------------
hifiles_externals_INCLUDES=""
hifiles_externals_LIBS=""

# Tecplot
# CONFIGURE_TECIO
# if (test $enabletecio = yes); then
#   hifiles_externals_INCLUDES="$TECIO_INCLUDE $hifiles_externals_INCLUDES"
#   hifiles_externals_LIBS="$TECIO_LIB $hifiles_externals_LIBS"
# else
#   hifiles_externals_INCLUDES="-DNO_TECIO $hifiles_externals_INCLUDES"
# fi
# AM_CONDITIONAL(BUILD_TECIO, test x$enabletecio = xyes)
# AC_CONFIG_FILES([externals/tecio/Makefile])

# Metis
# CONFIGURE_METIS
# if (test $enablemetis = yes); then
#   su2_externals_INCLUDES="$METIS_INCLUDE $su2_externals_INCLUDES"
#   su2_externals_LIBS="$METIS_LIB $su2_externals_LIBS"
# else
#   su2_externals_INCLUDES="-DNO_METIS $su2_externals_INCLUDES"
# fi
# AM_CONDITIONAL(BUILD_METIS, test x$enablemetis = xyes)
# AC_CONFIG_FILES([externals/metis/Makefile])
# 
# 
# AC_SUBST([hifiles_externals_INCLUDES])
# AC_SUBST([hifiles_externals_LIBS])

# --------------------------------------------------------------
# optional bindings
# --------------------------------------------------------------
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

AC_ARG_WITH(Metis-lib,
   	AS_HELP_STRING([--with-Metis-lib[=ARG]], [Metis library directory, ARG = path to libmetis.a]), 
    [with_Metis_lib=$withval], [with_Metis_lib="no"])
AC_ARG_WITH(Metis-include,
   	AS_HELP_STRING([--with-Metis-include[=ARG]], [Metis include directory, ARG = path to metis.h]), 
    [with_Metis_include=$withval], [with_Metis_include="no"]) 

AC_ARG_WITH(BLAS,
    AS_HELP_STRING([--with-BLAS[=ARG]], [Build using support for a BLAS variant, ARG = BLAS library to use]), 
    [with_BLAS=$withval], [with_BLAS="no"])
AC_ARG_WITH(BLAS-lib,
    AS_HELP_STRING([--with-BLAS-lib[=ARG]], [BLAS library directory, ARG = path to libcblas.a]), 
    [with_BLAS_lib=$withval], [with_BLAS_lib="no"])
AC_ARG_WITH(BLAS-include,
    AS_HELP_STRING([--with-BLAS-include[=ARG]], [BLAS include directory, ARG = path to cblas.h]), 
    [with_BLAS_include=$withval], [with_BLAS_include="no"])

AC_ARG_WITH(CXX,
    AS_HELP_STRING([--with-CXX[=ARG]], [Optional: Specify C++ compiler, ARG = C++ compiler]), [with_CXX=$withval], [with_CXX="GCC"])
    
########################### CUDA (GPU-Enabled)
# check if we're building GPU or CPU version
# first, check for [a] CUDA library in the given directory
CUDAlib=libcublas.so
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
        AC_CHECK_FILE([$with_CUDA_include/$CUDAheader],[have_CUDA="yes"],[have_CUDA="no"])
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
	CUDA_CXX="-D_GPU -I"$with_CUDA_include
    ARCH=`uname -m`
    if [[ $ARCH == "x86_64" ]];
    then
      SUFFIX="64"
    else
      SUFFIX=""
    fi
	CUDA_LDFLAGS="-L$with_CUDA_lib"
	CUDA_LIBS="-lcudart -lcublas -lcusparse -lm"
    NVCC=$with_CUDA
else
	AM_CONDITIONAL(BUILD_GPU,false)
	CUDA_CXX="-D_CPU"
	CUDA_LDFLAGS=
	CUDA_LIBS=
fi
AC_SUBST([NVCC])
AC_SUBST([CUDA_LIBS])
AC_SUBST([CUDA_LDFLAGS])
AC_SUBST([CUDA_CXX])
AC_SUBST([NVCCFLAGS])

########################### ParMETIS & METIS
# check for lib
ParMETISlib=libparmetis.a
ParMETISheader=parmetis.h
METISlib=libmetis.a
METISheader=metis.h
ParMETISversion=
have_ParMetis="no"

# check for library
if test "$with_ParMetis_lib" != "no" 
then
    # Optional value for separate Metis library file location
    if test "$with_Metis_lib" == "no"
    then
        with_Metis_lib=$with_ParMetis_lib
    fi    
    
  	AC_CHECK_FILE([$with_ParMetis_lib/$ParMETISlib],[have_ParMetis="yes"],[have_ParMetis="no"])   
	if test "$have_ParMetis" == "no"
	then
		AC_MSG_ERROR([ParMETIS requested but ParMETIS library file not found.])
	fi
	
	AC_CHECK_FILE([$with_ParMetis_lib/$METISlib],[have_ParMetis="yes"],[have_ParMetis="no"])   
	if test "$have_ParMetis" == "no"
	then
		AC_MSG_ERROR([ParMETIS requested but METIS library file not found.])
	fi
		
	AC_CHECK_FILE([$with_Metis_lib/$METISlib],[have_ParMetis="yes"],[have_ParMetis="no"])   
	if test "$have_ParMetis" == "no"
	then
		AC_MSG_ERROR([ParMETIS requested but METIS library file not found.])
	fi
fi

# check for header
if test "$have_ParMetis" != "no"
then
  if test "$with_ParMetis_include" != "no"
  then
    # Optional value for separate Metis header file location
    if test "$with_Metis_include" == "no"
    then
      with_Metis_include=$with_ParMetis_include
    fi  
    
    AC_CHECK_FILE([$with_ParMetis_include/$ParMETISheader],[have_ParMetis="yes"],[have_ParMetis="no"]) 
    if test "$have_ParMetis" == "no"
    then
      AC_MSG_ERROR([ParMETIS requested but ParMETIS header file not found.])
    fi
		
    AC_CHECK_FILE([$with_ParMetis_include/$ParMETISheader],[have_ParMetis="yes"],[have_ParMetis="no"]) 

    AC_CHECK_FILE([$with_Metis_include/$METISheader],[have_ParMetis="yes"],[have_ParMetis="no"]) 
    if test "$have_ParMetis" == "no"
    then
      AC_MSG_ERROR([ParMETIS requested but METIS header file not found.])
    fi

  else
    have_ParMetis="no"
  fi  
fi

if test "$have_ParMetis" != "no"
then
  AM_CONDITIONAL(BUILD_PARMETIS,true)	
  PARMETIS_CXX="-I$with_Metis_include -I$with_ParMetis_include"
  PARMETIS_LD="$with_ParMetis_lib/$ParMETISlib $with_Metis_lib/$METISlib"
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
if test "$with_BLAS" == "ACCELERATE"
then
    BLAS_CXX=" -flax-vector-conversions -D_ACCELERATE_BLAS"
    BLAS_LDFLAGS=" -framework Accelerate"
    BLAS_LD=
elif test "$with_BLAS" == "STANDARD"
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
		AC_MSG_ERROR([BLAS linking requested but library file not found (libcblas.a).])
	fi
elif test "$with_BLAS" == "ATLAS"
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
		AC_MSG_ERROR([BLAS linking requested but library files not found (libcblas.a, libatlas.a).])
	fi
# needs more work here (not terribly familiar with our MKL usage)
elif test "$with_BLAS" == "MKL"
then
    if test "$CXX" == "icpc"
    then
        BLAS_CXX="-mkl=parallel -D_MKL_BLAS"
        BLAS_LD=
        BLAS_INCLUDE=
        BLAS_LDFLAGS=
    fi
elif [test "$with_BLAS" == "NO"] || [test "$with_BLAS" == "no"]
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
AM_PROG_LIBTOOL
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
    CUDA support:         $have_CUDA
    ParMetis support:     $have_ParMetis
    TecIO support:        $have_Tecio

Additionally, don't forget to set environment variable HIFILES_HOME

])
