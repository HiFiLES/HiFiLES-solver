#####################################################
# \file configure.ac
# \brief autoconf file for HiFiLES
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
AC_INIT([HiFiLES], [0.1], [HiFiLES-dev@lists.stanford.edu],
        [HiFiLES], [http://hifiles.stanford.edu])
AC_PREREQ([2.59])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([subdir-objects foreign])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES(yes)])

top_dir=`pwd`
AC_SUBST([top_dir])

# --------------------------------------------------------------
# optional bindings
# --------------------------------------------------------------
AC_ARG_WITH(CUDA,
    AS_HELP_STRING([--with-CUDA[=ARG]], [Build GPU version with NVidia CUDA compiler, ARG = CUDA compiler]), 
    [with_CUDA=$withval], [with_CUDA="NO"])
AC_ARG_WITH(CUDA-lib,
    AS_HELP_STRING([--with-CUDA-lib[=ARG]], [CUDA library directory, ARG = path to libcuda.a]), 
    [with_CUDA_lib=$withval], [with_CUDA_lib="NO"])
AC_ARG_WITH(CUDA-include,
    AS_HELP_STRING([--with-CUDA-include[=ARG]], [CUDA include directory, ARG = path to cuda.h]), 
    [with_CUDA_include=$withval], [with_CUDA_include="NO"])
AC_ARG_WITH(CUDA-arch,
    AS_HELP_STRING([--with-CUDA-arch[=ARG]], [CUDA device architecture, compute_xx (default: 20)]), 
    [with_CUDA_arch=$withval], [with_CUDA_arch=20])

AC_ARG_WITH(MPI,
    AS_HELP_STRING([--with-MPI[=ARG]], [Build parallel version with MPI tools, ARG = MPI C++ compiler]), 
    [with_MPI=$withval], [with_MPI="NO"])
AC_ARG_WITH(MPI-include,
    AS_HELP_STRING([--with-MPI-include[=ARG]], [MPI include directory, ARG = path to mpi.h, needed for METIS]),
    [with_MPI_include=$withval], [with_MPI_include="NO"])

AC_ARG_WITH(Tecio-lib,
    AS_HELP_STRING([--with-Tecio-lib[=ARG]], [Tecio library directory, ARG = path to libtecio.a]), 
    [with_Tecio_lib=$withval], [with_Tecio_lib="NO"])
AC_ARG_WITH(Tecio-include,
    AS_HELP_STRING([--with-Tecio-include[=ARG]], [Tecio include directory, ARG = path to TECIO.h]), 
    [with_Tecio_include=$withval], [with_Tecio_include="NO"])

AC_ARG_WITH(ParMetis-lib,
   	AS_HELP_STRING([--with-ParMetis-lib[=ARG]], [ParMetis library directory, ARG = path to libparmetis.a]), 
    [with_ParMetis_lib=$withval], [with_ParMetis_lib="NO"])
AC_ARG_WITH(ParMetis-include,
   	AS_HELP_STRING([--with-ParMetis-include[=ARG]], [ParMetis include directory, ARG = path to parmetis.h]), 
    [with_ParMetis_include=$withval], [with_ParMetis_include="NO"])

AC_ARG_WITH(Metis-lib,
   	AS_HELP_STRING([--with-Metis-lib[=ARG]], [Metis library directory, ARG = path to libmetis.a]), 
    [with_Metis_lib=$withval], [with_Metis_lib="NO"])
AC_ARG_WITH(Metis-include,
   	AS_HELP_STRING([--with-Metis-include[=ARG]], [Metis include directory, ARG = path to metis.h]), 
    [with_Metis_include=$withval], [with_Metis_include="NO"]) 

AC_ARG_WITH(BLAS,
    AS_HELP_STRING([--with-BLAS[=ARG]], [Build using support for a BLAS variant, ARG = BLAS library to use]), 
    [with_BLAS=$withval], [with_BLAS="NO"])
AC_ARG_WITH(BLAS-lib,
    AS_HELP_STRING([--with-BLAS-lib[=ARG]], [BLAS library directory, ARG = path to libcblas.a]), 
    [with_BLAS_lib=$withval], [with_BLAS_lib="NO"])
AC_ARG_WITH(BLAS-include,
    AS_HELP_STRING([--with-BLAS-include[=ARG]], [BLAS include directory, ARG = path to cblas.h]), 
    [with_BLAS_include=$withval], [with_BLAS_include="NO"])

AC_ARG_WITH(CXX,
    AS_HELP_STRING([--with-CXX[=ARG]], [Optional: Specify C++ compiler, ARG = C++ compiler]), [with_CXX=$withval], [with_CXX="GCC"])

AC_ARG_ENABLE(release, 
    AS_HELP_STRING([--enable-release[=ARG]], [Optional: Specify whether to compile for DEBUG (default) or RELEASE]),
    [code_type=$enableval], [code_type="DEBUG"])

# -------------------------------------------------------------
# Specify whether to build debug or release version of the code
# -------------------------------------------------------------
if test "$code_type" == "RELEASE" ; then
  CXXFLAGS="-O3"
else
  CXXFLAGS="-g -O2"
fi

# --------------------------------------------------------------
# Build up contributed packages that we will build from source
# --------------------------------------------------------------
hifiles_externals_INCLUDES=""
hifiles_externals_LIBS=""

# Metis
CONFIGURE_METIS
if (test $with_MPI != "NO")
then
  if (test $enablemetis = "YES")
  then
    hifiles_externals_INCLUDES="$METIS_INCLUDE $hifiles_externals_INCLUDES"
    hifiles_externals_LIBS="$METIS_LIB $hifiles_externals_LIBS"
    AC_CONFIG_FILES([lib/parmetis-4.0.2/Makefile])
    AC_CONFIG_FILES([lib/parmetis-4.0.2/metis/Makefile])
    AM_CONDITIONAL(BUILD_METIS, test $enablemetis = "YES")
  else
    hifiles_externals_INCLUDES="-DNO_METIS $hifiles_externals_INCLUDES"
    AM_CONDITIONAL(BUILD_METIS, test $enablemetis = "YES")
  fi
else
  AM_CONDITIONAL(BUILD_METIS, test $with_MPI != "NO")
fi

AC_SUBST([hifiles_externals_INCLUDES])
AC_SUBST([hifiles_externals_LIBS])

########################### CUDA (GPU-Enabled)
# check if we're building GPU or CPU version
# first, check for [a] CUDA library in the given directory
CUDAlib=libcublas.so
CUDAheader=cuda.h
have_CUDA="NO"

if test "$with_CUDA" != "NO"
then
    AC_CHECK_FILE([$with_CUDA_lib/$CUDAlib],[have_CUDA="YES"],[have_CUDA="NO"])
	if test "$have_CUDA" == "NO"
	then
		AC_MSG_ERROR([GPU build requested but CUDA library files not found.])
	fi
fi

# next, check for CUDA header
if test "$have_CUDA" != "NO"
then
    if test "$with_CUDA_include" != "NO"
    then
        AC_CHECK_FILE([$with_CUDA_include/$CUDAheader],[have_CUDA="YES"],[have_CUDA="NO"])
 	    if test "$have_CUDA" == "NO"
	    then
	    	AC_MSG_ERROR([GPU build requested but CUDA header file not found.])
	    fi    
	else
	    have_CUDA="NO"
	fi
fi

if test "$have_CUDA" != "NO"
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
	CUDA_ARCH="-gencode=arch=compute_$with_CUDA_arch,code=sm_$with_CUDA_arch"
    NVCC=$with_CUDA
else
	AM_CONDITIONAL(BUILD_GPU,false)
	CUDA_CXX="-D_CPU"
	CUDA_LDFLAGS=
	CUDA_LIBS=
	CUDA_ARCH=
fi
AC_SUBST([NVCC])
AC_SUBST([CUDA_LIBS])
AC_SUBST([CUDA_LDFLAGS])
AC_SUBST([CUDA_CXX])
AC_SUBST([CUDA_ARCH])
AC_SUBST([NVCCFLAGS])

########################### ParMETIS
# check for lib
ParMETISlib=libparmetis.a
ParMETISheader=parmetis.h
METISlib=libmetis.a
METISheader=metis.h
ParMETISversion=
have_ParMetis="NO"

# If not building from lib folder, link to given libraries
if test $enablemetis != "YES"
then
  if test "$with_ParMetis_lib" != "NO" 
  then
    
    # Optional value for separate Metis library file location
    if test "$with_Metis_lib" == "NO"
    then
      with_Metis_lib=$with_ParMetis_lib
    fi    

    AC_CHECK_FILE([$with_ParMetis_lib/$ParMETISlib],[have_ParMetis="YES"],[have_ParMetis="NO"])   
    if test "$have_ParMetis" == "NO"
    then
      AC_MSG_ERROR([ParMETIS requested but library file not found.])
    fi

    AC_CHECK_FILE([$with_Metis_lib/$METISlib],[have_ParMetis="YES"],[have_ParMetis="NO"])   
    if test "$have_ParMetis" == "NO"
    then
      AC_MSG_ERROR([ParMETIS requested but METIS library file not found.])
    fi
  fi

  # check for header
  if test "$have_ParMetis" != "NO"
  then
    if test "$with_ParMetis_include" != "NO"
    then
      # Optional value for separate Metis header file location
      if test "$with_Metis_include" == "NO"
      then
        with_Metis_include=$with_ParMetis_include
      fi  

      AC_CHECK_FILE([$with_ParMetis_include/$ParMETISheader],[have_ParMetis="YES"],[have_ParMetis="NO"]) 
      if test "$have_ParMetis" == "NO"
      then
        AC_MSG_ERROR([ParMETIS requested but header file not found.])
      fi

      AC_CHECK_FILE([$with_Metis_include/$METISheader],[have_ParMetis="YES"],[have_ParMetis="NO"]) 
      if test "$have_ParMetis" == "NO"
      then
        AC_MSG_ERROR([ParMETIS requested but METIS header file not found.])
      fi
    else
      have_ParMetis="NO"
    fi  
  fi

  if test "$have_ParMetis" != "NO"
  then
    AM_CONDITIONAL(BUILD_PARMETIS,true)
    PARMETIS_CXX="-I$with_Metis_include -I$with_ParMetis_include"
    PARMETIS_LD="$with_ParMetis_lib/$ParMETISlib $with_Metis_lib/$METISlib"
  else
    AM_CONDITIONAL(BUILD_PARMETIS,false)
    PARMETIS_CXX=-DNO_PARMETIS
    PARMETIS_LD=
  fi
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
have_Tecio="NO"

# check for lib
if test "$with_Tecio_lib" != "NO" 
then
  AC_CHECK_FILE([$with_Tecio_lib/$Teciolib],[have_Tecio="YES"],[have_Tecio="NO"]) 
  if test "$have_Tecio" == "NO"
  then
    Teciolib=libtecio.a
    AC_CHECK_FILE([$with_Tecio_lib/$Teciolib],[have_Tecio="YES"],[have_Tecio="NO"])
  fi
  if test "$have_Tecio" == "NO"
  then
    AC_MSG_ERROR([Tecplot linking requested but library file not found.])
  fi
fi

# check for header
if test "$have_Tecio" != "NO"
then
  if test "$with_Tecio_include" != "NO"
  then
    AC_CHECK_FILE([$with_Tecio_include/$Tecioheader],[have_Tecio="YES"],[have_Tecio="NO"]) 
    if test "$have_Tecio" == "NO"
    then
      AC_MSG_ERROR([Tecplot linking requested but header file not found.])
    fi
  else
    have_Tecio="NO"
  fi  
fi

if test "$have_Tecio" != "NO"
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

have_MPI="NO"
if test "$with_MPI" != "NO"
then
  AC_CHECK_FILE([$with_MPI_include/mpi.h],[have_MPI="YES"],[have_MPI="NO"])
  if test "$have_MPI" == "YES"
  then
    CXX="$with_MPI"	
    have_MPI="YES"
    CXXFLAGS=$CXXFLAGS" -D_MPI"
    MPI_INCLUDE="-I$with_MPI_include"
    AM_CONDITIONAL(BUILD_MPI,true)
	
	# Add MPI flag to CUDA flag if needed
	if test "$have_CUDA" != "NO"
	then
		NVCCFLAGS=$NVCCFLAGS" -D_MPI $MPI_INCLUDE"
	fi
  else
    AC_MSG_ERROR([MPI requested but header file not found.])
  fi
else
  MPI_INCLUDE=
  AM_CONDITIONAL(BUILD_MPI,false)
fi
AC_SUBST([MPI_INCLUDE])



########################### BLAS
if test "$with_BLAS" == "ACCELERATE"
then
  BLAS_CXX=" -flax-vector-conversions -D_ACCELERATE_BLAS"
  BLAS_LDFLAGS=" -framework Accelerate"
  BLAS_LD=
elif test "$with_BLAS" == "STANDARD"
then
    AC_CHECK_FILE([$with_BLAS_lib/libcblas.a],[have_BLAS="YES"],[have_BLAS="NO"]) 
	if test "$have_BLAS" != "NO"
	then
		BLAS_CXX="-I $with_BLAS_include -D_STANDARD_BLAS"
        BLAS_LD="$with_BLAS_lib/libcblas.a"
        BLAS_LDFLAGS=
	fi
	if test "$have_BLAS" == "NO"
	then
		AC_MSG_ERROR([BLAS linking requested but library file not found (libcblas.a).])
	fi
elif test "$with_BLAS" == "ATLAS"
then
    AC_CHECK_FILE([$with_BLAS_lib/libcblas.a],[have_BLAS="YES"],[have_BLAS="NO"]) 
	if test "$have_BLAS" != "NO"
	then
	    AC_CHECK_FILE([$with_BLAS_lib/libatlas.a],[have_BLAS="YES"],[have_BLAS="NO"]) 
	fi
	if test "$have_BLAS" != "NO"
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
elif [test "$with_BLAS" == "NO"] || [test "$with_BLAS" == "NO"]
then
    BLAS_CXX="-D_NO_BLAS"
    BLAS_LD=
    BLAS_LDFLAGS=
    have_BLAS="NO"
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
        lib/Makefile \
	obj/Makefile)

abs_srcdir=`(cd $srcdir && pwd)`
abs_prefix=`(cd $prefix && pwd)`


AC_MSG_RESULT([

Build Configuration Summary:

    Source code location: ${abs_srcdir}
    Install location: 	  ${abs_prefix}
    Version:              ${VERSION}
    Code Type:            ${code_type}
    Compiler:             ${CXX}
    Preprocessor flags:   ${CPPFLAGS} ${BLAS_CXX} ${PARMETIS_CXX} ${CUDA_CXX} ${MPI_INCLUDE}
    Compiler flags:       ${CXXFLAGS} ${NVCCFLAGS}
    Linker flags:         ${LDFLAGS} ${BLAS_LDFLAGS} ${CUDA_LDFLSGS}
    Libraries:            ${BLAS_LD} ${PARMETIS_LD} ${CUDA_LIBS}
    BLAS support:         $have_BLAS
    MPI support:          $have_MPI
    CUDA support:         $have_CUDA $CUDA_ARCH
    TecIO support:        $have_Tecio

Additionally, don't forget to set the environment variable HIFILES_HOME

])
