# -------------------------------------------------------------
# Metis
# -------------------------------------------------------------
AC_DEFUN([CONFIGURE_METIS],
[
  AC_ARG_ENABLE(metis,
                AC_HELP_STRING([--enable-metis],
                               [build with Metis graph partitioning suppport]),
		[case "${enableval}" in
		  "YES")  enablemetis="YES" ;;
		   "NO")  enablemetis="NO" ;;
 		    *)  AC_MSG_ERROR(bad value ${enableval} for --enable-metis) ;;
		 esac],
		 [enablemetis="YES"])

  # Thanks, SU2, for providing this file
  # The METIS API is distributed with HiFiLES, so we don't have to guess
  # where it might be installed...
  if (test $enablemetis = "YES"); then

    # look for METIS build cppflags by honoring the --with-metis-cppflags="..." flag,
    # defaulting to what we know works
    AC_ARG_WITH([metis-cppflags],
                 AC_HELP_STRING([--with-metis-cppflags="-D_FILE_OFFSET_BITS=64 -DNDEBUG -DNDEBUG2 -DHAVE_EXECINFO_H -DHAVE_GETLINE"],
                                [Specific METIS C Preprocessor flags to use]),
                 [HIFILES_METIS_CPPFLAGS="$withval"],
                 [HIFILES_METIS_CPPFLAGS="-D_FILE_OFFSET_BITS=64 -DNDEBUG -DNDEBUG2 -DHAVE_EXECINFO_H -DHAVE_GETLINE"])


     METIS_INCLUDE="-DMETIS_5 -I\$(top_srcdir)/lib/parmetis-4.0.2/include -I\$(top_srcdir)/lib/parmetis-4.0.2/metis/include"
     METIS_LIB="\$(top_builddir)/lib/parmetis-4.0.2/libparmetis.a \$(top_builddir)/lib/parmetis-4.0.2/metis/libmetis.a"
     AC_DEFINE(HAVE_METIS, 1, [Flag indicating whether the library will be compiled with Metis support])
     AC_MSG_RESULT(<<< Configuring library with Metis support >>>)

     # look for thread-local storage
     AX_TLS
 else
     METIS_INCLUDE=""
     METIS_LIB=""
     HIFILES_METIS_CPPFLAGS=""
     enablemetis="NO"
  fi

  AC_SUBST(METIS_INCLUDE)
  AC_SUBST(METIS_LIB)
  AC_SUBST(HIFILES_METIS_CPPFLAGS)
])
