SUBDIRS = metis/
pkg_cppflags  = @HIFILES_METIS_CPPFLAGS@
pkg_cppflags += -I$(srcdir)/include -I$(srcdir)/libparmetis 
pkg_cppflags += -I$(srcdir)/metis/GKlib  -I$(srcdir)/metis/libmetis -I$(srcdir)/metis/include
pkg_cppflags += @MPI_INCLUDE@ 
pkg_sources = \
    libparmetis/akwayfm.c \
    libparmetis/ametis.c \
    libparmetis/balancemylink.c \
    libparmetis/comm.c \
    libparmetis/csrmatch.c \
    libparmetis/ctrl.c \
    libparmetis/debug.c \
    libparmetis/defs.h \
    libparmetis/diffutil.c \
    libparmetis/frename.c \
    libparmetis/gkmetis.c \
    libparmetis/gkmpi.c \
    libparmetis/graph.c \
    libparmetis/initbalance.c \
    libparmetis/initmsection.c \
    libparmetis/initpart.c \
    libparmetis/kmetis.c \
    libparmetis/kwayrefine.c \
    libparmetis/macros.h \
    libparmetis/match.c \
    libparmetis/mdiffusion.c \
    libparmetis/mesh.c \
    libparmetis/mmetis.c \
    libparmetis/move.c \
    libparmetis/msetup.c \
    libparmetis/node_refine.c \
    libparmetis/ometis.c \
    libparmetis/parmetislib.h \
    libparmetis/proto.h \
    libparmetis/pspases.c \
    libparmetis/redomylink.c \
    libparmetis/remap.c \
    libparmetis/rename.h \
    libparmetis/renumber.c \
    libparmetis/rmetis.c \
    libparmetis/selectq.c \
    libparmetis/serial.c \
    libparmetis/stat.c \
    libparmetis/struct.h \
    libparmetis/timer.c \
    libparmetis/util.c \
    libparmetis/wave.c \
    libparmetis/weird.c \
    libparmetis/wspace.c \
    libparmetis/xyzpart.c
    



######################################################################
AUTOMAKE_OPTIONS = subdir-objects

AM_CXX       = $(CXX) # Typically the MPICC compiler (choose with --with-MPI option)
AM_CPPFLAGS  = $(pkg_cppflags) # any defined package-specific include paths
AM_CFLAGS    =
AM_CXXFLAGS  =
AM_LDFLAGS   = $(srcdir)/metis/libmetis.a

noinst_LIBRARIES = libparmetis.a

libparmetis_a_SOURCES  = $(pkg_sources)
libparmetis_a_CPPFLAGS = $(CPPFLAGS_DBG) $(AM_CPPFLAGS)
libparmetis_a_CXXFLAGS = $(CXXFLAGS_DBG)
libparmetis_a_CFLAGS   = $(CFLAGS_DBG)
