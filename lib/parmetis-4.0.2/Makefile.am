pkg_cppflags  = @HIFILES_METIS_CPPFLAGS@
pkg_cppflags += -I$(srcdir)/include  -I$(srcdir)/GKlib -I$(srcdir)/libmetis
pkg_sources = \
    akwayfm.c \
    ametis.c \
    balancemylink.c \
    comm.c \
    csrmatch.c \
    ctrl.c \
    debug.c \
    diffutil.c \
    frename.c \
    gkmetis.c \
    gkmpi.c \
    graph.c \
    initbalance.c \
    initmsection.c \
    initpart.c \
    kmetic.c \
    kwayrefine.c \
    match.c \
    mdiffusion.c \
    mesh.c \
    mmetis.c \
    move.c \
    msetup.c \
    node_refine.c \
    ometis.c \
    pspases.c \
    redomylink.c \
    remap.c \
    renumber.c \
    rmetis.c \
    selectq.c \
    serial.c \
    stat.c \
    timer.c \
    util.c \
    wave.c \
    weird.c \
    wspace.c \
    xyapart.c
    



######################################################################
AUTOMAKE_OPTIONS = subdir-objects

AM_CPPFLAGS  = $(pkg_cppflags) # any defined package-specific include paths
AM_CFLAGS    =
AM_CXXFLAGS  =
AM_LDFLAGS   = $(srcdir)/metis/libmetis/libmetis.a

noinst_LIBRARIES = libparmetis.a

libparmetis_a_SOURCES  = $(pkg_sources)
libparmetis_a_CPPFLAGS = $(CPPFLAGS_DBG) $(AM_CPPFLAGS)
libparmetis_a_CXXFLAGS = $(CXXFLAGS_DBG)
libparmetis_a_CFLAGS   = $(CFLAGS_DBG)
