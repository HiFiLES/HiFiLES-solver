# Configuration options.
gdb        = not-set
assert     = not-set
assert2    = not-set
debug      = not-set
openmp     = not-set
prefix     = not-set
gklib_path = not-set
metis_path = not-set
shared     = not-set
cc         = mpicc
cxx        = mpicxx


# Basically proxies everything to the builddir cmake.

PKGNAME = parmetis-4.0.2

cputype = $(shell uname -m | sed "s/\\ /_/g")
systype = $(shell uname -s)

BUILDDIR = build/$(systype)-$(cputype)

# Process configuration options.
CONFIG_FLAGS = -DCMAKE_VERBOSE_MAKEFILE=1
ifeq ($(gklib_path), not-set)
    gklib_path = metis/GKlib
endif
ifeq ($(metis_path), not-set)
    metis_path = metis
endif
CONFIG_FLAGS += -DGKLIB_PATH=$(abspath $(gklib_path)) -DMETIS_PATH=$(abspath $(metis_path))
ifneq ($(gdb), not-set)
    CONFIG_FLAGS += -DGDB=$(gdb)
endif
ifneq ($(assert), not-set)
    CONFIG_FLAGS += -DASSERT=$(assert)
endif
ifneq ($(assert2), not-set)
    CONFIG_FLAGS += -DASSERT2=$(assert2)
endif
ifneq ($(debug), not-set)
    CONFIG_FLAGS += -DDEBUG=$(debug)
endif
ifneq ($(openmp), not-set)
    CONFIG_FLAGS += -DOPENMP=$(openmp)
endif
ifneq ($(prefix), not-set)
    CONFIG_FLAGS += -DCMAKE_INSTALL_PREFIX=$(prefix)
endif
ifneq ($(shared), not-set)
    CONFIG_FLAGS += -DSHARED=1
endif
ifneq ($(cc), not-set)
    CONFIG_FLAGS += -DCMAKE_C_COMPILER=$(cc)
endif
ifneq ($(cxx), not-set)
    CONFIG_FLAGS += -DCMAKE_CXX_COMPILER=$(cxx)
endif

define run-config
mkdir -p $(BUILDDIR)
cd $(BUILDDIR) && cmake $(CURDIR) $(CONFIG_FLAGS)
endef

all clean install:
	@if [ ! -f $(BUILDDIR)/Makefile ]; then \
		more BUILD.txt; \
	else \
	  	make -C $(BUILDDIR) $@ $(MAKEFLAGS); \
	fi

uninstall:
	xargs rm < $(BUILDDIR)/install_manifest.txt

config: distclean
	$(run-config)

distclean:
	rm -rf $(BUILDDIR)

remake:
	find . -name CMakeLists.txt -exec touch {} ';'

dist:
	util/mkdist.sh $(PKGNAME)


.PHONY: config distclean dist all clean install uninstall remake
