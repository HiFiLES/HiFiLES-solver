TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../HiFiLES/src/bdy_inters.cpp \
    ../HiFiLES/src/cubature_1d.cpp \
    ../HiFiLES/src/cubature_hexa.cpp \
    ../HiFiLES/src/cubature_quad.cpp \
    ../HiFiLES/src/cubature_tet.cpp \
    ../HiFiLES/src/cubature_tri.cpp \
    ../HiFiLES/src/eles_hexas.cpp \
    ../HiFiLES/src/eles_pris.cpp \
    ../HiFiLES/src/eles_quads.cpp \
    ../HiFiLES/src/eles_tets.cpp \
    ../HiFiLES/src/eles_tris.cpp \
    ../HiFiLES/src/eles.cpp \
    ../HiFiLES/src/flux.cpp \
    ../HiFiLES/src/funcs.cpp \
    ../HiFiLES/src/geometry.cpp \
    ../HiFiLES/src/global.cpp \
    ../HiFiLES/src/HiFiLES.cpp \
    ../HiFiLES/src/input.cpp \
    ../HiFiLES/src/int_inters.cpp \
    ../HiFiLES/src/inters.cpp \
    ../HiFiLES/src/mpi_inters.cpp \
    ../HiFiLES/src/output.cpp \
    ../HiFiLES/src/solver.cpp \
    ../HiFiLES/src/cuda_kernels.cu

HEADERS += \
    ../HiFiLES/include/array.h \
    ../HiFiLES/include/bdy_inters.h \
    ../HiFiLES/include/cubature_1d.h \
    ../HiFiLES/include/cubature_hexa.h \
    ../HiFiLES/include/cubature_quad.h \
    ../HiFiLES/include/cubature_tet.h \
    ../HiFiLES/include/cubature_tri.h \
    ../HiFiLES/include/cuda_kernels.h \
    ../HiFiLES/include/eles_hexas.h \
    ../HiFiLES/include/eles_pris.h \
    ../HiFiLES/include/eles_quads.h \
    ../HiFiLES/include/eles_tets.h \
    ../HiFiLES/include/eles_tris.h \
    ../HiFiLES/include/eles.h \
    ../HiFiLES/include/error.h \
    ../HiFiLES/include/flux.h \
    ../HiFiLES/include/funcs.h \
    ../HiFiLES/include/geometry.h \
    ../HiFiLES/include/global.h \
    ../HiFiLES/include/input.h \
    ../HiFiLES/include/int_inters.h \
    ../HiFiLES/include/inters.h \
    ../HiFiLES/include/macros.h \
    ../HiFiLES/include/mpi_inters.h \
    ../HiFiLES/include/output.h \
    ../HiFiLES/include/parmetisbin.h \
    ../HiFiLES/include/rename.h \
    ../HiFiLES/include/solution.h \
    ../HiFiLES/include/solver.h \
    ../HiFiLES/include/util.h

