TEMPLATE = app
CONFIG += console
CONFIG -= qt

TOP_DIR = .
SRC_DIR = $$TOP_DIR/src
INCLUDE_DIR = $$TOP_DIR/include
DATA_DIR = $$TOP_DIR/data

SOURCES += \
    $$SRC_DIR/solver.cpp \
    $$SRC_DIR/output.cpp \
    $$SRC_DIR/mpi_inters.cpp \
    $$SRC_DIR/int_inters.cpp \
    $$SRC_DIR/inters.cpp \
    $$SRC_DIR/input.cpp \
    $$SRC_DIR/HiFiLES.cpp \
    $$SRC_DIR/global.cpp \
    $$SRC_DIR/geometry.cpp \
    $$SRC_DIR/funcs.cpp \
    $$SRC_DIR/flux.cpp \
    $$SRC_DIR/eles_tris.cpp \
    $$SRC_DIR/eles_tets.cpp \
    $$SRC_DIR/eles_quads.cpp \
    $$SRC_DIR/eles_pris.cpp \
    $$SRC_DIR/eles_hexas.cpp \
    $$SRC_DIR/eles.cpp \
    $$SRC_DIR/cuda_kernels.cu \
    $$SRC_DIR/cubature_tri.cpp \
    $$SRC_DIR/cubature_tet.cpp \
    $$SRC_DIR/cubature_quad.cpp \
    $$SRC_DIR/cubature_hexa.cpp \
    $$SRC_DIR/cubature_1d.cpp \
    $$SRC_DIR/bdy_inters.cpp \
    src/vector_structure.cpp \
    src/mesh.cpp \
    src/matrix_structure.cpp \
    src/linear_solvers_structure.cpp \
    include/vector_structure.inl \
    include/linear_solvers_structure.inl \
    include/matrix_structure.inl \
    src/source.cpp

HEADERS += \
    $$INCLUDE_DIR/util.h \
    $$INCLUDE_DIR/solver.h \
    $$INCLUDE_DIR/solution.h \
    $$INCLUDE_DIR/rename.h \
    $$INCLUDE_DIR/parmetisbin.h \
    $$INCLUDE_DIR/output.h \
    $$INCLUDE_DIR/mpi_inters.h \
    $$INCLUDE_DIR/macros.h \
    $$INCLUDE_DIR/int_inters.h \
    $$INCLUDE_DIR/inters.h \
    $$INCLUDE_DIR/input.h \
    $$INCLUDE_DIR/global.h \
    $$INCLUDE_DIR/geometry.h \
    $$INCLUDE_DIR/funcs.h \
    $$INCLUDE_DIR/flux.h \
    $$INCLUDE_DIR/error.h \
    $$INCLUDE_DIR/eles_tris.h \
    $$INCLUDE_DIR/eles_tets.h \
    $$INCLUDE_DIR/eles_quads.h \
    $$INCLUDE_DIR/eles_pris.h \
    $$INCLUDE_DIR/eles_hexas.h \
    $$INCLUDE_DIR/eles.h \
    $$INCLUDE_DIR/cuda_kernels.h \
    $$INCLUDE_DIR/cubature_tri.h \
    $$INCLUDE_DIR/cubature_tet.h \
    $$INCLUDE_DIR/cubature_quad.h \
    $$INCLUDE_DIR/cubature_hexa.h \
    $$INCLUDE_DIR/cubature_1d.h \
    $$INCLUDE_DIR/bdy_inters.h \
    $$INCLUDE_DIR/array.h \
    include/vector_structure.hpp \
    include/linear_solvers_structure.hpp \
    include/matrix_structure.hpp \
    include/source.h

OTHER_FILES += \
    $$DATA_DIR/loc_tri_inter_pts.dat \
    $$DATA_DIR/loc_tri_alpha_pts.dat \
    $$DATA_DIR/loc_tet_inter_pts_old.dat \
    $$DATA_DIR/loc_tet_inter_pts.dat \
    $$DATA_DIR/loc_tet_alpha_pts.dat \
    $$DATA_DIR/loc_1d_gauss_pts.dat \
    $$DATA_DIR/loc_1d_gauss_lobatto_pts.dat \
    $$DATA_DIR/cubature_tri.dat \
    $$DATA_DIR/cubature_tet.dat \
    $$DATA_DIR/cubature_quad.dat \
    $$DATA_DIR/cubature_hexa.dat \
    $$DATA_DIR/cubature_1d.dat \
    $$TOP_DIR/makefile.in \
    $$TOP_DIR/Makefile
