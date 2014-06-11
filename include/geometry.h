/*!
 * \file geometry.h
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL)
 *                                Aero/Astro Department. Stanford University.
 * \version 0.1.0
 *
 * High Fidelity Large Eddy Simulation (HiFiLES) Code.
 * Copyright (C) 2014 Aerospace Computing Laboratory (ACL).
 *
 * HiFiLES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HiFiLES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HiFiLES.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <string>

#include "array.h"
#include "input.h"
#include "eles.h"
#include "eles_tris.h"
#include "eles_quads.h"
#include "eles_hexas.h"
#include "eles_tets.h"
#include "eles_pris.h"
#include "int_inters.h"
#include "bdy_inters.h"
#include "solution.h"

#ifdef _MPI
#include "mpi.h"
#include "mpi_inters.h"
#endif

/*!
 * \brief Set input values (config file) in FlowSol.
 * \param[in] FlowSol - Structure with the entire solution and mesh information.
 */
void SetInput(struct solution* FlowSol);

/*!
 * \brief Read the computational mesh.
 * \param[in] FlowSol - Structure with the entire solution and mesh information.
 */
void GeoPreprocess(struct solution* FlowSol);

/*!
 * \brief Method to read a mesh.
 * \param[in] in_file_name - _______________________.
 * \param[out] out_xv - _______________________.
 * \param[out] out_c2v - _______________________.
 * \param[out] out_c2n_v - _______________________.
 * \param[out] out_ctype - _______________________.
 * \param[out] out_ic2icg - _______________________.
 * \param[out] out_iv2ivg - _______________________.
 * \param[out] out_n_cells - _______________________.
 * \param[out] out_n_verts - _______________________.
 * \param[in] FlowSol - Structure with the entire solution and mesh information.
 */
void ReadMesh(string& in_file_name, array<double>& out_xv, array<int>& out_c2v, array<int>& out_c2n_v, array<int>& out_ctype, array<int>& out_ic2icg, array<int>& out_iv2ivg, int& out_n_cells, int& out_n_verts, struct solution* FlowSol) ;

/* method to read boundaries from mesh */
void ReadBound(string& in_file_name, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_ctype, array<int>& out_bctype, array<int>& in_ic2icg, array<int>& in_icvsta, array<int>&in_icvert, array<int>& in_iv2ivg, int& in_n_cells, int& in_n_verts, struct solution* FlowSol);

/* method to read position vertices in a gambit mesh */
void read_vertices_gambit(string& in_file_name, int in_n_verts, array<int> &in_iv2ivg, array<double> &out_xv, struct solution* FlowSol);

void read_vertices_gmsh(string& in_file_name, int in_n_verts, array<int> &in_iv2ivg, array<double> &out_xv, struct solution* FlowSol);

void create_iv2ivg(array<int> &inout_iv2ivg, array<int> &inout_c2v, int &out_n_verts, int in_n_cells);

/* method to read cell connectivity in a gambit mesh */
void read_connectivity_gambit(string& in_file_name, int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol);

/* method to read cell connectivity in a gmsh mesh */
void read_connectivity_gmsh(string& in_file_name, int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol);

/* method to read boundary faces in a gambit mesh */
void read_boundary_gambit(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& out_bctype);

void read_boundary_gmsh(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& out_bctype, array<int>& in_iv2ivg, int in_n_verts, array<int>& in_ctype, array<int>& in_icvsta, array<int>& in_icvert, struct solution* FlowSol);

/*! method to create list of faces from the mesh */
void CompConnectivity(array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_ctype, array<int>& out_c2f, array<int>& out_c2e, array<int>& out_f2c, array<int>& out_f2loc_f, array<int>& out_f2v, array<int>& out_f2nv, array<int>& out_rot_tag, array<int>& out_unmatched_faces, int& out_n_unmatched_faces, array<int>& out_icvsta, array<int>& out_icvert, int& out_n_faces, int& out_n_edges, struct solution* FlowSol);

/*! Method that returns list of local vertices associated to a particular local face */
void get_vlist_loc_face(int& in_ctype, int& in_nspt, int& in_face, array<int>& out_vlist_loc, int& num_v_per_f);

/*! Method that returns list of local vertices associated to a particular local edge*/
void get_vlist_loc_edge(int& in_ctype, int& in_nspt, int& in_edge, array<int>& out_vlist_loc);

/*! Method that return the shape point number associated with a vertex */
void get_vert_loc(int& in_ctype, int& in_nspts, int& in_vert, int& out_v);

/*! Method that compares the vertices from two faces to check if they match */
void compare_faces(array<int>& vlist1, array<int>& vlist2, int& num_v_per_f, int& found, int& rtag);

void compare_faces_boundary(array<int>& vlist1, array<int>& vlist2, int& num_v_per_f, int& found);

/*! Method that compares two cyclic faces and check if they should be matched */
void compare_cyclic_faces(array<double> &xvert1, array<double> &xvert2, int& num_v_per_f, int& rtag, array<double> &delta_cyclic, double tol, struct solution* FlowSol);

/*! Method that checks if two cyclic faces are distance delta_cyclic apart */
bool check_cyclic(array<double> &delta_cyclic, array<double> &loc_center_inter_0, array<double> &loc_center_inter_1, double tol, struct solution* FlowSol);

int get_bc_number(string& bcname);

#ifdef _MPI

/* method to repartition a mesh using ParMetis */
void repartition_mesh(int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol);

void match_mpifaces(array<int> &in_f2v, array<int> &in_f2nv, array<double>& in_xv, array<int>& inout_f_mpi2f, array<int>& out_mpifaces_part, array<double> &delta_cyclic, int n_mpi_faces, double tol, struct solution* FlowSol);

void find_rot_mpifaces(array<int> &in_f2v, array<int> &in_f2nv, array<double>& in_xv, array<int>& in_f_mpi2f, array<int> &out_rot_tag_mpi, array<int> &mpifaces_part, array<double> delta_cyclic, int n_mpi_faces, double tol, struct solution* FlowSol);

void compare_mpi_faces(array<double> &xvert1, array<double> &xvert2, int& num_v_per_f, int& rtag, array<double> &delta_cyclic, double tol, struct solution* FlowSol);

#endif
