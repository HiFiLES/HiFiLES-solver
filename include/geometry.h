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
#include "mesh.h"

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
 * \param[in] Mesh - Structure containing many details of the mesh
 */
void GeoPreprocess(struct solution* FlowSol, mesh &Mesh);

/*!
 * \brief Method to read a mesh.
 * \param[in] in_file_name - Name of mesh file to read.
 * \param[out] out_xv - Array of physical vertex locations (x,y,z).
 * \param[out] out_c2v - ID of vertices making up each cell.
 * \param[out] out_c2n_v - Number of vertices in each cell.
 * \param[out] out_ctype - Cell type.
 * \param[out] out_ic2icg - Index of cell on processor to index of cell globally.
 * \param[out] out_iv2ivg - Index of vertex on processor to index of vertex globally.
 * \param[out] out_n_cells - Number of cells assigned to processor.
 * \param[out] out_n_verts - Number of vertices assigned to processor.
 * \param[in] FlowSol - Structure with the entire solution and mesh information.
 */
void ReadMesh(string& in_file_name, array<double>& out_xv, array<int>& out_c2v, array<int>& out_c2n_v, array<int>& out_ctype, array<int>& out_ic2icg,
              array<int>& out_iv2ivg, int& out_n_cells, int& out_n_verts, int& out_n_verts_global, struct solution* FlowSol);

/*! method to read boundaries from mesh */
void ReadBound(string& in_file_name, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_c2f, array<int>& in_f2v, array<int>& in_f2nv,
               array<int>& in_ctype, array<int>& out_bctype, array<array<int> >& out_boundpts, array<int> &out_bc_list, array<int> &out_bound_flag,
               array<int>& in_ic2icg, array<int>& in_icvsta, array<int> &in_icvert, array<int> &in_iv2ivg, int &in_n_cells, int &in_n_verts, solution *FlowSol);

/*! method to read position vertices in a gambit mesh */
void read_vertices_gambit(string& in_file_name, int in_n_verts, int &out_n_verts_global, array<int> &in_iv2ivg, array<double> &out_xv, solution *FlowSol);

/*! method to read position vertices in a gmsh mesh */
void read_vertices_gmsh(string& in_file_name, int in_n_verts, int& out_n_verts_global, array<int> &in_iv2ivg, array<double> &out_xv, struct solution* FlowSol);

/*!
 * \brief Method to setup iv2ivg & modify c2v
 * \param[out] inout_iv2ivg - Array to store the global index of each vertex on processor
 * \param[in,out] inout_c2v - On input - stores global vertex indices; on ouput - stores local vertex indicies
 * \param[out] out_n_verts - Number of vertices which belong to the cells on processor
 * \param[in] in_n_cells - Number of cells on processor
 */
void create_iv2ivg(array<int> &inout_iv2ivg, array<int> &inout_c2v, int &out_n_verts, int in_n_cells);

/*! method to read cell connectivity in a gambit mesh */
void read_connectivity_gambit(string& in_file_name, int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol);

/*! method to read cell connectivity in a gmsh mesh */
void read_connectivity_gmsh(string& in_file_name, int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol);

/*! method to read boundary faces in a gambit mesh */
void read_boundary_gambit(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& out_bctype, array<int> &out_bclist, array<array<int> > &out_bccells, array<array<int> > &out_bcfaces);

/*! method to read boundary faces in a gmsh mesh */
void read_boundary_gmsh(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& out_bctype, array<int> &out_bclist, array<int> &out_bound_flag, array<array<int> > &out_boundpts, array<int> &in_iv2ivg, int in_n_verts, array<int>& in_ctype, array<int> &in_icvsta, array<int> &in_icvert, solution *FlowSol);

/*! method to create bounpts array from Gambit reader output (vertex id = boundpts(bcid,i_pt) */
void create_boundpts(array<array<int> >& out_boundpts, array<int> &in_bclist, array<int> &out_bound_flag, array<array<int> >& in_bccells, array<array<int> > &in_bcfaces, array<int>& in_c2f, array<int>& in_f2v, array<int> &in_f2nv);

/*!
 * \brief method to create list of faces from the mesh
 * \param[in] in_c2v - cell to vertex array
 * \param[in] in_c2n_v - cell to # of vertices
 * \param[in] in_ctype - type of cell
 * \param[out] out_c2f - cell to face
 * \param[out] out_c2e - cell to edges
 * \param[out] out_f2c - face to index of right and left cells
 * \param[out] out_f2loc_f - face to cell-local face index of right and left cells
 * \param[out] out_f2v - face to vertices
 * \param[out] out_f2nv - face to number of vertices
 * \param[out] out_rot_tag - local node id within face2 which matches local node 0 of face1
 * \param[out] out_unmatched_faces - faces in mesh which have no match yet
 * \param[out] out_n_unmatched_faces - number of faces which have not been matched yet
 * \param[out] out_icvsta - index of out_icvert corresponding to first cell touching vertex
 * \param[out] out_icvert - 1D list of cells touching each vertex
 * \param[out] out_n_faces - number of faces belonging to processor
 * \param[out] out_n_edges - number of edges belonging to processor
 * \param[in] FlowSol - Structure with the entire solution and mesh information.
 */
void CompConnectivity(array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_ctype, array<int>& out_c2f, array<int>& out_c2e, array<int>& out_f2c, array<int>& out_f2loc_f, array<int>& out_f2v, array<int>& out_f2nv,  array<int>& out_e2v, array<int>& out_v2n_e, array<array<int> >& out_v2e,array<int>& out_rot_tag, array<int>& out_unmatched_faces, int& out_n_unmatched_faces, array<int>& out_icvsta, array<int>& out_icvert, int& out_n_faces, int& out_n_edges, struct solution* FlowSol);

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
