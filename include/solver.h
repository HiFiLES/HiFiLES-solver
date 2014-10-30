/*!
 * \file solver.h
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

#include "array.h"
#include <string>
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
 * \brief Calculate the residual.
 * \param[in] FlowSol - Structure with the entire solution and mesh information.
 */
void CalcResidual(int in_file_num, int in_rk_stage, struct solution* FlowSol);

void set_rank_nproc(int in_rank, int in_nproc, struct solution* FlowSol);

/*! get pointer to transformed discontinuous solution at a flux point */
double* get_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int n_local_inter, int in_fpt, struct solution* FlowSol);

/*! get pointer to normal continuous transformed inviscid flux at a flux point */
double* get_norm_tconf_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol);

/*! get pointer to subgrid-scale flux at a flux point */
double* get_sgsf_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol);

/*! get pointer to determinant of jacobian at a flux point */
double* get_detjac_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to determinant of jacobian at a flux point (dynamic->static) */
double* get_detjac_dyn_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to magntiude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at a solution point */
double* get_tdA_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to the equivalent of 'dA' (face area) at a flux point in dynamic physical space */
double* get_ndA_dyn_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol);

/*! get pointer to normal at a flux point */
double* get_norm_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

/*! get pointer to normal at a flux point in dynamic space */
double* get_norm_dyn_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

/*! get CPU pointer to coordinates at a flux point */
double* get_loc_fpts_ptr_cpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

/*! get GPU pointer to coordinates at a flux point */
double* get_loc_fpts_ptr_gpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

/*! get CPU pointer to coordinates at a flux point */
double* get_pos_dyn_fpts_ptr_cpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

/*! get pointer to delta of the transformed discontinuous solution at a flux point */
double* get_delta_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int n_local_inter, int in_fpt, struct solution* FlowSol);

/*! get pointer to gradient of the discontinuous solution at a flux point */
double* get_grad_disu_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol);

/*! get pointer to the closest normal point of the discontinuous solution at a flux point */
double* get_normal_disu_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_fpt, struct solution* FlowSol, array<double> temp_loc, double temp_pos[3]);

/*! get pointer to the grid velocity at a flux point */
double* get_grid_vel_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol);

// Initialize the solution in the mesh
void InitSolution(struct solution* FlowSol);

/*! reading a restart file */
void read_restart(int in_file_num, int in_n_files, struct solution* FlowSol);






