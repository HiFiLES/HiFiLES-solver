/*!
 * \file solver.cpp
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

#include <iostream>
#include <sstream>
#include <cmath>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/input.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/funcs.h"
#include "../include/error.h"
#include "../include/solution.h"

#ifdef _TECIO
#include "TECIO.h"
#endif

#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif

#ifdef _GPU
#include "../include/util.h"
#endif

using namespace std;

#define MAX_V_PER_F 4
#define MAX_F_PER_C 6
#define MAX_E_PER_C 12
#define MAX_V_PER_C 27

// used to switch between single- and multi-zone tecplot binary output
#define MULTI_ZONE
//#define SINGLE_ZONE

void CalcResidual(int in_file_num, int in_rk_stage, struct solution* FlowSol) {

  int in_disu_upts_from = 0;        /*!< Define... */
  int in_div_tconf_upts_to = 0;     /*!< Define... */
  int i;                            /*!< Loop iterator */

  /*! If at first RK step and using certain LES models, compute some model-related quantities. */
  if(run_input.LES==1 && in_disu_upts_from==0) {
      if(run_input.SGS_model==2 || run_input.SGS_model==3 || run_input.SGS_model==4) {
          for(i=0; i<FlowSol->n_ele_types; i++)
            FlowSol->mesh_eles(i)->calc_sgs_terms(in_disu_upts_from);
        }
    }

  /*! Shock capturing part - only concentration method on GPU and on quads for now */
  /*! TO be added: Persson's method for triangles with artificial viscosity structure */

  if(run_input.ArtifOn) {

    // #ifdef _GPU

      if(run_input.artif_type == 1){
          /*! This routine does shock detection. For concentration method filter is also applied in this routine itself */
          for(i=0;i<FlowSol->n_ele_types;i++)
            FlowSol->mesh_eles(i)->shock_capture_concentration(in_disu_upts_from);
      }

    // #endif
  }

  /*! Compute the solution at the flux points. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->extrapolate_solution(in_disu_upts_from);

#ifdef _MPI
  /*! Send the solution at the flux points across the MPI interfaces. */
  if (FlowSol->nproc>1)
    for(i=0; i<FlowSol->n_mpi_inter_types; i++)
      FlowSol->mesh_mpi_inters(i).send_solution();
#endif

  if (FlowSol->viscous) {
      /*! Compute the uncorrected gradient of the solution at the solution points. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->calculate_gradient(in_disu_upts_from);
    }

  /*! Compute the inviscid flux at the solution points and store in total flux storage. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->evaluate_invFlux(in_disu_upts_from);


  // If running periodic channel or periodic hill cases,
  // calculate body forcing and add to source term
  if(run_input.forcing==1 and in_rk_stage==0 and run_input.equation==0 and FlowSol->n_dims==3) {

#ifdef _GPU
  // copy disu_upts for body force calculation
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
#endif

    for(i=0;i<FlowSol->n_ele_types;i++) {
      FlowSol->mesh_eles(i)->evaluate_body_force(in_file_num);
    }
  }

  /*! Compute the inviscid numerical fluxes.
   Compute the common solution and solution corrections (viscous only). */
  for(i=0; i<FlowSol->n_int_inter_types; i++)
    FlowSol->mesh_int_inters(i).calculate_common_invFlux();

  for(i=0; i<FlowSol->n_bdy_inter_types; i++)
    FlowSol->mesh_bdy_inters(i).evaluate_boundaryConditions_invFlux(FlowSol->time);

#ifdef _MPI
  /*! Send the previously computed values across the MPI interfaces. */
  if (FlowSol->nproc>1) {
      for(i=0; i<FlowSol->n_mpi_inter_types; i++)
        FlowSol->mesh_mpi_inters(i).receive_solution();

      for(i=0; i<FlowSol->n_mpi_inter_types; i++)
        FlowSol->mesh_mpi_inters(i).calculate_common_invFlux();
    }
#endif

  if (FlowSol->viscous) {
      /*! Compute corrected gradient of the solution at the solution and flux points. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->correct_gradient();

      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->extrapolate_corrected_gradient();

#ifdef _MPI
      /*! Send the corrected value and SGS flux across the MPI interface. */
      if (FlowSol->nproc>1) {
          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).send_corrected_gradient();

          if (run_input.LES) {
            for(i=0; i<FlowSol->n_mpi_inter_types; i++)
              FlowSol->mesh_mpi_inters(i).send_sgsf_fpts();
          }
        }
#endif

      /*! Compute discontinuous viscous flux at upts and add to inviscid flux at upts. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->evaluate_viscFlux(in_disu_upts_from);
    }

  /*! If using LES, compute the SGS flux at flux points. */
  if (run_input.LES) {
	  for(i=0; i<FlowSol->n_ele_types; i++)
			FlowSol->mesh_eles(i)->evaluate_sgsFlux();
  }

  /*! For viscous or inviscid, compute the normal discontinuous flux at flux points. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->extrapolate_totalFlux();

  /*! For viscous or inviscid, compute the divergence of flux at solution points. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calculate_divergence(in_div_tconf_upts_to);

  if (FlowSol->viscous) {
      /*! Compute normal interface viscous flux and add to normal inviscid flux. */
      for(i=0; i<FlowSol->n_int_inter_types; i++)
        FlowSol->mesh_int_inters(i).calculate_common_viscFlux();

      for(i=0; i<FlowSol->n_bdy_inter_types; i++)
        FlowSol->mesh_bdy_inters(i).evaluate_boundaryConditions_viscFlux(FlowSol->time);

#if _MPI
      /*! Evaluate the MPI interfaces. */
      if (FlowSol->nproc>1) {
          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).receive_corrected_gradient();

          if (run_input.LES) {
            for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).receive_sgsf_fpts();
          }

          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).calculate_common_viscFlux();
        }
#endif
    }

  /*! Compute the divergence of the transformed continuous flux. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calculate_corrected_divergence(in_div_tconf_upts_to);

  /*! Compute source term */
  if (run_input.turb_model==1) {
    for (i=0; i<FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->calc_src_upts_SA(in_disu_upts_from);
  }
}

#ifdef _MPI
void set_rank_nproc(int in_rank, int in_nproc, struct solution* FlowSol)
{
  FlowSol->rank = in_rank;
  FlowSol->nproc = in_nproc;
  FlowSol->error_states.setup(FlowSol->nproc);
}
#endif

// get pointer to transformed discontinuous solution at a flux point

double* get_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_disu_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
}

// get pointer to normal continuous transformed inviscid flux at a flux point

double* get_norm_tconf_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_norm_tconf_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
}

// get pointer to subgrid-scale flux at a flux point

double* get_sgsf_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol)
{
	return FlowSol->mesh_eles(in_ele_type)->get_sgsf_fpts_ptr(in_fpt,in_local_inter,in_field,in_dim,in_ele);
}

// get pointer to determinant of jacobian at a flux point

double* get_detjac_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_detjac_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}

// get pointer to determinant of jacobian at a flux point (dynamic->static)

double* get_detjac_dyn_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_detjac_dyn_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}

// get pointer to magntiude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at a flux point

double* get_tdA_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_tdA_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}

// get pointer to the equivalent of 'dA' (face area) at a flux point in dynamic physical space
double* get_ndA_dyn_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_ndA_dyn_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}

// get pointer to the normal at a flux point

double* get_norm_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_norm_fpts_ptr(in_fpt,in_local_inter,in_dim,in_ele);
}

// get pointer to the normal at a flux point in the dynamic space

double* get_norm_dyn_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_norm_dyn_fpts_ptr(in_fpt,in_local_inter,in_dim,in_ele);
}

// get CPU pointer to the coordinates at a flux point.
// See bdy_inters for reasons for this CPU/GPU split.

double* get_loc_fpts_ptr_cpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_loc_fpts_ptr_cpu(in_fpt,in_local_inter,in_dim,in_ele);
}

// get GPU pointer to the coordinates at a flux point

double* get_loc_fpts_ptr_gpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_loc_fpts_ptr_gpu(in_fpt,in_local_inter,in_dim,in_ele);
}

// get CPU pointer to the physical dynamic coordinates at a flux point.

double* get_pos_dyn_fpts_ptr_cpu(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_pos_dyn_fpts_ptr_cpu(in_fpt,in_local_inter,in_dim,in_ele);
}

// get pointer to normal continuous transformed viscous flux at a flux point

//double* get_norm_tconvisf_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt)
//{
//	return mesh_eles(in_ele_type)->get_norm_tconvisf_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
//}

// get pointer to delta of the transformed discontinuous solution at a flux point

double* get_delta_disu_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_delta_disu_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
}

// get pointer to gradient of the discontinuous solution at a flux point
double* get_grad_disu_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_dim, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_grad_disu_fpts_ptr(in_fpt,in_local_inter,in_dim,in_field,in_ele);
}

// get pointer to the discontinuous solution (close normal) at a flux point
double* get_normal_disu_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_field, int in_fpt, struct solution* FlowSol, array<double> temp_loc, double temp_pos[3])
{
  return FlowSol->mesh_eles(in_ele_type)->get_normal_disu_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele, temp_loc, temp_pos);
}

// get pointer to the grid velocity at a flux point
double* get_grid_vel_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_grid_vel_fpts_ptr(in_ele,in_local_inter,in_fpt,in_dim);
}

void InitSolution(struct solution* FlowSol)
{
  // set initial conditions
  if (FlowSol->rank==0) cout << "Setting initial conditions... " << endl;

  if (run_input.restart_flag==0) {
      for(int i=0;i<FlowSol->n_ele_types;i++) {
          if (FlowSol->mesh_eles(i)->get_n_eles()!=0)

            FlowSol->mesh_eles(i)->set_ics(FlowSol->time);
        }

      FlowSol->time = 0.;
    }
  else
    {
      FlowSol->ini_iter = run_input.restart_iter;
      read_restart(run_input.restart_iter,run_input.n_restart_files,FlowSol);
    }

  for (int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
          FlowSol->mesh_eles(i)->set_disu_upts_to_zero_other_levels();
        }
    }

  // copy solution to gpu
#ifdef _GPU
  for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
          FlowSol->mesh_eles(i)->cp_disu_upts_cpu_gpu();

        }
    }
#endif

}

void read_restart(int in_file_num, int in_n_files, struct solution* FlowSol)
{

  char file_name_s[50];
  char *file_name;
  ifstream restart_file;
  restart_file.precision(15);

  // Open the restart files and read info

  for (int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

          for (int j=0;j<in_n_files;j++)
            {
              sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,j);
              file_name = &file_name_s[0];
              restart_file.open(file_name);
              if (!restart_file)
                FatalError("Could not open restart file ");

              restart_file >> FlowSol->time;

              int info_found = FlowSol->mesh_eles(i)->read_restart_info(restart_file);
              restart_file.close();

              if (info_found)
                break;
            }
        }
    }

  // Now open all the restart files one by one and store data belonging to you

  for (int j=0;j<in_n_files;j++)
    {
      //cout <<  "Reading restart file " << j << endl;
      sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,j);
      file_name = &file_name_s[0];
      restart_file.open(file_name);

      if (restart_file.fail())
        FatalError(strcat((char *)"Could not open restart file ",file_name));

      for (int i=0;i<FlowSol->n_ele_types;i++)  {
          if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

              FlowSol->mesh_eles(i)->read_restart_data(restart_file);

            }
        }
      restart_file.close();
    }
  cout << "Rank=" << FlowSol->rank << " Done reading restart files" << endl;
}

