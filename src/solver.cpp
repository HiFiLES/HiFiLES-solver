/*!
 * \file solver.cpp
 * \brief _____________________________
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL) directed
 *                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
 * \version 1.0.0
 *
 * HiFiLES (High Fidelity Large Eddy Simulation).
 * Copyright (C) 2013 Aerospace Computing Laboratory.
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
#include "../include/cuda_kernels.h"

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

void CalcResidual(struct solution* FlowSol) {

  int in_disu_upts_from = 0;        /*!< Define... */
  int in_div_tconf_upts_to = 0;     /*!< Define... */
  int i;                            /*!< Loop iterator */

  /*! If using LES, filter the solution prior to everything else. If using Similarity
   LES model or explicit SVV filtering as a 'model'. */
  if(run_input.LES==1) {
      if(run_input.SGS_model==2 || run_input.SGS_model==3 || run_input.SGS_model==4) {
          for(i=0; i<FlowSol->n_ele_types; i++)
            FlowSol->mesh_eles(i)->calc_disuf_upts(in_disu_upts_from);
        }
    }

  /*! Compute the solution at the flux points. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calc_disu_fpts(in_disu_upts_from);

#ifdef _MPI
  /*! Send the solution at the flux points across the MPI interfaces. */
  if (FlowSol->nproc>1)
    for(i=0; i<FlowSol->n_mpi_inter_types; i++)
      FlowSol->mesh_mpi_inters(i).send_disu_fpts();
#endif

  if (FlowSol->viscous) {
      /*! Compute the uncorrected gradient of the solution at the solution points. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->calc_uncor_tgrad_disu_upts(in_disu_upts_from);
    }

  /*! Compute the inviscid flux at the solution points and store in total flux storage. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calc_tdisinvf_upts(in_disu_upts_from);

  /*! Calculate body forcing, if switched on, and add to flux. */
  if(run_input.equation==0 && run_input.run_type==0 && run_input.forcing==1) {
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->add_body_force_upts(FlowSol->body_force);
    }

  /*! Compute the inviscid numerical fluxes.
   Compute the common solution and solution corrections (viscous only). */
  for(i=0; i<FlowSol->n_int_inter_types; i++)
    FlowSol->mesh_int_inters(i).calc_norm_tconinvf_fpts();

  for(i=0; i<FlowSol->n_bdy_inter_types; i++)
    FlowSol->mesh_bdy_inters(i).calc_norm_tconinvf_fpts_boundary(FlowSol->time);

#ifdef _MPI
  /*! Send the previously computed values across the MPI interfaces. */
  if (FlowSol->nproc>1) {
      for(i=0; i<FlowSol->n_mpi_inter_types; i++)
        FlowSol->mesh_mpi_inters(i).receive_disu_fpts();

      for(i=0; i<FlowSol->n_mpi_inter_types; i++)
        FlowSol->mesh_mpi_inters(i).calc_norm_tconinvf_fpts_mpi();
    }
#endif

  if (FlowSol->viscous) {
      /*! Compute corrected gradient of the solution at the solution and flux points. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->calc_cor_grad_disu_upts();

      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->calc_cor_grad_disu_fpts();

#ifdef _MPI
      /*! Send the corrected value across the MPI interface. */
      if (FlowSol->nproc>1) {
          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).send_cor_grad_disu_fpts();
        }
#endif

#ifdef _GPU

      /*! Compute element-wise artificial viscosity co-efficients */
      for(i=0;i<FlowSol->n_ele_types;i++)
        FlowSol->mesh_eles(i)->calc_artivisc_coeff(in_disu_upts_from, FlowSol->epsilon_global_eles.get_ptr_gpu());

      /*! Compute vertex-wise artificial viscosity co-efficients for enforcing C0-continuity */
      calc_artivisc_coeff_verts(FlowSol);

      /*! Compute artificial viscosity co-efficients at solution and flux points */
      for(i=0;i<FlowSol->n_ele_types;i++)
        FlowSol->mesh_eles(i)->calc_artivisc_coeff_upts_fpts(FlowSol->epsilon_verts.get_ptr_gpu(), FlowSol->ele2vert.get_ptr_gpu(), FlowSol->num_eles);

#ifdef _MPI
  /*! Send the solution at the flux points across the MPI interfaces. */
  if (FlowSol->nproc>1){
    for(i=0; i<FlowSol->n_mpi_inter_types; i++)
      FlowSol->mesh_mpi_inters(i).send_epsilon_fpts();

    for(i=0; i<FlowSol->n_mpi_inter_types; i++)
      FlowSol->mesh_mpi_inters(i).receive_epsilon_fpts();
  }
#endif

#endif

      /*! Compute discontinuous viscous flux at upts and add to inviscid flux at upts. */
      for(i=0; i<FlowSol->n_ele_types; i++)
        FlowSol->mesh_eles(i)->calc_tdisvisf_upts(in_disu_upts_from);
    }

  /*! For viscous or inviscid, compute the divergence of flux at solution points. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calc_div_tdisf_upts(in_div_tconf_upts_to);

  /*! For viscous or inviscid, compute the normal discontinuous flux at flux points. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calc_norm_tdisf_fpts();

  if (FlowSol->viscous) {
      /*! Compute normal interface viscous flux and add to normal inviscid flux. */
      for(i=0; i<FlowSol->n_int_inter_types; i++)
        FlowSol->mesh_int_inters(i).calc_norm_tconvisf_fpts();

      for(i=0; i<FlowSol->n_bdy_inter_types; i++)
        FlowSol->mesh_bdy_inters(i).calc_norm_tconvisf_fpts_boundary(FlowSol->time);

#if _MPI
      /*! Evaluate the MPI interfaces. */
      if (FlowSol->nproc>1) {
          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).receive_cor_grad_disu_fpts();

          for(i=0; i<FlowSol->n_mpi_inter_types; i++)
            FlowSol->mesh_mpi_inters(i).calc_norm_tconvisf_fpts_mpi();
        }
#endif
    }

  /*! Compute the divergence of the transformed continuous flux. */
  for(i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->calc_div_tconf_upts(in_div_tconf_upts_to);

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

// get pointer to AV co-efficients at solution points
double* get_epsilon_ptr(int in_ele_type, int in_ele, struct solution* FlowSol)
{
        return FlowSol->mesh_eles(in_ele_type)->get_epsilon_ptr(in_ele);
}

// get pointer to AV co-efficient at flux points
double* get_epsilon_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
        return FlowSol->mesh_eles(in_ele_type)->get_epsilon_fpts_ptr(in_ele, in_local_inter, in_fpt);
}

// get pointer to normal continuous transformed inviscid flux at a flux point

double* get_norm_tconf_fpts_ptr(int in_ele_type, int in_ele, int in_field, int in_local_inter, int in_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_norm_tconf_fpts_ptr(in_fpt,in_local_inter,in_field,in_ele);
}

// get pointer to determinant of jacobian at a flux point

double* get_detjac_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_detjac_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}

// get pointer to magntiude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at a flux point

double* get_mag_tnorm_dot_inv_detjac_mul_jac_fpts_ptr(int in_ele_type, int in_ele, int in_ele_local_inter, int in_inter_local_fpt, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_mag_tnorm_dot_inv_detjac_mul_jac_fpts_ptr(in_inter_local_fpt,in_ele_local_inter,in_ele);
}

// get pointer to the normal at a flux point

double* get_norm_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_norm_fpts_ptr(in_fpt,in_local_inter,in_dim,in_ele);
}

// get pointer to the coordinates at a flux point

double* get_loc_fpts_ptr(int in_ele_type, int in_ele, int in_local_inter, int in_fpt, int in_dim, struct solution* FlowSol)
{
  return FlowSol->mesh_eles(in_ele_type)->get_loc_fpts_ptr(in_fpt,in_local_inter,in_dim,in_ele);
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
  cout << "rank=" << FlowSol->rank << " reading restart info" << endl;

  for (int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

          //cout << "Ele_type=" << i << "Reading restart file ";

          for (int j=0;j<in_n_files;j++)
            {
              cout << j << " ";
              sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,j);
              file_name = &file_name_s[0];
              cout<<"restart file name: "<<file_name_s<<endl;
              restart_file.open(file_name);
              if (!restart_file)
                FatalError("Could not open restart file ");

              restart_file >> FlowSol->time;

              int info_found = FlowSol->mesh_eles(i)->read_restart_info(restart_file);
              restart_file.close();

              if (info_found)
                break;
            }
          cout << endl;
        }
    }
  cout << "Rank=" << FlowSol->rank << " Done reading restart info" << endl;

  // Now open all the restart files one by one and store data belonging to you

  for (int j=0;j<in_n_files;j++)
    {
      //cout <<  "Reading restart file " << j << endl;
      sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,j);
      file_name = &file_name_s[0];
      restart_file.open(file_name);

      if (restart_file.fail())
        FatalError(strcat("Could not open restart file ",file_name));

      for (int i=0;i<FlowSol->n_ele_types;i++)  {
          if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

              FlowSol->mesh_eles(i)->read_restart_data(restart_file);

            }
        }

      restart_file.close();
    }
  cout << "Rank=" << FlowSol->rank << " Done reading restart data" << endl;


}

#ifdef _GPU
// Uses elemental artificial viscosity co-efficients to compute viscosity co-efficients at vertices
void calc_artivisc_coeff_verts(struct solution* FlowSol)
{
  calc_artivisc_coeff_verts_gpu_kernel_wrapper(FlowSol->num_verts, FlowSol->icvert.get_ptr_gpu(), FlowSol->icvsta.get_ptr_gpu(), FlowSol->epsilon_global_eles.get_ptr_gpu(), FlowSol->epsilon_verts.get_ptr_gpu());
}
#endif
