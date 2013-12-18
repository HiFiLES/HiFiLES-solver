/*!
 * \file mpi_inters.cpp
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
#include <cmath>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/inters.h"
#include "../include/mpi_inters.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/flux.h"
#include "../include/error.h"

#ifdef _MPI
#include "mpi.h"
#endif

#if defined _GPU
#include "../include/cuda_kernels.h" 
#endif

using namespace std;

// #### constructors ####

// default constructor

mpi_inters::mpi_inters() { }

mpi_inters::~mpi_inters() { }

// #### methods ####

// setup mpi_inters

void mpi_inters::setup(int in_n_inters, int in_inters_type, int in_run_type)
{
  (*this).setup_inters(in_n_inters,in_inters_type,in_run_type);

  if (in_run_type==0)
  {
    // Allocate memory for out_buffer etc
    out_buffer_disu.setup(in_n_inters*n_fpts_per_inter*n_fields);
    in_buffer_disu.setup(in_n_inters*n_fpts_per_inter*n_fields);

    if (viscous) {
      out_buffer_grad_disu.setup(in_n_inters*n_fpts_per_inter*n_fields*n_dims);
      in_buffer_grad_disu.setup(in_n_inters*n_fpts_per_inter*n_fields*n_dims);
    }

#ifdef _GPU
    // Here, data is copied but is meaningless. Just need to allocate on GPU
    out_buffer_disu.cp_cpu_gpu();
    in_buffer_disu.cp_cpu_gpu();

    if (viscous) {
      out_buffer_grad_disu.cp_cpu_gpu();
      in_buffer_grad_disu.cp_cpu_gpu();
    }
#endif 

	  disu_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields);
	  if(viscous)
	  {
	  	grad_disu_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
	  }
  }
}

void mpi_inters::set_nproc(int in_nproc, int in_rank)
{
  nproc = in_nproc;
  rank = in_rank;

  Nout_proc.setup(nproc);

  // Initialize Nout_proc to 0
  for (int p=0;p<in_nproc;p++) {
      Nout_proc(p)=0;
  } 

}

void mpi_inters::set_nout_proc(int in_nout,int in_p)
{
  Nout_proc(in_p) = in_nout;
}

void mpi_inters::set_mpi_requests(int in_number_of_requests)
{
#ifdef _MPI
  mpi_in_requests = (MPI_Request*) malloc(in_number_of_requests*sizeof(MPI_Request));
  mpi_out_requests = (MPI_Request*) malloc(in_number_of_requests*sizeof(MPI_Request));
#endif
  if (viscous)
  {
#ifdef _MPI
    mpi_in_requests_grad = (MPI_Request*) malloc(in_number_of_requests*sizeof(MPI_Request));
    mpi_out_requests_grad = (MPI_Request*) malloc(in_number_of_requests*sizeof(MPI_Request));
#endif
  }

}

// move all from cpu to gpu

void mpi_inters::mv_all_cpu_gpu(void)
{
	#ifdef _GPU
	
	disu_fpts_l.mv_cpu_gpu();
	norm_tconf_fpts_l.mv_cpu_gpu();
	//detjac_fpts_l.mv_cpu_gpu();
	mag_tnorm_dot_inv_detjac_mul_jac_fpts_l.mv_cpu_gpu();
	norm_fpts.mv_cpu_gpu();

  disu_fpts_r.mv_cpu_gpu();

  delta_disu_fpts_l.mv_cpu_gpu();

	if(viscous)
	{

		grad_disu_fpts_l.mv_cpu_gpu();
		grad_disu_fpts_r.mv_cpu_gpu();

		//norm_tconvisf_fpts_l.mv_cpu_gpu();

	}

	#endif
}


void mpi_inters::set_mpi(int in_inter, int in_ele_type_l, int in_ele_l, int in_local_inter_l, int rot_tag, int in_run_type, struct solution* FlowSol)
{
	int i,j,k;
	int i_rhs,j_rhs;

  if (in_run_type==0)
  {
	  get_lut(rot_tag);

	  for(j=0;j<n_fpts_per_inter;j++)
	  {
	    for(i=0;i<n_fields;i++)
	    {
	  		j_rhs=lut(j);

	  		disu_fpts_l(j,in_inter,i)=get_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);

#ifdef _GPU
	  		disu_fpts_r(j,in_inter,i)=in_buffer_disu.get_ptr_gpu(in_inter*n_fpts_per_inter*n_fields+i*n_fpts_per_inter+j_rhs);
#else 
	  		disu_fpts_r(j,in_inter,i)=in_buffer_disu.get_ptr_cpu(in_inter*n_fpts_per_inter*n_fields+i*n_fpts_per_inter+j_rhs);
#endif
	  	
	  		norm_tconf_fpts_l(j,in_inter,i)=get_norm_tconf_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
	  		
	  		if(viscous)
	  		{
	  			delta_disu_fpts_l(j,in_inter,i)=get_delta_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
	  		
	    		for(k=0; k<n_dims; k++) {
	    				grad_disu_fpts_l(j,in_inter,i,k) = get_grad_disu_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,k,j,FlowSol);
#ifdef _GPU
	    				grad_disu_fpts_r(j,in_inter,i,k) = in_buffer_grad_disu.get_ptr_gpu(in_inter*n_fpts_per_inter*n_fields*n_dims+k*n_fpts_per_inter*n_fields+i*n_fpts_per_inter+j_rhs);
#else
	    				grad_disu_fpts_r(j,in_inter,i,k) = in_buffer_grad_disu.get_ptr_cpu(in_inter*n_fpts_per_inter*n_fields*n_dims+k*n_fpts_per_inter*n_fields+i*n_fpts_per_inter+j_rhs);
#endif
	        }
	  		}
	  	}
	  }

	  for(i=0;i<n_fpts_per_inter;i++)
	  {	
	  	i_rhs=lut(i);
	  	
	  	mag_tnorm_dot_inv_detjac_mul_jac_fpts_l(i,in_inter)=get_mag_tnorm_dot_inv_detjac_mul_jac_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,FlowSol);
	  	
	  	for(j=0;j<n_dims;j++)
	  	{
	  		norm_fpts(i,in_inter,j)=get_norm_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,j,FlowSol);
	  	}
	  }	

  }

}


void mpi_inters::send_disu_fpts()
{

  if (n_inters!=0)
  {
	  // Pack out_buffer
#ifdef _CPU
	  int counter = 0;
	  for(int i=0;i<n_inters;i++) 
	  	for(int k=0;k<n_fields;k++) 
	  		for(int j=0;j<n_fpts_per_inter;j++)
	  			out_buffer_disu(counter++) = (*disu_fpts_l(j,i,k));
#endif 
#ifdef _GPU
    pack_out_buffer_disu_gpu_kernel_wrapper(n_fpts_per_inter,n_inters,n_fields,disu_fpts_l.get_ptr_gpu(),out_buffer_disu.get_ptr_gpu());

    // copy buffer from GPU to CPU
    out_buffer_disu.cp_gpu_cpu();

#endif

	  // Initiate mpi_send
	  Nmess = 0;
	  int sk = 0;
    int Nout;
    int request_count=0;
	  for (int p=0;p<nproc;p++) {
	  	Nout = Nout_proc(p)*n_fpts_per_inter*n_fields;
      //cout << "rank=" << rank << "p=" << p << "inters_type=" << inters_type << "Nout = " << Nout << endl;
	  	if (Nout) {
#ifdef _MPI
	  		MPI_Isend(out_buffer_disu.get_ptr_cpu(sk),Nout,MPI_DOUBLE,p,inters_type*10000+p   ,MPI_COMM_WORLD,&mpi_out_requests[request_count]);
	  		MPI_Irecv(in_buffer_disu.get_ptr_cpu(sk),Nout,MPI_DOUBLE,p,inters_type*10000+rank,MPI_COMM_WORLD,&mpi_in_requests[request_count]);
#endif
	  		sk+=Nout;
	  		Nmess++;
        request_count++;
	  	}
	  }

  }
}

void mpi_inters::receive_disu_fpts()
{

  if (n_inters!=0) {
	  // Receive in_buffer
#ifdef _MPI
	  MPI_Waitall(Nmess,mpi_in_requests,MPI_STATUSES_IGNORE);
	  MPI_Waitall(Nmess,mpi_out_requests,MPI_STATUSES_IGNORE);
#endif
#ifdef _GPU
    in_buffer_disu.cp_cpu_gpu();
#endif

  }  

}

void mpi_inters::send_cor_grad_disu_fpts()
{
  if (n_inters!=0)
  {
#ifdef _CPU
	  // Pack out_buffer
	  int counter = 0;
	  for(int i=0;i<n_inters;i++) 
      for (int m=0;m<n_dims;m++)
	  	  for(int k=0;k<n_fields;k++) 
	  		  for(int j=0;j<n_fpts_per_inter;j++)
	  			  out_buffer_grad_disu(counter++) = (*grad_disu_fpts_l(j,i,k,m));
#endif 

#ifdef _GPU
    if (n_inters!=0)
      pack_out_buffer_grad_disu_gpu_kernel_wrapper(n_fpts_per_inter,n_inters,n_fields,n_dims,grad_disu_fpts_l.get_ptr_gpu(),out_buffer_grad_disu.get_ptr_gpu());

    // copy buffer from GPU to CPU
    out_buffer_grad_disu.cp_gpu_cpu();
#endif

	  // Pack out_buffer
	  // Initiate mpi_send
	  Nmess = 0;
	  int sk = 0;
    int Nout;
    int request_count=0;
	  for (int p=0;p<nproc;p++) {
	  	Nout = Nout_proc(p)*n_fpts_per_inter*n_fields*n_dims;

	  	if (Nout) {
#ifdef _MPI
	  		MPI_Isend(out_buffer_grad_disu.get_ptr_cpu(sk),Nout,MPI_DOUBLE,p,inters_type*10000+p   ,MPI_COMM_WORLD,&mpi_out_requests_grad[request_count]);
	  		MPI_Irecv(in_buffer_grad_disu.get_ptr_cpu(sk),Nout,MPI_DOUBLE,p,inters_type*10000+rank,MPI_COMM_WORLD,&mpi_in_requests_grad[request_count]);
#endif
	  		sk+=Nout;
	  		Nmess++;
        request_count++;
	  	}
	  }
  }

}

void mpi_inters::receive_cor_grad_disu_fpts()
{
  if (n_inters!=0)
  {
#ifdef _MPI
	  MPI_Waitall(Nmess,mpi_in_requests_grad,MPI_STATUSES_IGNORE);
	  MPI_Waitall(Nmess,mpi_out_requests_grad,MPI_STATUSES_IGNORE);
#endif
#ifdef _GPU
    in_buffer_grad_disu.cp_cpu_gpu();
#endif
  }

}

// calculate normal transformed continuous inviscid flux at the flux points at mpi faces
void mpi_inters::calc_norm_tconinvf_fpts_mpi(void)
{

  #ifdef _CPU
  array<double> norm(n_dims), fn(n_fields);
  array<double> u_c(n_fields);
 
	for(int i=0;i<n_inters;i++)
	{
		for(int j=0;j<n_fpts_per_inter;j++)
		{

			// calculate discontinuous solution at flux points
			for(int k=0;k<n_fields;k++) {
				temp_u_l(k)=(*disu_fpts_l(j,i,k));
				temp_u_r(k)=(*disu_fpts_r(j,i,k));

        //if (rank==1 && i==3)
        //{
        //    cout << "temp_u_l=" << setprecision(10) << temp_u_l(k) << endl;
        //    cout << "temp_u_r=" << setprecision(10) << temp_u_r(k) << endl;
        //}
			}

      // storing normal components
      for (int m=0;m<n_dims;m++)
        norm(m) = *norm_fpts(j,i,m);

      if (run_input.riemann_solve_type==0)
      {
			  // calculate flux from discontinuous solution at flux points
			  if(n_dims==2) {
			  	calc_invf_2d(temp_u_l,temp_f_l);
			  	calc_invf_2d(temp_u_r,temp_f_r);
			  }
			  else if(n_dims==3) {
			  	calc_invf_3d(temp_u_l,temp_f_l);
			  	calc_invf_3d(temp_u_r,temp_f_r);
			  }
			  else
  		  	FatalError("ERROR: Invalid number of dimensions ... ");

        // Calling Riemann solver
        rusanov_flux(temp_u_l,temp_u_r,temp_f_l,temp_f_r,norm,fn,n_dims,n_fields,run_input.gamma);
      }
      else if (run_input.riemann_solve_type==1)
      {
        lax_friedrich(temp_u_l,temp_u_r,norm,fn,n_dims,n_fields,run_input.lambda,run_input.wave_speed);
      }
      else if (run_input.riemann_solve_type==2) { // ROE
        roe_flux(temp_u_l,temp_u_r,norm,fn,n_dims,n_fields,run_input.gamma);
      }
      else
        FatalError("Riemann solver not implemented");

      // Transform back to reference space  
  		for(int k=0;k<n_fields;k++) {
					(*norm_tconf_fpts_l(j,i,k))=fn(k)*(*mag_tnorm_dot_inv_detjac_mul_jac_fpts_l(j,i));
      }
 
      if(viscous)
      {
        // Calling viscous riemann solver
        if (run_input.vis_riemann_solve_type==0)
          ldg_solution(0,temp_u_l,temp_u_r,u_c,run_input.pen_fact,norm);
        else
          FatalError("Viscous Riemann solver not implemented");
		
			  for(int k=0;k<n_fields;k++) {
			  	*delta_disu_fpts_l(j,i,k) = (u_c(k) - temp_u_l(k));

        //if (rank==1 && i==3)
        //{
        //    cout << "delta=" << *delta_tisu_fpts_l(j,i,k) << endl;
        //}

        }
      }

    }
  }
	#endif
		
	#ifdef _GPU
  if (n_inters!=0) {
    calc_norm_tconinvf_fpts_mpi_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),disu_fpts_r.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),mag_tnorm_dot_inv_detjac_mul_jac_fpts_l.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),run_input.riemann_solve_type, delta_disu_fpts_l.get_ptr_gpu(),run_input.gamma,run_input.pen_fact, run_input.viscous, run_input.vis_riemann_solve_type, run_input.wave_speed(0), run_input.wave_speed(1), run_input.wave_speed(2), run_input.lambda);
  }
  #endif

}

void mpi_inters::calc_norm_tconvisf_fpts_mpi(void)
{

#ifdef _CPU

  array<double> norm(n_dims), fn(n_fields);
	
	for(int i=0;i<n_inters;i++)
	{
		for(int j=0;j<n_fpts_per_inter;j++)
		{
			// obtain discontinuous solution at flux points
			
			for(int k=0;k<n_fields;k++)
			{
				temp_u_l(k)=(*disu_fpts_l(j,i,k));
				temp_u_r(k)=(*disu_fpts_r(j,i,k));
			}
			
			// obtain gradient of discontinuous solution at flux points
			
			for(int k=0;k<n_dims;k++)
			{
				for(int l=0;l<n_fields;l++)
				{
					temp_grad_u_l(l,k) = *grad_disu_fpts_l(j,i,l,k);
					temp_grad_u_r(l,k) = *grad_disu_fpts_r(j,i,l,k);

          //if (rank==0) cout << "k=" << k << "l=" << l << "grad_l=" << temp_grad_u_l(l,k) << endl;
          //if (rank==0) cout << "k=" << k << "l=" << l << "grad_r=" << temp_grad_u_r(l,k) << endl;

				}
			}

			// calculate flux from discontinuous solution at flux points
			
			if(n_dims==2)
			{
				calc_visf_2d(temp_u_l,temp_grad_u_l,temp_f_l);
				calc_visf_2d(temp_u_r,temp_grad_u_r,temp_f_r);
			}
			else if(n_dims==3)
			{
				calc_visf_3d(temp_u_l,temp_grad_u_l,temp_f_l);
				calc_visf_3d(temp_u_r,temp_grad_u_r,temp_f_r);
			}
			else
  			FatalError("ERROR: Invalid number of dimensions ... ");


      // storing normal components
      for (int m=0;m<n_dims;m++)
        norm(m) = *norm_fpts(j,i,m);
			
      // Calling viscous riemann solver
      if (run_input.vis_riemann_solve_type==0)
        ldg_flux(0,temp_u_l,temp_u_r,temp_f_l,temp_f_r,norm,fn,n_dims,n_fields,run_input.tau,run_input.pen_fact);
      else
        FatalError("Viscous Riemann solver not implemented");

      // Transform back to reference space  
  		for(int k=0;k<n_fields;k++) {
					(*norm_tconf_fpts_l(j,i,k))+=fn(k)*(*mag_tnorm_dot_inv_detjac_mul_jac_fpts_l(j,i));
      }
			
		}
	}

  //cout << "done viscous mpi" << endl;
#endif

#ifdef _GPU
  if (n_inters!=0)
    calc_norm_tconvisf_fpts_mpi_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),disu_fpts_r.get_ptr_gpu(),grad_disu_fpts_l.get_ptr_gpu(),grad_disu_fpts_r.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),mag_tnorm_dot_inv_detjac_mul_jac_fpts_l.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),run_input.riemann_solve_type,run_input.vis_riemann_solve_type,run_input.pen_fact,run_input.tau,run_input.gamma,run_input.prandtl,run_input.rt_inf,run_input.mu_inf,run_input.c_sth,run_input.fix_vis,run_input.diff_coeff);
#endif
}

