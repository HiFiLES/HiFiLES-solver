/*!
 * \file eles.cpp
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
#include <iomanip>
#include <cmath>

#if defined _ACCELERATE_BLAS
#include <Accelerate/Accelerate.h>
#endif

#if defined _MKL_BLAS
#include "mkl.h"
#include "mkl_spblas.h"
#endif

#if defined _STANDARD_BLAS
extern "C"
{
#include "cblas.h"
}
#endif

#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif

#if defined _GPU
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "../include/cuda_kernels.h"
#endif

#include "../include/global.h"
#include "../include/array.h"
#include "../include/flux.h"
#include "../include/source.h"
#include "../include/eles.h"
#include "../include/funcs.h"

using namespace std;

// #### constructors ####

// default constructor

eles::eles()
{
}

// default destructor

eles::~eles() {}

// #### methods ####

// set number of elements

void eles::setup(int in_n_eles, int in_max_n_spts_per_ele, int in_run_type)
{

	n_eles=in_n_eles;

  if (n_eles!=0)
  {

	order=run_input.order;
	p_res=run_input.p_res;
	viscous = run_input.viscous;
	LES = run_input.LES;
	sgs_model = run_input.SGS_model;
	filter = 0;
	// SVV model requires filtered solution
	if(LES)
		if(sgs_model==3 || sgs_model==2 || sgs_model==4)
			filter = 1;

	wall_model = run_input.wall_model;
  inters_cub_order = run_input.inters_cub_order;
  volume_cub_order = run_input.volume_cub_order;
  n_bdy_eles=0;

  // Initialize the element specific static members
  (*this).setup_ele_type_specific(in_run_type);

  if (in_run_type==0)
  {
	  if(run_input.adv_type==0)
	  {
	  	n_adv_levels=1;
	  }
	  else if(run_input.adv_type==1)
	  {
	  	n_adv_levels=3;
	  }
	  else if(run_input.adv_type==2)
	  {
	  	n_adv_levels=4;
	  }
	  else if(run_input.adv_type==3)
	  {
	  	n_adv_levels=2;
	  }
	  else
	  {
	  	cout << "ERROR: Type of time integration scheme not recongized ... " << endl;
	  }
  }
  // For plotting, we just need 0 adv_level
  else if (in_run_type==1)
  {
      n_adv_levels=1;
  }

  // Allocate storage for solution
	disu_upts.setup(n_adv_levels);
	for(int i=0;i<n_adv_levels;i++)
	{
	  disu_upts(i).setup(n_upts_per_ele,n_eles,n_fields);
	}

  // Initialize to zero
  for (int m=0;m<n_adv_levels;m++)
  for (int i=0;i<n_upts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
      for (int k=0;k<n_fields;k++)
        disu_upts(m)(i,j,k) = 0.;

	// Allocate extra arrays for LES models
	if(LES) {

		sgsf_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
		sgsf_fpts.setup(n_fpts_per_ele,n_eles,n_fields,n_dims);

		// SVV model requires filtered solution
		if(sgs_model==3 || sgs_model==2 || sgs_model==4) {

			filter = 1;

			disuf_upts.setup(n_upts_per_ele,n_eles,n_fields);
		}
		// is this necessary?
		else {
			disuf_upts.setup(1);
		}

		// Similarity model requires product terms and Leonard tensors
		if(sgs_model==2 || sgs_model==4) {
			// Leonard tensor and velocity-velocity product for momentum SGS term
			if(n_dims==2) {
				Lu.setup(n_upts_per_ele,n_eles,3);
				uu.setup(n_upts_per_ele,n_eles,3);
			}
			else if(n_dims==3) {
				Lu.setup(n_upts_per_ele,n_eles,6);
				uu.setup(n_upts_per_ele,n_eles,6);
			}

			// Leonard tensor and velocity-energy product for energy SGS term
			Le.setup(n_upts_per_ele,n_eles,n_dims);
			ue.setup(n_upts_per_ele,n_eles,n_dims);
		}
		// is this necessary?
		else {
			Lu.setup(1);
			uu.setup(1);
			Le.setup(1);
			ue.setup(1);
		}
	}
	// Dummy arrays to pass to GPU kernel wrapper. is this necessary?
	else {
		disuf_upts.setup(1);
		Lu.setup(1);
		uu.setup(1);
		Le.setup(1);
		ue.setup(1);
	}

	// Allocate array for wall distance if using a wall model
	if(wall_model > 0) {
		wall_distance.setup(n_upts_per_ele,n_eles,n_dims);
		twall.setup(n_upts_per_ele,n_eles,n_fields);
		zero_array(wall_distance);
		zero_array(twall);
	}
	else {
		wall_distance.setup(1);
		twall.setup(1);
	}

	// Allocate small SGS flux array if using LES or wall model
	if(LES || wall_model > 0) {
		temp_sgsf.setup(n_fields,n_dims);
	}

	if (run_input.turb_model == 1)
	{
	  wall_distance.setup(n_upts_per_ele,n_eles,n_dims);
	  wall_distance_mag.setup(n_upts_per_ele,n_eles);
	}

	set_shape(in_max_n_spts_per_ele);
  ele2global_ele.setup(n_eles);
  bctype.setup(n_eles,n_inters_per_ele);

	// for mkl sparse blas
	matdescra[0]='G';
	matdescra[3]='F';
	transa='N';
	one=1.0;
	zero=0.0;

	n_fields_mul_n_eles=n_fields*n_eles;
	n_dims_mul_n_upts_per_ele=n_dims*n_upts_per_ele;

  if (in_run_type==0) {

	  div_tconf_upts.setup(n_adv_levels);
	  for(int i=0;i<n_adv_levels;i++)
	  {
	  	div_tconf_upts(i).setup(n_upts_per_ele,n_eles,n_fields);
	  }

    // Initialize to zero
    for (int m=0;m<n_adv_levels;m++)
    for (int i=0;i<n_upts_per_ele;i++)
      for (int j=0;j<n_eles;j++)
        for (int k=0;k<n_fields;k++)
          div_tconf_upts(m)(i,j,k) = 0.;

	  disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
	  tdisf_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
	  norm_tdisf_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
	  norm_tconf_fpts.setup(n_fpts_per_ele,n_eles,n_fields);

	  if(viscous)
	  {
	  	delta_disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
	  	grad_disu_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
	  	grad_disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields,n_dims);
	  }

	  // Initialize source term
	  src_term.setup(n_upts_per_ele, n_eles, n_fields);
    src_term.initialize_to_zero();

	  // Set connectivity array. Needed for Paraview output.
    if (ele_type==3) // prism
      connectivity_plot.setup(8,n_peles_per_ele);
    else
      connectivity_plot.setup(n_verts_per_ele,n_peles_per_ele);

		set_connectivity_plot();
  }
  else if (in_run_type==1)
  {
    ppt_to_pnode.setup(n_eles,n_ppts_per_ele);
    pos_ppts.setup(n_eles,n_ppts_per_ele);
    for (int i=0;i<n_eles;i++)
      for (int j=0;j<n_ppts_per_ele;j++)
        pos_ppts(i,j).setup(n_dims);

    if (ele_type==3) // prism
      connectivity_plot.setup(8,n_peles_per_ele);
    else
      connectivity_plot.setup(n_verts_per_ele,n_peles_per_ele);
  }

  }

}

void eles::set_disu_upts_to_zero_other_levels(void)
{

  if (n_eles!=0)
  {
    // Initialize to zero
    for (int m=1;m<n_adv_levels;m++)
    {
      for (int i=0;i<n_upts_per_ele;i++)
        for (int j=0;j<n_eles;j++)
          for (int k=0;k<n_fields;k++)
            disu_upts(m)(i,j,k) = 0.;


      #ifdef _GPU
      if (n_eles!=0)
      {
	      disu_upts(m).cp_cpu_gpu();
      }
      #endif
    }
  }
}

int* eles::get_connectivity_plot_ptr()
{
    return connectivity_plot.get_ptr_cpu();
}

array<int> eles::get_connectivity_plot()
{
    return connectivity_plot;
}

array<double> eles::calc_pos_pnode_vert(int in_ele, int in_vert)
{
	//cout << "in calc_pos_pnode_vert " << in_vert << ", " << n_verts_per_ele << endl;
  if (in_vert >= n_verts_per_ele)
    FatalError("in_vert >= n_verts_per_ele");

  int ppt = vert_to_ppt(in_vert);
  return pos_ppts(in_ele,ppt);
}

array<double> eles::calc_pos_pnode_edge(int in_ele, int in_edge, int in_edge_ppt)
{
  if (in_edge >= n_edges_per_ele || in_edge_ppt >= n_ppts_per_edge)
    FatalError("in_edge >= n_edges_per_ele");

  int ppt = edge_ppt_to_ppt(in_edge, in_edge_ppt);
  return pos_ppts(in_ele,ppt);
}


array<double> eles::calc_pos_pnode_face(int in_ele, int in_face, int in_face_ppt)
{
  if (in_face >= n_inters_per_ele || in_face_ppt >= n_ppts_per_face(in_face))
    FatalError("in_face >= n_inters_per_ele");

  int ppt = face_ppt_to_ppt(in_face)(in_face_ppt);
  return pos_ppts(in_ele,ppt);
}

array<double> eles::calc_pos_pnode_interior(int in_ele, int in_interior_ppt)
{
  if (in_interior_ppt >= n_interior_ppts)
    FatalError("in_interior_ppt >= n_interior_ppts");

  int ppt = interior_ppt_to_ppt(in_interior_ppt);
  return pos_ppts(in_ele,ppt);
}

void eles::set_pnode_vert(int in_ele, int in_vert, int in_pnode)
{
  int ppt = vert_to_ppt(in_vert);
  //cout << "in_ele=" << in_ele << " ppt=" << ppt << endl;
  ppt_to_pnode(in_ele,ppt) = in_pnode;

}

void eles::set_pnode_edge(int in_ele, int in_edge, int in_edge_ppt, int in_pnode)
{
  int ppt = edge_ppt_to_ppt(in_edge, in_edge_ppt);
  ppt_to_pnode(in_ele,ppt) = in_pnode;
}

void eles::set_pnode_face(int in_ele, int in_face, int in_face_ppt,  int in_pnode)
{
  int ppt = face_ppt_to_ppt(in_face)(in_face_ppt);
  ppt_to_pnode(in_ele,ppt) = in_pnode;
}

void eles::set_pnode_interior(int in_ele, int in_interior_ppt, int in_pnode)
{
  int ppt = interior_ppt_to_ppt(in_interior_ppt);
  ppt_to_pnode(in_ele,ppt) = in_pnode;
}

int eles::get_n_interior_ppts()
{
  return n_interior_ppts;
}

int eles::get_n_ppts_per_face(int in_face)
{
  return n_ppts_per_face(in_face);
}

int eles::get_max_n_ppts_per_face()
{
  return max_n_ppts_per_face;
}

int eles::calc_num_ppts(void)
{
    return n_ppts_per_ele*n_eles;
}


// set initial conditions

void eles::set_ics(double& time)
{
	int i,j,k;

	double rho,vx,vy,vz,p;
	double gamma=run_input.gamma;
  time = 0.;

	array<double> loc(n_dims);
	array<double> pos(n_dims);
	array<double> ics(n_fields);

  array<double> grad_rho(n_dims);

	for(i=0;i<n_eles;i++)
	{
		for(j=0;j<n_upts_per_ele;j++)
		{
			for(k=0;k<n_dims;k++)
			{
				loc(k)=loc_upts(k,j);
			}

			// calculate position of solution point

			calc_pos(loc,i,pos);

			// evaluate solution at solution point
			if(run_input.ic_form==0)
			{
        eval_isentropic_vortex(pos,time,rho,vx,vy,vz,p,n_dims);

			  ics(0)=rho;
			  ics(1)=rho*vx;
			  ics(2)=rho*vy;
			  if(n_dims==2)
			  {
			  	ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));
			  }
			  else if(n_dims==3)
			  {
			  	ics(3)=rho*vz;
			  	ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));
			  }
			  else
			  {
			  	cout << "ERROR: Invalid number of dimensions ... " << endl;
			  }
			}
			else if(run_input.ic_form==1)
			{
				rho=run_input.rho_c_ic;
				vx=run_input.u_c_ic;
				vy=run_input.v_c_ic;
				vz=run_input.w_c_ic;
				p=run_input.p_c_ic;

			  ics(0)=rho;
			  ics(1)=rho*vx;
			  ics(2)=rho*vy;
			  if(n_dims==2)
			  {
			  	ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));

			  	if(run_input.turb_model==1)
			  	{
			  	    ics(4) = run_input.mu_tilde_c_ic;
			  	}
			  }
			  else if(n_dims==3)
			  {
			  	ics(3)=rho*vz;
			  	ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));

			  	if(run_input.turb_model==1)
			  	{
			  	    ics(5) = run_input.mu_tilde_c_ic;
			  	}
			  }
			  else
			  {
			  	cout << "ERROR: Invalid number of dimensions ... " << endl;
			  }
			}
      else if (run_input.ic_form==2) // Sine wave (single)
      {
        eval_sine_wave_single(pos,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
        ics(0) = rho;
      }
      else if (run_input.ic_form==3) // Sine wave (group)
      {
        eval_sine_wave_group(pos,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
        ics(0) = rho;
      }
      else if (run_input.ic_form==4) // Spherical distribution
      {
        eval_sphere_wave(pos,run_input.wave_speed,time,rho,n_dims);
        ics(0) = rho;
      }
      else if (run_input.ic_form==5) // Constant for adv-diff
      {
        ics(0) = 1.0;
      }
      else if (run_input.ic_form==6) // Up to 4th order polynomials for u, v, w
      {
				rho=run_input.rho_c_ic;
				p=run_input.p_c_ic;
        eval_poly_ic(pos,rho,ics,n_dims);
        ics(0) = rho;
			  if(n_dims==2)
					ics(3)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2));
				else if(n_dims==3)
					ics(4)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2)+ics(3)*ics(3));
      }
      else if (run_input.ic_form==7) // Taylor-Green Vortex initial conditions
      {
				rho=run_input.rho_c_ic;
        ics(0) = rho;
			  if(n_dims==2)
				{
					// Simple 2D div-free vortex
					p = 0.0;
					ics(1) = sin(pos(0)/2.)*cos(pos(1)/2.);
					ics(2) = -1.0*cos(pos(0)/2.)*sin(pos(1)/2.);
					ics(3) = 0.0;
					ics(4)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2));
				}
				else if(n_dims==3)
				{
					// ONERA benchmark setup
					p = 100 + rho/16.0*(cos(2.0*pos(0)) + cos(2.0*pos(1)))*(cos(2.0*pos(2)) + 2.0);
					ics(1) = sin(pos(0))*cos(pos(1))*cos(pos(2));
					ics(2) = -1.0*cos(pos(0))*sin(pos(1))*cos(pos(2));
					ics(3) = 0.0;
					ics(4)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2)+ics(3)*ics(3));
				}
      }
			else
      {
				cout << "ERROR: Invalid form of initial condition ... (File: " << __FILE__ << ", Line: " << __LINE__ << ")" << endl;
				exit (1);
			}

			// Add perturbation to channel
			if(run_input.perturb_ic==1 and n_dims==3)
			{
				// Constructs the function
				// u = alpha exp(-((x-L_x/2)/L_x)^2) exp(-(y/L_y)^2) cos(4 pi z/L_z)
				// Where L_x, L_y, L_z are the domain dimensions, u_0=bulk velocity,
				// alpha=scale factor, u=wall normal velocity
				double alpha, L_x, L_y, L_z;
				alpha = 0.1;
				L_x = 2.*pi;
				L_y = pi;
				L_z = 2.;
				ics(3) += alpha*exp(pow(-((pos(0)-L_x/2.)/L_x),2))*exp(pow(-(pos(1)/L_y),2))*cos(4.*pi*pos(2)/L_z);
			}

			// set solution at solution point
			for(k=0;k<n_fields;k++)
			{
				disu_upts(0)(j,i,k)=ics(k);
			}
		}
	}
}


// set initial conditions


void eles::read_restart_data(ifstream& restart_file)
{

  if (n_eles==0) return;

  int num_eles_to_read;
  string ele_name,str;
  if (ele_type==0) ele_name="TRIS";
  else if (ele_type==1) ele_name="QUADS";
  else if (ele_type==2) ele_name="TETS";
  else if (ele_type==3) ele_name="PRIS";
  else if (ele_type==4) ele_name="HEXAS";

  // Move cursor to correct element type
  while(1) {
    getline(restart_file,str);
    if (str==ele_name) break;
    if (restart_file.eof()) return; // Restart file doesn't contain my elements
  }

  // Move cursor to n_eles
  while(1) {
    getline(restart_file,str);
    if (str=="n_eles") break;
  }

  // Read number of elements to read
  restart_file >> num_eles_to_read;
  getline(restart_file,str);

  //Skip ele2global_ele lines
  getline(restart_file,str);
  getline(restart_file,str);
  getline(restart_file,str);

  int ele,index;
  array<double> disu_upts_rest;
  disu_upts_rest.setup(n_upts_per_ele_rest,n_fields);

  for (int i=0;i<num_eles_to_read;i++)
  {
    restart_file >> ele ;
    index = index_locate_int(ele,ele2global_ele.get_ptr_cpu(),n_eles);

    if (index!=-1) // Ele belongs to processor
    {
      for (int j=0;j<n_upts_per_ele_rest;j++)
        for (int k=0;k<n_fields;k++)
           restart_file >> disu_upts_rest(j,k);

      // Now compute transformed solution at solution points using opp_r
      for (int m=0;m<n_fields;m++)
      {
        for (int j=0;j<n_upts_per_ele;j++)
        {
          double value = 0.;
          for (int k=0;k<n_upts_per_ele_rest;k++)
            value += opp_r(j,k)*disu_upts_rest(k,m);

          disu_upts(0)(j,index,m) = value;
        }
      }

    }
    else // Skip the data (doesn't belong to current processor)
    {
      // Skip rest of ele line
      getline(restart_file,str);
      for (int j=0;j<n_upts_per_ele_rest;j++)
        getline(restart_file,str);
    }
  }
}


void eles::write_restart_data(ofstream& restart_file)
{
  restart_file << "n_eles" << endl;
  restart_file << n_eles << endl;
  restart_file << "ele2global_ele array" << endl;
  for (int i=0;i<n_eles;i++)
    restart_file << ele2global_ele(i) << " ";
  restart_file << endl;

  restart_file << "data" << endl;

  for (int i=0;i<n_eles;i++)
  {
    restart_file << ele2global_ele(i) << endl;
    for (int j=0;j<n_upts_per_ele;j++)
    {
      for (int k=0;k<n_fields;k++)
      {
         restart_file << disu_upts(0)(j,i,k) << " ";
      }
      restart_file << endl;
    }
  }
  restart_file << endl;
}

// move all to from cpu to gpu

void eles::mv_all_cpu_gpu(void)
{
	#ifdef _GPU
  if (n_eles!=0)
  {
    disu_upts(0).cp_cpu_gpu();
	  div_tconf_upts(0).cp_cpu_gpu();

	  for(int i=1;i<n_adv_levels;i++)
	  {
	  	disu_upts(i).cp_cpu_gpu();
	  	div_tconf_upts(i).mv_cpu_gpu();
	  }

	  disu_fpts.mv_cpu_gpu();
	  tdisf_upts.mv_cpu_gpu();
	  norm_tdisf_fpts.mv_cpu_gpu();
	  norm_tconf_fpts.mv_cpu_gpu();

	  //TODO: mv instead of cp
    if(viscous)
	  {
      delta_disu_fpts.mv_cpu_gpu();
	  	grad_disu_upts.cp_cpu_gpu();
	  	grad_disu_fpts.mv_cpu_gpu();

	  	//tdisvisf_upts.mv_cpu_gpu();
	  	//norm_tdisvisf_fpts.mv_cpu_gpu();
	  	//norm_tconvisf_fpts.mv_cpu_gpu();
	  }

		// LES arrays
		//if(LES) {
			filter_upts.mv_cpu_gpu();
			disuf_upts.mv_cpu_gpu();
			sgsf_upts.mv_cpu_gpu();
			sgsf_fpts.mv_cpu_gpu();
			uu.mv_cpu_gpu();
			ue.mv_cpu_gpu();
			Lu.mv_cpu_gpu();
			Le.mv_cpu_gpu();
		//}
		// wall model array
		//if(wall_model > 0) {
			twall.mv_cpu_gpu();
		//}
  }
	#endif
}

// move wall distance array to gpu

void eles::mv_wall_distance_cpu_gpu(void)
{
	#ifdef _GPU

	wall_distance.mv_cpu_gpu();
	wall_distance_mag.mv_cpu_gpu();

	#endif
}

// copy discontinuous solution at solution points to cpu
void eles::cp_disu_upts_gpu_cpu(void)
{
	#ifdef _GPU
  if (n_eles!=0)
  {
	  disu_upts(0).cp_gpu_cpu();
  }
	#endif
}


// copy discontinuous solution at solution points to gpu
void eles::cp_disu_upts_cpu_gpu(void)
{
	#ifdef _GPU
  if (n_eles!=0)
  {
	  disu_upts(0).cp_cpu_gpu();
  }
	#endif
}


// copy gradient of discontinuous solution at solution points to cpu
void eles::cp_grad_disu_upts_gpu_cpu(void)
{
  if (n_eles!=0)
  {
	#ifdef _GPU
	grad_disu_upts.cp_gpu_cpu();
	#endif
  }
}

// copy determinant of jacobian at solution points to cpu

void eles::cp_detjac_upts_gpu_cpu(void)
{
	#ifdef _GPU

	detjac_upts.cp_gpu_cpu();

	#endif
}

// copy divergence at solution points to cpu

void eles::cp_div_tconf_upts_gpu_cpu(void)
{
  if (n_eles!=0)
  {
	#ifdef _GPU

	div_tconf_upts(0).cp_gpu_cpu();

	#endif
  }
}


// remove transformed discontinuous solution at solution points from cpu

void eles::rm_disu_upts_cpu(void)
{
	#ifdef _GPU

	disu_upts(0).rm_cpu();

	#endif
}

// remove determinant of jacobian at solution points from cpu

void eles::rm_detjac_upts_cpu(void)
{
	#ifdef _GPU

	detjac_upts.rm_cpu();

	#endif
}

// advance with rk11 (forwards euler)

void eles::advance_rk11(void)
{
  if (n_eles!=0)
  {

	/*!
	Performs B = B + (alpha*A) where: \n
	alpha = -run_input.dt \n
	A = div_tconf_upts(0)\n
	B = disu_upts(0)
	*/

	#ifdef _CPU
	for (int i=0;i<n_fields;i++)
	{
	  for (int ic=0;ic<n_eles;ic++)
	  {
			for (int inp=0;inp<n_upts_per_ele;inp++)
			{
        disu_upts(0)(inp,ic,i) -= run_input.dt*(div_tconf_upts(0)(inp,ic,i)/detjac_upts(inp,ic) - run_input.const_src_term - src_term(inp,ic,i));
			}
		}
	}
	#endif

	#ifdef _GPU
    RK11_update_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, disu_upts(0).get_ptr_gpu(), div_tconf_upts(0).get_ptr_gpu(), detjac_upts.get_ptr_gpu(), run_input.dt, run_input.const_src_term);
  #endif

  /*

	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

	cblas_daxpy(n_eles*n_fields*n_upts_per_ele,-run_input.dt,div_tconf_upts(0).get_ptr_cpu(),1,disu_upts(0).get_ptr_cpu(),1);


	#endif


	cublasDaxpy(n_eles*n_fields*n_upts_per_ele,-run_input.dt,div_tconf_upts(0).get_ptr_gpu(),1,disu_upts(0).get_ptr_gpu(),1);

	#endif
  */
  }
}

// advance with rk33 (three-stage third-order runge-kutta)

void eles::advance_rk33(int in_step)
{
  FatalError("Advance_rk33 not implemented");
}

// advance with rk44 (four-stage forth-order runge-kutta)

void eles::advance_rk44(int in_step)
{
  FatalError("Advance_rk44 not implemented");
}

// advance with rk45 (five-stage forth-order low-storage runge-kutta)

void eles::advance_rk45(int in_step)
{
  if (n_eles!=0)
  {

  double rk4a, rk4b;
  if (in_step==0) {
    	rk4a=    0.0;
    	rk4b=   0.149659021999229;
  }
  else if (in_step==1) {
    	rk4a=   -0.417890474499852;
    	rk4b=   0.379210312999627;
  }
  else if (in_step==2) {
    	rk4a=   -1.192151694642677;
    	rk4b=   0.822955029386982;
  }
  else if (in_step==3) {
    	rk4a=   -1.697784692471528;
    	rk4b=   0.699450455949122;
  }
  else if (in_step==4) {
    	rk4a=   -1.514183444257156;
    	rk4b=   0.153057247968152;
  }

	#ifdef _CPU

  double res, rhs;
	for (int ic=0;ic<n_eles;ic++)
	{
		for (int i=0;i<n_fields;i++)
		{
			for (int inp=0;inp<n_upts_per_ele;inp++)
			{
				rhs = -div_tconf_upts(0)(inp,ic,i)/detjac_upts(inp,ic) + run_input.const_src_term + src_term(inp,ic,i);
	  		res = disu_upts(1)(inp,ic,i);
        res = rk4a*res + run_input.dt*rhs;

	  		disu_upts(1)(inp,ic,i) = res;
        disu_upts(0)(inp,ic,i) += rk4b*res;
			}
		}
	}
  #endif

	#ifdef _GPU

  RK45_update_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, disu_upts(0).get_ptr_gpu(), disu_upts(1).get_ptr_gpu(), div_tconf_upts(0).get_ptr_gpu(), detjac_upts.get_ptr_gpu(), rk4a,  rk4b, run_input.dt, run_input.const_src_term);

  #endif

  }
}


// calculate the discontinuous solution at the flux points

void eles::calc_disu_fpts(int in_disu_upts_from)
{
  if (n_eles!=0) {

    /*!
	  Performs C = (alpha*A*B) + (beta*C) where: \n
	  alpha = 1.0 \n
	  beta = 0.0 \n
	  A = opp_0 \n
	  B = disu_upts(in_disu_upts_from) \n
	  C = disu_fpts
	  */

    Arows =	n_fpts_per_ele;
	  Acols = n_upts_per_ele;

	  Brows = Acols;
	  Bcols = n_fields*n_eles;

	  Astride = Arows;
	  Bstride = Brows;
	  Cstride = Arows;

	  #ifdef _CPU

	  if(opp_0_sparse==0) // dense
	  {
	  	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_0.get_ptr_cpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_cpu(),Bstride,0.0,disu_fpts.get_ptr_cpu(),Cstride);

	  	#endif
	  }
	  else if(opp_0_sparse==1) // mkl blas four-array csr format
	  {
	  	#if defined _MKL_BLAS
	  	mkl_dcsrmm(&transa, &n_fpts_per_ele, &n_fields_mul_n_eles, &n_upts_per_ele, &one, matdescra, opp_0_data.get_ptr_cpu(), opp_0_cols.get_ptr_cpu(), opp_0_b.get_ptr_cpu(), opp_0_e.get_ptr_cpu(), disu_upts(in_disu_upts_from).get_ptr_cpu(), &n_upts_per_ele, &zero, disu_fpts.get_ptr_cpu(), &n_fpts_per_ele);

	  	#endif
	  }
	  else { cout << "ERROR: Unknown storage for opp_0 ... " << endl; }

	  #endif

	  #ifdef _GPU
    if(opp_0_sparse==0)
    {
      cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_0.get_ptr_gpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_gpu(),Bstride,0.0,disu_fpts.get_ptr_gpu(),Cstride);
    }
    else if (opp_0_sparse==1)
    {
      bespoke_SPMV(n_fpts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_0_ell_data.get_ptr_gpu(), opp_0_ell_indices.get_ptr_gpu(), opp_0_nnz_per_row, disu_upts(in_disu_upts_from).get_ptr_gpu(), disu_fpts.get_ptr_gpu(), ele_type, order, 0);
    }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_0 ... " << endl;
	  }
	  #endif

  }

}

// calculate the transformed discontinuous inviscid flux at the solution points

void eles::calc_tdisinvf_upts(int in_disu_upts_from)
{
  if (n_eles!=0)
  {

	#ifdef _CPU

	int i,j,k,l,m;

	for(i=0;i<n_eles;i++)
	{
		for(j=0;j<n_upts_per_ele;j++)
		{
			for(k=0;k<n_fields;k++)
			{
				temp_u(k)=disu_upts(in_disu_upts_from)(j,i,k);
			}

			if(n_dims==2)
			{
				calc_invf_2d(temp_u,temp_f);
			}
			else if(n_dims==3)
			{
				calc_invf_3d(temp_u,temp_f);
			}
			else
			{
					cout << "ERROR: Invalid number of dimensions ... " << endl;
			}

			for(k=0;k<n_fields;k++)
			{
				for(l=0;l<n_dims;l++)
				{
					tdisf_upts(j,i,k,l)=0.;
					for(m=0;m<n_dims;m++)
					{
					  tdisf_upts(j,i,k,l)+=inv_detjac_mul_jac_upts(j,i,l,m)*temp_f(k,m);
					}
				}
			}
		}
	}

	#endif

	#ifdef _GPU
	calc_tdisinvf_upts_gpu_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, disu_upts(in_disu_upts_from).get_ptr_gpu(), tdisf_upts.get_ptr_gpu(), detjac_upts.get_ptr_gpu(), inv_detjac_mul_jac_upts.get_ptr_gpu(), run_input.gamma, run_input.equation, run_input.wave_speed(0), run_input.wave_speed(1), run_input.wave_speed(2));


  //tdisinvf_upts.cp_gpu_cpu();
	#endif
/*
  for (int i=0;i<n_upts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
      for (int k=0;k<n_fields;k++)
        for (int m=0;m<n_dims;m++)
          cout << "i=" << i << "j=" << j << "k=" << k << "m=" << m << " " << tdisinvf_upts(i,j,k,m) << endl;
          */
  }
}


// calculate the normal transformed discontinuous flux at the flux points

void eles::calc_norm_tdisf_fpts()
{
  if (n_eles!=0)
  {
	  #ifdef _CPU

	  if(opp_1_sparse==0) // dense
	  {
	  	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

	  	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(0).get_ptr_cpu(),n_fpts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,0),n_upts_per_ele,0.0,norm_tdisf_fpts.get_ptr_cpu(),n_fpts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
	  	  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(i).get_ptr_cpu(),n_fpts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele,1.0,norm_tdisf_fpts.get_ptr_cpu(),n_fpts_per_ele);
      }

	  	#endif
	  }
	  else if(opp_1_sparse==1) // mkl blas four-array csr format
	  {
	  	#if defined _MKL_BLAS

	  	mkl_dcsrmm(&transa, &n_fpts_per_ele, &n_fields_mul_n_eles, &n_upts_per_ele, &one, matdescra, opp_1_data(0).get_ptr_cpu(), opp_1_cols(0).get_ptr_cpu(), opp_1_b(0).get_ptr_cpu(), opp_1_e(0).get_ptr_cpu(), tdisf_upts.get_ptr_cpu(0,0,0,0), &n_upts_per_ele, &zero, norm_tdisf_fpts.get_ptr_cpu, &n_fpts_per_ele);

      for (int i=1;i<n_dims;i++) {
	  	  mkl_dcsrmm(&transa, &n_fpts_per_ele, &n_fields_mul_n_eles, &n_upts_per_ele, &one, matdescra, opp_1_data(i).get_ptr_cpu(), opp_1_cols(i).get_ptr_cpu(), opp_1_b(i).get_ptr_cpu(), opp_1_e(i).get_ptr_cpu(), tdisf_upts.get_ptr_cpu(0,0,0,i), &n_upts_per_ele, &one, norm_tdisf_fpts.get_ptr_cpu(), &n_fpts_per_ele);
      }

	  	#endif
	  }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_1 ... " << endl;
	  }

	  #endif

	  #ifdef _GPU

    if (opp_1_sparse==0)
    {
	    cublasDgemm('N','N',n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(0).get_ptr_gpu(),n_fpts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,0),n_upts_per_ele,0.0,norm_tdisf_fpts.get_ptr_gpu(),n_fpts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
	      cublasDgemm('N','N',n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(i).get_ptr_gpu(),n_fpts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele,1.0,norm_tdisf_fpts.get_ptr_gpu(),n_fpts_per_ele);
      }
    }
    else if (opp_1_sparse==1)
    {
      bespoke_SPMV(n_fpts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_1_ell_data(0).get_ptr_gpu(), opp_1_ell_indices(0).get_ptr_gpu(), opp_1_nnz_per_row(0), tdisf_upts.get_ptr_gpu(0,0,0,0), norm_tdisf_fpts.get_ptr_gpu(), ele_type, order, 0);
      for (int i=1;i<n_dims;i++)
      {
        bespoke_SPMV(n_fpts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_1_ell_data(i).get_ptr_gpu(), opp_1_ell_indices(i).get_ptr_gpu(), opp_1_nnz_per_row(i), tdisf_upts.get_ptr_gpu(0,0,0,i), norm_tdisf_fpts.get_ptr_gpu(), ele_type, order, 1);
      }
    }
	  #endif

  }

  /*
#ifdef _GPU
    tdisinvf_upts.cp_gpu_cpu();
#endif

    cout << "Before" << endl;
    for (int i=0;i<n_fpts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
    for (int k=0;k<n_fields;k++)
    for (int m=0;m<n_dims;m++)
    cout << setprecision(10)  << i << " " << j<< " " << k << " " << tdisinvf_upts(i,j,k,m) << endl;
    */

  /*
  cout << "After,ele_type =" << ele_type << endl;
#ifdef _GPU
    norm_tdisinvf_fpts.cp_gpu_cpu();
#endif

  for (int i=0;i<n_fpts_per_ele;i++)
  for (int j=0;j<n_eles;j++)
  for (int k=0;k<n_fields;k++)
  cout << setprecision(10)  << i << " " << j<< " " << k << " " << norm_tdisinvf_fpts(i,j,k) << endl;
  */
}


// calculate the divergence of the transformed discontinuous flux at the solution points

void eles::calc_div_tdisf_upts(int in_div_tconf_upts_to)
{
  if (n_eles!=0)
  {
	  #ifdef _CPU

	  if(opp_2_sparse==0) // dense
	  {
	  	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

	  	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(0).get_ptr_cpu(),n_upts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,0),n_upts_per_ele,0.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),n_upts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
	  	  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(i).get_ptr_cpu(),n_upts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),n_upts_per_ele);
      }

	  	#endif
	  }
	  else if(opp_2_sparse==1) // mkl blas four-array csr format
	  {
	  	#if defined _MKL_BLAS

	  	mkl_dcsrmm(&transa, &n_upts_per_ele, &n_fields_mul_n_eles, &n_upts_per_ele, &one, matdescra, opp_2_data(0).get_ptr_cpu(), opp_2_cols(0).get_ptr_cpu(), opp_2_b(0).get_ptr_cpu(), opp_2_e(0).get_ptr_cpu(), tdisf_upts.get_ptr_cpu(0,0,0,0), &n_upts_per_ele, &zero, div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(), &n_upts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
	  	  mkl_dcsrmm(&transa, &n_upts_per_ele, &n_fields_mul_n_eles, &n_upts_per_ele, &one, matdescra, opp_2_data(i).get_ptr_cpu(), opp_2_cols(i).get_ptr_cpu(), opp_2_b(i).get_ptr_cpu(), opp_2_e(i).get_ptr_cpu(), tdisf_upts.get_ptr_cpu(0,0,0,i), &n_upts_per_ele, &one, div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(), &n_upts_per_ele);
      }

	  	#endif
	  }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_2 ... " << endl;
	  }

	  #endif


	  #ifdef _GPU

    if (opp_2_sparse==0)
    {
	    cublasDgemm('N','N',n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(0).get_ptr_gpu(),n_upts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,0),n_upts_per_ele,0.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),n_upts_per_ele);
      for (int i=1;i<n_dims;i++) {
	      cublasDgemm('N','N',n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(i).get_ptr_gpu(),n_upts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),n_upts_per_ele);
      }
    }
    else if (opp_2_sparse==1)
    {
      bespoke_SPMV(n_upts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_2_ell_data(0).get_ptr_gpu(), opp_2_ell_indices(0).get_ptr_gpu(), opp_2_nnz_per_row(0), tdisf_upts.get_ptr_gpu(0,0,0,0), div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(), ele_type, order, 0);
      for (int i=1;i<n_dims;i++) {
        bespoke_SPMV(n_upts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_2_ell_data(i).get_ptr_gpu(), opp_2_ell_indices(i).get_ptr_gpu(), opp_2_nnz_per_row(i), tdisf_upts.get_ptr_gpu(0,0,0,i), div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(), ele_type, order, 1);
      }

    }
	  #endif

  }

  /*
  for (int j=0;j<n_eles;j++)
  for (int i=0;i<n_upts_per_ele;i++)
      //for (int k=0;k<n_fields;k++)
        cout << scientific << setw(16) << setprecision(12) << div_tconf_upts(0)(i,j,0) << endl;
  */
}


// calculate divergence of the transformed continuous flux at the solution points

void eles::calc_div_tconf_upts(int in_div_tconf_upts_to)
{
  if (n_eles!=0)
  {
	  #ifdef _CPU

	  #if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

	  cblas_daxpy(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_cpu(),1,norm_tconf_fpts.get_ptr_cpu(),1);

	  #endif

	  if(opp_3_sparse==0) // dense
	  {
	  	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

	  	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields*n_eles,n_fpts_per_ele,1.0,opp_3.get_ptr_cpu(),n_upts_per_ele,norm_tconf_fpts.get_ptr_cpu(),n_fpts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),n_upts_per_ele);

	  	#endif
	  }
	  else if(opp_3_sparse==1) // mkl blas four-array csr format
	  {
	  	#if defined _MKL_BLAS

	  	mkl_dcsrmm(&transa,&n_upts_per_ele,&n_fields_mul_n_eles,&n_fpts_per_ele,&one,matdescra,opp_3_data.get_ptr_cpu(),opp_3_cols.get_ptr_cpu(),opp_3_b.get_ptr_cpu(),opp_3_e.get_ptr_cpu(),norm_tconf_fpts.get_ptr_cpu(),&n_fpts_per_ele,&one,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),&n_upts_per_ele);

	  	#endif
	  }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_3 ... " << endl;
	  }

	  #endif

	  #ifdef _GPU

	  cublasDaxpy(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_gpu(),1,norm_tconf_fpts.get_ptr_gpu(),1);

    if (opp_3_sparse==0)
    {
	  cublasDgemm('N','N',n_upts_per_ele,n_fields*n_eles,n_fpts_per_ele,1.0,opp_3.get_ptr_gpu(),n_upts_per_ele,norm_tconf_fpts.get_ptr_gpu(),n_fpts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),n_upts_per_ele);
    }
    else if (opp_3_sparse==1)
    {
      bespoke_SPMV(n_upts_per_ele, n_fpts_per_ele, n_fields, n_eles, opp_3_ell_data.get_ptr_gpu(), opp_3_ell_indices.get_ptr_gpu(), opp_3_nnz_per_row, norm_tconf_fpts.get_ptr_gpu(), div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(), ele_type, order, 1);
    }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_3 ... " << endl;
	  }
	  #endif

  }

  /*
#ifdef _GPU
    norm_tconinvf_fpts.cp_gpu_cpu();
#endif

    for (int i=0;i<n_eles;i++)
    {
      for (int j=0;j<n_fpts_per_ele;j++)
      {
      for (int k=0;k<n_fields;k++)
      {
        cout << "i=" << i << "j=" << j << "k=" << k << "norm=" << setprecision(12) << norm_tconinvf_fpts(j,i,k) << endl;
      }
      }
    }
  */

  /*
  for (int j=0;j<n_eles;j++)
  for (int i=0;i<n_upts_per_ele;i++)
      //for (int k=0;k<n_fields;k++)
        cout << scientific << setw(16) << setprecision(12) << div_tconf_upts(0)(i,j,0) << endl;
        */

  /*
  cout << "OUTPUT" << endl;
#ifdef _GPU
  norm_tconinvf_fpts.cp_gpu_cpu();
#endif

  for (int i=0;i<n_fpts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
      for (int k=0;k<n_fields;k++)
        cout << setprecision(10)  << i << " " << j<< " " << k << " " << norm_tconinvf_fpts(i,j,k) << endl;
  */

  /*
  for (int j=0;j<n_eles;j++)
  for (int i=0;i<n_upts_per_ele;i++)
        cout << div_tconf_upts(0)(i,j,1) << endl;
  */

}


// calculate uncorrected transformed gradient of the discontinuous solution at the solution points
// (mixed derivative)

void eles::calc_uncor_tgrad_disu_upts(int in_disu_upts_from)
{
  if (n_eles!=0)
  {
	  Arows =	n_upts_per_ele;
	  Acols = n_upts_per_ele;

	  Brows = Acols;
	  Bcols = n_fields*n_eles;

	  Astride = Arows;
	  Bstride = Brows;
	  Cstride = Arows;

    #ifdef _CPU

	  if(opp_4_sparse==0) // dense
	  {
	  	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

      for (int i=0;i<n_dims;i++) {
	  	  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_4(i).get_ptr_cpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_cpu(),Bstride,0.0,grad_disu_upts.get_ptr_cpu(0,0,0,i),Cstride);
      }

	  	#endif
	  }
	  else if(opp_4_sparse==1) // mkl blas four-array csr format
	  {
	  	#if defined _MKL_BLAS

	  	// implement

	  	#endif
	  }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_4 ... " << endl;
	  }

	  #endif

	  #ifdef _GPU

    if (opp_4_sparse==0)
    {
      for (int i=0;i<n_dims;i++)
      {
	      cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_4(i).get_ptr_gpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_gpu(),Bstride,0.0,grad_disu_upts.get_ptr_gpu(0,0,0,i),Cstride);
      }
    }
    else if (opp_4_sparse==1)
    {
      for (int i=0;i<n_dims;i++)
      {
        bespoke_SPMV(Arows, Acols, n_fields, n_eles, opp_4_ell_data(i).get_ptr_gpu(), opp_4_ell_indices(i).get_ptr_gpu(), opp_4_nnz_per_row(i), disu_upts(in_disu_upts_from).get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(0,0,0,i), ele_type, order, 0);
      }
    }
	  #endif
  }

  /*
  cout << "OUTPUT" << endl;
  #ifdef _GPU
  grad_disu_upts.cp_gpu_cpu();
  #endif

  for (int i=0;i<n_upts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
      for (int k=0;k<n_fields;k++)
        for (int m=0;m<n_dims;m++)
        {
          if (ele2global_ele(j)==53)
            cout << setprecision(10)  << i << " " << ele2global_ele(j) << " " << k << " " << m << " " << grad_disu_upts(i,j,k,m) << endl;
        }
  */
}

// calculate corrected gradient of the discontinuous solution at solution points

void eles::calc_cor_grad_disu_upts(void)
{
  if (n_eles!=0)
  {
	  Arows =	n_upts_per_ele;
	  Acols = n_fpts_per_ele;

	  Brows = Acols;
	  Bcols = n_fields*n_eles;

	  Astride = Arows;
	  Bstride = Brows;
	  Cstride = Arows;

    #ifdef _CPU

	  if(opp_5_sparse==0) // dense
	  {
	  	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

      for (int i=0;i<n_dims;i++)
      {
	  	  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_5(i).get_ptr_cpu(),Astride,delta_disu_fpts.get_ptr_cpu(),Bstride,1.0,grad_disu_upts.get_ptr_cpu(0,0,0,i),Cstride);
      }

	  	#endif
	  }
	  else if(opp_5_sparse==1) // mkl blas four-array csr format
	  {
	  	#if defined _MKL_BLAS

	  	// impelement

	  	#endif
	  }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_5 ... " << endl;
	  }

    // Transform to physical space
    double detjac;
    double inv_detjac;
    double rx,ry,rz,sx,sy,sz,tx,ty,tz;
    double ur,us,ut;

    for (int i=0;i<n_eles;i++)
    {
      for (int j=0;j<n_upts_per_ele;j++)
      {
	      detjac = detjac_upts(j,i);
	      inv_detjac = 1.0/detjac;

	      rx = inv_detjac_mul_jac_upts(j,i,0,0);
	      ry = inv_detjac_mul_jac_upts(j,i,0,1);
	      sx = inv_detjac_mul_jac_upts(j,i,1,0);
	      sy = inv_detjac_mul_jac_upts(j,i,1,1);

	      //physical gradient
	      if(n_dims==2)
	      {
	      	for(int k=0;k<n_fields;k++)
	      	{
	      		ur = grad_disu_upts(j,i,k,0);
	      		us = grad_disu_upts(j,i,k,1);

	      		grad_disu_upts(j,i,k,0) = (1.0/detjac)*(ur*rx + us*sx) ;
	      		grad_disu_upts(j,i,k,1) = (1.0/detjac)*(ur*ry + us*sy) ;
	      	}
        }
        if (n_dims==3)
        {
	    	  rz = inv_detjac_mul_jac_upts(j,i,0,2);
	    		sz = inv_detjac_mul_jac_upts(j,i,1,2);

	    		tx = inv_detjac_mul_jac_upts(j,i,2,0);
	    		ty = inv_detjac_mul_jac_upts(j,i,2,1);
	    		tz = inv_detjac_mul_jac_upts(j,i,2,2);

          for (int k=0;k<n_fields;k++)
          {
	    		  ur = grad_disu_upts(j,i,k,0);
	    		  us = grad_disu_upts(j,i,k,1);
	    		  ut = grad_disu_upts(j,i,k,2);

	    		  grad_disu_upts(j,i,k,0) = (1.0/detjac)*(ur*rx + us*sx + ut*tx);
	    		  grad_disu_upts(j,i,k,1) = (1.0/detjac)*(ur*ry + us*sy + ut*ty);
	    		  grad_disu_upts(j,i,k,2) = (1.0/detjac)*(ur*rz + us*sz + ut*tz);
          }
        }
      }
    }

	  #endif

	  #ifdef _GPU

    if (opp_5_sparse==0)
    {
      for (int i=0;i<n_dims;i++)
      {
	      cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_5(i).get_ptr_gpu(),Astride,delta_disu_fpts.get_ptr_gpu(),Bstride,1.0,grad_disu_upts.get_ptr_gpu(0,0,0,i),Cstride);
      }
    }
    else if (opp_5_sparse==1)
    {
      for (int i=0;i<n_dims;i++)
      {
        bespoke_SPMV(Arows, Acols, n_fields, n_eles, opp_5_ell_data(i).get_ptr_gpu(), opp_5_ell_indices(i).get_ptr_gpu(), opp_5_nnz_per_row(i), delta_disu_fpts.get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(0,0,0,i), ele_type, order, 1);
      }
    }

	  transform_grad_disu_upts_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, grad_disu_upts.get_ptr_gpu(), detjac_upts.get_ptr_gpu(), inv_detjac_mul_jac_upts.get_ptr_gpu(), run_input.equation);

	  #endif

  }

  /*
  for (int i=0;i<n_fpts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
      for (int k=0;k<n_fields;k++)
      {
          if (ele2global_ele(j)==53)
          {
            cout << setprecision(10)  << i << " " << ele2global_ele(j) << " " << k << " " << " " << delta_disu_fpts(i,j,k) << endl;
          }
       }
  */

  /*
  cout << "OUTPUT" << endl;
  #ifdef _GPU
    grad_disu_upts.cp_gpu_cpu();
  #endif

  for (int i=0;i<n_upts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
      for (int k=0;k<n_fields;k++)
        for (int m=0;m<n_dims;m++)
        {
          if (ele2global_ele(j)==53)
          {
            cout << setprecision(10)  << i << " " << ele2global_ele(j) << " " << k << " " << m << " " << grad_disu_upts(i,j,k,m) << endl;
          }
        }
  */
}


// calculate corrected gradient of the discontinuous solution at flux points

void eles::calc_cor_grad_disu_fpts(void)
{
  if (n_eles!=0)
  {
	  Arows =	n_fpts_per_ele;
	  Acols = n_upts_per_ele;

	  Brows = Acols;
	  Bcols = n_fields*n_eles;

	  Astride = Arows;
	  Bstride = Brows;
	  Cstride = Arows;

    #ifdef _CPU

	  if(opp_6_sparse==0) // dense
	  {
	  	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

      for (int i=0;i<n_dims;i++)
      {
	  	  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_6.get_ptr_cpu(),Astride,grad_disu_upts.get_ptr_cpu(0,0,0,i),Bstride,0.0,grad_disu_fpts.get_ptr_cpu(0,0,0,i),Cstride);
      }
	  	#endif
	  }
	  else if(opp_6_sparse==1) // mkl blas four-array csr format
	  {
	  	#if defined _MKL_BLAS

	  	// implement

	  	#endif
	  }
	  else
	  {
	  	cout << "ERROR: Unknown storage for opp_6 ... " << endl;
	  }

	  #endif

	  #ifdef _GPU

    if (opp_6_sparse==0)
    {
      for (int i=0;i<n_dims;i++)
      {
	      cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_6.get_ptr_gpu(),Astride,grad_disu_upts.get_ptr_gpu(0,0,0,i),Bstride,0.0,grad_disu_fpts.get_ptr_gpu(0,0,0,i),Cstride);
      }
    }
    else if (opp_6_sparse==1)
    {
      for (int i=0;i<n_dims;i++)
      {
        bespoke_SPMV(Arows, Acols, n_fields, n_eles, opp_6_ell_data.get_ptr_gpu(), opp_6_ell_indices.get_ptr_gpu(), opp_6_nnz_per_row, grad_disu_upts.get_ptr_gpu(0,0,0,i), grad_disu_fpts.get_ptr_gpu(0,0,0,i), ele_type, order, 0);
      }
    }

	  #endif

  }

  /*
  cout << "OUTPUT" << endl;
#ifdef _GPU
  grad_disu_fpts.cp_gpu_cpu();
#endif

  for (int i=0;i<n_fpts_per_ele;i++)
    for (int j=0;j<n_eles;j++)
      for (int k=0;k<n_fields;k++)
        for (int m=0;m<n_dims;m++)
        cout << setprecision(10)  << i << " " << j<< " " << k << " " << m << " " << grad_disu_fpts(i,j,k,m) << endl;
  */
}

/*! If at first RK step and using certain LES models, compute some model-related quantities.
If using similarity or WALE-similarity (WSM) models, compute filtered solution and Leonard tensors.
If using spectral vanishing viscosity (SVV) model, compute filtered solution. */

void eles::calc_sgs_terms(int in_disu_upts_from)
{
  if (n_eles!=0) {

		int i,j,k,l;
		int dim3;
		double diag, rsq;
		array <double> utemp(n_fields);

		/*! Filter solution */

  	Arows =	n_upts_per_ele;
  	Acols = n_upts_per_ele;
  	Brows = Acols;
  	Bcols = n_fields*n_eles;

  	Astride = Arows;
  	Bstride = Brows;
  	Cstride = Arows;

		//zero_array(disuf_upts);
		//zero_array(Lu);
		//zero_array(Le);

#ifdef _CPU

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,filter_upts.get_ptr_cpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_cpu(),Bstride,0.0,disuf_upts.get_ptr_cpu(),Cstride);

#else

		/*! slow matrix multiplication */
		for(i=0;i<n_upts_per_ele;i++) {
			for(j=0;j<n_eles;j++) {
		  	for(k=0;k<n_fields;k++) {
					disuf_upts(i,j,k) = 0.0;
					for(l=0;l<n_upts_per_ele;l++) {
						disuf_upts(i,j,k) += filter_upts(i,l)*disu_upts(in_disu_upts_from)(l,j,k);
					}
				}
			}
		}

#endif

		/*! Check for NaNs */
		for(i=0;i<n_upts_per_ele;i++)
			for(j=0;j<n_eles;j++)
		  	for(k=0;k<n_fields;k++)
					if(isnan(disuf_upts(i,j,k)))
						FatalError("nan in filtered solution");

		/*! If SVV model, copy filtered solution back to solution */
		if(sgs_model==3)
			for(i=0;i<n_upts_per_ele;i++)
				for(j=0;j<n_eles;j++)
			  	for(k=0;k<n_fields;k++)
						disu_upts(in_disu_upts_from)(i,j,k) = disuf_upts(i,j,k);

		/*! If Similarity model, compute product terms and Leonard tensors */
		else if(sgs_model==2 || sgs_model==4) {

			/*! third dimension of Lu, uu arrays */
			if(n_dims==2)      dim3 = 3;
			else if(n_dims==3) dim3 = 6;

			/*! Calculate velocity and energy product arrays uu, ue */
			for(i=0;i<n_upts_per_ele;i++) {
				for(j=0;j<n_eles;j++) {
			  	for(k=0;k<n_fields;k++) {
						utemp(k) = disu_upts(in_disu_upts_from)(i,j,k);
					}

					rsq = utemp(0)*utemp(0);

					/*! note that product arrays are symmetric */
					if(n_dims==2) {
						/*! velocity-velocity product */
						uu(i,j,0) = utemp(1)*utemp(1)/rsq;
						uu(i,j,1) = utemp(2)*utemp(2)/rsq;
						uu(i,j,2) = utemp(1)*utemp(2)/rsq;

						/*! velocity-energy product */
						utemp(3) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2))/utemp(0); // internal energy*rho

						ue(i,j,0) = utemp(1)*utemp(3)/rsq;
						ue(i,j,1) = utemp(2)*utemp(3)/rsq;
					}
					else if(n_dims==3) {
						/*! velocity-velocity product */
						uu(i,j,0) = utemp(1)*utemp(1)/rsq;
						uu(i,j,1) = utemp(2)*utemp(2)/rsq;
						uu(i,j,2) = utemp(3)*utemp(3)/rsq;
						uu(i,j,3) = utemp(1)*utemp(2)/rsq;
						uu(i,j,4) = utemp(1)*utemp(3)/rsq;
						uu(i,j,5) = utemp(2)*utemp(3)/rsq;

						/*! velocity-energy product */
						utemp(4) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2)+utemp(3)*utemp(3))/utemp(0); // internal energy*rho

						ue(i,j,0) = utemp(1)*utemp(4)/rsq;
						ue(i,j,1) = utemp(2)*utemp(4)/rsq;
						ue(i,j,2) = utemp(3)*utemp(4)/rsq;
					}
				}
			}

			/*! Filter products uu and ue */

#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

  		Bcols = dim3*n_eles;

			cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,filter_upts.get_ptr_cpu(),Astride,uu.get_ptr_cpu(),Bstride,0.0,Lu.get_ptr_cpu(),Cstride);

  		Bcols = n_dims*n_eles;

			cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,filter_upts.get_ptr_cpu(),Astride,ue.get_ptr_cpu(),Bstride,0.0,Le.get_ptr_cpu(),Cstride);

#else

			/*! slow matrix multiplication */
			for(i=0;i<n_upts_per_ele;i++) {
				for(j=0;j<n_eles;j++) {

			  	for(k=0;k<dim3;k++)
						for(l=0;l<n_upts_per_ele;l++)
							Lu(i,j,k) += filter_upts(i,l)*uu(l,j,k);

			  	for(k=0;k<n_dims;k++)
						for(l=0;l<n_upts_per_ele;l++)
							Le(i,j,k) += filter_upts(i,l)*ue(l,j,k);

				}
			}

#endif

			/*! Subtract product of unfiltered quantities from Leonard tensors */
			for(i=0;i<n_upts_per_ele;i++) {
				for(j=0;j<n_eles;j++) {

					// filtered solution
			  	for(k=0;k<n_fields;k++)
						utemp(k) = disuf_upts(i,j,k);

					rsq = utemp(0)*utemp(0);

					if(n_dims==2) {

						Lu(i,j,0) -= (utemp(1)*utemp(1))/rsq;
						Lu(i,j,1) -= (utemp(2)*utemp(2))/rsq;
						Lu(i,j,2) -= (utemp(1)*utemp(2))/rsq;

						diag = (Lu(i,j,0)+Lu(i,j,1))/3.0;

						// internal energy*rho
						utemp(3) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2))/utemp(0);

						Le(i,j,0) = (Le(i,j,0) - utemp(1)*utemp(3))/rsq;
						Le(i,j,1) = (Le(i,j,1) - utemp(2)*utemp(3))/rsq;

					}
					else if(n_dims==3) {

						Lu(i,j,0) -= (utemp(1)*utemp(1))/rsq;
						Lu(i,j,1) -= (utemp(2)*utemp(2))/rsq;
						Lu(i,j,2) -= (utemp(3)*utemp(3))/rsq;
						Lu(i,j,3) -= (utemp(1)*utemp(2))/rsq;
						Lu(i,j,4) -= (utemp(1)*utemp(3))/rsq;
						Lu(i,j,5) -= (utemp(2)*utemp(3))/rsq;

						diag = (Lu(i,j,0)+Lu(i,j,1)+Lu(i,j,2))/3.0;

						// internal energy*rho
						utemp(4) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2)+utemp(3)*utemp(3))/utemp(0);

						Le(i,j,0) = (Le(i,j,0) - utemp(1)*utemp(4))/rsq;
						Le(i,j,1) = (Le(i,j,1) - utemp(2)*utemp(4))/rsq;
						Le(i,j,2) = (Le(i,j,2) - utemp(3)*utemp(4))/rsq;

					}

					/*! subtract diagonal from Lu */
					for (k=0;k<n_dims;++k) Lu(i,j,k) -= diag;

					//cout << "uf*uf = " << setprecision(10) << disuf_upts(i,j,1)*disuf_upts(i,j,1) <<", "<< disuf_upts(i,j,2)*disuf_upts(i,j,2) <<", "<< disuf_upts(i,j,1)*disuf_upts(i,j,2) << endl;
					//cout<<"uu = "<<setprecision(10)<<uu(i,j,0)<<", "<<uu(i,j,1)<<", "<<uu(i,j,2)<<endl;
					//cout<<"Leonard = "<<setprecision(10)<<Lu(i,j,0)<<", "<<Lu(i,j,1)<<", "<<Lu(i,j,2)<<endl;
				}
			}
		}

#endif

		/*! GPU version of the above */
#ifdef _GPU

		/*! Filter solution (CUDA BLAS library) */
		cublasDgemm('N','N',Arows,Bcols,Acols,1.0,filter_upts.get_ptr_gpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_gpu(),Bstride,0.0,disuf_upts.get_ptr_gpu(),Cstride);

		/*! Check for NaNs */
		/*disuf_upts.cp_gpu_cpu();

		for(i=0;i<n_upts_per_ele;i++)
			for(j=0;j<n_eles;j++)
		  	for(k=0;k<n_fields;k++)
					if(isnan(disuf_upts(i,j,k)))
						FatalError("nan in filtered solution");*/

		/*! If Similarity model */
		if(sgs_model==2 || sgs_model==4) {

			/*! compute product terms uu, ue (pass flag=0 to wrapper function) */
			calc_similarity_model_kernel_wrapper(0, n_fields, n_upts_per_ele, n_eles, n_dims, disu_upts(in_disu_upts_from).get_ptr_gpu(), disuf_upts.get_ptr_gpu(), uu.get_ptr_gpu(), ue.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu());

			/*! third dimension of Lu, uu arrays */
			if(n_dims==2)
				dim3 = 3;
			else if(n_dims==3)
				dim3 = 6;

  		Bcols = dim3*n_eles;

			/*! Filter product terms uu and ue */
			cublasDgemm('N','N',Arows,Bcols,Acols,1.0,filter_upts.get_ptr_gpu(),Astride,uu.get_ptr_gpu(),Bstride,0.0,Lu.get_ptr_gpu(),Cstride);

			Bcols = n_dims*n_eles;

			cublasDgemm('N','N',Arows,Bcols,Acols,1.0,filter_upts.get_ptr_gpu(),Astride,ue.get_ptr_gpu(),Bstride,0.0,Le.get_ptr_gpu(),Cstride);

		/*! compute Leonard tensors Lu, Le (pass flag=1 to wrapper function) */
			calc_similarity_model_kernel_wrapper(1, n_fields, n_upts_per_ele, n_eles, n_dims, disu_upts(in_disu_upts_from).get_ptr_gpu(), disuf_upts.get_ptr_gpu(), uu.get_ptr_gpu(), ue.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu());

		}

		/*! If SVV model, copy filtered solution back to original solution */
		else if(sgs_model==3) {
			for(i=0;i<n_upts_per_ele;i++) {
				for(j=0;j<n_eles;j++) {
			  	for(k=0;k<n_fields;k++) {
						disu_upts(in_disu_upts_from)(i,j,k) = disuf_upts(i,j,k);
					}
				}
			}
			/*! copy back to GPU */
			disu_upts(in_disu_upts_from).cp_cpu_gpu();
		}

#endif

  }
}

// calculate transformed discontinuous viscous flux at solution points

void eles::calc_tdisvisf_upts(int in_disu_upts_from)
{
  if (n_eles!=0)
  {
	  #ifdef _CPU

	  int i,j,k,l,m;
	  double detjac;
	  for(i=0;i<n_eles;i++)
	  {
			// Calculate viscous flux
	  	for(j=0;j<n_upts_per_ele;j++)
			{
				detjac = detjac_upts(j,i);
				//physical solution
				for(k=0;k<n_fields;k++)
				{
					temp_u(k)=disu_upts(in_disu_upts_from)(j,i,k);

					//physical gradient
					for (m=0;m<n_dims;m++)
					{
						temp_grad_u(k,m) = grad_disu_upts(j,i,k,m);
					}
				}
				if(n_dims==2)
				{
					calc_visf_2d(temp_u,temp_grad_u,temp_f);
				}
				else if(n_dims==3)
				{
					calc_visf_3d(temp_u,temp_grad_u,temp_f);
				}
				else
				{
					cout << "ERROR: Invalid number of dimensions ... " << endl;
				}

				// If LES or wall model, calculate SGS viscous flux
				if(LES || (wall_model > 0)) {

					calc_sgsf_upts(temp_u,temp_grad_u,detjac,i,j,temp_sgsf);

					// Add SGS or wall flux to viscous flux
					for(k=0;k<n_fields;k++)
						for(l=0;l<n_dims;l++)
							temp_f(k,l) += temp_sgsf(k,l);

				}

				// If LES, add SGS flux to global array (needed for interface flux calc)
				if(LES) {

					for(k=0;k<n_fields;k++) {
						for(l=0;l<n_dims;l++) {
							sgsf_upts(j,i,k,l) = 0.0;
							for(m=0;m<n_dims;m++) {
								sgsf_upts(j,i,k,l)+=inv_detjac_mul_jac_upts(j,i,l,m)*temp_sgsf(k,m);
							}
						}
					}
				}

				// Transform viscous flux
				for(k=0;k<n_fields;k++)
				{
					for(l=0;l<n_dims;l++)
					{
						for(m=0;m<n_dims;m++)
						{
							tdisf_upts(j,i,k,l)+=inv_detjac_mul_jac_upts(j,i,l,m)*temp_f(k,m);
						}
					}
				}
			}
		}
		#endif

		// TODO: modify GPU routine to account for new SGS flux array sgsf_upts
		#ifdef _GPU

		calc_tdisvisf_upts_gpu_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, ele_type, run_input.filter_ratio, LES, sgs_model, wall_model, run_input.wall_layer_t, wall_distance.get_ptr_gpu(), twall.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu(), disu_upts(in_disu_upts_from).get_ptr_gpu(), tdisf_upts.get_ptr_gpu(), sgsf_upts.get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(), detjac_upts.get_ptr_gpu(), inv_detjac_mul_jac_upts.get_ptr_gpu(), run_input.gamma, run_input.prandtl, run_input.rt_inf, run_input.mu_inf, run_input.c_sth, run_input.fix_vis, run_input.equation, run_input.diff_coeff);

		#endif

	}
}


// calculate source term for SA turbulence model at solution points
void eles::calc_src_term_SA()
{
    if (n_eles!=0)
    {
#ifdef _CPU

        int i,j,k,l,m;

        for(i=0; i<n_eles; i++)
        {
            for(j=0; j<n_upts_per_ele; j++)
            {
                // physical solution
                for(k=0; k<n_fields; k++)
                {
                    temp_u(k)=disu_upts(0)(j,i,k);
                }

                // physical gradient
                for(k=0; k<n_fields; k++)
                {
                    for (m=0; m<n_dims; m++)
                    {
                        temp_grad_u(k,m) = grad_disu_upts(j,i,k,m);
                    }
                }

                // source term
                if(n_dims==2)
                    calc_source_SA_2d(temp_u, temp_grad_u, wall_distance_mag(j,i), src_term(j,i,n_fields-1));
                else if(n_dims==3)
                    calc_source_SA_3d(temp_u, temp_grad_u, wall_distance_mag(j,i), src_term(j,i,n_fields-1));
                else
                    cout << "ERROR: Invalid number of dimensions ... " << endl;
            }
        }

#endif

//#ifdef _GPU
//        calc_tdisvisf_upts_gpu_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,disu_upts(in_disu_upts_from).get_ptr_gpu(),tdisf_upts.get_ptr_gpu(),grad_disu_upts.get_ptr_gpu(),detjac_upts.get_ptr_gpu(),inv_detjac_mul_jac_upts.get_ptr_gpu(),run_input.gamma,run_input.prandtl,run_input.rt_inf,run_input.mu_inf,run_input.c_sth,run_input.fix_vis,run_input.equation,run_input.diff_coeff);
//#endif

    }
}


// Calculate SGS flux
void eles::calc_sgsf_upts(array<double>& temp_u, array<double>& temp_grad_u, double& detjac, int ele, int upt, array<double>& temp_sgsf)
{
	int i,j,k;
	int eddy, sim, wall;
	double Cs;
	double diag=0.0;
	double Smod=0.0;
	double ke=0.0;
	double Pr=0.5; // turbulent Prandtl number
	double delta, mu, mu_t, vol;
	double rho, inte, rt_ratio;
	array<double> u(n_dims);
	array<double> drho(n_dims), dene(n_dims), dke(n_dims), de(n_dims);
	array<double> dmom(n_dims,n_dims), du(n_dims,n_dims), S(n_dims,n_dims);

	// quantities for wall model
  array<double> norm(n_dims);
  array<double> tau(n_dims,n_dims);
  array<double> Mrot(n_dims,n_dims);
  array<double> temp(n_dims,n_dims);
  array<double> urot(n_dims);
  array<double> tw(n_dims);
	double y, qw, utau, yplus;

	// primitive variables
	rho = temp_u(0);
	for (i=0;i<n_dims;i++) {
		u(i) = temp_u(i)/rho;
		ke += 0.5*pow(u(i),2);
	}
	inte = temp_u(n_fields-1)/rho - ke;

	// fluid properties
	rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
	mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
	mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

	// Initialize SGS flux array to zero
	zero_array(temp_sgsf);

	// Compute SGS flux using wall model if sufficiently close to solid boundary
	wall = 0;

	if(wall_model > 0) {

		// Magnitude of wall distance vector
		y = 0.0;
		for (i=0;i<n_dims;i++)
			y += wall_distance(upt,ele,i)*wall_distance(upt,ele,i);

		y = sqrt(y);

		// get subgrid momentum flux at previous timestep
		//utau = 0.0;
		for (i=0;i<n_dims;i++) {
			tw(i) = twall(upt,ele,i+1);
			//utau += tw(i)*tw(i);
		}
		// shear velocity
		//utau = pow((utau/rho/rho),0.25);

		// Wall distance in wall units
		//yplus = y*rho*utau/mu;

		if(y < run_input.wall_layer_t) wall = 1;
		//if(yplus < 100.0) wall = 1;
		//cout << "tw, y, y+ " << tw(0) << ", " << y << ", " << yplus << endl;
	}

	// calculate SGS flux from a wall model
	if(wall) {

		for (i=0;i<n_dims;i++) {
			// Get approximate normal from wall distance vector
			norm(i) = wall_distance(upt,ele,i)/y;
		}

		// subgrid energy flux from previous timestep
		qw = twall(upt,ele,n_fields-1);

		// Calculate local rotation matrix
		Mrot = calc_rotation_matrix(norm);

		// Rotate velocity to surface
		if(n_dims==2) {
			urot(0) = u(0)*Mrot(0,1)+u(1)*Mrot(1,1);
			urot(1) = 0.0;
		}
		else {
			urot(0) = u(0)*Mrot(0,1)+u(1)*Mrot(1,1)+u(2)*Mrot(2,1);
			urot(1) = u(0)*Mrot(0,2)+u(1)*Mrot(1,2)+u(2)*Mrot(2,2);
			urot(2) = 0.0;
		}

		// Calculate wall shear stress
		calc_wall_stress(rho,urot,inte,mu,run_input.prandtl,run_input.gamma,y,tw,qw);

		// correct the sign of wall shear stress and wall heat flux? - see SD3D

		// Set arrays for next timestep
		for(i=0;i<n_dims;++i) twall(upt,ele,i+1) = tw(i); // momentum flux

		twall(upt,ele,0)          = 0.0; // density flux
		twall(upt,ele,n_fields-1) = qw;  // energy flux

		// populate ndims*ndims rotated stress array
		zero_array(tau);

		for(i=0;i<n_dims-1;i++) tau(i+1,0) = tau(0,i+1) = tw(i);

		//cout << "tau " << setprecision(6) << endl;
		//tau.print();

		// rotate stress array back to Cartesian coordinates
		zero_array(temp);
		for(i=0;i<n_dims;++i)
			for(j=0;j<n_dims;++j)
				for(k=0;k<n_dims;++k)
					temp(i,j) += tau(i,k)*Mrot(k,j);

		zero_array(tau);
		for(i=0;i<n_dims;++i)
			for(j=0;j<n_dims;++j)
				for(k=0;k<n_dims;++k)
					tau(i,j) += Mrot(k,i)*temp(k,j);

		//cout << "tau " << setprecision(6) << endl;
		//tau.print();

		// set SGS fluxes
		for(i=0;i<n_dims;i++) {

			// density
			temp_sgsf(0,i) = 0.0;

			// velocity
			for(j=0;j<n_dims;j++) {
				temp_sgsf(j+1,i) = 0.5*(tau(i,j)+tau(j,i));
			}

			// energy
			temp_sgsf(n_fields-1,i) = qw*norm(i);
		}
		//cout<<"rank, wall flux: "<<rank<<endl;
		//temp_sgsf.print();
		//cout << endl;
	}

	// Free-stream SGS flux
	else {

		// Set wall shear stress to 0 to prevent NaNs
		//for(i=0;i<n_dims;++i)
			//twall(upt,ele,i) = 0.0;

		// 0: Smagorinsky, 1: WALE, 2: WALE-similarity, 3: SVV, 4: Similarity
		if(sgs_model==0) {
			eddy = 1;
			sim = 0;
		}
		else if(sgs_model==1) {
			eddy = 1;
			sim = 0;
		}
		else if(sgs_model==2) {
			eddy = 1;
			sim = 1;
		}
		else if(sgs_model==3) {
			eddy = 0;
			sim = 0;
		}
		else if(sgs_model==4) {
			eddy = 0;
			sim = 1;
		}
		else {
			FatalError("SGS model not implemented");
		}

		if(eddy==1) {

			// Delta is the cutoff length-scale representing local grid resolution.

			// OPTION 1. Approx resolution in 1D element. Interval is [-1:1]
			// Appropriate for quads, hexes and tris. Not sure about tets.
			//dlt = 2.0/order;

			// OPTION 2. Deardorff definition (Deardorff, JFM 1970)
			vol = (*this).calc_ele_vol(detjac);
			//delta = run_input.filter_ratio*pow(vol,1./n_dims);
			delta = pow(vol,1./n_dims);

			// OPTION 3. Suggested by Bardina, AIAA 1980:
			// delta = sqrt((dx^2+dy^2+dz^2)/3)

			// Filtered solution gradient
			for (i=0;i<n_dims;i++) {
				drho(i) = temp_grad_u(0,i); // density gradient
				dene(i) = temp_grad_u(n_fields-1,i); // energy gradient

				for (j=1;j<n_fields-1;j++) {
					dmom(i,j-1) = temp_grad_u(j,i); // momentum gradients
				}
			}

			// Velocity and energy gradients
			for (i=0;i<n_dims;i++) {
				dke(i) = ke*drho(i);

				for (j=0;j<n_dims;j++) {
					du(i,j) = (dmom(i,j)-u(j)*drho(j))/rho;
					dke(i) += rho*u(j)*du(i,j);
				}

				de(i) = (dene(i)-dke(i)-drho(i)*inte)/rho;
			}

			// Strain rate tensor
			for (i=0;i<n_dims;i++) {
				for (j=0;j<n_dims;j++) {
					S(i,j) = (du(i,j)+du(j,i))/2.0;
				}
				diag += S(i,i)/3.0;
			}

			// Subtract diag
			for (i=0;i<n_dims;i++) S(i,i) -= diag;

			// Strain modulus
			for (i=0;i<n_dims;i++)
				for (j=0;j<n_dims;j++)
					Smod += 2.0*S(i,j)*S(i,j);

			Smod = sqrt(Smod);

			// Eddy viscosity

			// Smagorinsky model
			if(sgs_model==0) {

				Cs=0.1;
				mu_t = rho*Cs*Cs*delta*delta*Smod;

			}

			//  Wall-Adapting Local Eddy-viscosity (WALE) SGS Model
			//
			//  NICOUD F., DUCROS F.: "Subgrid-Scale Stress Modelling Based on the Square
			//                         of the Velocity Gradient Tensor"
			//  Flow, Turbulence and Combustion 62: 183-200, 1999.
			//
			//                                            (sqij*sqij)^3/2
			//  Output: mu_t = rho*Cs^2*delta^2 * -----------------------------
			//                                     (Sij*Sij)^5/2+(sqij*sqij)^5/4
			//
			//  Typically Cw = 0.5.

			else if(sgs_model==1 || sgs_model==2) {

				Cs=0.5;
				double num=0.0;
				double denom=0.0;
				double eps=1.e-12;
				array<double> Sq(n_dims,n_dims);
				diag = 0.0;

				// Square of gradient tensor
				// This needs optimising!
				for (i=0;i<n_dims;i++) {
					for (j=0;j<n_dims;j++) {
						Sq(i,j) = 0.0;
						for (k=0;k<n_dims;++k) {
							Sq(i,j) += (du(i,k)*du(k,j)+du(j,k)*du(k,i))/2.0;
						}
						diag += du(i,j)*du(j,i)/3.0;
					}
				}

				// Subtract diag
				for (i=0;i<n_dims;i++) Sq(i,i) -= diag;

				// Numerator and denominator
				for (i=0;i<n_dims;i++) {
					for (j=0;j<n_dims;j++) {
						num += Sq(i,j)*Sq(i,j);
						denom += S(i,j)*S(i,j);
					}
				}

				denom = pow(denom,2.5) + pow(num,1.25);
				num = pow(num,1.5);
				mu_t = rho*Cs*Cs*delta*delta*num/(denom+eps);
			}

			// Add eddy-viscosity term to SGS fluxes
			for (j=0;j<n_dims;j++) {
				temp_sgsf(0,j) = 0.0; // Density flux
				temp_sgsf(n_fields-1,j) = -1.0*run_input.gamma*mu_t/Pr*de(j); // Energy flux

				for (i=1;i<n_fields-1;i++) {
					temp_sgsf(i,j) = -2.0*mu_t*S(i-1,j); // Velocity flux
				}
			}
		}

		// Add similarity term to SGS fluxes if WSM or Similarity model
		if(sim==1) {
			for (j=0;j<n_dims;j++) {
				temp_sgsf(0,j) += 0.0; // Density flux
				temp_sgsf(n_fields-1,j) += run_input.gamma*rho*Le(upt,ele,j); // Energy flux
			}

			// Momentum fluxes
			if(n_dims==2) {
				temp_sgsf(1,0) += rho*Lu(upt,ele,0);
				temp_sgsf(1,1) += rho*Lu(upt,ele,2);
				temp_sgsf(2,0) += temp_sgsf(1,1);
				temp_sgsf(2,1) += rho*Lu(upt,ele,1);
			}
			else if(n_dims==3) {
				temp_sgsf(1,0) += rho*Lu(upt,ele,0);
				temp_sgsf(1,1) += rho*Lu(upt,ele,3);
				temp_sgsf(1,2) += rho*Lu(upt,ele,4);
				temp_sgsf(2,0) += temp_sgsf(1,1);
				temp_sgsf(2,1) += rho*Lu(upt,ele,1);
				temp_sgsf(2,2) += rho*Lu(upt,ele,5);
				temp_sgsf(3,0) += temp_sgsf(1,2);
				temp_sgsf(3,1) += temp_sgsf(2,2);
				temp_sgsf(3,2) += rho*Lu(upt,ele,2);
			}

			//cout<<"Leonard = "<<setprecision(10)<<Lu(upt,ele,0)<<", "<<Lu(upt,ele,1)<<", "<<Lu(upt,ele,2)<<endl;

		}
	}
	//cout<<"rank, sgs flux: "<<rank<<endl;
	//temp_sgsf.print();
	//cout << endl;
}

/*! If using a RANS or LES near-wall model, calculate distance
of each solution point to nearest no-slip wall by a brute-force method */

void eles::calc_wall_distance(int n_seg_noslip_inters, int n_tri_noslip_inters, int n_quad_noslip_inters, array< array<double> > loc_noslip_bdy)
{
	if(n_eles!=0)
	{
		int i,j,k,m,n,p;
		int n_fpts_per_inter_seg = order+1;
		int n_fpts_per_inter_tri = (order+2)*(order+1)/2;
		int n_fpts_per_inter_quad = (order+1)*(order+1);
		double dist;
		double distmin;
	  array<double> pos(n_dims);
	  array<double> pos_bdy(n_dims);
	  array<double> vec(n_dims);
	  array<double> vecmin(n_dims);

		// hold our breath and go round the brute-force loop...
		for (i=0;i<n_eles;++i) {
			for (j=0;j<n_upts_per_ele;++j) {

				// get coords of current solution point
				calc_pos_upt(j,i,pos);
				//cout << "rank, coords " << rank << endl;
				//pos.print();

				// initialize wall distance
				distmin = 1e20;

				// line segment boundaries
				for (k=0;k<n_seg_noslip_inters;++k) {

					for (m=0;m<n_fpts_per_inter_seg;++m) {

						dist = 0.0;
						// get coords of boundary flux point
						for (n=0;n<n_dims;++n) {
							pos_bdy(n) = loc_noslip_bdy(0)(m,k,n);
							vec(n) = pos(n) - pos_bdy(n);
							dist += vec(n)*vec(n);
						}
						dist = sqrt(dist);

						//cout << "rank, coords_bdy " << rank << endl;
						//pos_bdy.print();

						// update shortest vector
						if (dist < distmin) {
							for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
							distmin = dist;
						}
					}
				}

				// tri boundaries
				for (k=0;k<n_tri_noslip_inters;++k) {

					for (m=0;m<n_fpts_per_inter_tri;++m) {

						dist = 0.0;
						// get coords of boundary flux point
						for (n=0;n<n_dims;++n) {
							pos_bdy(n) = loc_noslip_bdy(1)(m,k,n);
							vec(n) = pos(n) - pos_bdy(n);
							dist += vec(n)*vec(n);
						}
						dist = sqrt(dist);

						// update shortest vector
						if (dist < distmin) {
							for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
							distmin = dist;
						}
					}
				}

				// quad boundaries
				for (k=0;k<n_quad_noslip_inters;++k) {

					for (m=0;m<n_fpts_per_inter_quad;++m) {

						dist = 0.0;
						// get coords of boundary flux point
						for (n=0;n<n_dims;++n) {
							pos_bdy(n) = loc_noslip_bdy(2)(m,k,n);
							vec(n) = pos(n) - pos_bdy(n);
							dist += vec(n)*vec(n);
						}
						dist = sqrt(dist);

						// update shortest vector
						if (dist < distmin) {
							for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
							distmin = dist;
						}
					}
				}
			}

			for (n=0;n<n_dims;++n) wall_distance(j,i,n) = vecmin(n);
			wall_distance_mag(j,i) = distmin;

			//cout << "vec" << endl;
			//vecmin.print();
			//if(rank==0) cout << "rank, wall distance " << rank << ", " << distmin << endl;
			//if(rank==0) cout << endl;
		}
	}
}

#ifdef _MPI

void eles::calc_wall_distance_parallel(array<int> n_seg_inters_array, array<int> n_tri_inters_array, array<int> n_quad_inters_array, array< array<double> > loc_noslip_bdy_global, int nproc)
{
	if(n_eles!=0)
	{
		int i,j,k,m,n,p;
		int n_fpts_per_inter_seg = order+1;
		int n_fpts_per_inter_tri = (order+2)*(order+1)/2;
		int n_fpts_per_inter_quad = (order+1)*(order+1);
		double dist;
		double distmin;
	  array<double> pos(n_dims);
	  array<double> pos_bdy(n_dims);
	  array<double> vec(n_dims);
	  array<double> vecmin(n_dims);

		//if (rank==0) cout << "rank, nproc " << rank << endl;


		// hold our breath and go round the brute-force loop...
		for (i=0;i<n_eles;++i) {
			for (j=0;j<n_upts_per_ele;++j) {

				// get coords of current solution point
				calc_pos_upt(j,i,pos);
				//if (rank==0) cout << "rank, coords " << rank << endl;
				//if (rank==0) pos.print();

				// initialize wall distance
				distmin = 1e20;

				// loop over all partitions
				for (p=0;p<nproc;++p) {

					//if (rank==0) cout << "p, n_seg_inters(p) " << p << ", " << n_seg_inters_array(p) << endl;

					// line segment boundaries
					for (k=0;k<n_seg_inters_array(p);++k) {

						for (m=0;m<n_fpts_per_inter_seg;++m) {

							dist = 0.0;
							// get coords of boundary flux point
							for (n=0;n<n_dims;++n) {
								pos_bdy(n) = loc_noslip_bdy_global(0)(m,k,p*n_dims+n);
								vec(n) = pos(n) - pos_bdy(n);
								dist += vec(n)*vec(n);
							}
							dist = sqrt(dist);

							//if (rank==0) cout << "pos_bdy" << endl;
							//if (rank==0) pos_bdy.print();

							// update shortest vector
							if (dist < distmin) {
								for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
								distmin = dist;
							}
						}
					}

					// tri boundaries
					for (k=0;k<n_tri_inters_array(p);++k) {

						for (m=0;m<n_fpts_per_inter_tri;++m) {

							dist = 0.0;
							// get coords of boundary flux point
							for (n=0;n<n_dims;++n) {
								pos_bdy(n) = loc_noslip_bdy_global(1)(m,k,p*n_dims+n);
								vec(n) = pos(n) - pos_bdy(n);
								dist += vec(n)*vec(n);
							}
							dist = sqrt(dist);

							// update shortest vector
							if (dist < distmin) {
								for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
								distmin = dist;
							}
						}
					}

					// quad boundaries
					for (k=0;k<n_quad_inters_array(p);++k) {

						for (m=0;m<n_fpts_per_inter_quad;++m) {

							dist = 0.0;
							// get coords of boundary flux point
							for (n=0;n<n_dims;++n) {
								pos_bdy(n) = loc_noslip_bdy_global(2)(m,k,p*n_dims+n);
								vec(n) = pos(n) - pos_bdy(n);
								dist += vec(n)*vec(n);
							}
							dist = sqrt(dist);

							// update shortest vector
							if (dist < distmin) {
								for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
								distmin = dist;
							}
						}
					}
				}

				for (n=0;n<n_dims;++n) wall_distance(j,i,n) = vecmin(n);
				wall_distance_mag(j,i) = distmin;

				//if (rank==0) cout << "vecmin" << endl;
				//if (rank==0) vecmin.print();
				//if (rank==0) cout << "rank, wall distance " << rank << ", " << distmin << endl;
				//if (rank==0) cout << endl;
			}
		}
	}
}

#endif

array<double> eles::calc_rotation_matrix(array<double>& norm)
{
	array <double> mrot(n_dims,n_dims);
	double nn;

	//cout << "norm "<< norm(0) << ", " << norm(1) << endl;

	// Create rotation matrix
	if(n_dims==2) {
		if(abs(norm(1)) > 0.7) {
			mrot(0,0) = norm(0);
			mrot(1,0) = norm(1);
			mrot(0,1) = norm(1);
			mrot(1,1) = -norm(0);
		}
		else {
			mrot(0,0) = -norm(0);
			mrot(1,0) = -norm(1);
			mrot(0,1) = norm(1);
			mrot(1,1) = -norm(0);
		}
	}
	else if(n_dims==3) {
		if(abs(norm(2)) > 0.7) {
			nn = sqrt(norm(1)*norm(1)+norm(2)*norm(2));

			mrot(0,0) = norm(0)/nn;
			mrot(1,0) = norm(1)/nn;
			mrot(2,0) = norm(2)/nn;
			mrot(0,1) = 0.0;
			mrot(1,1) = -norm(2)/nn;
			mrot(2,1) = norm(1)/nn;
			mrot(0,2) = nn;
			mrot(1,2) = -norm(0)*norm(1)/nn;
			mrot(2,2) = -norm(0)*norm(2)/nn;
		}
		else {
			nn = sqrt(norm(0)*norm(0)+norm(1)*norm(1));

			mrot(0,0) = norm(0)/nn;
			mrot(1,0) = norm(1)/nn;
			mrot(2,0) = norm(2)/nn;
			mrot(0,1) = norm(1)/nn;
			mrot(1,1) = -norm(0)/nn;
			mrot(2,1) = 0.0;
			mrot(0,2) = norm(0)*norm(2)/nn;
			mrot(1,2) = norm(1)*norm(2)/nn;
			mrot(2,2) = -nn;
		}
	}

	return mrot;
}

void eles::calc_wall_stress(double rho, array<double>& urot, double ene, double mu, double Pr, double gamma, double y, array<double>& tau_wall, double q_wall)
{
	double eps = 1.e-10;
	double Rey, Rey_c, u, uplus, utau, tw, qw;
	double Pr_t = 0.9;
	double c0;
	double ymatch = 11.8;
	int i,j;

	// Magnitude of surface velocity
	u = 0.0;
	for(i=0;i<n_dims;++i) u += urot(i)*urot(i);

	u = sqrt(u);

	if(u > eps) {

		/*! Simple power-law wall model Werner and Wengle (1991)

		          u+ = y+               for y+ < 11.8
		          u+ = 8.3*(y+)^(1/7)   for y+ > 11.8
		*/

		if(run_input.wall_model == 1) {

			Rey_c = ymatch*ymatch;
			Rey = rho*u*y/mu;

			//cout << "Rey "<< Rey << endl;

			if(Rey < Rey_c) uplus = sqrt(Rey);
			else            uplus = pow(8.3,0.875)*pow(Rey,0.125);

			utau = u/uplus;
			tw = rho*utau*utau;

			//cout << "utau "<< utau << endl;

			for (i=0;i<n_dims;i++) tau_wall(i) = tw*urot(i)/u;

			// Wall heat flux
			if(Rey < Rey_c) q_wall = ene*gamma*tw / (Pr * u);
			else            q_wall = ene*gamma*tw / (Pr * (u + utau * sqrt(Rey_c) * (Pr/Pr_t-1.0)));
		}

		/*! Breuer-Rodi 3-layer wall model (Breuer and Rodi, 1996)

		          u+ = y+               for y+ <= 5.0
		          u+ = A*ln(y+)+B       for 5.0 < y+ <= 30.0
		          u+ = ln(E*y+)/k       for y+ > 30.0

		          k=0.42, E=9.8
		          A=(log(30.0*E)/k-5.0)/log(6.0)
		          B=5.0-A*log(5.0)

		Note: the law of wall is made algebraic by first guessing the friction
		velocity with the wall shear at the previous timestep

	  N.B. using a two-layer law to compute the wall heat flux
		*/

		else if(run_input.wall_model == 2) {

			double A, B, phi;
			double E = 9.8;
			double Rey0, ReyL, ReyH, ReyM;
			double yplus, yplusL, yplusH, yplusM, yplusN;
			double kappa = 0.42;
			double sign, s;
			int maxit = 0;
			int it;

			A = (log(30.0*E)/kappa - 5.0)/log(6.0);
			B = 5.0 - A*log(5.0);

			// compute wall distance in wall units
			phi = rho*y/mu;
			Rey0 = u*phi;
			utau = 0.0;
			for (i=0;i<n_dims;i++)
				utau += tau_wall(i)*tau_wall(i);

			utau = pow((utau/rho/rho),0.25);
			yplus = utau*phi;

			if(maxit > 0) {
				Rey = wallfn_br(yplus,A,B,E,kappa);

				// if in the
				if(Rey > Rey0) {
					yplusH = yplus;
					ReyH = Rey-Rey0;
					yplusL = yplus*Rey0/Rey;

					ReyL = wallfn_br(yplusL,A,B,E,kappa);
					ReyL -= Rey0;

					it = 0;
					while(ReyL*ReyH >= 0.0 && it < maxit) {

						yplusL -= 1.6*(yplusH-yplusL);
						ReyL = wallfn_br(yplusL,A,B,E,kappa);
						ReyL -= Rey0;
						++it;

					}
				}
				else {
					yplusH = yplus;
					ReyH = Rey-Rey0;

					if(Rey > eps) yplusH = yplus*Rey0/Rey;
					else yplusH = 2.0*yplusL;

					ReyH = wallfn_br(yplusH,A,B,E,kappa);
					ReyH -= Rey0;

					it = 0;
					while(ReyL*ReyH >= 0.0 && it < maxit) {

						yplusH += 1.6*(yplusH - yplusL);
						ReyH = wallfn_br(yplusH,A,B,E,kappa);
						ReyH -= Rey0;
						++it;

					}
				}

				// iterative solution by Ridders' Method

				yplus = 0.5*(yplusL+yplusH);

				for(it=0;it<maxit;++it) {

					yplusM = 0.5*(yplusL+yplusH);
					ReyM = wallfn_br(yplusM,A,B,E,kappa);
					ReyM -= Rey0;
					s = sqrt(ReyM*ReyM - ReyL*ReyH);
					if(s==0.0) break;

					sign = (ReyL-ReyH)/abs(ReyL-ReyH);
					yplusN = yplusM + (yplusM-yplusL)*(sign*ReyM/s);
					if(abs(yplusN-yplus) < eps) break;

					yplus = yplusN;
					Rey = wallfn_br(yplus,A,B,E,kappa);
					Rey -= Rey0;
					if(abs(Rey) < eps) break;

					if(Rey/abs(Rey)*ReyM != ReyM) {
						yplusL = yplusM;
						ReyL = ReyM;
						yplusH = yplus;
						ReyH = Rey;
					}
					else if(Rey/abs(Rey)*ReyL != ReyL) {
						yplusH = yplus;
						ReyH = Rey;
					}
					else if(Rey/abs(Rey)*ReyH != ReyH) {
						yplusL = yplus;
						ReyL = Rey;
					}

					if(abs(yplusH-yplusL) < eps) break;
				} // end for loop

				utau = u*yplus/Rey0;
			}

			// approximate solution using tw at previous timestep
			// Wang, Moin (2002), Phys.Fluids 14(7)
			else {
				if(Rey > eps) utau = u*yplus/Rey;
				else          utau = 0.0;
				yplus = utau*phi;
			}

			tw = rho*utau*utau;

			// why different to WW model?
			for (i=0;i<n_dims;i++) tau_wall(i) = abs(tw*urot(i)/u);

			// Wall heat flux
			if(yplus <= ymatch) q_wall = ene*gamma*tw / (Pr * u);
			else                q_wall = ene*gamma*tw / (Pr * (u + utau * ymatch * (Pr/Pr_t-1.0)));
		}
	}

	// if velocity is 0
	else {
		for (i=0;i<n_dims;i++) tau_wall(i) = 0.0;
		q_wall = 0.0;
	}
}

double eles::wallfn_br(double yplus, double A, double B, double E, double kappa) {
	double Rey;

	if     (yplus < 0.5)  Rey = yplus*yplus;
	else if(yplus > 30.0) Rey = yplus*log(E*yplus)/kappa;
	else                  Rey = yplus*(A*log(yplus)+B);

	return Rey;
}

/*! Calculate SGS flux at solution points */
void eles::calc_sgsf_fpts(void)
{
	if (n_eles!=0) {

		/*!
		Performs C = (alpha*A*B) + (beta*C) where: \n
		alpha = 1.0 \n
		beta = 0.0 \n
		A = opp_0 \n
		B = sgsf_upts \n
		C = sgsf_fpts
		*/

		Arows =	n_fpts_per_ele;
		Acols = n_upts_per_ele;

		Brows = Acols;
		Bcols = n_fields*n_eles;

		Astride = Arows;
		Bstride = Brows;
		Cstride = Arows;

		#ifdef _CPU

		if(opp_0_sparse==0) // dense
		{
			#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

			for (int i=0;i<n_dims;i++) {
				cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_0.get_ptr_cpu(),Astride,sgsf_upts.get_ptr_cpu(0,0,0,i),Bstride,0.0,sgsf_fpts.get_ptr_cpu(0,0,0,i),Cstride);
			}

			#endif
	  }
		else if(opp_0_sparse==1) // mkl blas four-array csr format
		{
			#if defined _MKL_BLAS

			for (int i=0;i<n_dims;i++) {
				mkl_dcsrmm(&transa, &n_fpts_per_ele, &n_fields_mul_n_eles, &n_upts_per_ele, &one, matdescra, opp_0_data.get_ptr_cpu(), opp_0_cols.get_ptr_cpu(), opp_0_b.get_ptr_cpu(), opp_0_e.get_ptr_cpu(), sgsf_upts.get_ptr_cpu(0,0,0,i), &n_upts_per_ele, &zero, sgsf_fpts.get_ptr_cpu(0,0,0,i), &n_fpts_per_ele);
			}

			#endif
		}
		else { cout << "ERROR: Unknown storage for opp_0 ... " << endl; }

		#endif

		#ifdef _GPU

		if(opp_0_sparse==0)
    {
			for (int i=0;i<n_dims;i++) {
				cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_0.get_ptr_gpu(),Astride,sgsf_upts.get_ptr_gpu(0,0,0,i),Bstride,0.0,sgsf_fpts.get_ptr_gpu(0,0,0,i),Cstride);
			}
		}
		else if (opp_0_sparse==1)
		{
			for (int i=0;i<n_dims;i++) {
	      bespoke_SPMV(n_fpts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_0_ell_data.get_ptr_gpu(), opp_0_ell_indices.get_ptr_gpu(), opp_0_nnz_per_row, sgsf_upts.get_ptr_gpu(0,0,0,i), sgsf_fpts.get_ptr_gpu(0,0,0,i), ele_type, order, 0);
			}
		}
		else
		{
			cout << "ERROR: Unknown storage for opp_0 ... " << endl;
		}
		#endif
	}
}

// get the type of element

int eles::get_ele_type(void)
{
	return ele_type;
}

// get number of elements

int eles::get_n_eles(void)
{
	return n_eles;
}

// get number of ppts_per_ele
int eles::get_n_ppts_per_ele(void)
{
	return n_ppts_per_ele;
}

// get number of peles_per_ele
int eles::get_n_peles_per_ele(void)
{
	return n_peles_per_ele;
}

// get number of verts_per_ele
int eles::get_n_verts_per_ele(void)
{
	return n_verts_per_ele;
}

// get number of elements

int eles::get_n_dims(void)
{
	return n_dims;
}

// get number of elements

int eles::get_n_fields(void)
{
	return n_fields;
}

// get number of solutions points per element

int eles::get_n_upts_per_ele(void)
{
	return n_upts_per_ele;
}

// set the shape array
void eles::set_shape(int in_max_n_spts_per_ele)
{
	shape.setup(n_dims,in_max_n_spts_per_ele,n_eles);
  n_spts_per_ele.setup(n_eles);
}

// set a shape node

void eles::set_shape_node(int in_spt, int in_ele, array<double>& in_pos)
{
	for(int i=0;i<n_dims;i++)
  {
		shape(i,in_spt,in_ele)=in_pos(i);
  }
}

void eles::set_rank(int in_rank)
{
    rank = in_rank;
}

// set bc type
void eles::set_bctype(int in_ele,int in_inter, int in_bctype)
{
  bctype(in_ele, in_inter) = in_bctype;
}

// set number of shape points

void eles::set_n_spts(int in_ele, int in_n_spts)
{
  n_spts_per_ele(in_ele) = in_n_spts;

  // Allocate storage for the s_nodal_basis

  d_nodal_s_basis.setup(in_n_spts,n_dims);

  int n_comp;
	if(n_dims == 2)
		n_comp = 3;
  else if(n_dims == 3)
		n_comp = 6;

  dd_nodal_s_basis.setup(in_n_spts,n_comp);

}

// set global element number

void eles::set_ele2global_ele(int in_ele, int in_global_ele)
{
  ele2global_ele(in_ele) = in_global_ele;
}


// set opp_0 (transformed discontinuous solution at solution points to transformed discontinuous solution at flux points)

void eles::set_opp_0(int in_sparse)
{
	int i,j,k;

	array<double> loc(n_dims);

	opp_0.setup(n_fpts_per_ele,n_upts_per_ele);

	for(i=0;i<n_upts_per_ele;i++)
	{
		for(j=0;j<n_fpts_per_ele;j++)
		{
			for(k=0;k<n_dims;k++)
			{
				loc(k)=tloc_fpts(k,j);
			}

			opp_0(j,i)=eval_nodal_basis(i,loc);
		}
	}

#ifdef _GPU
	  opp_0.cp_cpu_gpu();
#endif

//cout << "opp_0" << endl;
//cout << "ele_type=" << ele_type << endl;
//opp_0.print();
//cout << endl;

	if(in_sparse==0)
	{
		opp_0_sparse=0;
	}
	else if(in_sparse==1)
	{
		opp_0_sparse=1;

#ifdef _CPU
		array_to_mklcsr(opp_0,opp_0_data,opp_0_cols,opp_0_b,opp_0_e);
#endif

#ifdef _GPU
    array_to_ellpack(opp_0, opp_0_ell_data, opp_0_ell_indices, opp_0_nnz_per_row);
    opp_0_ell_data.cp_cpu_gpu();
    opp_0_ell_indices.cp_cpu_gpu();
#endif

  }
  else
  {
		cout << "ERROR: Invalid sparse matrix form ... " << endl;
	}



}

// set opp_1 (transformed discontinuous flux at solution points to normal transformed discontinuous flux at flux points)

void eles::set_opp_1(int in_sparse)
{
	int i,j,k,l;
	array<double> loc(n_dims);

  opp_1.setup(n_dims);
  for (int i=0;i<n_dims;i++)
	  opp_1(i).setup(n_fpts_per_ele,n_upts_per_ele);

	for(i=0;i<n_dims;i++)
	{
		for(j=0;j<n_upts_per_ele;j++)
		{
			for(k=0;k<n_fpts_per_ele;k++)
			{
				for(l=0;l<n_dims;l++)
				{
					loc(l)=tloc_fpts(l,k);
				}

				opp_1(i)(k,j)=eval_nodal_basis(j,loc)*tnorm_fpts(i,k);
			}
		}
    //cout << "opp_1,i =" << i << endl;
    //cout << "ele_type=" << ele_type << endl;
    //opp_1(i).print();
    //cout << endl;
	}

#ifdef _GPU
    for (int i=0;i<n_dims;i++)
	    opp_1(i).cp_cpu_gpu();
#endif


	if(in_sparse==0)
	{
		opp_1_sparse=0;
	}
	else if(in_sparse==1)
	{
		opp_1_sparse=1;

#ifdef _CPU
    for (int i=0;i<n_dims;i++) {
		  array_to_mklcsr(opp_1(i),opp_1_data(i),opp_1_cols(i),opp_1_b(i),opp_1_e(i));
    }
#endif

#ifdef _GPU
    opp_1_ell_data.setup(n_dims);
    opp_1_ell_indices.setup(n_dims);
    opp_1_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_1(i), opp_1_ell_data(i), opp_1_ell_indices(i), opp_1_nnz_per_row(i));
      opp_1_ell_data(i).cp_cpu_gpu();
      opp_1_ell_indices(i).cp_cpu_gpu();
    }
#endif

  }
	else
	{
		cout << "ERROR: Invalid sparse matrix form ... " << endl;
	}
}

// set opp_2 (transformed discontinuous flux at solution points to divergence of transformed discontinuous flux at solution points)

void eles::set_opp_2(int in_sparse)
{

	int i,j,k,l;

	array<double> loc(n_dims);

  opp_2.setup(n_dims);
  for (int i=0;i<n_dims;i++)
	  opp_2(i).setup(n_upts_per_ele,n_upts_per_ele);

	for(i=0;i<n_dims;i++)
	{
		for(j=0;j<n_upts_per_ele;j++)
		{
			for(k=0;k<n_upts_per_ele;k++)
			{
				for(l=0;l<n_dims;l++)
				{
					loc(l)=loc_upts(l,k);
				}

				opp_2(i)(k,j)=eval_d_nodal_basis(j,i,loc);
			}
		}

    //cout << "opp_2,i =" << i << endl;
    //cout << "ele_type=" << ele_type << endl;
    //opp_2(i).print();
    //cout << endl;

    //cout << "opp_2,i=" << i << endl;
    //opp_2(i).print();

	}

#ifdef _GPU
    for (int i=0;i<n_dims;i++)
	    opp_2(i).cp_cpu_gpu();
#endif

  //cout << "opp 2" << endl;
  //opp_2.print();

	if(in_sparse==0)
	{
		opp_2_sparse=0;
	}
	else if(in_sparse==1)
	{
		opp_2_sparse=1;

#ifdef _CPU
    for (int i=0;i<n_dims;i++) {
		  array_to_mklcsr(opp_2(i),opp_2_data(i),opp_2_cols(i),opp_2_b(i),opp_2_e(i));
    }
#endif

#ifdef _GPU
    opp_2_ell_data.setup(n_dims);
    opp_2_ell_indices.setup(n_dims);
    opp_2_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_2(i), opp_2_ell_data(i), opp_2_ell_indices(i), opp_2_nnz_per_row(i));
      opp_2_ell_data(i).cp_cpu_gpu();
      opp_2_ell_indices(i).cp_cpu_gpu();
    }
#endif
	}
	else
	{
		cout << "ERROR: Invalid sparse matrix form ... " << endl;
	}
}

// set opp_3 (normal transformed correction flux at edge flux points to divergence of transformed correction flux at solution points)

void eles::set_opp_3(int in_sparse)
{

	opp_3.setup(n_upts_per_ele,n_fpts_per_ele);
  (*this).fill_opp_3(opp_3);

  //cout << "OPP_3" << endl;
  //cout << "ele_type=" << ele_type << endl;
  //opp_3.print();
  //cout << endl;

#ifdef _GPU
	opp_3.cp_cpu_gpu();
#endif

	if(in_sparse==0)
	{
		opp_3_sparse=0;
	}
	else if(in_sparse==1)
	{
		opp_3_sparse=1;

#ifdef _CPU
		array_to_mklcsr(opp_3,opp_3_data,opp_3_cols,opp_3_b,opp_3_e);
#endif

#ifdef _GPU
    array_to_ellpack(opp_3, opp_3_ell_data, opp_3_ell_indices, opp_3_nnz_per_row);
    opp_3_ell_data.cp_cpu_gpu();
    opp_3_ell_indices.cp_cpu_gpu();
#endif
  }
  else
	{
		cout << "ERROR: Invalid sparse matrix form ... " << endl;
	}
}

// set opp_4 (transformed solution at solution points to transformed gradient of transformed solution at solution points)

void eles::set_opp_4(int in_sparse)
{
	int i,j,k,l;

	array<double> loc(n_dims);

  opp_4.setup(n_dims);
  for (int i=0;i<n_dims;i++)
	  opp_4(i).setup(n_upts_per_ele, n_upts_per_ele);

	for(i=0; i<n_dims; i++)
	{
		for(j=0; j<n_upts_per_ele; j++)
		{
			for(k=0; k<n_upts_per_ele; k++)
			{
				for(l=0; l<n_dims; l++)
				{
					loc(l)=loc_upts(l,k);
				}

				opp_4(i)(k,j) = eval_d_nodal_basis(j,i,loc);
			}
		}
	}

#ifdef _GPU
    for (int i=0;i<n_dims;i++)
	    opp_4(i).cp_cpu_gpu();
#endif

	if(in_sparse==0)
	{
		opp_4_sparse=0;
	}
	else if(in_sparse==1)
	{
		opp_4_sparse=1;

#ifdef _CPU
    for (int i=0;i<n_dims;i++)
    {
		  array_to_mklcsr(opp_4(i),opp_4_data(i),opp_4_cols(i),opp_4_b(i),opp_4_e(i));
    }
#endif

#ifdef _GPU
    opp_4_ell_data.setup(n_dims);
    opp_4_ell_indices.setup(n_dims);
    opp_4_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_4(i), opp_4_ell_data(i), opp_4_ell_indices(i), opp_4_nnz_per_row(i));
      opp_4_ell_data(i).cp_cpu_gpu();
      opp_4_ell_indices(i).cp_cpu_gpu();
    }
#endif
	}
	else
	{
		cout << "ERROR: Invalid sparse matrix form ... " << endl;
	}
}

// transformed solution correction at flux points to transformed gradient correction at solution points

void eles::set_opp_5(int in_sparse)
{
	int i,j,k,l;

	array<double> loc(n_dims);

  opp_5.setup(n_dims);
  for (int i=0;i<n_dims;i++)
	  opp_5(i).setup(n_upts_per_ele, n_fpts_per_ele);

	for(i=0;i<n_dims;i++)
	{
		for(j=0;j<n_fpts_per_ele;j++)
		{
			for(k=0;k<n_upts_per_ele;k++)
			{
        /*
				for(l=0;l<n_dims;l++)
				{
					loc(l)=loc_upts(l,k);
				}
        */

				//opp_5(i)(k,j) = eval_div_vcjh_basis(j,loc)*tnorm_fpts(i,j);
				opp_5(i)(k,j) = opp_3(k,j)*tnorm_fpts(i,j);
			}
		}
	}

#ifdef _GPU
    for (int i=0;i<n_dims;i++)
	    opp_5(i).cp_cpu_gpu();
#endif

  //cout << "opp_5" << endl;
  //opp_5.print();

	if(in_sparse==0)
	{
		opp_5_sparse=0;
	}
	else if(in_sparse==1)
	{
		opp_5_sparse=1;

#ifdef _CPU
    for (int i=0;i<n_dims;i++) {
		  array_to_mklcsr(opp_5(i),opp_5_data(i),opp_5_cols(i),opp_5_b(i),opp_5_e(i));
    }
#endif

#ifdef _GPU
    opp_5_ell_data.setup(n_dims);
    opp_5_ell_indices.setup(n_dims);
    opp_5_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_5(i), opp_5_ell_data(i), opp_5_ell_indices(i), opp_5_nnz_per_row(i));
      opp_5_ell_data(i).cp_cpu_gpu();
      opp_5_ell_indices(i).cp_cpu_gpu();
    }
#endif
	}
	else
	{
		cout << "ERROR: Invalid sparse matrix form ... " << endl;
	}
}

// transformed gradient at solution points to transformed gradient at flux points

void eles::set_opp_6(int in_sparse)
{
	int i,j,k,l,m;

	array<double> loc(n_dims);

	opp_6.setup(n_fpts_per_ele, n_upts_per_ele);

	for(j=0; j<n_upts_per_ele; j++)
	{
	  for(l=0; l<n_fpts_per_ele; l++)
		{
		  for(m=0; m<n_dims; m++)
			{
			  loc(m) = tloc_fpts(m,l);
		  }
		  opp_6(l,j) = eval_nodal_basis(j,loc);
		}
	}

  //cout << "opp_6" << endl;
  //opp_6.print();

#ifdef _GPU
	opp_6.cp_cpu_gpu();
#endif

	if(in_sparse==0)
	{
		opp_6_sparse=0;
	}
	else if(in_sparse==1)
	{
		opp_6_sparse=1;

#ifdef _CPU
		array_to_mklcsr(opp_6,opp_6_data,opp_6_cols,opp_6_b,opp_6_e);
#endif

#ifdef _GPU
    array_to_ellpack(opp_6, opp_6_ell_data, opp_6_ell_indices, opp_6_nnz_per_row);
    opp_6_ell_data.cp_cpu_gpu();
    opp_6_ell_indices.cp_cpu_gpu();
#endif

	}
	else
	{
		cout << "ERROR: Invalid sparse matrix form ... " << endl;
	}
}

// set opp_p (solution at solution points to solution at plot points)

void eles::set_opp_p(void)
{
	int i,j,k;

	array<double> loc(n_dims);

	opp_p.setup(n_ppts_per_ele,n_upts_per_ele);

	for(i=0;i<n_upts_per_ele;i++)
	{
		for(j=0;j<n_ppts_per_ele;j++)
		{
			for(k=0;k<n_dims;k++)
			{
				loc(k)=loc_ppts(k,j);
			}

			opp_p(j,i)=eval_nodal_basis(i,loc);
		}
	}

}

// set opp_cub_f (solution at solution points to solution at plot points)

void eles::set_opp_inters_cubpts(void)
{

	int i,j,k,l;

	array<double> loc(n_dims);

  opp_inters_cubpts.setup(n_inters_per_ele);

  for (int i=0;i<n_inters_per_ele;i++)
  {
      opp_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_upts_per_ele);
  }

  for(l=0;l<n_inters_per_ele;l++)
  {
	  for(i=0;i<n_upts_per_ele;i++)
	  {
	  	for(j=0;j<n_cubpts_per_inter(l);j++)
	  	{
	  		for(k=0;k<n_dims;k++)
	  		{
	  			loc(k)=loc_inters_cubpts(l)(k,j);
	  		}

	  		opp_inters_cubpts(l)(j,i)=eval_nodal_basis(i,loc);
	  	}
	  }
  }

}

void eles::set_opp_volume_cubpts(void)
{

	int i,j,k,l;
	array<double> loc(n_dims);
  opp_volume_cubpts.setup(n_cubpts_per_ele,n_upts_per_ele);

	  for(i=0;i<n_upts_per_ele;i++)
	  {
	  	for(j=0;j<n_cubpts_per_ele;j++)
	  	{
	  		for(k=0;k<n_dims;k++)
	  		{
	  			loc(k)=loc_volume_cubpts(k,j);
	  		}

	  		opp_volume_cubpts(j,i)=eval_nodal_basis(i,loc);
	  	}
	  }
}


// set opp_r (solution at restart points to solution at solution points)

void eles::set_opp_r(void)
{
	int i,j,k;

	array<double> loc(n_dims);

	opp_r.setup(n_upts_per_ele,n_upts_per_ele_rest);

	for(i=0;i<n_upts_per_ele_rest;i++)
	{
		for(j=0;j<n_upts_per_ele;j++)
		{
			for(k=0;k<n_dims;k++)
				loc(k)=loc_upts(k,j);

			opp_r(j,i)=eval_nodal_basis_restart(i,loc);
		}
	}
}

array<double> eles::get_pos_ppt(int in_ele, int in_ppt)
{
  return pos_ppts(in_ele,in_ppt);
}

void eles::set_pos_ppts()
{
	array<double> pos_ppts_temp;
  pos_ppts_temp.setup(n_ppts_per_ele,n_dims);

  for (int i=0;i<n_eles;i++)
  {
    calc_pos_ppts(i,pos_ppts_temp);
    for (int j=0;j<n_ppts_per_ele;j++)
      for (int k=0;k<n_dims;k++)
        pos_ppts(i,j)(k) = pos_ppts_temp(j,k);
  }

}


// get position of the plot points
void eles::get_pos_ppts(int in_ele, array<double>& out_pos_ppts)
{
  for (int i=0;i<n_ppts_per_ele;i++)
    for (int j=0;j<n_dims;j++)
      out_pos_ppts(i,j) = pos_ppts(in_ele,i)(j);
}

// calculate position of the plot points

void eles::calc_pos_ppts(int in_ele, array<double>& out_pos_ppts)
{
	int i,j;

	array<double> loc(n_dims);
	array<double> pos(n_dims);

	for(i=0;i<n_ppts_per_ele;i++)
	{
		for(j=0;j<n_dims;j++)
		{
			loc(j)=loc_ppts(j,i);
		}

		calc_pos(loc,in_ele,pos);

		for(j=0;j<n_dims;j++)  // TODO: can this be made more efficient/simpler?
		{
			out_pos_ppts(i,j)=pos(j);
		}
	}
}

void eles::get_plotq_ppts(int in_ele, array<double> &out_plotq_ppts, array<double>& plotq_pnodes)
{

	for(int i=0;i<run_input.n_plot_quantities;i++)
	{
		for(int j=0;j<n_ppts_per_ele;j++)
		{
			 out_plotq_ppts(j,i)=plotq_pnodes(ppt_to_pnode(in_ele,j),i);
		}
	}

}

void eles::get_face_pnode_list(array<int>& out_inter_pnodes, int ic_l, int loc_f, int& out_n_inter_pnodes)
{
  out_n_inter_pnodes=n_ppts_per_face2(loc_f);

  for (int i=0;i<out_n_inter_pnodes;i++)
  {
    //cout << "ele_type= " << ele_type << "loc_f=" << loc_f << " i=" << i << endl;
    //
    //
    int cell_ppt = face2_ppt_to_ppt(loc_f)(i);
    out_inter_pnodes(i) = ppt_to_pnode(ic_l,cell_ppt);
    //cout << "after" << endl;
  }

}

// calculate solution at the plot points
void eles::calc_disu_ppts(int in_ele, array<double>& out_disu_ppts)
{
  if (n_eles!=0)
  {

	int i,j,k;

	array<double> disu_upts_plot(n_upts_per_ele,n_fields);

	for(i=0;i<n_fields;i++)
	{
		for(j=0;j<n_upts_per_ele;j++)
		{
			disu_upts_plot(j,i)=disu_upts(0)(j,in_ele,i);
		}
	}


	// HACK to show wall distance in Paraview
	//for(i=1;i<n_fields-1;i++)
		//for(j=0;j<n_upts_per_ele;j++)
			//disu_upts_plot(j,i)=wall_distance(j,in_ele,i-1);

	//for(j=0;j<n_upts_per_ele;j++) {
		//disu_upts_plot(j,0)=1.0;
		//disu_upts_plot(j,n_fields-1)=1.0;
	//}

	#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,disu_upts_plot.get_ptr_cpu(),n_upts_per_ele,0.0,out_disu_ppts.get_ptr_cpu(),n_ppts_per_ele);

	#else

	//HACK (inefficient, but useful if cblas is unavailible)

	for(i=0;i<n_ppts_per_ele;i++)
	{
		for(k=0;k<n_fields;k++)
		{
			out_disu_ppts(i,k) = 0.;

			for(j=0;j<n_upts_per_ele;j++)
			{
				out_disu_ppts(i,k) += opp_p(i,j)*disu_upts_plot(j,k);
			}
		}
	}

	#endif

  }
}

// calculate position of solution point

void eles::calc_pos_upt(int in_upt, int in_ele, array<double>& out_pos)
{
	int i;

	array<double> loc(n_dims);

	for(i=0;i<n_dims;i++)
	{
		loc(i)=loc_upts(i,in_upt);
	}

	calc_pos(loc,in_ele,out_pos);
}

double eles::get_loc_upt(int in_upt, int in_dim)
{
  return loc_upts(in_dim,in_upt);
}

// set transforms

void eles::set_transforms(int in_run_type)
{
  if (n_eles!=0)
  {

	int i,j,k;

	int n_comp;

	if(n_dims == 2)
	{
		n_comp = 3;
	}
  else if(n_dims == 3)
	{
		n_comp = 6;
	}

	array<double> loc(n_dims);
	array<double> pos(n_dims);
	array<double> d_pos(n_dims,n_dims);
	array<double> dd_pos(n_dims,n_comp);
	array<double> tnorm_dot_inv_detjac_mul_jac(n_dims);

	double xr, xs, xt;
	double yr, ys, yt;
	double zr, zs, zt;

	double xrr, xss, xtt, xrs, xrt, xst;
	double yrr, yss, ytt, yrs, yrt, yst;
	double zrr, zss, ztt, zrs, zrt, zst;

	detjac_upts.setup(n_upts_per_ele,n_eles);
	inv_detjac_mul_jac_upts.setup(n_upts_per_ele,n_eles,n_dims,n_dims);

  if (in_run_type==0) {

    if (viscous) {
	  	tgrad_detjac_upts.setup(n_upts_per_ele,n_eles,n_dims);
    }
  }

  if (rank==0) {
    cout << " at solution points" << endl;
  }

	for(i=0;i<n_eles;i++)
	{
    if ((i%1000)==0 && rank==0)
      cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;

	  for(j=0;j<n_upts_per_ele;j++)
	  {
			// get coordinates of the solution point

			for(k=0;k<n_dims;k++)
			{
				loc(k)=loc_upts(k,j);
			}

			// calculate first derivatives of shape functions at the solution point
			calc_d_pos(loc,i,d_pos);

			// calculate second derivatives of shape functions at the solution point
		  if (viscous && in_run_type==0)
			  calc_dd_pos(loc,i,dd_pos);

			// store quantities at the solution point

			if(n_dims==2)
			{
				xr = d_pos(0,0);
				xs = d_pos(0,1);

				yr = d_pos(1,0);
				ys = d_pos(1,1);

				// store determinant of jacobian at solution point
				detjac_upts(j,i)= xr*ys - xs*yr;

        if (detjac_upts(j,i) < 0)
        {
          FatalError("Negative Jacobian at solution points");
        }

				// store inverse of determinant of jacobian multiplied by jacobian at the solution point
				inv_detjac_mul_jac_upts(j,i,0,0)= ys;
				inv_detjac_mul_jac_upts(j,i,0,1)= -xs;
				inv_detjac_mul_jac_upts(j,i,1,0)= -yr;
				inv_detjac_mul_jac_upts(j,i,1,1)= xr;

        if (in_run_type==0)
        {

				  // gradient of detjac at solution point

				  if(viscous)
				  {
				  	xrr = dd_pos(0,0);
				  	xss = dd_pos(0,1);
				  	xrs = dd_pos(0,2);

				  	yrr = dd_pos(1,0);
				  	yss = dd_pos(1,1);
				  	yrs = dd_pos(1,2);

				  	tgrad_detjac_upts(j,i,0) = xrr*ys + yrs*xr - yrr*xs - xrs*yr;
				  	tgrad_detjac_upts(j,i,1) = yss*xr + xrs*ys - xss*yr - yrs*xs;
				  }
        }

			}
			else if(n_dims==3)
			{
				xr = d_pos(0,0);
				xs = d_pos(0,1);
				xt = d_pos(0,2);

				yr = d_pos(1,0);
				ys = d_pos(1,1);
				yt = d_pos(1,2);

				zr = d_pos(2,0);
				zs = d_pos(2,1);
				zt = d_pos(2,2);

				// store determinant of jacobian at solution point

				detjac_upts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

        //cout << "jac=" << detjac_upts(j,i) << endl;

				inv_detjac_mul_jac_upts(j,i,0,0) = ys*zt - yt*zs;
				inv_detjac_mul_jac_upts(j,i,0,1) = xt*zs - xs*zt;
				inv_detjac_mul_jac_upts(j,i,0,2) = xs*yt - xt*ys;
				inv_detjac_mul_jac_upts(j,i,1,0) = yt*zr - yr*zt;
				inv_detjac_mul_jac_upts(j,i,1,1) = xr*zt - xt*zr;
				inv_detjac_mul_jac_upts(j,i,1,2) = xt*yr - xr*yt;
				inv_detjac_mul_jac_upts(j,i,2,0) = yr*zs - ys*zr;
				inv_detjac_mul_jac_upts(j,i,2,1) = xs*zr - xr*zs;
				inv_detjac_mul_jac_upts(j,i,2,2) = xr*ys - xs*yr;

        if (in_run_type==0)
        {
				  // store inverse of determinant of jacobian multiplied by jacobian at the solution point


				  // gradient of detjac at solution point

				  if(viscous)
				  {
				  	xrr = dd_pos(0,0);
				  	xss = dd_pos(0,1);
				  	xtt = dd_pos(0,2);
				  	xrs = dd_pos(0,3);
				  	xrt = dd_pos(0,4);
				  	xst = dd_pos(0,5);

				  	yrr = dd_pos(1,0);
				  	yss = dd_pos(1,1);
				  	ytt = dd_pos(1,2);
				  	yrs = dd_pos(1,3);
				  	yrt = dd_pos(1,4);
				  	yst = dd_pos(1,5);

				  	zrr = dd_pos(2,0);
				  	zss = dd_pos(2,1);
				  	ztt = dd_pos(2,2);
				  	zrs = dd_pos(2,3);
				  	zrt = dd_pos(2,4);
				  	zst = dd_pos(2,5);

				  	tgrad_detjac_upts(j,i,0) = xrt*(zs*yr - ys*zr) - xrs*(zt*yr - yt*zr) + xrr*(zt*ys - yt*zs) +
				  								xr*(-zs*yrt + ys*zrt + zt*yrs - yt*zrs) - xs*(-zr*yrt + yr*zrt + zt*yrr - yt*zrr) + xt*(-zr*yrs + yr*zrs + zs*yrr - ys*zrr);
				  	tgrad_detjac_upts(j,i,1) = -xss*(zt*yr - yt*zr) + xst*(zs*yr - ys*zr) + xrs*(zt*ys - yt*zs) +
				  								xr*(-zs*yst + ys*zst + zt*yss - yt*zss) - xs*(zst*yr - yst*zr + zt*yrs - yt*zrs) + xt*(zss*yr - yss*zr + zs*yrs - ys*zrs);
				  	tgrad_detjac_upts(j,i,2) = -xst*(zt*yr - yt*zr) + xtt*(zs*yr - ys*zr) + xrt*(zt*ys - yt*zs) +
				  							xr*(ztt*ys - ytt*zs + zt*yst - yt*zst) - xs*(ztt*yr - ytt*zr + zt*yrt - yt*zrt) + xt*(zst*yr - yst*zr + zs*yrt - ys*zrt);
				  }
        }
			}
			else
			{
				cout << "ERROR: Invalid number of dimensions ... " << endl;
			}
    }
  }

#ifdef _GPU
  if (in_run_type==0)
  {
	  detjac_upts.cp_cpu_gpu(); // Copy since need in write_tec
	  inv_detjac_mul_jac_upts.mv_cpu_gpu();
    /*
    if (viscous) {
	  	tgrad_detjac_upts.mv_cpu_gpu();
    }
    */
  }
#endif

  if (in_run_type==0) // Compute metrics term at flux points
  {
	  detjac_fpts.setup(n_fpts_per_ele,n_eles);
	  inv_detjac_mul_jac_fpts.setup(n_fpts_per_ele,n_eles,n_dims,n_dims);
	  mag_tnorm_dot_inv_detjac_mul_jac_fpts.setup(n_fpts_per_ele,n_eles);
	  norm_fpts.setup(n_fpts_per_ele,n_eles,n_dims);
	  loc_fpts.setup(n_fpts_per_ele,n_eles,n_dims);

    if (viscous)
    {
	  	tgrad_detjac_fpts.setup(n_fpts_per_ele,n_eles,n_dims);
    }

    if (rank==0)
      cout << endl << " at flux points"  << endl;

	  for(i=0;i<n_eles;i++)
	  {
      if ((i%1000)==0 && rank==0)
        cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;

	    for(j=0;j<n_fpts_per_ele;j++)
	    {
	  		// get coordinates of the flux point

	  		for(k=0;k<n_dims;k++)
	  		{
	  			loc(k)=tloc_fpts(k,j);
	  		}

        calc_pos(loc,i,pos);

        for(k=0;k<n_dims;k++)
	  		{
	  			loc_fpts(j,i,k)=pos(k);
	  		}

	  		// calculate first derivatives of shape functions at the flux points

	  		calc_d_pos(loc,i,d_pos);

	  		// calculate second derivatives of shape functions at the flux point

	  		if(viscous)
	  		  calc_dd_pos(loc,i,dd_pos);

	  		// store quantities at the flux point

	  		if(n_dims==2)
	  		{
	  			xr = d_pos(0,0);
	  			xs = d_pos(0,1);

	  			yr = d_pos(1,0);
	  			ys = d_pos(1,1);

	  			// store determinant of jacobian at flux point

	  			detjac_fpts(j,i)= xr*ys - xs*yr;

        if (detjac_fpts(j,i) < 0)
        {
          FatalError("Negative Jacobian at flux points");
        }

	  			// store inverse of determinant of jacobian multiplied by jacobian at the flux point

	  			inv_detjac_mul_jac_fpts(j,i,0,0)= ys;
	  			inv_detjac_mul_jac_fpts(j,i,0,1)= -xs;
	  			inv_detjac_mul_jac_fpts(j,i,1,0)= -yr;
	  			inv_detjac_mul_jac_fpts(j,i,1,1)= xr;

	  			// gradient of detjac at the flux point

	  			if(viscous)
	  			{
	  				xrr = dd_pos(0,0);
	  				xss = dd_pos(0,1);
	  				xrs = dd_pos(0,2);

	  				yrr = dd_pos(1,0);
	  				yss = dd_pos(1,1);
	  				yrs = dd_pos(1,2);

	  				tgrad_detjac_fpts(j,i,0) = xrr*ys + yrs*xr - yrr*xs - xrs*yr;
	  				tgrad_detjac_fpts(j,i,1) = yss*xr + xrs*ys - xss*yr - yrs*xs;
	  			}

	  			// temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

	  			tnorm_dot_inv_detjac_mul_jac(0)=(tnorm_fpts(0,j)*d_pos(1,1))-(tnorm_fpts(1,j)*d_pos(1,0));
	  			tnorm_dot_inv_detjac_mul_jac(1)=-(tnorm_fpts(0,j)*d_pos(0,1))+(tnorm_fpts(1,j)*d_pos(0,0));

	  			// store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

	  			mag_tnorm_dot_inv_detjac_mul_jac_fpts(j,i)=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                                          tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1));


	  			// store normal at flux point

	  			norm_fpts(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm_dot_inv_detjac_mul_jac_fpts(j,i);
	  			norm_fpts(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm_dot_inv_detjac_mul_jac_fpts(j,i);
	  		}
	  		else if(n_dims==3)
	  		{
	  			xr = d_pos(0,0);
	  			xs = d_pos(0,1);
	  			xt = d_pos(0,2);

	  			yr = d_pos(1,0);
	  			ys = d_pos(1,1);
	  			yt = d_pos(1,2);

	  			zr = d_pos(2,0);
	  			zs = d_pos(2,1);
	  			zt = d_pos(2,2);

	  			// store determinant of jacobian at flux point

	  			detjac_fpts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

	  			// store inverse of determinant of jacobian multiplied by jacobian at the flux point

	  			inv_detjac_mul_jac_fpts(j,i,0,0) = ys*zt - yt*zs;
	  			inv_detjac_mul_jac_fpts(j,i,0,1) = xt*zs - xs*zt;
	  			inv_detjac_mul_jac_fpts(j,i,0,2) = xs*yt - xt*ys;
	  			inv_detjac_mul_jac_fpts(j,i,1,0) = yt*zr - yr*zt;
	  			inv_detjac_mul_jac_fpts(j,i,1,1) = xr*zt - xt*zr;
	  			inv_detjac_mul_jac_fpts(j,i,1,2) = xt*yr - xr*yt;
	  			inv_detjac_mul_jac_fpts(j,i,2,0) = yr*zs - ys*zr;
	  			inv_detjac_mul_jac_fpts(j,i,2,1) = xs*zr - xr*zs;
	  			inv_detjac_mul_jac_fpts(j,i,2,2) = xr*ys - xs*yr;

	  			// gradient of detjac at the flux point

	  			if(viscous)
	  			{
	  				xrr = dd_pos(0,0);
	  				xss = dd_pos(0,1);
	  				xtt = dd_pos(0,2);
	  				xrs = dd_pos(0,3);
	  				xrt = dd_pos(0,4);
	  				xst = dd_pos(0,5);

	  				yrr = dd_pos(1,0);
	  				yss = dd_pos(1,1);
	  				ytt = dd_pos(1,2);
	  				yrs = dd_pos(1,3);
	  				yrt = dd_pos(1,4);
	  				yst = dd_pos(1,5);

	  				zrr = dd_pos(2,0);
	  				zss = dd_pos(2,1);
	  				ztt = dd_pos(2,2);
	  				zrs = dd_pos(2,3);
	  				zrt = dd_pos(2,4);
	  				zst = dd_pos(2,5);

	  				tgrad_detjac_fpts(j,i,0) = xrt*(zs*yr - ys*zr) - xrs*(zt*yr - yt*zr) + xrr*(zt*ys - yt*zs) +
	  											xr*(-zs*yrt + ys*zrt + zt*yrs - yt*zrs) - xs*(-zr*yrt + yr*zrt + zt*yrr - yt*zrr) + xt*(-zr*yrs + yr*zrs + zs*yrr - ys*zrr);
	  				tgrad_detjac_fpts(j,i,1) = -xss*(zt*yr - yt*zr) + xst*(zs*yr - ys*zr) + xrs*(zt*ys - yt*zs) +
	  											xr*(-zs*yst + ys*zst + zt*yss - yt*zss) - xs*(zst*yr - yst*zr + zt*yrs - yt*zrs) + xt*(zss*yr - yss*zr + zs*yrs - ys*zrs);
	  				tgrad_detjac_fpts(j,i,2) = -xst*(zt*yr - yt*zr) + xtt*(zs*yr - ys*zr) + xrt*(zt*ys - yt*zs) +
	  											xr*(ztt*ys - ytt*zs + zt*yst - yt*zst) - xs*(ztt*yr - ytt*zr + zt*yrt - yt*zrt) + xt*(zst*yr - yst*zr + zs*yrt - ys*zrt);
	  			}

	  			// temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

	  			tnorm_dot_inv_detjac_mul_jac(0)=((tnorm_fpts(0,j)*(d_pos(1,1)*d_pos(2,2)-d_pos(1,2)*d_pos(2,1)))+(tnorm_fpts(1,j)*(d_pos(1,2)*d_pos(2,0)-d_pos(1,0)*d_pos(2,2)))+(tnorm_fpts(2,j)*(d_pos(1,0)*d_pos(2,1)-d_pos(1,1)*d_pos(2,0))));
	  			tnorm_dot_inv_detjac_mul_jac(1)=((tnorm_fpts(0,j)*(d_pos(0,2)*d_pos(2,1)-d_pos(0,1)*d_pos(2,2)))+(tnorm_fpts(1,j)*(d_pos(0,0)*d_pos(2,2)-d_pos(0,2)*d_pos(2,0)))+(tnorm_fpts(2,j)*(d_pos(0,1)*d_pos(2,0)-d_pos(0,0)*d_pos(2,1))));
	  			tnorm_dot_inv_detjac_mul_jac(2)=((tnorm_fpts(0,j)*(d_pos(0,1)*d_pos(1,2)-d_pos(0,2)*d_pos(1,1)))+(tnorm_fpts(1,j)*(d_pos(0,2)*d_pos(1,0)-d_pos(0,0)*d_pos(1,2)))+(tnorm_fpts(2,j)*(d_pos(0,0)*d_pos(1,1)-d_pos(0,1)*d_pos(1,0))));

	  			// store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point

	  			mag_tnorm_dot_inv_detjac_mul_jac_fpts(j,i)=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                                     tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1)+
                                                     tnorm_dot_inv_detjac_mul_jac(2)*tnorm_dot_inv_detjac_mul_jac(2));

	  			// store normal at flux point

	  			norm_fpts(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm_dot_inv_detjac_mul_jac_fpts(j,i);
	  			norm_fpts(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm_dot_inv_detjac_mul_jac_fpts(j,i);
	  			norm_fpts(j,i,2)=tnorm_dot_inv_detjac_mul_jac(2)/mag_tnorm_dot_inv_detjac_mul_jac_fpts(j,i);
	  		}
	  		else
	  		{
	  			cout << "ERROR: Invalid number of dimensions ... " << endl;
	  		}
	  	}
	  }

#ifdef _GPU
	  mag_tnorm_dot_inv_detjac_mul_jac_fpts.mv_cpu_gpu();
	  norm_fpts.mv_cpu_gpu();
	  loc_fpts.cp_cpu_gpu();

    /*
	  inv_detjac_mul_jac_fpts.mv_cpu_gpu();
	  detjac_fpts.mv_cpu_gpu();
    if (viscous)
    {
	  	tgrad_detjac_fpts.mv_cpu_gpu();
    }
    */
#endif


  }

   if (rank==0) cout << endl;
  } // if n_eles!=0
}

void eles::add_contribution_to_pnodes(array<double> &plotq_pnodes)
{

	double xr, xs, xt;
	double yr, ys, yt;
	double zr, zs, zt;
  double wx,wy,wz;

  double irho;
  double dudx, dudy, dudz;
  double dvdx, dvdy, dvdz;
  double dwdx, dwdy, dwdz;

	array<double> loc(n_dims);
	array<double> d_pos(n_dims,n_dims);

  // First obtain disu at the ppts
  array<double> disu_ppt(n_fields);
  array<double> grad_disu_ppt(n_fields,n_dims);
	array<double> grad_disu_upts_temp(n_upts_per_ele,n_fields,n_dims);
  double plotq_ppt;
  double u,v,w;

  double pressure,v_sq;

  int compute_grad = 0;
  for (int k=0;k<run_input.n_plot_quantities;k++)
  {
    if (run_input.plot_quantities(k)=="vorticity" || run_input.plot_quantities(k)=="q_criterion")
    {
      compute_grad=1;
    }
  }

  // Then loop over cells and divide by jacobian at the ppts
  double detjac_ppt;
  for(int i=0;i<n_eles;i++)
  {
    if (compute_grad)
    {
      // Get grad disu at solution points
      for (int j=0;j<n_upts_per_ele;j++)
      {
        for (int k=0;k<n_fields;k++)
        {
          for (int k2=0;k2<n_dims;k2++)
          {
            double value=0;
            for (int m=0;m<n_upts_per_ele;m++)
              value += opp_4(k2)(j,m)*disu_upts(0)(m,i,k);

            grad_disu_upts_temp(j,k,k2) = value;
          }
        }
      }

      // Now transform to physical space
      double detjac;
      double inv_detjac;
      double rx,ry,rz,sx,sy,sz,tx,ty,tz;
      double ur,us,ut;

      for (int j=0;j<n_upts_per_ele;j++)
      {
	      detjac = detjac_upts(j,i);
	      inv_detjac = 1.0/detjac;

	      rx = inv_detjac_mul_jac_upts(j,i,0,0);
	      ry = inv_detjac_mul_jac_upts(j,i,0,1);
	      sx = inv_detjac_mul_jac_upts(j,i,1,0);
	      sy = inv_detjac_mul_jac_upts(j,i,1,1);

	      //physical gradient
	      if(n_dims==2)
	      {
	      	for(int k=0;k<n_fields;k++)
	      	{
	      		ur = grad_disu_upts_temp(j,k,0);
	      		us = grad_disu_upts_temp(j,k,1);

	      		grad_disu_upts_temp(j,k,0) = (1.0/detjac)*(ur*rx + us*sx) ;
	      		grad_disu_upts_temp(j,k,1) = (1.0/detjac)*(ur*ry + us*sy) ;
	      	}
        }
        if (n_dims==3)
        {
	    	  rz = inv_detjac_mul_jac_upts(j,i,0,2);
	    		sz = inv_detjac_mul_jac_upts(j,i,1,2);

	    		tx = inv_detjac_mul_jac_upts(j,i,2,0);
	    		ty = inv_detjac_mul_jac_upts(j,i,2,1);
	    		tz = inv_detjac_mul_jac_upts(j,i,2,2);

          for (int k=0;k<n_fields;k++)
          {
	    		  ur = grad_disu_upts_temp(j,k,0);
	    		  us = grad_disu_upts_temp(j,k,1);
	    		  ut = grad_disu_upts_temp(j,k,2);

	    		  grad_disu_upts_temp(j,k,0) = (1.0/detjac)*(ur*rx + us*sx + ut*tx);
	    		  grad_disu_upts_temp(j,k,1) = (1.0/detjac)*(ur*ry + us*sy + ut*ty);
	    		  grad_disu_upts_temp(j,k,2) = (1.0/detjac)*(ur*rz + us*sz + ut*tz);
          }
        }
      }
    }

    // Get disu at ppt, using opp_p
    for(int j=0;j<n_ppts_per_ele;j++)
    {
      // Get disu at ppt, using opp_p
      for (int k=0;k<n_fields;k++)
      {
        double value = 0.;
        for (int m=0;m<n_upts_per_ele;m++)
          value += opp_p(j,m)*disu_upts(0)(m,i,k);

        disu_ppt(k) = value;
      }

      if (compute_grad==1)
      {
        // Get grad_disu at ppt, using opp_p
        for (int k=0;k<n_fields;k++)
        {
          for (int k2=0;k2<n_dims;k2++)
          {
            double value = 0.;
            for (int m=0;m<n_upts_per_ele;m++)
              value += opp_p(j,m)*grad_disu_upts_temp(m,k,k2);

            grad_disu_ppt(k,k2) = value;
          }
        }
      }

      // Now compute plotting quantities
      int pnode = ppt_to_pnode(i,j);

      // Compute velocity square
      v_sq = 0.;
      for (int m=0;m<n_dims;m++)
        v_sq += (disu_ppt(m+1)*disu_ppt(m+1));
      v_sq /= disu_ppt(0)*disu_ppt(0);

      // Compute pressure
	    pressure = (run_input.gamma-1.0)*( disu_ppt(n_dims+1) - 0.5*disu_ppt(0)*v_sq);

      for (int k=0;k<run_input.n_plot_quantities;k++)
      {
        if (run_input.plot_quantities(k)=="rho")
          plotq_ppt = disu_ppt(0);
        else if (run_input.plot_quantities(k)=="u")
          plotq_ppt = disu_ppt(1)/disu_ppt(0);
        else if (run_input.plot_quantities(k)=="v")
          plotq_ppt = disu_ppt(2)/disu_ppt(0);
        else if (run_input.plot_quantities(k)=="w")
        {
          if (n_dims==2)
            plotq_ppt = 0.;
          else if (n_dims==3)
            plotq_ppt = disu_ppt(3)/disu_ppt(0);
        }
        else if (run_input.plot_quantities(k)=="energy")
        {
          if (n_dims==2)
            plotq_ppt = disu_ppt(3);
          else if (n_dims==3)
            plotq_ppt = disu_ppt(4);
        }
        else if (run_input.plot_quantities(k)=="mach")
        {
          plotq_ppt = sqrt( v_sq / (run_input.gamma*pressure/disu_ppt(0)) );
        }
        else if (run_input.plot_quantities(k)=="pressure")
        {
          plotq_ppt = pressure;
        }
        else if (run_input.plot_quantities(k)=="pressure")
        {
          plotq_ppt = pressure;
        }
        else if (run_input.plot_quantities(k)=="vorticity" || run_input.plot_quantities(k)=="q_criterion")
        {
          irho = 1./disu_ppt(0);
          u = disu_ppt(1)*irho;
          v = disu_ppt(2)*irho;

          dudx = irho*(grad_disu_ppt(1,0) - u*grad_disu_ppt(0,0));
          dudy = irho*(grad_disu_ppt(1,1) - u*grad_disu_ppt(0,1));
          dvdx = irho*(grad_disu_ppt(2,0) - v*grad_disu_ppt(0,0));
          dvdy = irho*(grad_disu_ppt(2,1) - v*grad_disu_ppt(0,1));

          if (n_dims==2)
          {
            if (run_input.plot_quantities(k) == "vorticity")
            {
              plotq_ppt = abs(dvdx-dudy);
            }
            else if (run_input.plot_quantities(k) == "q_criterion")
            {
              FatalError("Not implemented in 2D");
            }
          }
          else if (n_dims==3)
          {
            w = disu_ppt(3)*irho;

            dudz = irho*(grad_disu_ppt(1,2) - u*grad_disu_ppt(0,2));
            dvdz = irho*(grad_disu_ppt(2,2) - v*grad_disu_ppt(0,2));

            dwdx = irho*(grad_disu_ppt(3,0) - w*grad_disu_ppt(0,0));
            dwdy = irho*(grad_disu_ppt(3,1) - w*grad_disu_ppt(0,1));
            dwdz = irho*(grad_disu_ppt(3,2) - w*grad_disu_ppt(0,2));

            wx = dwdy - dvdz;
            wy = dudz - dwdx;
            wz = dvdx - dudy;

            if (run_input.plot_quantities(k) == "vorticity")
            {
              plotq_ppt = sqrt(wx*wx+wy*wy+wz*wz);
            }
            else if (run_input.plot_quantities(k) == "q_criterion")
            {

              wx *= 0.5;
              wy *= 0.5;
              wz *= 0.5;

              double Sxx,Syy,Szz,Sxy,Sxz,Syz,SS,OO;
              Sxx = dudx;
              Syy = dvdy;
              Szz = dwdz;
              Sxy = 0.5*(dudy+dvdx);
              Sxz = 0.5*(dudz+dwdx);
              Syz = 0.5*(dvdz+dwdy);

              SS = Sxx*Sxx + Syy*Syy + Szz*Szz + 2*Sxy*Sxy + 2*Sxz*Sxz + 2*Syz*Syz;
              OO = 2*wx*wx + 2*wy*wy + 2*wz*wz;

              plotq_ppt = 0.5*(OO-SS);

            }
          }
        }
        else
          FatalError("plot_quantity not recognized");

         if (isnan(plotq_ppt))
           FatalError("NaN");

        plotq_pnodes(pnode,k) += plotq_ppt;
      }

    }
  }
}

void eles::set_bdy_ele2ele(void)
{

  n_bdy_eles=0;
  // Count the number of bdy_eles
  for (int i=0;i<n_eles;i++) {
    for (int j=0;j<n_inters_per_ele;j++) {
      if (bctype(i,j) != 0) {
        n_bdy_eles++;
        break;
      }
    }
  }

  if (n_bdy_eles!=0) {

    bdy_ele2ele.setup(n_bdy_eles);

    n_bdy_eles=0;
    for (int i=0;i<n_eles;i++) {
      for (int j=0;j<n_inters_per_ele;j++) {
        if (bctype(i,j) != 0) {
          bdy_ele2ele(n_bdy_eles++) = i;
          break;
        }
      }
    }

  }

}


// set transforms

void eles::set_transforms_inters_cubpts(void)
{
  if (n_eles!=0)
  {
	int i,j,k;
	int n_comp;

	double xr, xs, xt;
	double yr, ys, yt;
	double zr, zs, zt;

	double xrr, xss, xtt, xrs, xrt, xst;
	double yrr, yss, ytt, yrs, yrt, yst;
	double zrr, zss, ztt, zrs, zrt, zst;

  // Initialize bdy_ele2ele array
  (*this).set_bdy_ele2ele();

	if(n_dims == 2)
	{
		n_comp = 3;
	}
	if(n_dims == 3)
	{
		n_comp = 6;
	}
  double mag_tnorm;

	array<double> loc(n_dims);
	array<double> d_pos(n_dims,n_dims);
	array<double> tnorm_dot_inv_detjac_mul_jac(n_dims);

	inter_detjac_inters_cubpts.setup(n_inters_per_ele);
	norm_inters_cubpts.setup(n_inters_per_ele);
	vol_detjac_inters_cubpts.setup(n_inters_per_ele);

  for (int i=0;i<n_inters_per_ele;i++)
  {
	  inter_detjac_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles);
	  norm_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles,n_dims);
	  vol_detjac_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles);
  }

	for(i=0;i<n_bdy_eles;i++)
	{
    for (int l=0;l<n_inters_per_ele;l++)
    {
	    for(j=0;j<n_cubpts_per_inter(l);j++)
	    {
		  	// get coordinates of the flux point

		  	for(k=0;k<n_dims;k++)
		  	{
		  		loc(k)=loc_inters_cubpts(l)(k,j);
		  	}

		  	// calculate first derivatives of shape functions at the flux points

        // TODO: Need mapping between bdy_interfaces and ele
		  	calc_d_pos(loc,bdy_ele2ele(i),d_pos);

		  	// store quantities at the flux point

		  	if(n_dims==2)
		  	{

   				xr = d_pos(0,0);
   				xs = d_pos(0,1);

   				yr = d_pos(1,0);
   				ys = d_pos(1,1);

   				// store determinant of jacobian at flux point
   				vol_detjac_inters_cubpts(l)(j,i)= xr*ys - xs*yr;

		  		// temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
		  		tnorm_dot_inv_detjac_mul_jac(0)=(tnorm_inters_cubpts(l)(0,j)*d_pos(1,1))-(tnorm_inters_cubpts(l)(1,j)*d_pos(1,0));
		  		tnorm_dot_inv_detjac_mul_jac(1)=-(tnorm_inters_cubpts(l)(0,j)*d_pos(0,1))+(tnorm_inters_cubpts(l)(1,j)*d_pos(0,0));

		  		// store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
		  		mag_tnorm = sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                                          tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1));

		  		// store normal at flux point
		  		norm_inters_cubpts(l)(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm;
		  		norm_inters_cubpts(l)(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm;

          inter_detjac_inters_cubpts(l)(j,i) = compute_inter_detjac_inters_cubpts(l,d_pos);


		  	}
		  	else if(n_dims==3)
		  	{

  				xr = d_pos(0,0);
  				xs = d_pos(0,1);
  				xt = d_pos(0,2);

  				yr = d_pos(1,0);
  				ys = d_pos(1,1);
  				yt = d_pos(1,2);

  				zr = d_pos(2,0);
  				zs = d_pos(2,1);
  				zt = d_pos(2,2);

  				// store determinant of jacobian at flux point
  				vol_detjac_inters_cubpts(l)(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

		  		// temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
		  		tnorm_dot_inv_detjac_mul_jac(0)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(1,1)*d_pos(2,2)-d_pos(1,2)*d_pos(2,1)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(1,2)*d_pos(2,0)-d_pos(1,0)*d_pos(2,2)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(1,0)*d_pos(2,1)-d_pos(1,1)*d_pos(2,0))));
		  		tnorm_dot_inv_detjac_mul_jac(1)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(0,2)*d_pos(2,1)-d_pos(0,1)*d_pos(2,2)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(0,0)*d_pos(2,2)-d_pos(0,2)*d_pos(2,0)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(0,1)*d_pos(2,0)-d_pos(0,0)*d_pos(2,1))));
		  		tnorm_dot_inv_detjac_mul_jac(2)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(0,1)*d_pos(1,2)-d_pos(0,2)*d_pos(1,1)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(0,2)*d_pos(1,0)-d_pos(0,0)*d_pos(1,2)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(0,0)*d_pos(1,1)-d_pos(0,1)*d_pos(1,0))));

		  		// store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
		  		mag_tnorm=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                                     tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1)+
                                                     tnorm_dot_inv_detjac_mul_jac(2)*tnorm_dot_inv_detjac_mul_jac(2));

		  		// store normal at flux point
		  		norm_inters_cubpts(l)(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm;
		  		norm_inters_cubpts(l)(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm;
		  		norm_inters_cubpts(l)(j,i,2)=tnorm_dot_inv_detjac_mul_jac(2)/mag_tnorm;

          inter_detjac_inters_cubpts(l)(j,i) = compute_inter_detjac_inters_cubpts(l,d_pos);
		  	}
		  	else
		  	{
		  		FatalError("ERROR: Invalid number of dimensions ... ");
		  	}
		  }
    }
	}

  } // if n_eles!=0
}

// Set transforms at volume cubature points
void eles::set_transforms_vol_cubpts(void)
{
	if(n_eles!=0)
	{
		int i,j,m;
		array<double> d_pos(n_dims,n_dims);
	  array<double> loc(n_dims);
	  array<double> pos(n_dims);

		vol_detjac_vol_cubpts.setup(n_cubpts_per_ele);

	  for (i=0;i<n_cubpts_per_ele;i++)
		  vol_detjac_vol_cubpts(i).setup(n_eles);

		for (i=0;i<n_eles;i++)
		{
			for (j=0;j<n_cubpts_per_ele;j++)
			{
	      // Get jacobian determinant at cubpts
  	    for (m=0;m<n_dims;m++)
	        loc(m) = loc_volume_cubpts(m,j);

  	    calc_pos(loc,i,pos);
  	    calc_d_pos(loc,i,d_pos);

  	    if (n_dims==2)
  	    {
  	      vol_detjac_vol_cubpts(j)(i) = d_pos(0,0)*d_pos(1,1) - d_pos(0,1)*d_pos(1,0);
  	    }
  	    else if (n_dims==3)
  	    {
					vol_detjac_vol_cubpts(j)(i) = d_pos(0,0)*(d_pos(1,1)*d_pos(2,2) - d_pos(1,2)*d_pos(2,1))
																		 - d_pos(0,1)*(d_pos(1,0)*d_pos(2,2) - d_pos(1,2)*d_pos(2,0))
																		 + d_pos(0,2)*(d_pos(1,0)*d_pos(2,1) - d_pos(1,1)*d_pos(2,0));
				}
			}
		}
	}
}

// get a pointer to the transformed discontinuous solution at a flux point

double* eles::get_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return disu_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
	return disu_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
}

// get a pointer to the normal transformed continuous inviscid flux at a flux point

double* eles::get_norm_tconf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return norm_tconf_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
	return norm_tconf_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif

}

// get a pointer to the determinant of the jacobian at a flux point

double* eles::get_detjac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return detjac_fpts.get_ptr_gpu(fpt,in_ele);
#else
	return detjac_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get a pointer to the magnitude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at flux points

double* eles::get_mag_tnorm_dot_inv_detjac_mul_jac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return mag_tnorm_dot_inv_detjac_mul_jac_fpts.get_ptr_gpu(fpt,in_ele);
#else
	return mag_tnorm_dot_inv_detjac_mul_jac_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get a pointer to the normal at a flux point

double* eles::get_norm_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return norm_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
#else
	return norm_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
#endif
}

// get a pointer to the coordinates at a flux point

double* eles::get_loc_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

// WHY DOESN'T THE GPU POINTER WORK WHEN CALLED FROM GEOMETRY
// BUT IT DOES WORK WHEN CALLED FROM BDY_INTERS???
//#ifdef _GPU
	//return loc_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
//#else
	return loc_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
//#endif
}

// get a pointer to delta of the transformed discontinuous solution at a flux point

double* eles::get_delta_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

  //if (ele2global_ele(in_ele)==53)
  //{
  //  cout << "HERE" << endl;
  //  cout << "local_face=" << in_ele_local_inter << endl;
  //}

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return delta_disu_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
	return delta_disu_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
}

// get a pointer to gradient of discontinuous solution at a flux point

double* eles::get_grad_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_field, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return grad_disu_fpts.get_ptr_gpu(fpt,in_ele,in_field,in_dim);
#else
	return grad_disu_fpts.get_ptr_cpu(fpt,in_ele,in_field,in_dim);
#endif
}

// get a pointer to the normal transformed continuous viscous flux at a flux point
/*
double* eles::get_norm_tconvisf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return norm_tconvisf_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
	return norm_tconvisf_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
}
*/

// get a pointer to the subgrid-scale flux at a flux point
double* eles::get_sgsf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_dim, int in_ele)
{
	int i;

	int fpt;

	fpt=in_inter_local_fpt;

	for(i=0;i<in_ele_local_inter;i++)
	{
		fpt+=n_fpts_per_inter(i);
	}

#ifdef _GPU
	return sgsf_fpts.get_ptr_gpu(fpt,in_ele,in_field,in_dim);
#else
	return sgsf_fpts.get_ptr_cpu(fpt,in_ele,in_field,in_dim);
#endif
}

//#### helper methods ####

// calculate position

void eles::calc_pos(array<double> in_loc, int in_ele, array<double>& out_pos)
{
	int i,j;

	for(i=0;i<n_dims;i++)
	{
		out_pos(i)=0.0;

		for(j=0;j<n_spts_per_ele(in_ele);j++)
    {
			out_pos(i)+=eval_nodal_s_basis(j,in_loc,n_spts_per_ele(in_ele))*shape(i,j,in_ele);
		}
	}

}

// calculate derivative of position - NEEDS TO BE OPTIMIZED

void eles::calc_d_pos(array<double> in_loc, int in_ele, array<double>& out_d_pos)
{
	int i,j,k;

  eval_d_nodal_s_basis(d_nodal_s_basis,in_loc,n_spts_per_ele(in_ele));

	for(j=0;j<n_dims;j++)
	{
		for(k=0;k<n_dims;k++)
		{
			out_d_pos(j,k)=0.0;
			for(i=0;i<n_spts_per_ele(in_ele);i++)
			{
				//out_d_pos(j,k)+=eval_d_nodal_s_basis(i,k,in_loc,n_spts_per_ele(in_ele))*shape(j,i,in_ele);
				out_d_pos(j,k)+=d_nodal_s_basis(i,k)*shape(j,i,in_ele);
			}
		}
	}
}

// calculate second derivative of position

void eles::calc_dd_pos(array<double> in_loc, int in_ele, array<double>& out_dd_pos)
{
	int i,j,k;
	int n_comp;

	if(n_dims == 2)
		n_comp = 3;
  else if(n_dims == 3)
		n_comp = 6;

  eval_dd_nodal_s_basis(dd_nodal_s_basis,in_loc,n_spts_per_ele(in_ele));

	for(j=0;j<n_dims;j++)
	{
		for(k=0;k<n_comp;k++)
		{
			out_dd_pos(j,k)=0.0;

			for(i=0;i<n_spts_per_ele(in_ele);i++)
			{
				out_dd_pos(j,k)+=dd_nodal_s_basis(i,k)*shape(j,i,in_ele);
			}
		}
	}
}

/*! Calculate residual sum for monitoring purposes */
double eles::compute_res_upts(int in_norm_type, int in_field) {

  int i, j;
  double sum = 0.;
  double cell_sum = 0.;

  // NOTE: div_tconf_upts must be on CPU

  for (i=0; i<n_eles; i++) {
    cell_sum=0;
    for (j=0; j<n_upts_per_ele; j++) {
        if (in_norm_type == 1) {
          cell_sum += abs(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-run_input.const_src_term-src_term(j,i,in_field));
        }
        else if (in_norm_type == 2) {
          cell_sum += (div_tconf_upts(0)(j, i, in_field)/detjac_upts(j,i)-run_input.const_src_term-src_term(j,i,in_field))*(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-run_input.const_src_term-src_term(j,i,in_field));
        }
    }
    sum += cell_sum;
  }

  return sum;

}


array<double> eles::compute_error(int in_norm_type, double& time)
{
  array<double> disu_cubpt(n_fields);
  array<double> grad_disu_cubpt(n_fields,n_dims);
  double detjac;
  array<double> pos(n_dims);

  array<double> error(2,n_fields);  //storage
  array<double> error_sum(2,n_fields);  //output

  for (int i=0; i<n_fields; i++)
  {
    error_sum(0,i) = 0.;
    error_sum(1,i) = 0.;
  }

  for (int i=0;i<n_eles;i++)
  {
    for (int j=0;j<n_cubpts_per_ele;j++)
    {
      // Get jacobian determinant at cubpts
			detjac = vol_detjac_vol_cubpts(j)(i);

      // Get the solution at cubature point
      for (int m=0;m<n_fields;m++)
      {
        disu_cubpt(m) = 0.;
        for (int k=0;k<n_upts_per_ele;k++)
        {
          disu_cubpt(m) += opp_volume_cubpts(j,k)*disu_upts(0)(k,i,m);
        }
      }

      // Get the gradient at cubature point
      if (viscous==1)
      {
        for (int m=0;m<n_fields;m++) {
          for (int n=0;n<n_dims;n++) {
            double value=0.;
            for (int k=0;k<n_upts_per_ele;k++) {
                value += opp_volume_cubpts(j,k)*grad_disu_upts(k,i,m,n);
            }
            grad_disu_cubpt(m,n) = value;
            //cout << value << endl;
          }
        }
      }

      error = get_pointwise_error(disu_cubpt,grad_disu_cubpt,pos,time,in_norm_type);

      for (int m=0;m<n_fields;m++) {
          error_sum(0,m) += error(0,m)*weight_volume_cubpts(j)*detjac;
          error_sum(1,m) += error(1,m)*weight_volume_cubpts(j)*detjac;
      }
    }
  }

  cout << "time   " << time << endl;

  return error_sum;
}


array<double> eles::get_pointwise_error(array<double>& sol, array<double>& grad_sol, array<double>& loc, double& time, int in_norm_type)
{
  array<double> error(2,n_fields);  //output

  array<double> error_sol(n_fields);
  array<double> error_grad_sol(n_fields,n_dims);

  for (int i=0; i<n_fields; i++) {
    error_sol(i) = 0.;

    error(0,i) = 0.;
    error(1,i) = 0.;

    for (int j=0; j<n_dims; j++) {
      error_grad_sol(i,j) = 0.;
    }
  }

  if (run_input.test_case==1) // Isentropic vortex
  {
    // Computing error in x-momentum
    double rho,vx,vy,vz,p;
    eval_isentropic_vortex(loc,time,rho,vx,vy,vz,p,n_dims);

    error_sol(1) = sol(1) - rho*vx;
  }
  else if (run_input.test_case==2) // Sine Wave (single)
  {
    double rho;
    array<double> grad_rho(n_dims);

    if(viscous) {
      eval_sine_wave_single(loc,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
    }
    else {
      eval_sine_wave_single(loc,run_input.wave_speed,0.,time,rho,grad_rho,n_dims);
    }

    error_sol(0) = sol(0) - rho;

    for (int j=0; j<n_dims; j++) {
       error_grad_sol(0,j) = grad_sol(0,j) - grad_rho(j);
    }

  }
  else if (run_input.test_case==3) // Sine Wave (group)
  {
    double rho;
    array<double> grad_rho(n_dims);

    if(viscous) {
      eval_sine_wave_group(loc,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
    }
    else {
      eval_sine_wave_group(loc,run_input.wave_speed,0.,time,rho,grad_rho,n_dims);
    }

    error_sol(0) = sol(0) - rho;

    for (int j=0; j<n_dims; j++) {
       error_grad_sol(0,j) = grad_sol(0,j) - grad_rho(j);
    }
  }
  else if (run_input.test_case==4) // Sphere Wave
  {
    double rho;
    eval_sphere_wave(loc,run_input.wave_speed,time,rho,n_dims);
    error_sol(0) = sol(0) - rho;
  }
  else if (run_input.test_case==5) // Couette flow
  {
		int ind;
    double ene, u_wall;
    array<double> grad_ene(n_dims);

    u_wall = run_input.v_wall(0);

    eval_couette_flow(loc,run_input.gamma, run_input.R_ref, u_wall, run_input.T_wall, run_input.p_bound, run_input.prandtl, time, ene, grad_ene, n_dims);

		ind = n_dims+1;

    error_sol(ind) = sol(ind) - ene;

    for (int j=0; j<n_dims; j++) {
       error_grad_sol(ind,j) = grad_sol(ind,j) - grad_ene(j);
    }
  }
  else {
    FatalError("Test case not recognized in compute error, exiting");
  }

  if (in_norm_type==1)
  {
    for (int m=0;m<n_fields;m++) {
      error(0,m) += abs(error_sol(m));

      for(int n=0;n<n_dims;n++) {
        error(1,m) += abs(error_grad_sol(m,n)); //might be incorrect
      }
    }
  }

  if (in_norm_type==2)
  {
    for (int m=0;m<n_fields;m++) {
      error(0,m) += error_sol(m)*error_sol(m);

      for(int n=0;n<n_dims;n++) {
        error(1,m) += error_grad_sol(m,n)*error_grad_sol(m,n);
      }
    }
  }

  return error;
}

// Calculate body forcing term for periodic channel flow. HARDCODED FOR THE CHANNEL!

void eles::calc_body_force_upts(array <double>& vis_force, array <double>& body_force)
{
 #ifdef _CPU

	if (n_eles!=0) {
		int i,j,k,l,m;
		double lenx, areayz, vol, detjac, sum;
		array <double> cubwgts(n_cubpts_per_ele);
		double mdot, ubulk, wallf, dpdx;
		double alpha=0.1; // relaxation parameter
		double mdot0 = 1.0; // initial mass flow rate
		array <double> disu_cubpt(2);
		array <double> solint(2);
		array <int> inflowinters(n_bdy_eles,n_inters_per_ele);
		array <double> norm(n_dims), flow(n_dims);
		double wgt;

		solint(0)=0.0; solint(1)=0.0;

		// Calculate mass flux as area integral
		for (i=0;i<n_bdy_eles;i++)
			for (l=0;l<n_inters_per_ele;l++)
				inflowinters(i,l)=0;
		sum=0;
		// Mass flux on inflow boundary
		// Integrate density and x-velocity over inflow area
	  for (i=0;i<n_bdy_eles;i++)
		{
			int ele = bdy_ele2ele(i);
			for (l=0;l<n_inters_per_ele;l++)
			{
				if(inflowinters(i,l)!=1) // only unflagged inters
				{
					// Inlet has either Sub_In_Simp (1), Sub_In_Char (3), Sup_In (5) or Cyclic (9) BC
					if(bctype(ele,l) == 1 || bctype(ele,l) == 3 || bctype(ele,l) == 5 || bctype(ele,l) == 9)
					{
						// HACK NUMBER UMPTEEN: inflow plane normal direction is -x
						if(bctype(ele,l) == 9)
						{
		          // Get the normal
		          for (m=0;m<n_dims;m++)
								norm(m) = norm_inters_cubpts(l)(0,i,m);

							if(norm(0)==-1)
							{
								inflowinters(i,l)=1; // Flag this interface
							}
							else
							{
								inflowinters(i,l)=0;
							}
						}
						else
						{
							inflowinters(i,l)=1; // Flag this interface
						}
					}
				}
				sum+=inflowinters(i,l);
			}
		}

		// Now loop over flagged inters
	  for (i=0;i<n_bdy_eles;i++)
		{
			int ele = bdy_ele2ele(i);
			for (l=0;l<n_inters_per_ele;l++)
			{
				if(inflowinters(i,l)==1)
				{
					for (j=0;j<n_cubpts_per_inter(l);j++)
					{
						wgt = weight_inters_cubpts(l)(j);
						detjac = inter_detjac_inters_cubpts(l)(j,i);

			      // Get the solution at cubature point
		 	      disu_cubpt(0) = 0.; disu_cubpt(1) = 0.;
						for (k=0;k<n_upts_per_ele;k++)
						{
							disu_cubpt(0) += opp_inters_cubpts(l)(j,k)*disu_upts(0)(k,ele,0); // density
							disu_cubpt(1) += opp_inters_cubpts(l)(j,k)*disu_upts(0)(k,ele,1); // x-momentum
						}
						solint(0) += wgt*disu_cubpt(0)*detjac;
						solint(1) += wgt*disu_cubpt(1)*detjac;
					}
				}
			}
		}

		if(solint(0)==0.0 or solint(1)==0.0)
		{
			cout<<"Body force error diagnostics:"<<endl;
			cout<<disu_cubpt(0)<<", "<<disu_cubpt(1)<<", "<<wgt<<", "<<detjac<<", "<<solint(0)<<", "<<solint(1)<<", "<<sum<<", "<<n_bdy_eles<<endl;
		}

		lenx = 2*pi;
		areayz = 2*pi;
		vol = areayz*lenx;
		mdot = solint(1)/areayz;
		if(solint(0)==0)
			ubulk = 0.0;
		else
			ubulk = solint(1)/solint(0);

		body_force(1) = vis_force(0)/vol; // x-momentum forcing
		body_force(n_fields-1) = body_force(1)*ubulk; // energy forcing

/*#ifdef _MPI
	  array<double> bodyforce_global(n_fields);
	  for (m=0;m<n_fields;m++)
		{
	    bodyforce_global(m) = 0.;
			MPI_Reduce(&bodyforce(m),&bodyforce_global(m),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			bodyforce(m) = bodyforce_global(m);
		}
#endif*/

		cout<<"Channel body force: " << setprecision(10)<<body_force(1)<<", "<<body_force(n_fields-1)<<endl;
		if(isnan(body_force(1)) or isnan(body_force(n_fields-1)))
		{
			cout<<"ERROR: NaN body force, exiting"<<endl;
			cout<<"timestep, vol: "<< run_input.dt<<", "<<vol<<endl;
			cout<<"Channel mass flux: " << setprecision(10)<<mdot<<endl;
			cout<<"Channel bulk velocity: " << setprecision(10)<<ubulk<<endl;
			cout<<"Channel viscous force: " << setprecision(10)<<vis_force(0)<<endl;
			exit(1);
		}
	}
#endif
}

void eles::add_body_force_upts(array <double>& body_force)
{
	if (n_eles!=0) {
		int i,j,k,l,m;
		// Add to viscous flux at solution points
	  for (i=0;i<n_eles;i++)
		  for (j=0;j<n_upts_per_ele;j++)
		  	for(k=0;k<n_fields;k++)
		 			for(l=0;l<n_dims;l++)
			 			for(m=0;m<n_dims;m++)
		 					tdisf_upts(j,i,k,l)+=inv_detjac_mul_jac_upts(j,i,l,m)*body_force(k);
	}
}

// Compute diagnostic quantities
void eles::CalcDiagnostics(int n_diagnostics, array <double>& diagnostic_array)
{
  array<double> disu_cubpt(n_fields);
  array<double> grad_disu_cubpt(n_fields,n_dims);
  array<double> S(n_dims,n_dims);
  double wx, wy, wz;
  double dudx, dudy, dudz;
  double dvdx, dvdy, dvdz;
  double dwdx, dwdy, dwdz;
	double diagnostic, tke, pressure, diag, irho, detjac;

	// Sum over elements
  for (int i=0;i<n_eles;i++)
  {
    for (int j=0;j<n_cubpts_per_ele;j++)
    {
      // Get jacobian determinant at cubpts
			detjac = vol_detjac_vol_cubpts(j)(i);

      // Get the solution at cubature point
      for (int m=0;m<n_fields;m++)
      {
        disu_cubpt(m) = 0.;
        for (int k=0;k<n_upts_per_ele;k++)
        {
          disu_cubpt(m) += opp_volume_cubpts(j,k)*disu_upts(0)(k,i,m);
        }
			}
			// Get the solution gradient at cubature point
			for (int m=0;m<n_fields;m++)
			{
				for (int n=0;n<n_dims;n++)
				{
					grad_disu_cubpt(m,n)=0.;
					for (int k=0;k<n_upts_per_ele;k++)
					{
						grad_disu_cubpt(m,n) += opp_volume_cubpts(j,k)*grad_disu_upts(k,i,m,n);
					}
				}
			}
      irho = 1./disu_cubpt(0);
      dudx = irho*(grad_disu_cubpt(1,0) - disu_cubpt(1)*irho*grad_disu_cubpt(0,0));
      dudy = irho*(grad_disu_cubpt(1,1) - disu_cubpt(1)*irho*grad_disu_cubpt(0,1));
      dvdx = irho*(grad_disu_cubpt(2,0) - disu_cubpt(2)*irho*grad_disu_cubpt(0,0));
      dvdy = irho*(grad_disu_cubpt(2,1) - disu_cubpt(2)*irho*grad_disu_cubpt(0,1));

			if (n_dims==3)
			{
  	    dudz = irho*(grad_disu_cubpt(1,2) - disu_cubpt(1)*irho*grad_disu_cubpt(0,2));
  	    dvdz = irho*(grad_disu_cubpt(2,2) - disu_cubpt(2)*irho*grad_disu_cubpt(0,2));
  	    dwdx = irho*(grad_disu_cubpt(3,0) - disu_cubpt(3)*irho*grad_disu_cubpt(0,0));
  	    dwdy = irho*(grad_disu_cubpt(3,1) - disu_cubpt(3)*irho*grad_disu_cubpt(0,1));
  	    dwdz = irho*(grad_disu_cubpt(3,2) - disu_cubpt(3)*irho*grad_disu_cubpt(0,2));
			}

			// Now calculate diagnostic quantities
			for (int m=0;m<n_diagnostics;++m)
			{
				diagnostic = 0.0;
		    if (run_input.diagnostics(m)=="kineticenergy")
				{
					// Compute kinetic energy
					tke = 0.0;
					for (int n=1;n<n_fields-1;n++)
						tke += 0.5*disu_cubpt(n)*disu_cubpt(n);

					diagnostic = irho*tke;
				}
				else if (run_input.diagnostics(m)=="vorticity")
				{
					// Compute vorticity squared
 	        wz = dvdx - dudy;
          diagnostic = wz*wz;
					if (n_dims==3)
					{
  	        wx = dwdy - dvdz;
  	        wy = dudz - dwdx;
						diagnostic += wx*wx+wy*wy;
					}
					diagnostic *= 0.5/irho;
				}
				else if (run_input.diagnostics(m)=="pressuredilatation")
				{
					// Kinetic energy
					tke = 0.0;
					for (int n=1;n<n_fields-1;n++)
						tke += 0.5*disu_cubpt(n)*disu_cubpt(n);

		      // Compute pressure
			    pressure = (run_input.gamma-1.0)*(disu_cubpt(n_fields-1) - irho*tke);

					// Multiply pressure by divergence of velocity
					if (n_dims==2) {
						diagnostic = pressure*(dudx+dvdy);
					}
					else if (n_dims==3) {
						diagnostic = pressure*(dudx+dvdy+dwdz);
					}
				}
				else if (run_input.diagnostics(m)=="straincolonproduct" || run_input.diagnostics(m)=="devstraincolonproduct")
				{
					// Rate of strain tensor
					S(0,0) = dudx;
					S(0,1) = (dudy+dvdx)/2.0;
					S(1,0) = S(0,1);
					S(1,1) = dvdy;
					diag = (S(0,0)+S(1,1))/3.0;

					if (n_dims==3)
					{
						S(0,2) = (dudz+dwdx)/2.0;
						S(1,2) = (dvdz+dwdy)/2.0;
						S(2,0) = S(0,2);
						S(2,1) = S(1,2);
						S(2,2) = dwdz;
						diag += S(2,2)/3.0;
					}

					// Subtract diag if deviatoric strain
					if (run_input.diagnostics(m)=="devstraincolonproduct") {
						for (int i=0;i<n_dims;i++)
							S(i,i) -= diag;
					}

					for (int i=0;i<n_dims;i++)
						for (int j=0;j<n_dims;j++)
							diagnostic += S(i,j)*S(i,j);

				}
				else
				{
					cout<<"Error: diagnostic quantity not recognized"<<endl;
					exit(1);
				}
				// Add contribution to global integral
				diagnostic_array(m) += diagnostic*weight_volume_cubpts(j)*detjac;
      }
    }
  }
}

void eles::compute_wall_forces( array<double>& inv_force, array<double>& vis_force, ofstream& cp_file)
{

  array<double> u_l(n_fields),norm(n_dims);
  double p_l,v_sq,vn_l;
  array<double> inv_force_on_inter(n_dims);
  array<double> grad_u_l(n_fields,n_dims);
  array<double> dv(n_dims,n_dims);
  array<double> de(n_dims);
  array<double> drho(n_dims);
  double diag, inte,mu,rt_ratio;
  array<double> taun(n_dims), Fvis(n_dims);
  array<double> vis_force_on_inter_new(n_dims);
  array<double> vis_force_on_inter(n_dims);
  array<double> loc(n_dims),pos(n_dims);
	double wgt, detjac;
  double gamma = run_input.gamma;

  for (int m=0;m<n_dims;m++)
  {
      inv_force(m) = 0.;
      vis_force(m) = 0.;
  }

  for (int i=0;i<n_bdy_eles;i++) {

    int ele = bdy_ele2ele(i);
    for (int l=0;l<n_inters_per_ele;l++) {

      if (bctype(ele,l) == 7 || bctype(ele,l) == 11 || bctype(ele,l) == 12 || bctype(ele,l)==16) {

        for (int m=0;m<n_dims;m++)
        {
          inv_force_on_inter(m) = 0.;
          vis_force_on_inter(m) = 0.;
					vis_force_on_inter_new(m) = 0.;
        }

        // Compute force on this interface
        for (int j=0;j<n_cubpts_per_inter(l);j++) {

					// Get determinant of Jacobian
					detjac = inter_detjac_inters_cubpts(l)(j,i);

					// Get cubature weight
					wgt = weight_inters_cubpts(l)(j);

          // Get position of cubature point
          for (int m=0;m<n_dims;m++)
            loc(m) = loc_inters_cubpts(l)(m,j);

          calc_pos(loc,ele,pos);

          // Compute solution at current cubature point
          for (int m=0;m<n_fields;m++) {
            double value = 0.;
            for (int k=0;k<n_upts_per_ele;k++) {
              value += opp_inters_cubpts(l)(j,k)*disu_upts(0)(k,ele,m);
            }
            u_l(m) = value;
          }

          // If viscous, extrapolate the gradient at the cubature points
          if (viscous==1)
          {
            for (int m=0;m<n_fields;m++) {
              for (int n=0;n<n_dims;n++) {
                double value=0.;
                for (int k=0;k<n_upts_per_ele;k++) {
                  value += opp_inters_cubpts(l)(j,k)*grad_disu_upts(k,ele,m,n);
                }
                grad_u_l(m,n) = value;
              }
            }
          }

          // Get the normal
          for (int m=0;m<n_dims;m++)
          {
           norm(m) = norm_inters_cubpts(l)(j,i,m);
          }

          // Add contribution from current cubature point
          // Get pressure

          // Not dual consistent
          if (bctype(ele,l)!=16) {
            v_sq = 0.;
            for (int m=0;m<n_dims;m++)
              v_sq += (u_l(m+1)*u_l(m+1));
	          p_l   = (gamma-1.0)*( u_l(n_dims+1) - 0.5*v_sq/u_l(0));
          }
          else
          {
            //Dual consistent approach
            vn_l = 0.;
            for (int m=0;m<n_dims;m++)
              vn_l += u_l(m+1)*norm(m);
            vn_l /= u_l(0);

            for (int m=0;m<n_dims;m++)
              u_l(m+1) = u_l(m+1)-(vn_l)*norm(m);

            v_sq = 0.;
            for (int m=0;m<n_dims;m++)
              v_sq += (u_l(m+1)*u_l(m+1));
	          p_l   = (gamma-1.0)*( u_l(n_dims+1) - 0.5*v_sq/u_l(0));
          }

          cp_file << scientific << setw(18) << setprecision(12) << pos(0) << " " << setw(18) << setprecision(12) << (p_l-run_input.p_c_ic)/(0.5*run_input.rho_c_ic*(run_input.u_c_ic*run_input.u_c_ic+run_input.v_c_ic*run_input.v_c_ic+run_input.w_c_ic*run_input.w_c_ic));

          if (viscous==1)
          {
            // TODO: Have a function that returns tau given u and grad_u
            // Computing the n_dims derivatives of rho,u,v,w and ene
            for (int m=0;m<n_dims;m++)
            {
              drho(m) = grad_u_l(0,m);
              for (int n=0;n<n_dims;n++)
              {
                dv(n,m) = 1.0/u_l(0)*(grad_u_l(n+1,m)-drho(m)*u_l(n+1));;
              }
              de(m) = 1.0/u_l(0)*(grad_u_l(n_dims+1,m)-drho(m)*u_l(n_dims+1));;
            }

            diag = 0.;
            for (int m=0;m<n_dims;m++)
              diag += dv(m,m);

	          inte = u_l(n_dims+1)/u_l(0);
            for (int m=0;m<n_dims;m++)
            {
              inte -= 0.5*u_l(m+1)*u_l(m+1);
            }

	          rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
            mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
            mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

            // Compute the resulting shear stress

            if (n_dims==2)
            {
              taun(0) = 2.0*(dv(0,0)-diag)*norm(0) + (dv(0,1)+dv(1,0))*norm(1);
              taun(1) = 2.0*(dv(1,1)-diag)*norm(1) + (dv(0,1)+dv(1,0))*norm(0);
            }
            else if (n_dims==3)
            {
              taun(0) = 2.0*(dv(0,0)-diag)*norm(0) + (dv(0,1)+dv(1,0))*norm(1) + (dv(0,2)+dv(2,0))*norm(2);
              taun(1) = 2.0*(dv(1,1)-diag)*norm(1) + (dv(0,1)+dv(1,0))*norm(0) + (dv(1,2)+dv(2,1))*norm(2);
              taun(2) = 2.0*(dv(2,2)-diag)*norm(2) + (dv(0,2)+dv(2,0))*norm(0) + (dv(1,2)+dv(2,1))*norm(1);
            }

            // Compute the coefficient of friction and wall shear stress
            if (n_dims==2)
            {
              double magnorm= norm(0)*norm(0)+norm(1)*norm(1);
							// DUDN IS WRONG - WHAT IS NORM(2) DOING IN HERE????
              double dudn = ((dv(0,0)*norm(1)-dv(1,0)*norm(0))*norm(0)+ (dv(0,1)*norm(2)-dv(1,1)*norm(0))*norm(1))*magnorm;
							// WHAT I THINK IT SHOULD BE:
              //double dudn = ((dv(0,0)*norm(1)-dv(1,0)*norm(0))*norm(0)+ (dv(0,1)*norm(1)-dv(1,1)*norm(0))*norm(1))*magnorm;
              double cf = 2.*mu*dudn/(run_input.rho_c_ic*(run_input.u_c_ic*run_input.u_c_ic+run_input.v_c_ic*run_input.v_c_ic));
							//Fvis(0) = wgt*mu*taun(0)*detjac;
							//Fvis(1) = wgt*mu*taun(1)*detjac;
							/*cout<< setw(18) <<setprecision(12)<<"norm: "<<endl;
							norm.print();
							cout<< setw(18) <<setprecision(12)<<"dudn: "<<dudn<<endl;
							cout<< setw(18) <<setprecision(12)<<"cf: "<<cf<<endl;
							cout<< setw(18) <<setprecision(12)<<"Fvis: "<<endl;
							Fvis.print();*/
              cp_file << " " << setw(18) <<setprecision(12) << cf ;

							// And now wall shear stress
							double taundotn = taun(0)*norm(0)+taun(1)*norm(1);
							taun(0) -= taundotn*norm(0)/magnorm;
							taun(1) -= taundotn*norm(1)/magnorm;
							//double taumag = sqrt(pow(taun(0),2)+pow(taun(1),2));
							Fvis(0) = wgt*mu*taun(0)*detjac;
							Fvis(1) = wgt*mu*taun(1)*detjac;
							/*cout<< setw(18) <<setprecision(12)<<"taundotn: "<<taundotn<<endl;
							cout<< setw(18) <<setprecision(12)<<"taun(0): "<<taun(0)<<endl;
							cout<< setw(18) <<setprecision(12)<<"taun(1): "<<taun(1)<<endl;
							cout<< setw(18) <<setprecision(12)<<"taumag: "<<taumag<<endl;
							cout<< setw(18) <<setprecision(12)<<"Fvis: "<<endl;
							Fvis.print();*/
            }

            if (n_dims==3)
            {
              double magnorm= norm(0)*norm(0)+norm(1)*norm(1)+norm(2)*norm(2);
							// DUDN IS WRONG - WHAT SHOULD 3D EXPRESSION BE?
              double dudn = ((dv(0,0)*norm(1)-dv(1,0)*norm(0))*norm(0)+ (dv(0,1)*norm(2)-dv(1,1)*norm(0))*norm(1))*magnorm;
              double cf = 2.*mu*dudn/(run_input.rho_c_ic*(run_input.u_c_ic*run_input.u_c_ic+run_input.v_c_ic*run_input.v_c_ic));
              cp_file << " " << setw(18) <<setprecision(12) << cf ;

							// And now wall shear stress
							double taundotn = taun(0)*norm(0)+taun(1)*norm(1)+taun(2)*norm(2);
							taun(0) -= taundotn*norm(0)/magnorm;
							taun(1) -= taundotn*norm(1)/magnorm;
							taun(2) -= taundotn*norm(2)/magnorm;
							Fvis(0) = wgt*mu*taun(0)*detjac;
							Fvis(1) = wgt*mu*taun(1)*detjac;
							Fvis(2) = wgt*mu*taun(2)*detjac;
						}
          } // End of if viscous

          cp_file << endl;

          for (int m=0;m<n_dims;m++)
          {
            inv_force_on_inter(m) += wgt*p_l*norm(m)*detjac;
            if (viscous==1)
            {
							// Patrice's version:
              //vis_force_on_inter(m) -= wgt*mu*taun(m)*inter_detjac_inters_cubpts(l)(j,i);
							// Jonathan's version:
              vis_force_on_inter_new(m) -= Fvis(m);
            }
          }
        }
        // Add contribution from current face
        for (int m=0;m<n_dims;m++) {
          inv_force(m) += inv_force_on_inter(m);
          vis_force(m) += vis_force_on_inter_new(m);
        }
      }
    }
  }
}

