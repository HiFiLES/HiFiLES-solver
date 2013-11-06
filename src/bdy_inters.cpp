/*!
 * \file bdy_inters.cpp
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
#include "../include/bdy_inters.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/flux.h"
#include "../include/error.h"

#if defined _GPU
#include "../include/cuda_kernels.h"
#endif

#ifdef _MPI
#include "mpi.h"
#endif

using namespace std;

// #### constructors ####

// default constructor

bdy_inters::bdy_inters()
{	
	order=run_input.order;
	viscous=run_input.viscous;
}

bdy_inters::~bdy_inters() { }

// #### methods ####

// setup inters

void bdy_inters::setup(int in_n_inters, int in_inters_type, int in_run_type)
{

  (*this).setup_inters(in_n_inters,in_inters_type,in_run_type);

  boundary_type.setup(in_n_inters);
  set_bdy_params();

}

void bdy_inters::set_bdy_params()
{
  max_bdy_params=30;
  bdy_params.setup(max_bdy_params);

  bdy_params(0) = run_input.rho_bound;
  bdy_params(1) = run_input.v_bound(0);
  bdy_params(2) = run_input.v_bound(1);
  bdy_params(3) = run_input.v_bound(2);
  bdy_params(4) = run_input.p_bound;

  if(viscous)
  {
    bdy_params(5) = run_input.v_wall(0);
    bdy_params(6) = run_input.v_wall(1);
    bdy_params(7) = run_input.v_wall(2);
    bdy_params(8) = run_input.T_wall;
  }
}

void bdy_inters::set_boundary(int in_inter, int bdy_type, int in_ele_type_l, int in_ele_l, int in_local_inter_l, int in_run_type, struct solution* FlowSol)
{
  boundary_type(in_inter) = bdy_type;

  if (in_run_type==0)
  {
	  for(int i=0;i<n_fields;i++)
	  {
	  	for(int j=0;j<n_fpts_per_inter;j++)
	  	{
	  		disu_fpts_l(j,in_inter,i)=get_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
	  	
	  		norm_tconf_fpts_l(j,in_inter,i)=get_norm_tconf_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
	  		
	  		if(viscous)
	  		{
	  			delta_disu_fpts_l(j,in_inter,i)=get_delta_disu_fpts_ptr(in_ele_type_l,in_ele_l,i,in_local_inter_l,j,FlowSol);
	  		}
	  	}
	  }
	  
	  for(int i=0;i<n_fields;i++)
	  {
	  	for(int j=0;j<n_fpts_per_inter;j++)
	  	{
	  		for(int k=0; k<n_dims; k++)
	  		{
	  			if(viscous)
	  			{
	  				grad_disu_fpts_l(j,in_inter,i,k) =get_grad_disu_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,k,j,FlowSol);
	  			}
	  		}
	  	}
	  }

	  for(int i=0;i<n_fpts_per_inter;i++)
	  {
	  	mag_tnorm_dot_inv_detjac_mul_jac_fpts_l(i,in_inter)=get_mag_tnorm_dot_inv_detjac_mul_jac_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,FlowSol);
	  	
	  	for(int j=0;j<n_dims;j++)
	  	{
	  		norm_fpts(i,in_inter,j)=get_norm_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,j,FlowSol);
	  		loc_fpts(i,in_inter,j)=get_loc_fpts_ptr(in_ele_type_l,in_ele_l,in_local_inter_l,i,j,FlowSol);
	  	}

	  }	
  }

}

// move all from cpu to gpu

void bdy_inters::mv_all_cpu_gpu(void)
{
	#ifdef _GPU
	
	disu_fpts_l.mv_cpu_gpu();
	norm_tconf_fpts_l.mv_cpu_gpu();
	mag_tnorm_dot_inv_detjac_mul_jac_fpts_l.mv_cpu_gpu();
	norm_fpts.mv_cpu_gpu();
	loc_fpts.mv_cpu_gpu();
	
  delta_disu_fpts_l.mv_cpu_gpu();

  if(viscous)
	{
		grad_disu_fpts_l.mv_cpu_gpu();
		//norm_tconvisf_fpts_l.mv_cpu_gpu();
	}
	//detjac_fpts_l.mv_cpu_gpu();

  boundary_type.mv_cpu_gpu();
  bdy_params.mv_cpu_gpu();

	#endif
}


// calculate normal transformed continuous inviscid flux at the flux points on the boundaries
void bdy_inters::calc_norm_tconinvf_fpts_boundary(double time_bound)
{

  #ifdef _CPU
  array<double> norm(n_dims), fn(n_fields);
  
  //viscous
  int bdy_spec, flux_spec;
  array<double> u_c(n_fields);


	for(int i=0;i<n_inters;i++)
	{
		for(int j=0;j<n_fpts_per_inter;j++)
		{

      // storing normal components
      for (int m=0;m<n_dims;m++)
        norm(m) = *norm_fpts(j,i,m);

			// calculate discontinuous solution at flux points
			for(int k=0;k<n_fields;k++) 
				temp_u_l(k)=(*disu_fpts_l(j,i,k));
  
      for (int m=0;m<n_dims;m++)
        temp_loc(m) = *loc_fpts(j,i,m);

      set_inv_boundary_conditions(boundary_type(i),temp_u_l.get_ptr_cpu(),temp_u_r.get_ptr_cpu(),norm.get_ptr_cpu(),temp_loc.get_ptr_cpu(),bdy_params.get_ptr_cpu(),n_dims,n_fields,run_input.gamma,run_input.R_ref,time_bound,run_input.equation);

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


      if (boundary_type(i)==16) // Dual consistent BC
      {
        // Set Normal flux to be right flux
        right_flux(temp_f_l,norm,fn,n_dims,n_fields,run_input.gamma); 
      }
      else // Call Riemann solver
      {
        // Calling Riemann solver
        if (run_input.riemann_solve_type==0) { //Rusanov
          rusanov_flux(temp_u_l,temp_u_r,temp_f_l,temp_f_r,norm,fn,n_dims,n_fields,run_input.gamma);
        }
        else if (run_input.riemann_solve_type==1) { // Lax-Friedrich
          lax_friedrich(temp_u_l,temp_u_r,norm,fn,n_dims,n_fields,run_input.lambda,run_input.wave_speed);
        }
        else if (run_input.riemann_solve_type==2) { // ROE
          roe_flux(temp_u_l,temp_u_r,norm,fn,n_dims,n_fields,run_input.gamma);
        }
        else
          FatalError("Riemann solver not implemented");
      }

      // Transform back to reference space  
  		for(int k=0;k<n_fields;k++) 
					(*norm_tconf_fpts_l(j,i,k))=fn(k)*(*mag_tnorm_dot_inv_detjac_mul_jac_fpts_l(j,i));

      if(viscous)
      {
        //boundary specification
        bdy_spec = boundary_type(i);
    
        if(bdy_spec == 12 || bdy_spec == 14)
          flux_spec = 2;
        else
          flux_spec = 1;

        // Calling viscous riemann solver
        if (run_input.vis_riemann_solve_type==0)
          ldg_solution(flux_spec,temp_u_l,temp_u_r,u_c,run_input.pen_fact,norm);
        else
          FatalError("Viscous Riemann solver not implemented");

		    for(int k=0;k<n_fields;k++)
		      *delta_disu_fpts_l(j,i,k) = (u_c(k) - temp_u_l(k));
      }

    }
  }

	#endif
		
	#ifdef _GPU
  if (n_inters!=0)
    calc_norm_tconinvf_fpts_boundary_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),mag_tnorm_dot_inv_detjac_mul_jac_fpts_l.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),loc_fpts.get_ptr_gpu(),boundary_type.get_ptr_gpu(),bdy_params.get_ptr_gpu(),run_input.riemann_solve_type,delta_disu_fpts_l.get_ptr_gpu(),run_input.gamma,run_input.R_ref,viscous,run_input.vis_riemann_solve_type, time_bound, run_input.wave_speed(0),run_input.wave_speed(1),run_input.wave_speed(2),run_input.lambda,run_input.equation);
  #endif
}

void bdy_inters::set_inv_boundary_conditions(int bdy_type, double* u_l, double* u_r, double *norm, double *loc, double *bdy_params, int n_dims, int n_fields, double gamma, double R_ref, double time_bound, int equation)
{

  
  double p_l, p_r;
  double T_l, T_r;
  double v_sq,vn_l;
  double rho_bound = bdy_params[0];
  double* v_bound = &bdy_params[1];
  double p_bound = bdy_params[4];
  double* v_wall = &bdy_params[5];
  double T_wall = bdy_params[8];
  
  double x,y,z;
  
  x = loc[0];
  y = loc[1];
  
  if(n_dims==3)
    z = loc[2];
  
  //printf("x = %6.10f\n",x);
  //printf("y = %6.10f\n",y);
  
  if(equation==0) //Navier-Stokes BC's
  {
    // Compute pressure on left side
    v_sq = 0.;
    for (int i=0;i<n_dims;i++)
      v_sq += (u_l[i+1]*u_l[i+1]);
	  p_l   = (gamma-1.0)*( u_l[n_dims+1] - 0.5*v_sq/u_l[0]);
    
    // Compute normal velocity on left side
    vn_l = 0.;
    for (int i=0;i<n_dims;i++)
      vn_l += u_l[i+1]*norm[i];
    vn_l /= u_l[0];
    
	  if(bdy_type == 1)
      // subsonic inflow simple (free pressure)
	  {
	  	// fix density and velocity
	  	u_r[0] =  rho_bound;
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = v_bound[i];
      
	  	// extrapolate pressure
	  	p_r   = p_l;
      
	  	// compute energy
      v_sq = 0.;
      for (int i=0;i<n_dims;i++)
        v_sq += (u_r[i+1]*u_r[i+1]);
	    u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];
	  }
    
	  // subsonic outflow simple (fixed pressure)
	  else if(bdy_type == 2)
	  {
	  	// extrapolate density and velocity
	  	u_r[0]    = u_l[0];
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = u_l[i+1];
      
	  	// fix pressure
	  	p_r = p_bound;
      
	  	// compute energy
      v_sq = 0.;
      for (int i=0;i<n_dims;i++)
        v_sq += (u_r[i+1]*u_r[i+1]);
	    u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];
      
	  }
    
	  // subsonic inflow characteristic
	  else if(bdy_type == 3)
	  {
	  	// TODO: Implement characteristic subsonic inflow BC
	  	printf("subsonic inflow char not implemented in 3d");
	  }
	  //subsonic outflow characteristic
	  else if(bdy_type == 4)
	  {
	  	printf("subsonic outflow char not implemented in 3d");
	  }
    
	  // supersonic inflow
	  else if(bdy_type == 5)
	  {
	  	// fix density and velocity
	  	u_r[0] =  rho_bound;
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = v_bound[i];
      
	  	// fix pressure
	  	p_r = p_bound;
      
	  	// compute energy
      v_sq = 0.;
      for (int i=0;i<n_dims;i++)
        v_sq += (u_r[i+1]*u_r[i+1]);
	    u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];
      
      /*
       for (int i=0;i<n_dims;i++)
       {
       cout << "norm=" << scientific << setprecision(12) << setw(18) << norm[i] << endl;
       }
       for (int i=0;i<n_fields;i++)
       {
       cout << "u_l=" << scientific << setprecision(12) << setw(18) << u_l[i] << endl;
       cout << "u_r=" << scientific << setprecision(12) << setw(18) << u_r[i] << endl;
       }
       */
      
	  }
    
	  // supersonic outflow
	  else if(bdy_type == 6)
	  {
	  	// extrapolate density, velocity
	  	u_r[0] = u_l[0];
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = u_l[i+1];
      
	  	// pressure and energy
      u_r[n_dims+1]=u_l[n_dims+1];
	  }
    
	  // slip wall
	  else if(bdy_type == 7)
	  {
	  	// extrapolate density
	  	u_r[0] = u_l[0];
      
	  	// reflect normal momentum
      for (int i=0;i<n_dims;i++)
	  	  u_r[i+1] = u_l[i+1]-2.0*vn_l*u_l[0]*norm[i];
      
	  	// extrapolate energy
	  	u_r[n_dims+1] = u_l[n_dims+1];
	  }
	  
    // isothermal, no-slip wall (fixed)
    else if(bdy_type == 11)
	  {
	  	// extrapolate pressure
	  	p_r = p_l;
      //p_r = p_bound; //HACK
      
      // isothermal temperature
      T_r = T_wall;
      //T_l = p_l/(u_l[0]*R_ref);
      
      // density
      u_r[0] = p_r/(R_ref*T_r);
      
      // no-slip
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = 0.;
      
	  	// energy
      v_sq = 0.;
      for (int i=0;i<n_dims;i++)
        v_sq += (u_r[i+1]*u_r[i+1]);
	    u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];
	  }
    
    // adiabatic, no-slip wall (fixed)
    else if(bdy_type == 12)
    {
      // extrapolate density
      u_r[0] = u_l[0];
      
      // extrapolate pressure
	  	p_r = p_l;
      
      // no-slip
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = 0.;
	  	
      // energy
      v_sq = 0.;
      for (int i=0;i<n_dims;i++)
        v_sq += (u_r[i+1]*u_r[i+1]);
	    u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];
    }
    
    // isothermal, no-slip wall (moving)
    else if(bdy_type == 13)
    {
      // extrapolate pressure
	  	p_r = p_l;
      //p_r = p_bound; //HACK
      
      // isothermal temperature
      T_r = T_wall;
      //T_l = p_l/(u_l[0]*R_ref);
      
      // density
      u_r[0] = p_r/(R_ref*T_r);
      
      // no-slip
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = u_r[0]*v_wall[i];
      
	  	// energy
      v_sq = 0.;
      for (int i=0;i<n_dims;i++)
        v_sq += (u_r[i+1]*u_r[i+1]);
	    u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];
    }
    
    // adiabatic, no-slip wall (moving)
    else if(bdy_type == 14)
    {
      // extrapolate density
      u_r[0] = u_l[0];
      
      // extrapolate pressure
	  	p_r = p_l;
      
      // no-slip
      for (int i=0;i<n_dims;i++)
        u_r[i+1] = u_r[0]*v_wall[i];
	  	
      // energy
      v_sq = 0.;
      for (int i=0;i<n_dims;i++)
        v_sq += (u_r[i+1]*u_r[i+1]);
	    u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];
    }
    
	  else if (bdy_type == 15) // Characteristic
	  {
	  	double c_star;
	  	double vn_star;
      double vn_bound;
      double vt_star;
	  	double r_plus,r_minus;
	  	
	  	double one_over_s;
	  	double h_free_stream;
      
      vn_bound = 0;
      for (int i=0;i<n_dims;i++)
        vn_bound += v_bound[i]*norm[i];
      
	  	r_plus  = vn_l + 2./(gamma-1.)*sqrt(gamma*p_l/u_l[0]);
	  	r_minus = vn_bound - 2./(gamma-1.)*sqrt(gamma*p_bound/rho_bound);
      
	  	c_star = 0.25*(gamma-1.)*(r_plus-r_minus);
	  	vn_star = 0.5*(r_plus+r_minus);
      
	  	//Works only for 2D and quasi-2D
      
	  	if (vn_l<0) // Inflow
	  	{
	  		vt_star = (v_bound[0]*norm[1] - v_bound[1]*norm[0]);
	  		//assumes quasi-2D boundary i.e. norm[2] == 0;
        
        // HACK
	  		one_over_s = pow(rho_bound,gamma)/p_bound;
        
	  		// freestream total enthalpy
        v_sq = 0.;
        for (int i=0;i<n_dims;i++)
          v_sq += v_bound[i]*v_bound[i];
        
	  		h_free_stream = gamma/(gamma-1.)*p_bound/rho_bound+ 0.5*v_sq;
	  		u_r[0] = pow(1./gamma*(one_over_s*c_star*c_star),1./(gamma-1.));
        
      	u_r[1] = u_r[0]*(norm[0]*vn_star + norm[1]*vt_star);
        u_r[2] = u_r[0]*(norm[1]*vn_star - norm[0]*vt_star);
	  		
	  		if(n_dims==3)
	  		{
	  			u_r[3] = 0.0; //no cross-flow
	  		}
        
	  		p_r = u_r[0]/gamma*c_star*c_star;
	  		u_r[n_dims+1] = u_r[0]*h_free_stream - p_r;
	  	}
	  	else  // Outflow
	  	{
	  		vt_star = (u_l[1]*norm[1] - u_l[2]*norm[0])/u_l[0];
        
	  		one_over_s = pow( u_l[0], gamma)/p_l;
        
	  		// freestream total enthalpy
	  		u_r[0] = pow( (1./gamma*(one_over_s*c_star*c_star)) , (1./(gamma-1.)));
        
        u_r[1] = u_r[0]*(norm[0]*vn_star + norm[1]*vt_star);
        u_r[2] = u_r[0]*(norm[1]*vn_star - norm[0]*vt_star);
        
	  		// no cross-flow
	  		if(n_dims==3)
	  		{
	  			u_r[3] = 0.0;
	  		}
        
	  		p_r = u_r[0]/gamma*c_star*c_star;
        v_sq = 0.;
        for (int i=0;i<n_dims;i++)
          v_sq += u_r[i+1]*u_r[i+1];
        
	  		u_r[n_dims+1] = (p_r/(gamma-1.0)) + 0.5*v_sq/u_r[0];;
	  	}
      
	  }
    else if (bdy_type==16) // Dual consistent BC
    {
      /*
       // Compute normal velocity on left side
       vn_l = 0.;
       for (int i=0;i<n_dims;i++)
       vn_l += u_l[i+1]*norm[i];
       vn_l /= u_l[0];
       
       // extrapolate density
       u_r[0] = u_l[0];
       
       // set u = u - (vn_l)nx
       // set v = v - (vn_l)ny
       // set w = w - (vn_l)nz
       
       for (int i=0;i<n_dims;i++)
       u_r[i+1] = u_l[i+1] - vn_l*norm[i]*u_r[0];
       
       // extrapolate energy
       u_r[n_dims+1] = u_l[n_dims+1];
       
       */
      
      
      // "DUAL-CONSISTENT" WALL BC
      
      u_r[0]   = u_l[0];
      u_r[1]  = (1-norm[0]*norm[0])*u_l[1] - norm[0]*norm[1]*u_l[2];
      u_r[2]  = (1-norm[1]*norm[1])*u_l[2] - norm[0]*norm[1]*u_l[1];
      u_r[3]  = u_l[3];
      
      /*
       uB     = QB(2)/rhoB
       vB     = QB(3)/rhoB
       pB     = (gam-1) * (QB(4)-0.5*rhoB*(uB**2+vB**2))
       */
      
      
      
      
    }
	  else if (bdy_type == 17) // Characteristic Lala
	  {
      
      if (n_dims==3)
        printf("Char BDY does not work in 3D");
      //cout << scientific << setw(12) << setprecision(8) << norm[0] << endl;
      //cout << scientific << setw(12) << setprecision(8) << norm[1] << endl;
      ////cout << scientific << setw(12) << setprecision(8) << u_l[0] << endl;
      //cout << scientific << setw(12) << setprecision(8) << u_l[1]/u_l[0] << endl;
      //cout << scientific << setw(12) << setprecision(8) << u_l[2]/u_l[0] << endl;
      //cout << scientific << setw(12) << setprecision(8) << u_l[3] << endl;
      
      double RHOL,UL,VL,PL,VNL,VTL,CL,SL;
      double VNI,VTI,CI,SI,RM,RP,VNB,VTB,SB,CB,RHOB,UB,VB,PB;
      
      RHOL  = u_l[0];
      UL    = u_l[1]/RHOL;
      VL    = u_l[2]/RHOL;
      PL    = (gamma-1) * (u_l[3]-0.5*RHOL*(UL*UL+VL*VL));
      VNL   = UL*norm[0] + VL*norm[1];
      VTL   = VL*norm[0] - UL*norm[1];
      CL    = sqrt(gamma*PL/RHOL);
      SL    = PL/pow(RHOL,gamma);
      
      //printf("v_bound= %f %f, v_l = %f %f\n",v_bound[0],v_bound[1],UL,VL);
      //printf("p_bound= %f,  p_l = %f\n",p_bound,PL);
      //printf("rho_bound= %f,  rho_l = %f\n",rho_bound,RHOL);
      
      // FLOW PROPERTIES IN THE FARFIELD
      VNI   = v_bound[0]*norm[0] + v_bound[1]*norm[1];
      VTI   = v_bound[1]*norm[0] - v_bound[0]*norm[1];
      CI    = sqrt(gamma*p_bound/rho_bound);
      SI    = p_bound/pow(rho_bound,gamma);
      
      // CALCULATE THE RIEMANN INVARIANT
      RM    = VNI - 2./(gamma-1.)*CI;
      RP    = VNL + 2./(gamma-1.)*CL;
      VNB   = (RP + RM)/2.;
      
      //printf("norm= %f %f, vnb=%f \n",norm[0],norm[1],VNB);
      
      if( VNB <0 ) { // INFLOW
        VTB      = VTI;
        SB       = SI;
      }
      else {// OUTFLOW
        VTB      = VTL;
        SB       = SL;
      }
      
      CB    = 0.25*(gamma-1.)*(RP - RM);
      
      RHOB = pow(CB*CB/(gamma*SB),1./(gamma-1.));
      PB    = SB*pow(RHOB,gamma);
      UB    = norm[0]*VNB - norm[1]*VTB;
      VB    = norm[1]*VNB + norm[0]*VTB;
      
      u_r[0] = RHOB;
      u_r[1] = RHOB*UB;
      u_r[2] = RHOB*VB;
      u_r[3] = PB/(gamma-1.) + 0.5*RHOB*(UB*UB + VB*VB);
      
	  }
    
    /*
     else if (flag==100)
     {
     
     if (x>0)
     {
     cout << "100,x>0" << endl;
     // fix density and velocity
     rho_r =  rho_free_stream;
     vx_r  =  u_free_stream;
     vy_r  =  v_free_stream;
     // fix pressure
     p_r = p_free_stream;
     
     // fix energy
     Energy_r = (p_r/(gam-1.0)) + 0.5*rho_r*( vx_r*vx_r + vy_r*vy_r );
     }
     else
     {
     cout << "100,x<0" << endl;
     rho_r = rho_l;
     vx_r = vx_l;
     vy_r = vy_l;
     Energy_r = Energy_l;
     }
     }
     else if (flag==200)
     {
     
     if (y>1/sqrt(3.))
     {
     cout << "200,y>" << endl;
     // fix density and velocity
     rho_r =  rho_free_stream;
     vx_r  =  u_free_stream;
     vy_r  =  v_free_stream;
     // fix pressure
     p_r = p_free_stream;
     
     // fix energy
     Energy_r = (p_r/(gam-1.0)) + 0.5*rho_r*( vx_r*vx_r + vy_r*vy_r );
     }
     else
     {
     cout << "200,y<" << endl;
     rho_r = rho_l;
     vx_r = vx_l;
     vy_r = vy_l;
     Energy_r = Energy_l;
     }
     
     }
     else if (flag==300)
     {
     
     if (y<1/sqrt(3.))
     {
     cout << "300,y<" << endl;
     // fix density and velocity
     rho_r =  rho_free_stream;
     vx_r  =  u_free_stream;
     vy_r  =  v_free_stream;
     // fix pressure
     p_r = p_free_stream;
     
     // fix energy
     Energy_r = (p_r/(gam-1.0)) + 0.5*rho_r*( vx_r*vx_r + vy_r*vy_r );
     }
     else
     {
     cout << "300,y>" << endl;
     rho_r = rho_l;
     vx_r = vx_l;
     vy_r = vy_l;
     Energy_r = Energy_l;
     }
     }
     */
    
    /*
     else if (flag==200)
     {
     
     // fix density and velocity
     rho_r =  rho_free_stream;
     vx_r  =  u_free_stream;
     vy_r  =  v_free_stream;
     // fix pressure
     p_r = p_free_stream;
     
     // fix energy
     Energy_r = (p_r/(gam-1.0)) + 0.5*rho_r*( vx_r*vx_r + vy_r*vy_r );
     }
     else if (flag==100)
     {
     rho_r = rho_l;
     vx_r = vx_l;
     vy_r = vy_l;
     Energy_r = Energy_l;
     
     }
     else if (flag==300)
     {
     rho_r = rho_l;
     vx_r = vx_l;
     vy_r = vy_l;
     Energy_r = Energy_l;
     }
     */
    
    
	  else
	  {
	  	// Boundary condition not implemented yet
	  	printf("bdy_type=%d\n",bdy_type);
	  	printf("Boundary conditions yet to be implemented");
	  }
  }
  
  
  if(equation==1) //Advection/Advection-Diffusion BC's
  {
    if(bdy_type==50) //Trivial dirichlet
    {
      u_r[0]=0.0; 
    }
  }
  
}


// calculate normal transformed continuous viscous flux at the flux points on the boundaries
void bdy_inters::calc_norm_tconvisf_fpts_boundary(double time_bound)
{

  #ifdef _CPU
  int bdy_spec, flux_spec;
  array<double> norm(n_dims), fn(n_fields);
	
	for(int i=0;i<n_inters;i++)
	{
    //boundary specification
    bdy_spec = boundary_type(i);
		
    if(bdy_spec == 12 || bdy_spec == 14)
      flux_spec = 2;
    else
      flux_spec = 1;

    for(int j=0;j<n_fpts_per_inter;j++)
		{
      // storing normal components
      for (int m=0;m<n_dims;m++)
        norm(m) = *norm_fpts(j,i,m);
			
      // obtain discontinuous solution at flux points
			for(int k=0;k<n_fields;k++)
				temp_u_l(k)=(*disu_fpts_l(j,i,k));
      
      for (int m=0;m<n_dims;m++)
        temp_loc(m) = *loc_fpts(j,i,m);

      set_inv_boundary_conditions(bdy_spec,temp_u_l.get_ptr_cpu(),temp_u_r.get_ptr_cpu(),norm.get_ptr_cpu(),temp_loc.get_ptr_cpu(),bdy_params.get_ptr_cpu(),n_dims,n_fields,run_input.gamma,run_input.R_ref,time_bound,run_input.equation);
			
			// obtain gradient of discontinuous solution at flux points
			for(int k=0;k<n_dims;k++)
			{
				for(int l=0;l<n_fields;l++)
				{
					temp_grad_u_l(l,k) = *grad_disu_fpts_l(j,i,l,k);
				}
			}
      
      //Right gradient
      if(flux_spec == 2)
      {
			  // Extrapolate
			  for(int k=0;k<n_dims;k++)
			  {
			  	for(int l=0;l<n_fields;l++)
			  	{
			  		temp_grad_u_r(l,k) = temp_grad_u_l(l,k);
			  	}
			  }

        set_vis_boundary_conditions(bdy_spec,temp_u_l.get_ptr_cpu(),temp_u_r.get_ptr_cpu(),temp_grad_u_r.get_ptr_cpu(),norm.get_ptr_cpu(),temp_loc.get_ptr_cpu(),bdy_params.get_ptr_cpu(),n_dims,n_fields,run_input.gamma,run_input.R_ref,time_bound,run_input.equation);
      }

			// calculate flux from discontinuous solution at flux points
			if(n_dims==2) {
				
        if(flux_spec == 1)
        {
          calc_visf_2d(temp_u_l,temp_grad_u_l,temp_f_l);
        }
        else if(flux_spec == 2)
        {
				  calc_visf_2d(temp_u_r,temp_grad_u_r,temp_f_r);
        }
        else
  			  FatalError("Invalid viscous flux specification");
			}
			else if(n_dims==3)  {
        
        if(flux_spec == 1)
        {
          calc_visf_3d(temp_u_l,temp_grad_u_l,temp_f_l);
        }
        else if(flux_spec == 2)
        {
				  calc_visf_3d(temp_u_r,temp_grad_u_r,temp_f_r);
        }
        else
  			  FatalError("Invalid viscous flux specification");
			}
			else
  			FatalError("ERROR: Invalid number of dimensions ... ");


      // Calling viscous riemann solver
      if (run_input.vis_riemann_solve_type==0)
        ldg_flux(flux_spec,temp_u_l,temp_u_r,temp_f_l,temp_f_r,norm,fn,n_dims,n_fields,run_input.tau,run_input.pen_fact);
      else
        FatalError("Viscous Riemann solver not implemented");

      // Transform back to reference space  
  		for(int k=0;k<n_fields;k++) 
					(*norm_tconf_fpts_l(j,i,k))+=fn(k)*(*mag_tnorm_dot_inv_detjac_mul_jac_fpts_l(j,i));
		}
	}

	#endif
	
	#ifdef _GPU
  if (n_inters!=0)
    calc_norm_tconvisf_fpts_boundary_gpu_kernel_wrapper(n_fpts_per_inter,n_dims,n_fields,n_inters,disu_fpts_l.get_ptr_gpu(),grad_disu_fpts_l.get_ptr_gpu(),norm_tconf_fpts_l.get_ptr_gpu(),mag_tnorm_dot_inv_detjac_mul_jac_fpts_l.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),loc_fpts.get_ptr_gpu(),boundary_type.get_ptr_gpu(),bdy_params.get_ptr_gpu(),delta_disu_fpts_l.get_ptr_gpu(),run_input.riemann_solve_type,run_input.vis_riemann_solve_type,run_input.R_ref,run_input.pen_fact,run_input.tau,run_input.gamma,run_input.prandtl,run_input.rt_inf,run_input.mu_inf,run_input.c_sth,run_input.fix_vis, time_bound, run_input.equation, run_input.diff_coeff);
	#endif
}


void bdy_inters::set_vis_boundary_conditions(int bdy_type, double* u_l, double* u_r, double* grad_u, double *norm, double *loc, double *bdy_params, int n_dims, int n_fields, double gamma, double R_ref, double time_bound, int equation)
{
  int cpu_flag;
  cpu_flag = 1;
  
  
  double v_sq;
  double inte;
  double p_l, p_r;
  
  double grad_vel[n_dims*n_dims];
  
  
  // Adiabatic wall
  if(bdy_type == 12 || bdy_type == 14)
  {
    v_sq = 0.;
    for (int i=0;i<n_dims;i++)
      v_sq += (u_l[i+1]*u_l[i+1]);
    p_l   = (gamma-1.0)*( u_l[n_dims+1] - 0.5*v_sq/u_l[0]);
    p_r = p_l;
    
    inte = p_r/((gamma-1.0)*u_r[0]);
    
    if(cpu_flag)
    {
      // Velocity gradients
      for (int j=0;j<n_dims;j++)
      {
        for (int i=0;i<n_dims;i++)
          grad_vel[j*n_dims + i] = (grad_u[i*n_fields + (j+1)] - grad_u[i*n_fields + 0]*u_r[j+1]/u_r[0])/u_r[0];
      }
      
      // Energy gradients (grad T = 0)
      if(n_dims == 2)
      {
        for (int i=0;i<n_dims;i++)
          grad_u[i*n_fields + 3] = inte*grad_u[i*n_fields + 0] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2])/(u_r[0]*u_r[0]))*grad_u[i*n_fields + 0] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]);
      }
      else if(n_dims == 3)
      {
        for (int i=0;i<n_dims;i++)
          grad_u[i*n_fields + 4] = inte*grad_u[i*n_fields + 0] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2]+u_r[3]*u_r[3])/(u_r[0]*u_r[0]))*grad_u[i*n_fields + 0] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]+(u_r[3]/u_r[0])*grad_vel[2*n_dims + i]);
      }
    }
    else
    {
      // Velocity gradients
      for (int j=0;j<n_dims;j++)
      {
        for (int i=0;i<n_dims;i++)
          grad_vel[j*n_dims + i] = (grad_u[(j+1)*n_dims + i] - grad_u[0*n_dims + i]*u_r[j+1]/u_r[0])/u_r[0];
      }
      
      if(n_dims == 2)
      {
        // Total energy gradient
        for (int i=0;i<n_dims;i++)
          grad_u[3*n_dims + i] = inte*grad_u[0*n_dims + i] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2])/(u_r[0]*u_r[0]))*grad_u[0*n_dims + i] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]);
      }
      else if(n_dims == 3)
      {
        for (int i=0;i<n_dims;i++)
          grad_u[4*n_dims + i] = inte*grad_u[0*n_dims + i] + 0.5*((u_r[1]*u_r[1]+u_r[2]*u_r[2]+u_r[3]*u_r[3])/(u_r[0]*u_r[0]))*grad_u[0*n_dims + i] + u_r[0]*((u_r[1]/u_r[0])*grad_vel[0*n_dims + i]+(u_r[2]/u_r[0])*grad_vel[1*n_dims + i]+(u_r[3]/u_r[0])*grad_vel[2*n_dims + i]);
      }
    }
    
  }   
  
}

