/*!
 * \file cuda_kernels.cu
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

#define HALFWARP 16
#include <iostream>

using namespace std;

#include "../include/cuda_kernels.h"
#include "../include/error.h"
#include "../include/util.h"

#ifdef _MPI
#include "mpi.h"
#endif

//Key

// met[0][0] = rx
// met[1][0] = sx
// met[0][1] = ry
// met[1][1] = sy

// Add a bespoke_MV_kernel to do non-sparse matrix-vector multiplication

template<int n_fields> 
__global__ void bespoke_SPMV_kernel(double *g_c, double *g_b, double *g_cont_mat, int *g_col_mat, const int n_nz, const int n_cells, const int dim1, const int dim0, const int cells_per_block, const int stride_n, const int stride_m, int add_flag)
{
	extern __shared__ double s_b[];

	const int tid = threadIdx.x;
	const int ic_loc = tid/dim0;
	const int ifp = tid-ic_loc*dim0;
	const int ic = blockIdx.x*cells_per_block+ ic_loc;
	const int stride_shared = cells_per_block*dim1+ (cells_per_block*dim1)/HALFWARP;
	int i_qpt, m, m1;

  double q[n_fields];

  #pragma unroll
  for (int j=0;j<n_fields;j++)
    q[j] = 0.;

	double mat_entry;

  	if (tid < cells_per_block*dim0 && ic < n_cells)
  	{

		// Fetching data to shared memory
  		int n_fetch_loops = (dim1-1)/(dim0)+1;

		// Since n_qpts might be larger than dim0 
        	// each thread might have to fetch more than n_fields values
	  	for (int i=0;i<n_fetch_loops;i++)
  		{
     			i_qpt= i*dim0+ifp;

     			if (i_qpt<dim1)
     		 	{
				    // Fetch the four field values of solution point i_qpt
				    m  = ic_loc *dim1+i_qpt;
				    m += m/HALFWARP;

				    m1 = ic     *dim1+i_qpt;
            #pragma unroll  
            for (int j=0;j<n_fields;j++)
            {
				      s_b[m] = g_b[m1]; // copy global B vector to shared B vector
              m += stride_shared; 
              m1 += stride_n;
            }
			    }
     	} 
   	}

   	__syncthreads(); // make sure memory copy is complete in all threads

   	if (tid < cells_per_block*dim0 && ic < n_cells)
   	{

  		// With data in shared memory, perform matrix multiplication
  		// 1 thread per flux point
  		for (int i=0;i<n_nz;i++)
  		{
			  m = i*dim0+ifp;
			  m1 = dim1*ic_loc + g_col_mat[m];
			  //m1 = n_qpts*ic_loc + tex1Dfetch(t_col_mat,m);
			  m1 += m1/HALFWARP;

			  mat_entry = g_cont_mat[m];
			  //mat_entry = fetch_double(t_cont_mat,m);

        #pragma unroll
        for (int j=0;j<n_fields;j++)
        {
			    q[j] += mat_entry*s_b[m1];	
          m1 += stride_shared;
        }

  		}
	
		// Store in global memory
		 m = ic*dim0+ifp;
    #pragma unroll
     for (int j=0;j<n_fields;j++)
      {
       if (add_flag==0)
        g_c[m] = q[j];
       else if (add_flag==1)
        g_c[m] += q[j];

       m += stride_m;
     }
  	} 
	
}


template<int n_dims, int n_fields> 
__device__ void set_inv_boundary_conditions_kernel(int bdy_type, double* u_l, double* u_r, double* norm, double* loc, double *bdy_params, double gamma, double R_ref, double time_bound, int equation)
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


      // "DUAL-CONSISTENT" WALL BC

      u_r[0]   = u_l[0];
      u_r[1]  = (1-norm[0]*norm[0])*u_l[1] - norm[0]*norm[1]*u_l[2];
      u_r[2]  = (1-norm[1]*norm[1])*u_l[2] - norm[0]*norm[1]*u_l[1];
      u_r[3]  = u_l[3];


    }
	  else if (bdy_type == 17) // Characteristic Lala
	  {

      if (n_dims==3)
        printf("Char BDY does not work in 3D");

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


template<int n_dims, int n_fields>
__device__ void set_vis_boundary_conditions_kernel(int bdy_type, double* u_l, double* u_r, double* grad_u, double *norm, double *loc, double *bdy_params, double gamma, double R_ref, double time_bound, int equation)
{
  int cpu_flag;
  cpu_flag = 0;
  
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


template<int in_n_dims>
__device__ void inv_NS_flux(double* q, double *p, double* f, double in_gamma, int in_field)
{
  if(in_n_dims==2) {
    
    if (in_field==-1) {
		  (*p) = (in_gamma-1.0)*(q[3]-0.5*(q[1]*q[1]+q[2]*q[2])/q[0]);
    }
    else if (in_field==0) {
		  f[0] = q[1];
		  f[1] = q[2];
    }
    else if (in_field==1) {
		  f[0]  = (*p)+(q[1]*q[1]/q[0]);
		  f[1]  = q[2]*q[1]/q[0];
    }
    else if (in_field==2) {
		  f[0]  = q[1]*q[2]/q[0];
		  f[1]  = (*p) + (q[2]*q[2]/q[0]);
    }
    else if (in_field==3) {
		  f[0]  = q[1]/q[0]*(q[3]+(*p));
		  f[1]  = q[2]/q[0]*(q[3]+(*p));
    }
  }
  else if(in_n_dims==3)
  {
    if (in_field==-1) {
			(*p) = (in_gamma-1.0)*(q[4]-0.5*(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/q[0]);
    }
    else if (in_field==0) {
			f[0] = q[1];
			f[1] = q[2];
			f[2] = q[3]; 
    }
    else if (in_field==1) {
	    f[0] = (*p)+(q[1]*q[1]/q[0]);
			f[1] = q[2]*q[1]/q[0];
			f[2] = q[3]*q[1]/q[0];
    }
    else if (in_field==2) {
			f[0] = q[1]*q[2]/q[0];
			f[1] = (*p)+(q[2]*q[2]/q[0]);
			f[2] = q[3]*q[2]/q[0];
    }
    else if (in_field==3) {
			f[0] = q[1]*q[3]/q[0];
			f[1] = q[2]*q[3]/q[0];
			f[2] = (*p) + (q[3]*q[3]/q[0]);
    }
    else if (in_field==4) {
	 		f[0] = q[1]/q[0]*(q[4]+(*p));
			f[1] = q[2]/q[0]*(q[4]+(*p));
			f[2] = q[3]/q[0]*(q[4]+(*p));
    }
  }
}


template<int in_n_dims>
__device__ void vis_NS_flux(double* q, double* grad_q, double* grad_vel, double* grad_ene, double* stensor, double* f, double* inte, double* mu, double in_prandtl, double in_gamma, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int in_field)
{
  double diag;
  double rt_ratio;

  if(in_n_dims==2) {
      
    if(in_field==-1) {
    
      // Internal energy
      (*inte) = (q[3]/q[0])-0.5*((q[1]*q[1]+q[2]*q[2])/(q[0]*q[0]));
    
      // Viscosity
      rt_ratio = (in_gamma-1.)*(*inte)/(in_rt_inf);
      (*mu) = in_mu_inf*pow(rt_ratio,1.5)*(1.+in_c_sth)/(rt_ratio+in_c_sth);
      (*mu) = (*mu) + in_fix_vis*(in_mu_inf - (*mu));
      
      // Velocity gradients
      #pragma unroll
      for (int j=0;j<in_n_dims;j++) 
      {
        #pragma unroll
        for (int i=0;i<in_n_dims;i++) 
          grad_vel[j*in_n_dims + i] = (grad_q[(j+1)*in_n_dims + i] - grad_q[0*in_n_dims + i]*q[j+1]/q[0])/q[0];
      }
      
      // Kinetic energy gradient
      #pragma unroll
      for (int i=0;i<in_n_dims;i++) 
        grad_ene[i] = 0.5*((q[1]*q[1]+q[2]*q[2])/(q[0]*q[0]))*grad_q[0*in_n_dims + i] + q[0]*((q[1]/q[0])*grad_vel[0*in_n_dims + i]+(q[2]/q[0])*grad_vel[1*in_n_dims + i]); 

      // Total energy gradient
      #pragma unroll
      for (int i=0;i<in_n_dims;i++) 
        grad_ene[i] = (grad_q[3*in_n_dims + i] - grad_ene[i] - grad_q[0*in_n_dims + i]*(*inte))/q[0];

      diag = (grad_vel[0*in_n_dims + 0] + grad_vel[1*in_n_dims + 1])/3.0;

      // Stress tensor
      #pragma unroll
      for (int i=0;i<in_n_dims;i++) 
        stensor[i] = 2.0*(*mu)*(grad_vel[i*in_n_dims + i] - diag);
      
      stensor[2] = (*mu)*(grad_vel[0*in_n_dims + 1] + grad_vel[1*in_n_dims + 0]); 

    }
    else if (in_field==0) {
      f[0] = 0.0;
      f[1] = 0.0;
    }
    else if (in_field==1) {
		  f[0]  = -stensor[0];
		  f[1]  = -stensor[2];
    }
    else if (in_field==2) {
		  f[0]  = -stensor[2];
		  f[1]  = -stensor[1];
    }
    else if (in_field==3) {
		  f[0]  = -((q[1]/q[0])*stensor[0] + (q[2]/q[0])*stensor[2] + (*mu)*in_gamma*grad_ene[0]/in_prandtl);
		  f[1]  = -((q[1]/q[0])*stensor[2] + (q[2]/q[0])*stensor[1] + (*mu)*in_gamma*grad_ene[1]/in_prandtl);
    }
  }
  else if(in_n_dims==3)
  {
    if(in_field==-1) {
      
      // Internal energy
      (*inte) = (q[4]/q[0])-0.5*((q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/(q[0]*q[0]));
    
      // Viscosity
      rt_ratio = (in_gamma-1.)*(*inte)/(in_rt_inf);
      (*mu) = in_mu_inf*pow(rt_ratio,1.5)*(1.+in_c_sth)/(rt_ratio+in_c_sth);
      (*mu) = (*mu) + in_fix_vis*(in_mu_inf - (*mu));
      
      // Velocity gradients
      #pragma unroll
      for (int j=0;j<in_n_dims;j++) 
      {
        #pragma unroll
        for (int i=0;i<in_n_dims;i++) 
          grad_vel[j*in_n_dims + i] = (grad_q[(j+1)*in_n_dims + i] - grad_q[0*in_n_dims + i]*q[j+1]/q[0])/q[0];
      }
      
      // Kinetic energy gradient
      #pragma unroll
      for (int i=0;i<in_n_dims;i++) 
        grad_ene[i] = 0.5*((q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/(q[0]*q[0]))*grad_q[0*in_n_dims + i] + q[0]*((q[1]/q[0])*grad_vel[0*in_n_dims + i]+(q[2]/q[0])*grad_vel[1*in_n_dims + i]+(q[3]/q[0])*grad_vel[2*in_n_dims + i]); 

      // Total energy gradient
      #pragma unroll
      for (int i=0;i<in_n_dims;i++) 
        grad_ene[i] = (grad_q[4*in_n_dims + i] - grad_ene[i] - grad_q[0*in_n_dims + i]*(*inte))/q[0];

      diag = (grad_vel[0*in_n_dims + 0] + grad_vel[1*in_n_dims + 1] + grad_vel[2*in_n_dims + 2])/3.0;

      // Stress tensor
      #pragma unroll
      for (int i=0;i<in_n_dims;i++) 
        stensor[i] = 2.0*(*mu)*(grad_vel[i*in_n_dims + i] - diag);
      
      stensor[3] = (*mu)*(grad_vel[0*in_n_dims + 1] + grad_vel[1*in_n_dims + 0]); 
      stensor[4] = (*mu)*(grad_vel[0*in_n_dims + 2] + grad_vel[2*in_n_dims + 0]); 
      stensor[5] = (*mu)*(grad_vel[1*in_n_dims + 2] + grad_vel[2*in_n_dims + 1]); 
    }
    else if (in_field==0) {
      f[0] = 0.0;
			f[1] = 0.0;
			f[2] = 0.0; 
    }
    else if (in_field==1) {
      f[0]  = -stensor[0];
			f[1]  = -stensor[3];
			f[2]  = -stensor[4];
    }
    else if (in_field==2) {
			f[0] = -stensor[3];
			f[1] = -stensor[1];
			f[2] = -stensor[5];
    }
    else if (in_field==3) {
			f[0] = -stensor[4];
			f[1] = -stensor[5];
			f[2] = -stensor[2];
    }
    else if (in_field==4) {
	 		f[0] = -((q[1]/q[0])*stensor[0]+(q[2]/q[0])*stensor[3]+(q[3]/q[0])*stensor[4] + (*mu)*in_gamma*grad_ene[0]/in_prandtl);
			f[1] = -((q[1]/q[0])*stensor[3]+(q[2]/q[0])*stensor[1]+(q[3]/q[0])*stensor[5] + (*mu)*in_gamma*grad_ene[1]/in_prandtl);
			f[2] = -((q[1]/q[0])*stensor[4]+(q[2]/q[0])*stensor[5]+(q[3]/q[0])*stensor[2] + (*mu)*in_gamma*grad_ene[2]/in_prandtl);
    }
  }
}

__device__ double SGS_filter_width(double in_detjac, int in_ele_type, int in_n_dims, double in_filter_ratio)
{
	// Define filter width by Deardorff's unstructured element method
	double delta, vol;

	if (in_ele_type==0) // triangle
	{
		vol = in_detjac*2.0;
	}
	else if (in_ele_type==1) // quads
	{
		vol = in_detjac*4.0;
	}
	else if (in_ele_type==2) // tets
	{
		vol = in_detjac*8.0/6.0;
	}
	else if (in_ele_type==4) // hexas
	{
		vol = in_detjac*8.0;
	}

	delta = in_filter_ratio*pow(vol,1./in_n_dims);
	//printf("vol: %6.10f",vol);
	//printf("delta: %6.10f",delta);
	return delta;
}

template<int in_n_dims>
__device__ void SGS_flux_kernel(double* q, double* grad_q, double* grad_vel, double* grad_ene, double* sdtensor, double* straintensor, double* Leonard_mom, double* Leonard_ene, double* f, int SGS_model, double delta, double in_gamma, int in_field)
{
	int i, j;
	int eddy, sim;
	double Cs, mu_t;
	double Smod=0.0;
	double Pr=0.5; // turbulent Prandtl number
	double num=0.0;
	double denom=0.0;
	double diag=0.0;
	double eps=1.e-10;

	// Set flags depending on which SGS model we are using
	// 0: Smagorinsky, 1: WALE, 2: WALE-similarity, 3: SVV, 4: Similarity
	if(SGS_model==0) {
		eddy = 1;
		sim = 0;
	}
	else if(SGS_model==1) {
		eddy = 1;
		sim = 0;
	}
	else if(SGS_model==2) {
		eddy = 1;
		sim = 1;
	}
	else if(SGS_model==3) {
		eddy = 0;
		sim = 0;
	}
	else if(SGS_model==4) {
		eddy = 0;
		sim = 1;
	}

	// Calculate eddy viscosity

	// Smagorinsky model
	if(SGS_model==0) {

		Cs=0.1;

		// Calculate modulus of strain rate tensor
		#pragma unroll
		for (i=0;i<in_n_dims;i++) {
			Smod += 2.0*straintensor[i]*straintensor[i];
		}

		// Now the off-diagonal components of strain tensor:
	  if(in_n_dims==2) {
			Smod += 4.0*straintensor[2]*straintensor[2];
		}
	  else if(in_n_dims==3) {
			Smod += 4.0*(straintensor[3]*straintensor[3] + straintensor[4]*straintensor[4] + straintensor[5]*straintensor[5]);
		}

		// Finally, the modulus of strain rate tensor
		Smod = sqrt(Smod);

		mu_t = q[0]*Cs*Cs*delta*delta*Smod;
	}

	// WALE or WSM model
	else if(SGS_model==1 || SGS_model==2) {

		Cs=0.5;

		// Square of velocity gradient tensor
		#pragma unroll
		for (i=0;i<in_n_dims;i++) {
			sdtensor[i] = 0.0;
			#pragma unroll
			for (j=0;j<in_n_dims;j++) {
				diag += grad_vel[i*in_n_dims + j]*grad_vel[j*in_n_dims + i]/3.0;
				sdtensor[i] += grad_vel[i*in_n_dims + j]*grad_vel[j*in_n_dims + i];
			}
		}

		// subtract trace from diagonal entries of tensor
		#pragma unroll
		for (i=0;i<in_n_dims;i++)
			sdtensor[i] -= diag;

		// off-diagonal terms of tensor
	  if(in_n_dims==2) {
			sdtensor[2] = 0.0;
			#pragma unroll
			for (j=0;j<in_n_dims;j++) {
				sdtensor[2] += (grad_vel[0*in_n_dims + j]*grad_vel[j*in_n_dims + 1] + grad_vel[1*in_n_dims + j]*grad_vel[j*in_n_dims + 0])/2.0;
			}
		}
	  else if(in_n_dims==3) {
			sdtensor[3] = 0.0;
			sdtensor[4] = 0.0;
			sdtensor[5] = 0.0;
			#pragma unroll
			for (j=0;j<in_n_dims;j++) {
				sdtensor[3] += (grad_vel[0*in_n_dims + j]*grad_vel[j*in_n_dims + 1] + grad_vel[1*in_n_dims + j]*grad_vel[j*in_n_dims + 0])/2.0;

				sdtensor[4] += (grad_vel[0*in_n_dims + j]*grad_vel[j*in_n_dims + 2] + grad_vel[2*in_n_dims + j]*grad_vel[j*in_n_dims + 0])/2.0;

				sdtensor[5] += (grad_vel[1*in_n_dims + j]*grad_vel[j*in_n_dims + 2] + grad_vel[2*in_n_dims + j]*grad_vel[j*in_n_dims + 1])/2.0;
			}
		}

		// numerator and denominator of eddy viscosity term
		#pragma unroll
		for (i=0;i<in_n_dims;i++) {
			num += sdtensor[i]*sdtensor[i];
			denom += straintensor[i]*straintensor[i];
		}

		if(in_n_dims==2) {
			num += 2.0*sdtensor[2]*sdtensor[2];
			denom += 2.0*straintensor[2]*straintensor[2];
		}
		else if(in_n_dims==3) {
			num += 2.0*(sdtensor[3]*sdtensor[3] + sdtensor[4]*sdtensor[4] + sdtensor[5]*sdtensor[5]);
			denom += 2.0*(straintensor[3]*straintensor[3] + straintensor[4]*straintensor[4] + straintensor[5]*straintensor[5]);
		}

		denom = pow(denom,2.5) + pow(num,1.25);
		num = pow(num,1.5);
		mu_t = q[0]*Cs*Cs*delta*delta*num/(denom+eps);
	}

	// Now set the flux values
	if (eddy==1) {
		if (in_n_dims==2) {

			// Density
			if (in_field==0) {
				f[0] = 0.0;
				f[1] = 0.0;
			}
			// u
			else if (in_field==1) {
				f[0] = -2.0*mu_t*straintensor[0];
				f[1] = -2.0*mu_t*straintensor[2];
			}
			// v
			else if (in_field==2) {
				f[0] = -2.0*mu_t*straintensor[2];
				f[1] = -2.0*mu_t*straintensor[1];
			}
			// Energy
			else if (in_field==3) {
				f[0] = -1.0*in_gamma*mu_t/Pr*grad_ene[0];
				f[1] = -1.0*in_gamma*mu_t/Pr*grad_ene[1];
			}
		}
		else if(in_n_dims==3) {

			// Density
			if (in_field==0) {
				f[0] = 0.0;
				f[1] = 0.0;
				f[2] = 0.0; 
			}
			// u
			else if (in_field==1) {
				f[0] = -2.0*mu_t*straintensor[0];
				f[1] = -2.0*mu_t*straintensor[3];
				f[2] = -2.0*mu_t*straintensor[4];
			}
			// v
			else if (in_field==2) {
				f[0] = -2.0*mu_t*straintensor[3];
				f[1] = -2.0*mu_t*straintensor[1];
				f[2] = -2.0*mu_t*straintensor[5];
			}
			// w
			else if (in_field==3) {
				f[0] = -2.0*mu_t*straintensor[4];
				f[1] = -2.0*mu_t*straintensor[5];
				f[2] = -2.0*mu_t*straintensor[2];
			}
			// Energy
			else if (in_field==4) {
				f[0] = -1.0*in_gamma*mu_t/Pr*grad_ene[0];
				f[1] = -1.0*in_gamma*mu_t/Pr*grad_ene[1];
				f[2] = -1.0*in_gamma*mu_t/Pr*grad_ene[2];
			}
		}
	}
	// Add similarity term to SGS fluxes if WSM or Similarity model
	if (sim==1)
	{
		if(in_n_dims==2) {

			// Density
			if (in_field==0) {
				f[0] += 0.0;
				f[1] += 0.0;
			}

			// u
			if (in_field==1) {
				f[0] += q[0]*Leonard_mom[0];
				f[1] += q[0]*Leonard_mom[2];
			}
			// v
			else if (in_field==2) {
				f[0] += q[0]*Leonard_mom[2];
				f[1] += q[0]*Leonard_mom[1];
			}
			// Energy
			else if (in_field==3) {
				f[0] += q[0]*in_gamma*Leonard_ene[0];
				f[1] += q[0]*in_gamma*Leonard_ene[1];
			}
		}
		else if(in_n_dims==3)
		{
			// u
			if (in_field==1) {
				f[0] += q[0]*Leonard_mom[0];
				f[1] += q[0]*Leonard_mom[3];
				f[2] += q[0]*Leonard_mom[4];
			}
			// v
			else if (in_field==2) {
				f[0] += q[0]*Leonard_mom[3];
				f[1] += q[0]*Leonard_mom[1];
				f[2] += q[0]*Leonard_mom[5];
			}
			// w
			else if (in_field==3) {
				f[0] += q[0]*Leonard_mom[4];
				f[1] += q[0]*Leonard_mom[5];
				f[2] += q[0]*Leonard_mom[2];
			}
			// Energy
			else if (in_field==4) {
				f[0] += q[0]*in_gamma*Leonard_ene[0];
				f[1] += q[0]*in_gamma*Leonard_ene[1];
				f[2] += q[0]*in_gamma*Leonard_ene[2];
			}
		}
	}
}

template<int in_n_fields, int in_n_dims>
__device__ __host__ void rusanov_flux(double* q_l, double *q_r, double *norm, double *fn, double in_gamma)
{
  double vn_l, vn_r;
  double vn_av_mag, c_av;
  double p_l, p_r,f_l,f_r;
  
  double f[in_n_dims];

  // Compute normal velocity
  vn_l = 0.;
  vn_r = 0.;
  #pragma unroll
  for (int i=0;i<in_n_dims;i++) {
    vn_l += q_l[i+1]/q_l[0]*norm[i];
    vn_r += q_r[i+1]/q_r[0]*norm[i];
  }

  // Flux prep
  inv_NS_flux<in_n_dims>(q_l,&p_l,f,in_gamma,-1);
  inv_NS_flux<in_n_dims>(q_r,&p_r,f,in_gamma,-1);
	  		
	vn_av_mag=sqrt(0.25*(vn_l+vn_r)*(vn_l+vn_r));
	c_av=sqrt((in_gamma*(p_l+p_r))/(q_l[0]+q_r[0]));

  #pragma unroll
  for (int i=0;i<in_n_fields;i++)
  {
    // Left normal flux
    inv_NS_flux<in_n_dims>(q_l,&p_l,f,in_gamma,i);
    
    f_l = f[0]*norm[0] + f[1]*norm[1];
    if(in_n_dims==3)
      f_l += f[2]*norm[2];
      
    // Right normal flux
    inv_NS_flux<in_n_dims>(q_r,&p_r,f,in_gamma,i);
    
    f_r = f[0]*norm[0] + f[1]*norm[1];
    if(in_n_dims==3)
      f_r += f[2]*norm[2];
    
    // Common normal flux
    fn[i] = 0.5*(f_l+f_r) - 0.5*(vn_av_mag+c_av)*(q_r[i]-q_l[i]);
  }
}


template<int in_n_fields, int in_n_dims>
__device__ __host__ void right_flux(double *q_r, double *norm, double *fn, double in_gamma)
{

  double p_r,f_r;
  double f[in_n_dims];

  // Flux prep
  inv_NS_flux<in_n_dims>(q_r,&p_r,f,in_gamma,-1);

  #pragma unroll
  for (int i=0;i<in_n_fields;i++)
  {
    //Right normal flux
    inv_NS_flux<in_n_dims>(q_r,&p_r,f,in_gamma,i);
    
    f_r = f[0]*norm[0] + f[1]*norm[1];
    if(in_n_dims==3)
      f_r += f[2]*norm[2];
    
    fn[i] = f_r;
  }
}


template<int n_fields, int n_dims>
__device__ __host__ void roe_flux(double* u_l, double *u_r, double *norm, double *fn, double in_gamma)
{
	double p_l,p_r;
  double h_l, h_r;
  double sq_rho,rrho,hm,usq,am,am_sq,unm;
  double lambda0,lambdaP,lambdaM;
  double rhoun_l, rhoun_r,eps;
  double a1,a2,a3,a4,a5,a6,aL1,bL1;
  double v_l[n_dims],v_r[n_dims],um[n_dims],du[n_fields];
  //array<double> um(n_dims);

  // velocities
  #pragma unroll
  for (int i=0;i<n_dims;i++)  {
    v_l[i] = u_l[i+1]/u_l[0];
    v_r[i] = u_r[i+1]/u_r[0];
  }

  if (n_dims==2) {
	  p_l=(in_gamma-1.0)*(u_l[3]-(0.5*u_l[0]*((v_l[0]*v_l[0])+(v_l[1]*v_l[1]))));
	  p_r=(in_gamma-1.0)*(u_r[3]-(0.5*u_r[0]*((v_r[0]*v_r[0])+(v_r[1]*v_r[1]))));
  }
  else
    printf("Roe not implemented in 3D\n");

  h_l = (u_l[n_dims+1]+p_l)/u_l[0];
  h_r = (u_r[n_dims+1]+p_r)/u_r[0];

  sq_rho = sqrt(u_r[0]/u_l[0]);

  rrho = 1./(sq_rho+1.);

  #pragma unroll
  for (int i=0;i<n_dims;i++)
    um[i] = rrho*(v_l[i]+sq_rho*v_r[i]);

  hm      = rrho*(h_l     +sq_rho*h_r);

  //if (flag)
  //  printf("hm = %16.12f, um=%16.12f %16.12f %16.12f\n",hm,um[0],um[1],um[2]);

  usq=0.;
  #pragma unroll
  for (int i=0;i<n_dims;i++)
    usq += 0.5*um[i]*um[i];

  am_sq   = (in_gamma-1.)*(hm-usq);
  am  = sqrt(am_sq);

  unm = 0.;
  #pragma unroll
  for (int i=0;i<n_dims;i++)
    unm += um[i]*norm[i];

  //if (flag)
  //  printf("unm=%16.12f, usq=%16.12f\n",unm,usq);

  // Compute Euler flux (first part)
  rhoun_l = 0.;
  rhoun_r = 0.;

  #pragma unroll
  for (int i=0;i<n_dims;i++)
  {
    rhoun_l += u_l[i+1]*norm[i];
    rhoun_r += u_r[i+1]*norm[i];
  }

  if (n_dims==2)
  {
    fn[0] = rhoun_l + rhoun_r;
    fn[1] = rhoun_l*v_l[0] + rhoun_r*v_r[0] + (p_l+p_r)*norm[0];
    fn[2] = rhoun_l*v_l[1] + rhoun_r*v_r[1] + (p_l+p_r)*norm[1];
    fn[3] = rhoun_l*h_l   +rhoun_r*h_r;

    //if (flag)
    //  printf("fn=%16.12f %16.12f %16.12f %16.12f\n",fn[0],fn[1],fn[2],fn[3]);
  }
  else
    printf("Roe not implemented in 3D\n");

  #pragma unroll
  for (int i=0;i<n_fields;i++)
  {
    du[i] = u_r[i]-u_l[i];
    //if (flag)
    //  printf("du=%16.12f\n",du[i]);
  }

  lambda0 = abs(unm);
  lambdaP = abs(unm+am);
  lambdaM = abs(unm-am);

  // Entropy fix
  eps = 0.5*(abs(rhoun_l/u_l[0]-rhoun_r/u_r[0])+ abs(sqrt(in_gamma*p_l/u_l[0])-sqrt(in_gamma*p_r/u_r[0])));
  if(lambda0 < 2.*eps) 
    lambda0 = 0.25*lambda0*lambda0/eps + eps;
  if(lambdaP < 2.*eps) 
    lambdaP = 0.25*lambdaP*lambdaP/eps + eps;
  if(lambdaM < 2.*eps) 
    lambdaM = 0.25*lambdaM*lambdaM/eps + eps;


  a2 = 0.5*(lambdaP+lambdaM)-lambda0;
  a3 = 0.5*(lambdaP-lambdaM)/am;
  a1 = a2*(in_gamma-1.)/am_sq;
  a4 = a3*(in_gamma-1.);

    //if (flag)
    //  printf("ndims=%d\n",n_dims);

  if (n_dims==2)
  {
    //if (flag)
    //  printf("inside");

    //if (flag)
    //  printf("%16.12f %16.12f %16.12f %16.12f %16.12f %16.12f %16.12f %16.12f\n",usq,du[0],um[0],du[1],um[1],du[2],du[3]);


    a5 = usq*du[0]-um[0]*du[1]-um[1]*du[2]+du[3];
    a6 = unm*du[0]-norm[0]*du[1]-norm[1]*du[2];
  }
  else if (n_dims==3)
  {
    a5 = usq*du[0]-um[0]*du[1]-um[1]*du[2]-um[2]*du[3]+du[4];
    a6 = unm*du[0]-norm[0]*du[1]-norm[1]*du[2]-norm[2]*du[3];
  }

  //if (flag)
  // printf("a=%16.12f %16.12f %16.12f %16.12f %16.12f %16.12f \n",a2,a3,a1,a4,a5,a6);

  aL1 = a1*a5 - a3*a6;
  bL1 = a4*a5 - a2*a6;

  //if (flag)
  // printf("aL1=%16.12f %16.12f \n",aL1,bL1);

  // Compute Euler flux (second part)
  if (n_dims==2)
  {
    fn[0] = fn[0] - (lambda0*du[0]+aL1);
    fn[1] = fn[1] - (lambda0*du[1]+aL1*um[0]+bL1*norm[0]);
    fn[2] = fn[2] - (lambda0*du[2]+aL1*um[1]+bL1*norm[1]);
    fn[3] = fn[3] - (lambda0*du[3]+aL1*hm   +bL1*unm);

    //if (flag)
    //  printf("fn=%16.12f %16.12f %16.12f %16.12f\n",fn[0],fn[1],fn[2],fn[3]);
  }
  else if (n_dims==3)
  {
    fn[0] = fn[0] - (lambda0*du[0]+aL1);
    fn[1] = fn[1] - (lambda0*du[1]+aL1*um[0]+bL1*norm[0]);
    fn[2] = fn[2] - (lambda0*du[2]+aL1*um[1]+bL1*norm[1]);
    fn[3] = fn[3] - (lambda0*du[3]+aL1*um[2]+bL1*norm[2]);
    fn[4] = fn[4] - (lambda0*du[4]+aL1*hm   +bL1*unm);
  }

  #pragma unroll
  for (int i=0;i<n_fields;i++)
    fn[i] =  0.5*fn[i];

}


template<int n_dims>
__device__ __host__ void lax_friedrichs_flux(double* u_l, double *u_r, double *norm, double *fn, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
		double u_av, u_diff;
		double norm_speed;
    
		u_av = 0.5*(u_r[0]+u_l[0]);
    u_diff = u_l[0]-u_r[0]; 

    norm_speed=0.;
    if (n_dims==2)
      norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1];
    else if (n_dims==3)
      norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2];
	  		
    // Compute common interface flux
    fn[0] = 0.;
    if (n_dims==2)
      fn[0] += (wave_speed_x*norm[0] + wave_speed_y*norm[1])*u_av;
    else if (n_dims==3)
      fn[0] += (wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2])*u_av;
    fn[0] += 0.5*lambda*abs(norm_speed)*u_diff;
}


template<int in_n_dims, int n_fields, int flux_spec>
__device__ void ldg_solution(double* q_l, double* q_r, double* norm, double* q_c, double in_pen_fact)
{
  if(flux_spec==0) // Interior, mpi
  {
    // Choosing a unique direction for the switch

		if(in_n_dims==2)
		{
    	if ((norm[0]+norm[1]) < 0.)
      	in_pen_fact=-in_pen_fact;
		}
		if(in_n_dims==3)
		{
    	if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
      	in_pen_fact=-in_pen_fact;
		}

    #pragma unroll
    for (int i=0;i<n_fields;i++) 
      q_c[i] = 0.5*(q_l[i]+q_r[i]) - in_pen_fact*(q_l[i]-q_r[i]);
  }
  else if(flux_spec==1) // Dirichlet
  {
    #pragma unroll
    for (int i=0;i<n_fields;i++) 
      q_c[i] = q_r[i];
  }
  else if(flux_spec==2) // von Neumann
  {
    #pragma unroll
    for (int i=0;i<n_fields;i++) 
      q_c[i] = q_l[i];
  }
}


template<int in_n_dims, int in_flux_spec>
__device__ __host__ void ldg_flux(double q_l, double q_r, double* f_l, double* f_r, double* f_c, double* norm, double in_pen_fact, double in_tau)
{
  if(in_flux_spec==0) //Interior, mpi
  {
		if(in_n_dims==2)
		{
    	if ((norm[0]+norm[1]) < 0.)
      	in_pen_fact=-in_pen_fact;
		}
		if(in_n_dims==3)
		{
    	if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
      	in_pen_fact=-in_pen_fact;
		}

    // Compute common interface flux
    #pragma unroll
    for (int i=0;i<in_n_dims;i++)
    {
      f_c[i] = 0.5*(f_l[i] + f_r[i]) + in_tau*norm[i]*(q_l - q_r);
      #pragma unroll
      for (int k=0;k<in_n_dims;k++)
        f_c[i] += in_pen_fact*norm[i]*norm[k]*(f_l[k] - f_r[k]);
    }
  }
  else if(in_flux_spec==1) // Dirichlet
  {
    #pragma unroll
    for (int i=0;i<in_n_dims;i++)
      f_c[i] = f_l[i] + in_tau*norm[i]*(q_l - q_r);
  }
  else if(in_flux_spec==2) // von Neumann
  {
    #pragma unroll
    for (int i=0;i<in_n_dims;i++)
      f_c[i] = f_r[i];
  }
}


template< int n_fields >
__global__ void RK11_update_kernel(double *g_q_qpts, double *g_div_tfg_con_qpts, double *g_jac_det_qpts, 
								const int n_cells, const int n_qpts, const double dt, const double const_src_term)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;
	const int m = n; 
	double jac;
  int stride = n_cells*n_qpts;
	 
	if (n<n_cells*n_qpts)
	{
		jac = g_jac_det_qpts[m];
		// Update 5 fields
    #pragma unroll
		for (int i=0;i<n_fields;i++)
		{
	  	g_q_qpts[n] -= dt*(g_div_tfg_con_qpts[n]/jac - const_src_term);
			n += stride;
		}
	}
}


template< int n_fields >
__global__ void RK45_update_kernel(double *g_q_qpts, double *g_div_tfg_con_qpts, double *g_res_qpts, double *g_jac_det_qpts, 
								const int n_cells, const int n_qpts, const double fa, const double fb, const double dt, const double const_src_term)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;
	const int m = n; 
	double rhs,res,jac;
  int stride = n_cells*n_qpts;
	 
	if (n<n_cells*n_qpts)
	{
		jac = g_jac_det_qpts[m];
		// Update 5 fields
    #pragma unroll
		for (int i=0;i<n_fields;i++)
		{
			rhs = -(g_div_tfg_con_qpts[n]/jac - const_src_term);
	  	res = g_res_qpts[n];
	  	res = fa*res + dt*rhs;
	 		g_res_qpts[n] = res;
	  	g_q_qpts[n] += fb*res;
			n += stride;
		}
	}
}


// gpu kernel to calculate transformed discontinuous inviscid flux at solution points for the wave equation
// otherwise, switch to one thread per output?
template<int in_n_dims>
__global__ void calc_tdisinvf_upts_AD_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, double wave_speed_x, double wave_speed_y, double wave_speed_z)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double q;
  double f[in_n_dims];	
  double met[in_n_dims][in_n_dims];

	int stride = in_n_upts_per_ele*in_n_eles;

 	if(thread_id<(in_n_upts_per_ele*in_n_eles))
 	{
    q = in_disu_upts_ptr[thread_id];

    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
      #pragma unroll
      for (int j=0;j<in_n_dims;j++) 
        met[j][i] = in_inv_detjac_mul_jac_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

    int index;

    if (in_n_dims==2)
    {
		  f[0] = wave_speed_x*q;
		  f[1] = wave_speed_y*q;

		  index = thread_id;
		  out_tdisf_upts_ptr[index       ] = met[0][0]*f[0] + met[0][1]*f[1]; 
		  out_tdisf_upts_ptr[index+stride] = met[1][0]*f[0] + met[1][1]*f[1];
    }
    else if (in_n_dims==3)
    {
			f[0] = wave_speed_x*q;
			f[1] = wave_speed_y*q;
			f[2] = wave_speed_z*q;

		  index = thread_id;
			out_tdisf_upts_ptr[index          ] = met[0][0]*f[0] + met[0][1]*f[1] + met[0][2]*f[2]; 
			out_tdisf_upts_ptr[index+  stride ] = met[1][0]*f[0] + met[1][1]*f[1] + met[1][2]*f[2]; 
			out_tdisf_upts_ptr[index+2*stride ] = met[2][0]*f[0] + met[2][1]*f[1] + met[2][2]*f[2]; 

    }
  }
}
/*! gpu kernel to calculate velocity and energy product terms for similarity model */
template<int in_n_fields>
__global__ void calc_similarity_terms_kernel(int in_n_upts_per_ele, int in_n_eles, int in_n_dims, double* in_disu_upts_ptr, double* in_uu_ptr, double* in_ue_ptr)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

	int stride = in_n_upts_per_ele*in_n_eles;
	int i;
  double q[in_n_fields];
	double rsq;

 	if(thread_id<in_n_upts_per_ele*in_n_eles) {

    // Solution
    #pragma unroll
    for (i=0;i<in_n_fields;i++) {
      q[i] = in_disu_upts_ptr[thread_id + i*stride];
		}

		rsq = q[0]*q[0];

		if(in_n_dims==2) {
			/*! velocity-velocity product */
			in_uu_ptr[thread_id + 0*stride] = q[1]*q[1]/rsq;
			in_uu_ptr[thread_id + 1*stride] = q[2]*q[2]/rsq;
			in_uu_ptr[thread_id + 2*stride] = q[1]*q[2]/rsq;

			/*! velocity-energy product */
			q[3] -= 0.5*(q[1]*q[1] + q[2]*q[2])/q[0]; // internal energy*rho

			in_ue_ptr[thread_id + 0*stride] = q[1]*q[3]/rsq; // subtract kinetic energy
			in_ue_ptr[thread_id + 1*stride] = q[2]*q[3]/rsq;
		}
		else if(in_n_dims==3) {
			/*! velocity-velocity product */
			in_uu_ptr[thread_id + 0*stride] = q[1]*q[1]/rsq;
			in_uu_ptr[thread_id + 1*stride] = q[2]*q[2]/rsq;
			in_uu_ptr[thread_id + 2*stride] = q[3]*q[3]/rsq;
			in_uu_ptr[thread_id + 3*stride] = q[1]*q[2]/rsq;
			in_uu_ptr[thread_id + 4*stride] = q[1]*q[3]/rsq;
			in_uu_ptr[thread_id + 5*stride] = q[2]*q[3]/rsq;

			/*! velocity-energy product */
			q[4] -= 0.5*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])/q[0]; // internal energy*rho

			in_ue_ptr[thread_id + 0*stride] = q[1]*q[4]/rsq; // subtract kinetic energy
			in_ue_ptr[thread_id + 1*stride] = q[2]*q[4]/rsq;
			in_ue_ptr[thread_id + 2*stride] = q[3]*q[4]/rsq;
		}
	}
}

/*! gpu kernel to calculate Leonard tensors for similarity model */
template<int in_n_fields>
__global__ void calc_similarity_model_kernel(int in_n_upts_per_ele, int in_n_eles, int in_n_dims, double* in_disuf_upts_ptr, double* in_Lu_ptr, double* in_Le_ptr)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

	int stride = in_n_upts_per_ele*in_n_eles;
	int i;
  double q[in_n_fields];
	double diag, rsq;

 	if(thread_id<in_n_upts_per_ele*in_n_eles) {
	  // filtered solution
  	#pragma unroll
  	for (i=0;i<in_n_fields;i++) {
  	  q[i] = in_disuf_upts_ptr[thread_id + i*stride];
		}

		rsq = q[0]*q[0];

		/*! subtract product of filtered solution terms from Leonard tensors */
		if(in_n_dims==2) {
			in_Lu_ptr[thread_id + 0*stride] = (in_Lu_ptr[thread_id + 0*stride] - q[1]*q[1])/rsq;
			in_Lu_ptr[thread_id + 1*stride] = (in_Lu_ptr[thread_id + 1*stride] - q[2]*q[2])/rsq;
			in_Lu_ptr[thread_id + 2*stride] = (in_Lu_ptr[thread_id + 2*stride] - q[1]*q[2])/rsq;

			diag = (in_Lu_ptr[thread_id + 0*stride] + in_Lu_ptr[thread_id + 1*stride])/3.0;

			q[3] -= 0.5*(q[1]*q[1] + q[2]*q[2])/q[0]; // internal energy*rho

			in_Le_ptr[thread_id + 0*stride] = (in_Le_ptr[thread_id + 0*stride] - q[1]*q[3])/rsq;
			in_Le_ptr[thread_id + 1*stride] = (in_Le_ptr[thread_id + 1*stride] - q[2]*q[3])/rsq;
		}
		else if(in_n_dims==3) {
			in_Lu_ptr[thread_id + 0*stride] = (in_Lu_ptr[thread_id + 0*stride] - q[1]*q[1])/rsq;
  	  in_Lu_ptr[thread_id + 1*stride] = (in_Lu_ptr[thread_id + 1*stride] - q[2]*q[2])/rsq;
  	  in_Lu_ptr[thread_id + 2*stride] = (in_Lu_ptr[thread_id + 2*stride] - q[3]*q[3])/rsq;
			in_Lu_ptr[thread_id + 3*stride] = (in_Lu_ptr[thread_id + 3*stride] - q[1]*q[2])/rsq;
  	  in_Lu_ptr[thread_id + 4*stride] = (in_Lu_ptr[thread_id + 4*stride] - q[1]*q[3])/rsq;
  	  in_Lu_ptr[thread_id + 5*stride] = (in_Lu_ptr[thread_id + 5*stride] - q[2]*q[3])/rsq;

			diag = (in_Lu_ptr[thread_id + 0*stride] + in_Lu_ptr[thread_id + 1*stride] + in_Lu_ptr[thread_id + 2*stride])/3.0;

			q[4] -= 0.5*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])/q[0]; // internal energy*rho

  	  in_Le_ptr[thread_id + 0*stride] = (in_Le_ptr[thread_id + 0*stride] - q[1]*q[4])/rsq;
  	  in_Le_ptr[thread_id + 1*stride] = (in_Le_ptr[thread_id + 1*stride] - q[2]*q[4])/rsq;
  	  in_Le_ptr[thread_id + 2*stride] = (in_Le_ptr[thread_id + 2*stride] - q[3]*q[4])/rsq;
		}

		/*! subtract diagonal from Lu */
  	#pragma unroll
		for (i=0;i<in_n_dims;++i) {
			in_Lu_ptr[thread_id + i*stride] -= diag;
		}
		// subtract diagonal from Le?
	}
}

/*! wrapper for gpu kernel to calculate terms for similarity model */
void calc_similarity_model_kernel_wrapper(int in_n_fields, int in_n_upts_per_ele, int in_n_eles, int in_n_dims, double* in_disu_upts_ptr, double* in_disuf_upts_ptr, double* in_uu_ptr, double* in_ue_ptr, double* in_Lu_ptr, double* in_Le_ptr, double* in_filter_upts_ptr, int flag)
{
  check_cuda_error("Before", __FILE__, __LINE__);

	// HACK: fix 256 threads per block
	int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

	/*! Calculate product terms uu, ue */
	if (flag==0) {
		// fixed n_fields at 4 for 2d and 5 for 3d
		if(in_n_dims==2) {
			calc_similarity_terms_kernel <4> <<< n_blocks,256>>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disu_upts_ptr, in_uu_ptr, in_ue_ptr);
		}
		else if(in_n_dims==3) {
			calc_similarity_terms_kernel <5> <<< n_blocks,256>>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disu_upts_ptr, in_uu_ptr, in_ue_ptr);
		}
	}
	/*! Calculate Leonard tensors Lu, Le */
	else if (flag==1) {
		// fixed n_fields at 4 for 2d and 5 for 3d
		if(in_n_dims==2) {
			calc_similarity_model_kernel <4> <<< n_blocks,256>>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disuf_upts_ptr, in_Lu_ptr, in_Le_ptr);
		}
		else if(in_n_dims==3) {
			calc_similarity_model_kernel <5> <<< n_blocks,256>>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disuf_upts_ptr, in_Lu_ptr, in_Le_ptr);
		}
	}

  check_cuda_error("After",__FILE__, __LINE__);
}

// gpu kernel to calculate transformed discontinuous inviscid flux at solution points for the Navier-Stokes equation
// otherwise, switch to one thread per output?
template<int in_n_dims, int in_n_fields>
__global__ void calc_tdisinvf_upts_NS_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, double in_gamma)
{

	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double q[in_n_fields];
  double f[in_n_dims];	
  double met[in_n_dims][in_n_dims];

	double p;
	int stride = in_n_upts_per_ele*in_n_eles;

 	if(thread_id<(in_n_upts_per_ele*in_n_eles))
 	{
    // Solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
      q[i] = in_disu_upts_ptr[thread_id + i*stride];

    // Metric terms
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
      #pragma unroll
      for (int j=0;j<in_n_dims;j++) 
        met[j][i] = in_inv_detjac_mul_jac_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

    // Flux prep
    inv_NS_flux<in_n_dims>(q,&p,f,in_gamma,-1);

    int index;

    // Flux computation
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      inv_NS_flux<in_n_dims>(q,&p,f,in_gamma,i);
      
      index = thread_id+i*stride;
    
      if (in_n_dims==2) {
		    out_tdisf_upts_ptr[index                    ] = met[0][0]*f[0] + met[0][1]*f[1]; 
		    out_tdisf_upts_ptr[index+stride*in_n_fields ] = met[1][0]*f[0] + met[1][1]*f[1];
      }
      else if(in_n_dims==3)
      {
			  out_tdisf_upts_ptr[index                      ] = met[0][0]*f[0] + met[0][1]*f[1] + met[0][2]*f[2]; 
			  out_tdisf_upts_ptr[index+  stride*in_n_fields ] = met[1][0]*f[0] + met[1][1]*f[1] + met[1][2]*f[2]; 
			  out_tdisf_upts_ptr[index+2*stride*in_n_fields ] = met[2][0]*f[0] + met[2][1]*f[1] + met[2][2]*f[2]; 
      }
    }
  
  }
}


// gpu kernel to calculate normal transformed continuous inviscid flux at the flux points
template <int in_n_dims, int in_n_fields, int in_riemann_solve_type, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconinvf_fpts_NS_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double** in_delta_disu_fpts_r_ptr, double in_gamma, double in_pen_fact, int in_viscous)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l[in_n_fields]; 
  double q_r[in_n_fields]; 
  double fn[in_n_fields];
  double norm[in_n_dims];
  
  double q_c[in_n_fields];

  double jac;

	if(thread_id<stride)
  {  
    // Compute left state solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride])); 

    // Compute right state solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_r[i]=(*(in_disu_fpts_r_ptr[thread_id+i*stride])); 

    // Compute normal
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);

    if (in_riemann_solve_type==0)
      rusanov_flux<in_n_fields,in_n_dims> (q_l,q_r,norm,fn,in_gamma);
    else if (in_riemann_solve_type==2)
      roe_flux<in_n_fields,in_n_dims> (q_l,q_r,norm,fn,in_gamma);
    
    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))=jac*fn[i];

    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_r_ptr[thread_id+i*stride]))=-jac*fn[i];

    // Viscous solution correction
    if(in_viscous)
    {
      if(in_vis_riemann_solve_type==0)
        ldg_solution<in_n_dims,in_n_fields,0> (q_l,q_r,norm,q_c,in_pen_fact);

      #pragma unroll
      for (int i=0;i<in_n_fields;i++) 
        (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);   

      #pragma unroll
      for (int i=0;i<in_n_fields;i++) 
        (*(in_delta_disu_fpts_r_ptr[thread_id+i*stride])) = (q_c[i]-q_r[i]);
    }

  }
}


template <int in_n_dims, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconinvf_fpts_lax_friedrich_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double** in_delta_disu_fpts_r_ptr, double in_pen_fact, int in_viscous, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l; 
  double q_r; 
  double fn,u_av,u_diff;
  double norm_speed;
  double norm[in_n_dims];
  
  double q_c;
  double jac;

	if(thread_id<stride)
  {  
    // Compute left state solution
  	q_l=(*(in_disu_fpts_l_ptr[thread_id])); 

    // Compute right state solution
  	q_r=(*(in_disu_fpts_r_ptr[thread_id])); 
 
    // Compute normal
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);

    u_av = 0.5*(q_r+q_l);
    u_diff = q_l-q_r; 

    norm_speed=0.;
    if (in_n_dims==2)
      norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1];
    else if (in_n_dims==3)
      norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2];
	  		
    // Compute common interface flux
    fn = 0.;
    if (in_n_dims==2)
      fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1])*u_av;
    else if (in_n_dims==3)
      fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2])*u_av;
    fn += 0.5*lambda*abs(norm_speed)*u_diff;

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
	  (*(in_norm_tconf_fpts_l_ptr[thread_id]))=jac*fn;

    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr[thread_id]));     
	  (*(in_norm_tconf_fpts_r_ptr[thread_id]))=-jac*fn;

    // viscous solution correction
    if(in_viscous)
    {
      //if(in_vis_riemann_solve_type==0)
      //  ldg_solution<in_n_dims,1,0> (&q_l,&q_r,norm,&q_c,in_pen_fact);

			if(in_n_dims==2)
			{
    		if ((norm[0]+norm[1]) < 0.)
    	  	in_pen_fact=-in_pen_fact;
			}
			if(in_n_dims==3)
			{
    		if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
    	  	in_pen_fact=-in_pen_fact;
			}

      q_c = 0.5*(q_l+q_r) - in_pen_fact*(q_l-q_r);

      //printf("%4.2f \n", q_c);

      (*(in_delta_disu_fpts_l_ptr[thread_id])) = (q_c-q_l);   
     
      (*(in_delta_disu_fpts_r_ptr[thread_id])) = (q_c-q_r);
    }
  }

}


// kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries
template<int in_n_dims, int in_n_fields, int in_riemann_solve_type, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconinvf_fpts_boundary_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_loc_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_R_ref, int in_viscous, double in_time_bound, double in_wave_speed_x, double in_wave_speed_y, double in_wave_speed_z, double in_lambda, int in_equation)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;

  int bdy_spec;
 
  double q_l[in_n_fields]; 
  double q_r[in_n_fields]; 
  double fn[in_n_fields];
  double norm[in_n_dims];
  double loc[in_n_dims];
  double q_c[in_n_fields];

  double jac;

	if(thread_id<stride)
  {  
    // Compute left solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride])); 

    // Compute normal
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);

    // Compute location
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  loc[i]=*(in_loc_fpts_ptr[thread_id + i*stride]);

    // Set boundary condition
    bdy_spec = in_boundary_type[thread_id/in_n_fpts_per_inter];
    set_inv_boundary_conditions_kernel<in_n_dims,in_n_fields>(bdy_spec,q_l,q_r,norm,loc,in_bdy_params,in_gamma, in_R_ref, in_time_bound, in_equation);

    if (bdy_spec==16) // Dual consistent
    {
    //  right_flux<in_n_fields,in_n_dims> (q_r,norm,fn,in_gamma);
        roe_flux<in_n_fields,in_n_dims> (q_l,q_r,norm,fn,in_gamma);
    }
    else
    {
      if (in_riemann_solve_type==0)
        rusanov_flux<in_n_fields,in_n_dims> (q_l,q_r,norm,fn,in_gamma);
			else if (in_riemann_solve_type==1)
				lax_friedrichs_flux<in_n_dims> (q_l,q_r,norm,fn,in_wave_speed_x,in_wave_speed_y,in_wave_speed_z,in_lambda);
      else if (in_riemann_solve_type==2)
        roe_flux<in_n_fields,in_n_dims> (q_l,q_r,norm,fn,in_gamma);
    }

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))=jac*fn[i];

    // Viscous solution correction
    if(in_viscous)
    {
      if(bdy_spec == 12 || bdy_spec == 14) // Adiabatic
      {
        if (in_vis_riemann_solve_type==0)
          ldg_solution<in_n_dims,in_n_fields,2> (q_l,q_r,norm,q_c,0);
      }
      else        
      {
        if (in_vis_riemann_solve_type==0)
          ldg_solution<in_n_dims,in_n_fields,1> (q_l,q_r,norm,q_c,0);
      }
      
      #pragma unroll
      for (int i=0;i<in_n_fields;i++) 
        (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);
    }

  }
}


// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int in_n_dims, int in_n_fields, int in_n_comp>
__global__ void calc_tdisvisf_upts_NS_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, int in_ele_type, double in_filter_ratio, int LES, int SGS_model, double* Leonard_mom, double* Leonard_ene, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double q[in_n_fields];
  double f[in_n_dims];
  double sgsf[in_n_dims];	// subgrid-scale flux set by SGS model
  double met[in_n_dims][in_n_dims];
  double stensor[in_n_comp]; // viscous stress tensor
  double straintensor[in_n_comp]; // for SGS models
  double sdtensor[in_n_comp]; // for WALE SGS model
	double lmtensor[in_n_comp]; // local Leonard tensor for momentum
	double letensor[in_n_dims]; // local Leonard tensor for energy
  double grad_ene[in_n_dims];
  double grad_vel[in_n_dims*in_n_dims];
  double grad_q[in_n_fields*in_n_dims];  
  double inte, mu, jac, delta;

	int i, j, ind;
	int stride = in_n_upts_per_ele*in_n_eles;

 	if(thread_id<(in_n_upts_per_ele*in_n_eles))
 	{
    // Physical solution
    #pragma unroll
    for (i=0;i<in_n_fields;i++) {
      q[i] = in_disu_upts_ptr[thread_id + i*stride];
		}

    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      #pragma unroll
      for (j=0;j<in_n_dims;j++) {
        met[j][i] = in_inv_detjac_mul_jac_upts_ptr[thread_id + (i*in_n_dims+j)*stride];
    	}
		}

    // Physical gradient
    #pragma unroll
    for (i=0;i<in_n_fields;i++)
    {
      ind = thread_id + i*stride;
      grad_q[i*in_n_dims + 0] = in_grad_disu_upts_ptr[ind];
      grad_q[i*in_n_dims + 1] = in_grad_disu_upts_ptr[ind + stride*in_n_fields];
      
      if(in_n_dims==3)
        grad_q[i*in_n_dims + 2] = in_grad_disu_upts_ptr[ind + 2*stride*in_n_fields];
    }

    // Flux prep
    vis_NS_flux<in_n_dims>(q, grad_q, grad_vel, grad_ene, stensor, f, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

		int index;

    // Flux computation
    #pragma unroll
    for (i=0;i<in_n_fields;i++) {

      vis_NS_flux<in_n_dims>(q, grad_q, grad_vel, grad_ene, stensor, f, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);

			// If LES, calculate SGS stress tensor
			if(LES) {
				// Calculate strain rate tensor from viscous stress tensor
				#pragma unroll
				for (j=0;j<in_n_comp;j++)
					straintensor[j] = stensor[j]/2.0/mu;

				// Calculate filter width
				jac = in_detjac_upts_ptr[thread_id];

				delta = SGS_filter_width(jac, in_ele_type, in_n_dims, in_filter_ratio);

				// Local Leonard tensors
				#pragma unroll
				for (j=0;j<in_n_dims;j++) {
					// energy
					letensor[j] = Leonard_ene[thread_id + j*stride];
				}

				#pragma unroll
				for (j=0;j<in_n_comp;j++) {
					// momentum
					lmtensor[j] = Leonard_mom[thread_id + j*stride];
				}

				SGS_flux_kernel<in_n_dims>(q, grad_q, grad_vel, grad_ene, sdtensor, straintensor, lmtensor, letensor, sgsf, SGS_model, delta, in_gamma, i);
				//printf("Leonard2 = %10.10f, %10.10f, %10.10f\n", lmtensor[0], lmtensor[1], lmtensor[2]);
				//printf("sim flux: %10.10f\n", sgsf);

				// Add SGS flux to viscous flux
				#pragma unroll
				for (j=0;j<in_n_dims;j++) {
					f[j] += sgsf[j];
				}
			}

      index = thread_id+i*stride;
      
      if(in_n_dims==2) {
		    out_tdisf_upts_ptr[index                   ] += met[0][0]*f[0] + met[0][1]*f[1]; 
		    out_tdisf_upts_ptr[index+stride*in_n_fields] += met[1][0]*f[0] + met[1][1]*f[1]; 
      }
      else if(in_n_dims==3) {
			  out_tdisf_upts_ptr[index                     ] += met[0][0]*f[0] + met[0][1]*f[1] + met[0][2]*f[2]; 
			  out_tdisf_upts_ptr[index+  stride*in_n_fields] += met[1][0]*f[0] + met[1][1]*f[1] + met[1][2]*f[2]; 
			  out_tdisf_upts_ptr[index+2*stride*in_n_fields] += met[2][0]*f[0] + met[2][1]*f[1] + met[2][2]*f[2]; 
      }
    }

  }
}


// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int in_n_dims>
__global__ void calc_tdisvisf_upts_AD_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, double diff_coeff)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double f[in_n_dims];	
  double met[in_n_dims][in_n_dims];
  double grad_q[in_n_dims];

  int ind;
  int index;
	int stride = in_n_upts_per_ele*in_n_eles;

 	if(thread_id<(in_n_upts_per_ele*in_n_eles))
 	{
    // Metric terms
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
      #pragma unroll
      for (int j=0;j<in_n_dims;j++) 
        met[j][i] = in_inv_detjac_mul_jac_upts_ptr[thread_id + (i*in_n_dims+j)*stride];
    
    // Physical gradient
    ind = thread_id;
    grad_q[0] = in_grad_disu_upts_ptr[ind];
    grad_q[1] = in_grad_disu_upts_ptr[ind + stride];
      
    if(in_n_dims==3)
      grad_q[2] = in_grad_disu_upts_ptr[ind + 2*stride];


    // Flux computation
    f[0] = -diff_coeff*grad_q[0];
    f[1] = -diff_coeff*grad_q[1];
    
    if(in_n_dims==3)
      f[2] = -diff_coeff*grad_q[2];    

    index = thread_id;
      
    if(in_n_dims==2) {
      out_tdisf_upts_ptr[index       ] += met[0][0]*f[0] + met[0][1]*f[1]; 
      out_tdisf_upts_ptr[index+stride] += met[1][0]*f[0] + met[1][1]*f[1]; 
    }
    else if(in_n_dims==3) {
      out_tdisf_upts_ptr[index         ] += met[0][0]*f[0] + met[0][1]*f[1] + met[0][2]*f[2]; 
      out_tdisf_upts_ptr[index+  stride] += met[1][0]*f[0] + met[1][1]*f[1] + met[1][2]*f[2]; 
      out_tdisf_upts_ptr[index+2*stride] += met[2][0]*f[0] + met[2][1]*f[1] + met[2][2]*f[2]; 
    }

  }
}

// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int in_n_dims, int in_n_fields>
__global__ void transform_grad_disu_upts_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double dq[in_n_dims];
  double met[in_n_dims][in_n_dims];

  double jac;
  int ind;

	int stride = in_n_upts_per_ele*in_n_eles;

 	if(thread_id<(in_n_upts_per_ele*in_n_eles))
 	{
    // Obtain metric terms
    jac = in_detjac_upts_ptr[thread_id];

    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
      #pragma unroll
      for (int j=0;j<in_n_dims;j++) 
        met[j][i] = in_inv_detjac_mul_jac_upts_ptr[thread_id + (i*in_n_dims+j)*stride];
    
    // Compute physical gradient
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      ind = thread_id + i*stride;
      dq[0] = in_grad_disu_upts_ptr[ind];
      dq[1] = in_grad_disu_upts_ptr[ind + stride*in_n_fields];
    
      if(in_n_dims==2)
      {
        in_grad_disu_upts_ptr[ind                   ] = (1./jac)*(dq[0]*met[0][0] + dq[1]*met[1][0]);
        in_grad_disu_upts_ptr[ind+stride*in_n_fields] = (1./jac)*(dq[0]*met[0][1] + dq[1]*met[1][1]);
      }
      if(in_n_dims==3)
      {
        dq[2] = in_grad_disu_upts_ptr[ind + 2*stride*in_n_fields];
        
        in_grad_disu_upts_ptr[ind                     ] = (1./jac)*(dq[0]*met[0][0] + dq[1]*met[1][0] + dq[2]*met[2][0]);
        in_grad_disu_upts_ptr[ind+stride*in_n_fields  ] = (1./jac)*(dq[0]*met[0][1] + dq[1]*met[1][1] + dq[2]*met[2][1]);
        in_grad_disu_upts_ptr[ind+2*stride*in_n_fields] = (1./jac)*(dq[0]*met[0][2] + dq[1]*met[1][2] + dq[2]*met[2][2]);
      }
    }
  }

}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims, int in_n_fields, int in_n_comp, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconvisf_fpts_NS_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l[in_n_fields]; 
  double q_r[in_n_fields]; 
  double f_l[in_n_fields][in_n_dims]; 
  double f_r[in_n_fields][in_n_dims]; 
  double f_c[in_n_fields][in_n_dims]; 
  
  double fn[in_n_fields];
  double norm[in_n_dims];
  
  double grad_ene[in_n_dims];
  double grad_vel[in_n_dims*in_n_dims];
  double grad_q[in_n_fields*in_n_dims];
  
  double stensor[in_n_comp];
  
  double jac;
  double inte, mu;

	if(thread_id<stride)
  {  
    // Left solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride])); 

    // Left solution gradient
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      #pragma unroll
      for(int j=0;j<in_n_dims;j++)
        grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
    }

    // Normal vector
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
    
    // Left flux prep
    vis_NS_flux<in_n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

    // Left flux computation
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
      vis_NS_flux<in_n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, f_l[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);
    

    // Right solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_r[i]=(*(in_disu_fpts_r_ptr[thread_id+i*stride])); 

    // Right solution gradient
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      #pragma unroll
      for(int j=0;j<in_n_dims;j++)
        grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_r_ptr[thread_id + (j*in_n_fields + i)*stride]);
    }
    
    // Right flux prep
    vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);
    
    // Right flux computation
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
      vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f_r[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);

    // Compute common flux
    if(in_vis_riemann_solve_type == 0)
    {
      #pragma unroll
      for (int i=0;i<in_n_fields;i++)
        ldg_flux<in_n_dims,0>(q_l[i],q_r[i],f_l[i],f_r[i],f_c[i],norm,in_pen_fact,in_tau);
    }

    // Compute common normal flux
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      fn[i] = f_c[i][0]*norm[0];
      #pragma unroll
      for (int j=1;j<in_n_dims;j++)
        fn[i] += f_c[i][j]*norm[j];
    }

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];

    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_r_ptr[thread_id+i*stride]))+=-jac*fn[i];

  }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims>
__global__ void calc_norm_tconvisf_fpts_AD_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, double in_pen_fact, double in_tau, double diff_coeff)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l; 
  double q_r; 
  double f_l[in_n_dims]; 
  double f_r[in_n_dims]; 
  double f_c[in_n_dims]; 
  
  double fn;
  double norm[in_n_dims];
  
  double grad_q[in_n_dims];
  double jac;

	if(thread_id<stride)
  {  
    // Left solution
    q_l=(*(in_disu_fpts_l_ptr[thread_id])); 

    // Left solution gradient
    #pragma unroll
    for(int j=0;j<in_n_dims;j++)
      grad_q[j] = *(in_grad_disu_fpts_l_ptr[thread_id + j*stride]);

    // Normal vector
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);

    // Left flux computation
    f_l[0] = -diff_coeff*grad_q[0];
    f_l[1] = -diff_coeff*grad_q[1];
    
    if (in_n_dims==3)
      f_l[2] = -diff_coeff*grad_q[2];


    // Right solution
    q_r=(*(in_disu_fpts_r_ptr[thread_id])); 

    // Right solution gradient
    #pragma unroll
    for(int j=0;j<in_n_dims;j++)
      grad_q[j] = *(in_grad_disu_fpts_r_ptr[thread_id + j*stride]);
    
    // Right flux computation
    f_r[0] = -diff_coeff*grad_q[0];
    f_r[1] = -diff_coeff*grad_q[1];
    
    if (in_n_dims==3)
      f_r[2] = -diff_coeff*grad_q[2];

    // Compute common flux
    ldg_flux<in_n_dims,0>(q_l,q_r,f_l,f_r,f_c,norm,in_pen_fact,in_tau);

    // Compute common normal flux
    fn = f_c[0]*norm[0];
    #pragma unroll
    for (int j=1;j<in_n_dims;j++)
      fn += f_c[j]*norm[j];

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    (*(in_norm_tconf_fpts_l_ptr[thread_id]))+=jac*fn;

    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr[thread_id]));     
    (*(in_norm_tconf_fpts_r_ptr[thread_id]))+=-jac*fn;

  }
}



// kernel to calculate normal transformed continuous viscous flux at the flux points at boundaries
template<int in_n_dims, int in_n_fields, int in_n_comp, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconvisf_fpts_boundary_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_grad_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_loc_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, double in_R_ref, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_time_bound, int in_equation, double diff_coeff)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;

  int bdy_spec;

  double q_l[in_n_fields]; 
  double q_r[in_n_fields]; 
  
  double f[in_n_fields][in_n_dims]; 
  double f_c[in_n_fields][in_n_dims]; 
  
  double fn[in_n_fields];
  double norm[in_n_dims];
  double loc[in_n_dims];
  
  double grad_ene[in_n_dims];
  double grad_vel[in_n_dims*in_n_dims];
  double grad_q[in_n_fields*in_n_dims];
  
  double stensor[in_n_comp];

  double jac;
  double inte, mu;

	if(thread_id<stride)
  {  
    // Left solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride])); 
      
    // Left solution gradient
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      #pragma unroll
      for(int j=0;j<in_n_dims;j++)
        grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
    }
    
    // Normal vector
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
    
    // Compute location
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  loc[i]=*(in_loc_fpts_ptr[thread_id + i*stride]);
    
    // Right solution
    bdy_spec = in_boundary_type[thread_id/in_n_fpts_per_inter];
    set_inv_boundary_conditions_kernel<in_n_dims,in_n_fields>(bdy_spec,q_l,q_r,norm,loc,in_bdy_params,in_gamma,in_R_ref,in_time_bound,in_equation);


    // Compute common flux
    if(bdy_spec == 12 || bdy_spec == 14) 
    {
      // Right solution gradient
      set_vis_boundary_conditions_kernel<in_n_dims,in_n_fields>(bdy_spec,q_l,q_r,grad_q,norm,loc,in_bdy_params,in_gamma,in_R_ref,in_time_bound,in_equation);
    
			if(in_equation==0)
			{
      	// Right flux prep
      	vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);
    
      	// Right flux computation
      	#pragma unroll
      	for (int i=0;i<in_n_fields;i++)
        	vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);
			}
			if(in_equation==1)
			{
    		f[0][0] = -diff_coeff*grad_q[0];
    		f[0][1] = -diff_coeff*grad_q[1];
    
				if(in_n_dims==3)
      		f[0][2] = -diff_coeff*grad_q[2];
			}
	
      if (in_vis_riemann_solve_type==0)
      {
        #pragma unroll
        for (int i=0;i<in_n_fields;i++)
          ldg_flux<in_n_dims,2>(q_l[i],q_r[i],NULL,f[i],f_c[i],norm,in_pen_fact,in_tau); // von Neumann
      }
    }
    else
    {
			if(in_equation==0)
			{
      	// Left flux prep
      	vis_NS_flux<in_n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

      	// Left flux computation
      	#pragma unroll
      	for (int i=0;i<in_n_fields;i++)
        	vis_NS_flux<in_n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, f[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);
			}
			if(in_equation==1)
			{
    		f[0][0] = -diff_coeff*grad_q[0];
    		f[0][1] = -diff_coeff*grad_q[1];
    
				if(in_n_dims==3)
      		f[0][2] = -diff_coeff*grad_q[2];
			}        

      if (in_vis_riemann_solve_type==0)
      {
        #pragma unroll
        for (int i=0;i<in_n_fields;i++)
          ldg_flux<in_n_dims,1>(q_l[i],q_r[i],f[i],NULL,f_c[i],norm,in_pen_fact,in_tau); // Dirichlet
      }
    }

    // compute common normal flux
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      fn[i] = f_c[i][0]*norm[0];
      #pragma unroll
      for (int j=1;j<in_n_dims;j++)
        fn[i] += f_c[i][j]*norm[j];
    }

    // store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];

  }
}


#ifdef _MPI

// gpu kernel to calculate normal transformed continuous inviscid flux at the flux points for mpi faces
template <int in_n_dims, int in_n_fields, int in_riemann_solve_type, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconinvf_fpts_NS_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_pen_fact, int in_viscous)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l[in_n_fields]; 
  double q_r[in_n_fields]; 
  double fn[in_n_fields];
  double norm[in_n_dims];

  double q_c[in_n_fields];

  double jac;

	if(thread_id<stride)
  {  
    // Compute left state solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride])); 

    // Compute right state solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_r[i]=*(in_disu_fpts_r_ptr[thread_id+i*stride]);

    // Compute normal
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);

    if (in_riemann_solve_type==0)
      rusanov_flux<in_n_fields,in_n_dims> (q_l,q_r,norm,fn,in_gamma);
    else if (in_riemann_solve_type==2)
      roe_flux<in_n_fields,in_n_dims> (q_l,q_r,norm,fn,in_gamma);

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))=jac*fn[i];

    // viscous solution correction
    if(in_viscous)
    {
      if(in_vis_riemann_solve_type==0)
        ldg_solution<in_n_dims,in_n_fields,0> (q_l,q_r,norm,q_c,in_pen_fact);

      #pragma unroll
      for (int i=0;i<in_n_fields;i++) 
      {
        (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);   
      }
    }

  }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims, int in_n_fields, int in_n_comp, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconvisf_fpts_NS_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l[in_n_fields]; 
  double q_r[in_n_fields]; 
  double f_l[in_n_fields][in_n_dims]; 
  double f_r[in_n_fields][in_n_dims]; 
  double f_c[in_n_fields][in_n_dims]; 
  
  double fn[in_n_fields];
  double norm[in_n_dims];
  
  double grad_ene[in_n_dims];
  double grad_vel[in_n_dims*in_n_dims];
  double grad_q[in_n_fields*in_n_dims];
  
  double stensor[in_n_comp];
  
  double jac;
  double inte, mu;

	if(thread_id<stride)
  {  
    // Left solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride])); 

    // Left solution gradient
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      #pragma unroll
      for(int j=0;j<in_n_dims;j++)
        grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
    }

    // Normal vector
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
    
    // Left flux prep
    vis_NS_flux<in_n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

    // Left flux computation
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
      vis_NS_flux<in_n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, f_l[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);


    // Right solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_r[i]=(*(in_disu_fpts_r_ptr[thread_id+i*stride]));// don't divide by jac, since points to buffer

    // Right solution gradient
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      #pragma unroll
      for(int j=0;j<in_n_dims;j++)
        grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_r_ptr[thread_id + (j*in_n_fields + i)*stride]);
    }
    
    // Right flux prep
    vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);
    
    // Right flux computation
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
      vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f_r[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);

   
    // Compute common flux
    if(in_vis_riemann_solve_type == 0)
    {
      #pragma unroll
      for (int i=0;i<in_n_fields;i++)
        ldg_flux<in_n_dims,0>(q_l[i],q_r[i],f_l[i],f_r[i],f_c[i],norm,in_pen_fact,in_tau);
    }

    // Compute common normal flux
    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
    {
      fn[i] = f_c[i][0]*norm[0];
      #pragma unroll
      for (int j=1;j<in_n_dims;j++)
        fn[i] += f_c[i][j]*norm[j];
    }

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
	    (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];

  }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims>
__global__ void calc_norm_tconvisf_fpts_AD_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double in_pen_fact, double in_tau, double diff_coeff)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l; 
  double q_r; 
  double f_l[in_n_dims]; 
  double f_r[in_n_dims]; 
  double f_c[in_n_dims]; 
  
  double fn;
  double norm[in_n_dims];
  
  double grad_q[in_n_dims];
  double jac;

	if(thread_id<stride)
  {  
    // Left solution
    q_l=(*(in_disu_fpts_l_ptr[thread_id])); 

    // Left solution gradient
    #pragma unroll
    for(int j=0;j<in_n_dims;j++)
      grad_q[j] = *(in_grad_disu_fpts_l_ptr[thread_id + j*stride]);

    // Normal vector
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);

    // Left flux computation
    f_l[0] = -diff_coeff*grad_q[0];
    f_l[1] = -diff_coeff*grad_q[1];
    
    if (in_n_dims==3)
      f_l[2] = -diff_coeff*grad_q[2];


    // Right solution
    q_r=(*(in_disu_fpts_r_ptr[thread_id])); 

    // Right solution gradient
    #pragma unroll
    for(int j=0;j<in_n_dims;j++)
      grad_q[j] = *(in_grad_disu_fpts_r_ptr[thread_id + j*stride]);
    
    // Right flux computation
    f_r[0] = -diff_coeff*grad_q[0];
    f_r[1] = -diff_coeff*grad_q[1];
    
    if (in_n_dims==3)
      f_r[2] = -diff_coeff*grad_q[2];

    // Compute common flux
    ldg_flux<in_n_dims,0>(q_l,q_r,f_l,f_r,f_c,norm,in_pen_fact,in_tau);

    // Compute common normal flux
    fn = f_c[0]*norm[0];
    #pragma unroll
    for (int j=1;j<in_n_dims;j++)
      fn += f_c[j]*norm[j];

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
    (*(in_norm_tconf_fpts_l_ptr[thread_id]))+=jac*fn;

  }
}


template <int in_n_dims, int in_vis_riemann_solve_type>
__global__ void calc_norm_tconinvf_fpts_lax_friedrich_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double in_pen_fact, int in_viscous, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;
	
  double q_l; 
  double q_r; 
  double fn,u_av,u_diff;
  double norm_speed;
  double norm[in_n_dims];
  
  double q_c;
  double jac;

	if(thread_id<stride)
  {  

   // Compute left state solution
  	q_l=(*(in_disu_fpts_l_ptr[thread_id])); 

    // Compute right state solution
  	q_r=(*(in_disu_fpts_r_ptr[thread_id]));

    // Compute normal
    #pragma unroll
    for (int i=0;i<in_n_dims;i++) 
  	  norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);

    u_av = 0.5*(q_r+q_l);
    u_diff = q_l-q_r; 

    norm_speed=0.;
    if (in_n_dims==2)
      norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1];
    else if (in_n_dims==3)
      norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2];
	  		
    // Compute common interface flux
    fn = 0.;
    if (in_n_dims==2)
      fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1])*u_av;
    else if (in_n_dims==3)
      fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2])*u_av;
    fn += 0.5*lambda*abs(norm_speed)*u_diff;

    // Store transformed flux
    jac = (*(in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr[thread_id]));     
	  (*(in_norm_tconf_fpts_l_ptr[thread_id]))=jac*fn;

    // viscous solution correction
    if(in_viscous)
    {
			if(in_n_dims==2)
			{
    		if ((norm[0]+norm[1]) < 0.)
    	  	in_pen_fact=-in_pen_fact;
			}
			if(in_n_dims==3)
			{
    		if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
    	  	in_pen_fact=-in_pen_fact;
			}

      q_c = 0.5*(q_l+q_r) - in_pen_fact*(q_l-q_r);

      /*
      if(in_vis_riemann_solve_type==0)
        ldg_solution<in_n_dims,1,0> (&q_l,&q_r,norm,&q_c,in_pen_fact);
      */

      (*(in_delta_disu_fpts_l_ptr[thread_id])) = (q_c-q_l);   
    }
  }
}


template <int in_n_fields>
__global__ void  pack_out_buffer_disu_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double* in_out_buffer_disu_ptr)
{

  double q_l[in_n_fields];

	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int inter = thread_id/(in_n_fpts_per_inter);
  const int fpt = thread_id - inter*in_n_fpts_per_inter;
  const int stride=in_n_fpts_per_inter*in_n_inters;

  if (thread_id < stride)
  {
    // Compute left state solution
    #pragma unroll
    for (int i=0;i<in_n_fields;i++) 
  	  q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride])); 

    #pragma unroll
    for (int i=0;i<in_n_fields;i++)
      in_out_buffer_disu_ptr[inter*in_n_fpts_per_inter*in_n_fields+i*in_n_fpts_per_inter+fpt]=q_l[i];

  }

}


template <int in_n_fields, int in_n_dims>
__global__ void  pack_out_buffer_grad_disu_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_grad_disu_fpts_l_ptr, double* in_out_buffer_grad_disu_ptr)
{

  double dq[in_n_fields][in_n_dims];

	const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int inter = thread_id/(in_n_fpts_per_inter);
  const int fpt = thread_id - inter*in_n_fpts_per_inter;
  const int stride=in_n_fpts_per_inter*in_n_inters;

  if (thread_id < stride)
  {
    // Compute left state solution
    #pragma unroll
    for (int j=0;j<in_n_dims;j++)
      #pragma unroll
      for (int i=0;i<in_n_fields;i++) 
  	    dq[i][j]=(*(in_grad_disu_fpts_l_ptr[thread_id+(j*in_n_fields+i)*stride]));

    #pragma unroll
    for (int j=0;j<in_n_dims;j++)
      #pragma unroll
      for (int i=0;i<in_n_fields;i++)
        in_out_buffer_grad_disu_ptr[inter*in_n_fpts_per_inter*in_n_fields*in_n_dims+j*in_n_fpts_per_inter*in_n_fields+i*in_n_fpts_per_inter+fpt]=dq[i][j];

  }

}

#endif



void RK45_update_kernel_wrapper(int in_n_upts_per_ele,int in_n_dims,int in_n_fields,int in_n_eles,double* in_disu0_upts_ptr,double* in_disu1_upts_ptr,double* in_div_tconf_upts_ptr, double* in_detjac_upts_ptr, double in_rk4a, double in_rk4b, double in_dt, double in_const_src_term)
{

	// HACK: fix 256 threads per block
	int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  if (in_n_fields==1)
  {
	  RK45_update_kernel <1> <<< n_blocks,256>>> (in_disu0_upts_ptr, in_div_tconf_upts_ptr, in_disu1_upts_ptr, in_detjac_upts_ptr, in_n_eles, in_n_upts_per_ele, in_rk4a, in_rk4b, in_dt, in_const_src_term);
  }
  else if (in_n_fields==4)
  {
	  RK45_update_kernel <4> <<< n_blocks,256>>> (in_disu0_upts_ptr, in_div_tconf_upts_ptr, in_disu1_upts_ptr, in_detjac_upts_ptr, in_n_eles, in_n_upts_per_ele, in_rk4a, in_rk4b, in_dt, in_const_src_term);
  }
  else if (in_n_fields==5)
  {
	  RK45_update_kernel <5> <<< n_blocks,256>>> (in_disu0_upts_ptr, in_div_tconf_upts_ptr, in_disu1_upts_ptr, in_detjac_upts_ptr, in_n_eles, in_n_upts_per_ele, in_rk4a, in_rk4b, in_dt, in_const_src_term);
  }
  else 
    FatalError("n_fields not supported");

}

void RK11_update_kernel_wrapper(int in_n_upts_per_ele,int in_n_dims,int in_n_fields,int in_n_eles,double* in_disu0_upts_ptr,double* in_div_tconf_upts_ptr, double* in_detjac_upts_ptr, double in_dt, double in_const_src_term)
{

	// HACK: fix 256 threads per block
	int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  if (in_n_fields==1)
  {
	  RK11_update_kernel <1> <<< n_blocks,256>>> (in_disu0_upts_ptr, in_div_tconf_upts_ptr, in_detjac_upts_ptr, in_n_eles, in_n_upts_per_ele, in_dt, in_const_src_term);
  }
  else if (in_n_fields==4)
  {
	  RK11_update_kernel <4> <<< n_blocks,256>>> (in_disu0_upts_ptr, in_div_tconf_upts_ptr, in_detjac_upts_ptr, in_n_eles, in_n_upts_per_ele, in_dt, in_const_src_term);
  }
  else if (in_n_fields==5)
  {
	  RK11_update_kernel <5> <<< n_blocks,256>>> (in_disu0_upts_ptr, in_div_tconf_upts_ptr, in_detjac_upts_ptr, in_n_eles, in_n_upts_per_ele, in_dt, in_const_src_term);
  }
  else 
    FatalError("n_fields not supported");

}


// wrapper for gpu kernel to calculate transformed discontinuous inviscid flux at solution points
void calc_tdisinvf_upts_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, double in_gamma, int equation, double wave_speed_x, double wave_speed_y, double wave_speed_z)
{
	// HACK: fix 256 threads per block
	int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (equation==0)
  {
    if (in_n_dims==2)
	    calc_tdisinvf_upts_NS_gpu_kernel<2,4> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr,in_gamma);
    else if (in_n_dims==3)
	    calc_tdisinvf_upts_NS_gpu_kernel<3,5> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr,in_gamma);
    else
		  FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else if (equation==1)
  {
    if (in_n_dims==2)
	    calc_tdisinvf_upts_AD_gpu_kernel<2> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr,wave_speed_x,wave_speed_y,wave_speed_z);
    else if (in_n_dims==3)
	    calc_tdisinvf_upts_AD_gpu_kernel<3> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr,wave_speed_x,wave_speed_y,wave_speed_z);
    else
		  FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else 
  {
    FatalError("equation not recognized");
  }

  check_cuda_error("After",__FILE__, __LINE__);
}



// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points
void calc_norm_tconinvf_fpts_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_norm_tconinvf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double** in_delta_disu_fpts_r_ptr, double in_gamma, double in_pen_fact, int in_viscous, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
	// HACK: fix 256 threads per block
	int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_riemann_solve_type==0) // Rusanov 
  {
    if(in_vis_riemann_solve_type==0) //LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_NS_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_NS_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if ( in_riemann_solve_type==2) // Roe
  {
    if(in_vis_riemann_solve_type==0) //LDG
    {  
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_NS_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_NS_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if (in_riemann_solve_type==1) // Lax-Friedrich
  {
    if(in_vis_riemann_solve_type==0) //LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_lax_friedrich_gpu_kernel<2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_lax_friedrich_gpu_kernel<3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else
    FatalError("ERROR: Riemann solver type not recognized ... ");
  
  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries
void calc_norm_tconinvf_fpts_boundary_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_loc_fpts_ptr, int* in_boundary_type, double* in_bdy_params, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_R_ref, int in_viscous, int in_vis_riemann_solve_type, double in_time_bound, double in_wave_speed_x, double in_wave_speed_y, double in_wave_speed_z, double in_lambda, int in_equation)
{

  check_cuda_error("Before", __FILE__, __LINE__);
	// HACK: fix 256 threads per block
	int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  if (in_riemann_solve_type==0)  // Rusanov
  {
    if (in_vis_riemann_solve_type==0) // LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_boundary_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_boundary_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
  }
  else if (in_riemann_solve_type==1)  // Lax-Friedrichs
  {
    if (in_vis_riemann_solve_type==0) // LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_boundary_gpu_kernel<2,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_boundary_gpu_kernel<3,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
  }
  else if (in_riemann_solve_type==2) // Roe
  {
    if (in_vis_riemann_solve_type==0) // LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_boundary_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_boundary_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
  }
  else
  {
		FatalError("ERROR: Riemann solver type not recognized in bdy riemann solver");
  }

  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate transformed discontinuous viscous flux at solution points
void calc_tdisvisf_upts_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, int in_ele_type, double in_filter_ratio, int LES, int SGS_model, double* in_Lu_ptr, double* in_Le_ptr, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double in_diff_coeff)
{
	// HACK: fix 256 threads per block
	int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (equation==0)
  {
    if (in_n_dims==2)
	    calc_tdisvisf_upts_NS_gpu_kernel<2,4,3> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_ele_type, in_filter_ratio, LES, SGS_model, in_Lu_ptr, in_Le_ptr, in_disu_upts_ptr, out_tdisf_upts_ptr, in_grad_disu_upts_ptr, in_detjac_upts_ptr, in_inv_detjac_mul_jac_upts_ptr, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis);
    else if (in_n_dims==3)
	    calc_tdisvisf_upts_NS_gpu_kernel<3,5,6> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_ele_type, in_filter_ratio, LES, SGS_model, in_Lu_ptr, in_Le_ptr, in_disu_upts_ptr, out_tdisf_upts_ptr, in_grad_disu_upts_ptr, in_detjac_upts_ptr, in_inv_detjac_mul_jac_upts_ptr, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis);
    else
		  FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else if (equation==1)
  {
    if (in_n_dims==2)
	    calc_tdisvisf_upts_AD_gpu_kernel<2> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_disu_upts_ptr, out_tdisf_upts_ptr, in_grad_disu_upts_ptr, in_detjac_upts_ptr, in_inv_detjac_mul_jac_upts_ptr, in_diff_coeff);
    else if (in_n_dims==3)
	    calc_tdisvisf_upts_AD_gpu_kernel<3> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_disu_upts_ptr, out_tdisf_upts_ptr, in_grad_disu_upts_ptr, in_detjac_upts_ptr, in_inv_detjac_mul_jac_upts_ptr, in_diff_coeff);
    else
		  FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else 
    FatalError("equation not recognized");

  check_cuda_error("After",__FILE__, __LINE__);
}

// wrapper for gpu kernel to transform gradient at sol points to physical gradient
void transform_grad_disu_upts_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_inv_detjac_mul_jac_upts_ptr, int equation) 
{
	// HACK: fix 256 threads per block
	int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(equation == 0) {
    if (in_n_dims==2)
	    transform_grad_disu_upts_kernel<2,4> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr);
    else if (in_n_dims==3)
	    transform_grad_disu_upts_kernel<3,5> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr);
    else
		  FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else if(equation == 1) {
    if (in_n_dims==2)
	    transform_grad_disu_upts_kernel<2,1> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr);
    else if (in_n_dims==3)
	    transform_grad_disu_upts_kernel<3,1> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_inv_detjac_mul_jac_upts_ptr);
    else
		  FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else
    FatalError("equation not recognized");

  check_cuda_error("After",__FILE__, __LINE__);
}


// wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points
void calc_norm_tconvisf_fpts_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr, double** in_norm_fpts_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double in_diff_coeff)
{
	// HACK: fix 256 threads per block
	int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(equation==0)
  {
    if (in_vis_riemann_solve_type==0) // LDG
    {
      if (in_n_dims==2)
        calc_norm_tconvisf_fpts_NS_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_norm_tconf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf, in_mu_inf,in_c_sth,in_fix_vis);
      else if (in_n_dims==3)
        calc_norm_tconvisf_fpts_NS_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_norm_tconf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf, in_mu_inf,in_c_sth,in_fix_vis);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if(equation==1)
  {
    if (in_vis_riemann_solve_type==0) // LDG
    {
      if (in_n_dims==2)
        calc_norm_tconvisf_fpts_AD_gpu_kernel<2> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_norm_tconf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_diff_coeff);
      else if (in_n_dims==3)
        calc_norm_tconvisf_fpts_AD_gpu_kernel<3> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_norm_tconf_fpts_r_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_r_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_diff_coeff);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else 
    FatalError("equation not recognized");


  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points at boundaries
void calc_norm_tconvisf_fpts_boundary_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_grad_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_loc_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_R_ref, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_time_bound, int in_equation, double in_diff_coeff)
{

	// HACK: fix 256 threads per block
	int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_vis_riemann_solve_type==0) // LDG
  {
		if(in_equation==0)
		{
    	if (in_n_dims==2)
      	calc_norm_tconvisf_fpts_boundary_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_grad_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type,in_bdy_params,in_delta_disu_fpts_l_ptr,in_R_ref,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf,in_mu_inf,in_c_sth,in_fix_vis, in_time_bound, in_equation, in_diff_coeff);
    	else if (in_n_dims==3)
      	calc_norm_tconvisf_fpts_boundary_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_grad_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type,in_bdy_params,in_delta_disu_fpts_l_ptr,in_R_ref,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf,in_mu_inf,in_c_sth,in_fix_vis, in_time_bound, in_equation, in_diff_coeff);
  	}
		else if(in_equation==1)
		{
    	if (in_n_dims==2)
      	calc_norm_tconvisf_fpts_boundary_gpu_kernel<2,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_grad_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type,in_bdy_params,in_delta_disu_fpts_l_ptr,in_R_ref,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf,in_mu_inf,in_c_sth,in_fix_vis, in_time_bound, in_equation, in_diff_coeff);
    	else if (in_n_dims==3)
      	calc_norm_tconvisf_fpts_boundary_gpu_kernel<3,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_grad_disu_fpts_l_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_loc_fpts_ptr,in_boundary_type,in_bdy_params,in_delta_disu_fpts_l_ptr,in_R_ref,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf,in_mu_inf,in_c_sth,in_fix_vis, in_time_bound, in_equation, in_diff_coeff);
		}
	}
  else
		FatalError("ERROR: Viscous riemann solver type not recognized ... ");

  check_cuda_error("After", __FILE__, __LINE__);
}


#ifdef _MPI

void pack_out_buffer_disu_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,double** in_disu_fpts_l_ptr, double* in_out_buffer_disu_ptr)
{
  int block_size=256;
	int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_n_fields==1)
    pack_out_buffer_disu_gpu_kernel<1> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_out_buffer_disu_ptr);
  else if (in_n_fields==4)
    pack_out_buffer_disu_gpu_kernel<4> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_out_buffer_disu_ptr);
  else if (in_n_fields==5)
    pack_out_buffer_disu_gpu_kernel<5> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_out_buffer_disu_ptr);
  else
    FatalError("Number of fields not supported in pack_out_buffer");

  check_cuda_error("After", __FILE__, __LINE__);

}

void pack_out_buffer_grad_disu_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,int in_n_dims, double** in_grad_disu_fpts_l_ptr, double* in_out_buffer_grad_disu_ptr)
{
  int block_size=256;
	int n_blocks=((in_n_inters*in_n_fpts_per_inter*in_n_dims-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_n_fields==1)
  {
    if (in_n_dims==2) {
      pack_out_buffer_grad_disu_gpu_kernel<1,2> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_grad_disu_fpts_l_ptr,in_out_buffer_grad_disu_ptr);
    }
    else if (in_n_dims==3) {
      pack_out_buffer_grad_disu_gpu_kernel<1,3> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_grad_disu_fpts_l_ptr,in_out_buffer_grad_disu_ptr);
    }

  }
  else if (in_n_fields==4)
  {
    pack_out_buffer_grad_disu_gpu_kernel<4,2> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_grad_disu_fpts_l_ptr,in_out_buffer_grad_disu_ptr);
  }
  else if (in_n_fields==5)
  {
    pack_out_buffer_grad_disu_gpu_kernel<5,3> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_grad_disu_fpts_l_ptr,in_out_buffer_grad_disu_ptr);
  }
  else
    FatalError("Number of fields not supported in pack_out_buffer");

  check_cuda_error("After", __FILE__, __LINE__);

}

// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points
void calc_norm_tconinvf_fpts_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr,int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_pen_fact,  int in_viscous, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
  
  int block_size=256;
	int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_riemann_solve_type==0 ) // Rusanov
  {
    if (in_vis_riemann_solve_type==0 ) //LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_NS_mpi_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_NS_mpi_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if (in_riemann_solve_type==2 ) // Roe
  {
    if (in_vis_riemann_solve_type==0 ) //LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_NS_mpi_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_NS_mpi_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if (in_riemann_solve_type==1) // Lax-Friedrich
  {
    if(in_vis_riemann_solve_type==0) //LDG
    {
      if (in_n_dims==2)
	      calc_norm_tconinvf_fpts_lax_friedrich_mpi_gpu_kernel<2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
      else if (in_n_dims==3)
	      calc_norm_tconinvf_fpts_lax_friedrich_mpi_gpu_kernel<3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
    }
    else
		  FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else
  {
		FatalError("ERROR: Riemann solver type not recognized ... ");
  }

  check_cuda_error("After", __FILE__, __LINE__);

}


// wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points
void calc_norm_tconvisf_fpts_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr, double** in_norm_fpts_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_diff_coeff)
{
 	// HACK: fix 256 threads per block
	int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);
  
	if (in_riemann_solve_type==0 ) // Rusanov
  {
  	if (in_vis_riemann_solve_type==0) // LDG
  	{
  	  if (in_n_dims==2)
  	    calc_norm_tconvisf_fpts_NS_mpi_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf, in_mu_inf,in_c_sth,in_fix_vis);
  	  else if (in_n_dims==3)
  	    calc_norm_tconvisf_fpts_NS_mpi_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_gamma,in_prandtl,in_rt_inf, in_mu_inf,in_c_sth,in_fix_vis);
  	}
  	else
			FatalError("ERROR: Viscous riemann solver type not recognized ... ");
	}
  else if (in_riemann_solve_type==1) // Lax-Friedrich
  {
    if (in_vis_riemann_solve_type==0) // LDG
    {
      if (in_n_dims==2)
        calc_norm_tconvisf_fpts_AD_mpi_gpu_kernel<2> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_diff_coeff);
      else if (in_n_dims==3)
        calc_norm_tconvisf_fpts_AD_mpi_gpu_kernel<3> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_mag_tnorm_dot_inv_detjac_mul_jac_fpts_l_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_diff_coeff);
    }
  	else
			FatalError("ERROR: Viscous riemann solver type not recognized ... ");
	}
  else
  {
		FatalError("ERROR: Riemann solver type not recognized ... ");
  }

  check_cuda_error("After", __FILE__, __LINE__);

}

#endif

void bespoke_SPMV(int m, int n, int n_fields, int n_eles, double* opp_ell_data_ptr, int* opp_ell_indices_ptr, int nnz_per_row, double* b_ptr, double *c_ptr, int cell_type, int order, int add_flag)
{

  int eles_per_block=2; // allows up to 128 DOFs per element
  int grid_size = (n_eles-1)/(eles_per_block)+1; 
  int block_size = eles_per_block*m;
  int shared_mem = n*eles_per_block*n_fields;
  shared_mem += shared_mem/HALFWARP;

  if (n_fields==1)
  {
    bespoke_SPMV_kernel<1> <<<grid_size, block_size, shared_mem*sizeof(double) >>> (c_ptr, b_ptr, opp_ell_data_ptr, opp_ell_indices_ptr, nnz_per_row, n_eles, n, m, eles_per_block,n_eles*n,n_eles*m,add_flag);
  }
  else if (n_fields==4)
  {
    bespoke_SPMV_kernel<4> <<<grid_size, block_size, shared_mem*sizeof(double) >>> (c_ptr, b_ptr, opp_ell_data_ptr, opp_ell_indices_ptr, nnz_per_row, n_eles, n, m, eles_per_block,n_eles*n,n_eles*m,add_flag);
  }
  else if (n_fields==5)
  {
    bespoke_SPMV_kernel<5> <<<grid_size, block_size, shared_mem*sizeof(double) >>> (c_ptr, b_ptr, opp_ell_data_ptr, opp_ell_indices_ptr, nnz_per_row, n_eles, n, m, eles_per_block,n_eles*n,n_eles*m,add_flag);
  }

}


