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

  if (tid < cells_per_block*dim0&& ic < n_cells)
    {

      // Fetching data to shared memory
      int n_fetch_loops = (dim1-1)/(dim0)+1;

      // Since n_qpts might be larger than dim0
      // each thread might have to fetch more than n_fields values
#pragma unroll
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
                  s_b[m] = g_b[m1];
                  m += stride_shared;
                  m1 += stride_n;
                }
            }
        }
    }

  __syncthreads();

  if (tid < cells_per_block*dim0&& ic < n_cells)
    {

      // With data in shared memory, perform matrix multiplication
      // 1 thread per flux point
#pragma unroll
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
__device__ void set_inv_boundary_conditions_kernel(int bdy_type, double* u_l, double* u_r, double* v_g, double* norm, double* loc, double *bdy_params, double gamma, double R_ref, double time_bound, int equation, int turb_model)
{
  double rho_l, rho_r;
  double v_l[n_dims], v_r[n_dims];
  double e_l, e_r;
  double p_l, p_r;
  double T_r;
  double vn_l;
  double v_sq;
  double rho_bound = bdy_params[0];
  double* v_bound = &bdy_params[1];
  double p_bound = bdy_params[4];
  double* v_wall = &bdy_params[5];
  double T_wall = bdy_params[8];

  // Navier-Stokes Boundary Conditions
  if(equation==0)
  {
    // Store primitive variables for clarity
    rho_l = u_l[0];
    for (int i=0; i<n_dims; i++)
      v_l[i] = u_l[i+1]/u_l[0];
    e_l = u_l[n_dims+1];

    // Compute pressure on left side
    v_sq = 0.;
    for (int i=0; i<n_dims; i++)
      v_sq += (v_l[i]*v_l[i]);
    p_l = (gamma-1.0)*(e_l - 0.5*rho_l*v_sq);

    // Subsonic inflow simple (free pressure) //CONSIDER DELETING
    if(bdy_type == 1)
    {
      // fix density and velocity
      rho_r = rho_bound;
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_bound[i];

      // extrapolate pressure
      p_r = p_l;

      // compute energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += (v_r[i]*v_r[i]);
      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

      // SA model
      if (turb_model == 1)
      {
        // set turbulent eddy viscosity
        double mu_tilde_inf = bdy_params[14];
        u_r[n_dims+2] = mu_tilde_inf;
      }
    }

    // Subsonic outflow simple (fixed pressure) //CONSIDER DELETING
    else if(bdy_type == 2)
    {
      // extrapolate density and velocity
      rho_r = rho_l;
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_l[i];

      // fix pressure
      p_r = p_bound;

      // compute energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += (v_r[i]*v_r[i]);
      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

      // SA model
      if (turb_model == 1)
      {
        // extrapolate turbulent eddy viscosity
        u_r[n_dims+2] = u_l[n_dims+2];
      }
    }

    // Subsonic inflow characteristic
    // there is one outgoing characteristic (u-c), therefore we can specify
    // all but one state variable at the inlet. The outgoing Riemann invariant
    // provides the final piece of info. Adapted from an implementation in
    // SU2.
    else if(bdy_type == 3)
    {
      double V_r;
      double c_l, c_r_sq, c_total_sq;
      double R_plus, h_total;
      double aa, bb, cc, dd;
      double Mach_sq, alpha;

      // Specify Inlet conditions
      double p_total_bound = bdy_params[9];
      double T_total_bound = bdy_params[10];
      double *n_free_stream = &bdy_params[11];

      // Compute normal velocity on left side
      vn_l = 0.;
      for (int i=0; i<n_dims; i++)
        vn_l += v_l[i]*norm[i];

      // Compute speed of sound
      c_l = sqrt(gamma*p_l/rho_l);

      // Extrapolate Riemann invariant
      R_plus = vn_l + 2.0*c_l/(gamma-1.0);

      // Specify total enthalpy
      h_total = gamma*R_ref/(gamma-1.0)*T_total_bound;

      // Compute total speed of sound squared
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += v_l[i]*v_l[i];
      c_total_sq = (gamma-1.0)*(h_total - (e_l/rho_l + p_l/rho_l) + 0.5*v_sq) + c_l*c_l;

      // Dot product of normal flow velocity
      alpha = 0.;
      for (int i=0; i<n_dims; i++)
        alpha += norm[i]*n_free_stream[i];

      // Coefficients of quadratic equation
      aa = 1.0 + 0.5*(gamma-1.0)*alpha*alpha;
      bb = -(gamma-1.0)*alpha*R_plus;
      cc = 0.5*(gamma-1.0)*R_plus*R_plus - 2.0*c_total_sq/(gamma-1.0);

      // Solve quadratic equation for velocity on right side
      // (Note: largest value will always be the positive root)
      // (Note: Will be set to zero if NaN)
      dd = bb*bb - 4.0*aa*cc;
      dd = sqrt(max(dd, 0.0));
      V_r = (-bb + dd)/(2.0*aa);
      V_r = max(V_r, 0.0);
      v_sq = V_r*V_r;

      // Compute speed of sound
      c_r_sq = c_total_sq - 0.5*(gamma-1.0)*v_sq;

      // Compute Mach number (cutoff at Mach = 1.0)
      Mach_sq = v_sq/(c_r_sq);
      Mach_sq = min(Mach_sq, 1.0);
      v_sq = Mach_sq*c_r_sq;
      V_r = sqrt(v_sq);
      c_r_sq = c_total_sq - 0.5*(gamma-1.0)*v_sq;

      // Compute velocity (based on free stream direction)
      for (int i=0; i<n_dims; i++)
        v_r[i] = V_r*n_free_stream[i];

      // Compute temperature
      T_r = c_r_sq/(gamma*R_ref);

      // Compute pressure
      p_r = p_total_bound*pow(T_r/T_total_bound, gamma/(gamma-1.0));

      // Compute density
      rho_r = p_r/(R_ref*T_r);

      // Compute energy
      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

      // SA model
      if (turb_model == 1)
      {
        // set turbulent eddy viscosity
        double mu_tilde_inf = bdy_params[14];
        u_r[n_dims+2] = mu_tilde_inf;
      }
    }

    // Subsonic outflow characteristic
    // there is one incoming characteristic, therefore one variable can be
    // specified (back pressure) and is used to update the conservative
    // variables. Compute the entropy and the acoustic Riemann variable.
    // These invariants, as well as the tangential velocity components,
    // are extrapolated. Adapted from an implementation in SU2.
    else if(bdy_type == 4)
    {
      double c_l, c_r;
      double R_plus, s;
      double vn_r;

      // Compute normal velocity on left side
      vn_l = 0.;
      for (int i=0; i<n_dims; i++)
        vn_l += v_l[i]*norm[i];

      // Compute speed of sound
      c_l = sqrt(gamma*p_l/rho_l);

      // Extrapolate Riemann invariant
      R_plus = vn_l + 2.0*c_l/(gamma-1.0);

      // Extrapolate entropy
      s = p_l/pow(rho_l,gamma);

      // fix pressure on the right side
      p_r = p_bound;

      // Compute density
      rho_r = pow(p_r/s, 1.0/gamma);

      // Compute speed of sound
      c_r = sqrt(gamma*p_r/rho_r);

      // Compute normal velocity
      vn_r = R_plus - 2.0*c_r/(gamma-1.0);

      // Compute velocity and energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
      {
        v_r[i] = v_l[i] + (vn_r - vn_l)*norm[i];
        v_sq += (v_r[i]*v_r[i]);
      }
      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

      // SA model
      if (turb_model == 1)
      {
        // extrapolate turbulent eddy viscosity
        u_r[n_dims+2] = u_l[n_dims+2];
      }
    }

    // Supersonic inflow
    else if(bdy_type == 5)
    {
      // fix density and velocity
      rho_r = rho_bound;
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_bound[i];

      // fix pressure
      p_r = p_bound;

      // compute energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += (v_r[i]*v_r[i]);
      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;
    }


    // Supersonic outflow
    else if(bdy_type == 6)
    {
      // extrapolate density, velocity, energy
      rho_r = rho_l;
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_l[i];
      e_r = e_l;
    }

    // Slip wall
    else if(bdy_type == 7)
    {
      // extrapolate density
      rho_r = rho_l;

      // Compute normal velocity on left side
      vn_l = 0.;
      for (int i=0; i<n_dims; i++)
        vn_l += (v_l[i]-v_g[i])*norm[i];

      // reflect normal velocity
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_l[i] - 2.0*vn_l*norm[i];

      // extrapolate energy
      e_r = e_l;
    }

    // Isothermal, no-slip wall (fixed)
    else if(bdy_type == 11)
    {
      // Set state for the right side
      // extrapolate pressure
      p_r = p_l;

      // isothermal temperature
      T_r = T_wall;

      // density
      rho_r = p_r/(R_ref*T_r);

      // no-slip
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_g[i];

      // energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += (v_r[i]*v_r[i]);

      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

      // SA model
      if (turb_model == 1)
      {
        // zero turbulent eddy viscosity at the wall
        u_r[n_dims+2] = 0.0;
      }
    }

    // Adiabatic, no-slip wall (fixed)
    else if(bdy_type == 12)
    {
      // extrapolate density
      rho_r = rho_l; // only useful part

      // extrapolate pressure
      p_r = p_l;

      // no-slip
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_g[i];

      // energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += (v_r[i]*v_r[i]);

      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

      // SA model
      if (turb_model == 1)
      {
        // zero turbulent eddy viscosity at the wall
        u_r[n_dims+2] = 0.0;
      }
    }

    // Isothermal, no-slip wall (moving)
    else if(bdy_type == 13)
    {
      // extrapolate pressure
      p_r = p_l;

      // isothermal temperature
      T_r = T_wall;

      // density
      rho_r = p_r/(R_ref*T_r);

      // no-slip
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_wall[i] + v_g[i];

      // energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += (v_r[i]*v_r[i]);

      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;
    }

    // Adiabatic, no-slip wall (moving)
    else if(bdy_type == 14)
    {
      // extrapolate density
      rho_r = rho_l;

      // extrapolate pressure
      p_r = p_l;

      // no-slip
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_wall[i] + v_g[i];

      // energy
      v_sq = 0.;
      for (int i=0; i<n_dims; i++)
        v_sq += (v_r[i]*v_r[i]);
      e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;
    }

    // Characteristic
    else if (bdy_type == 15)
    {
      double c_star;
      double vn_star;
      double vn_bound;
      double r_plus,r_minus;

      double one_over_s;
      double h_free_stream;

      // Compute normal velocity on left side
      vn_l = 0.;
      for (int i=0; i<n_dims; i++)
        vn_l += v_l[i]*norm[i];

      vn_bound = 0;
      for (int i=0; i<n_dims; i++)
        vn_bound += v_bound[i]*norm[i];

      r_plus  = vn_l + 2./(gamma-1.)*sqrt(gamma*p_l/rho_l);
      r_minus = vn_bound - 2./(gamma-1.)*sqrt(gamma*p_bound/rho_bound);

      c_star = 0.25*(gamma-1.)*(r_plus-r_minus);
      vn_star = 0.5*(r_plus+r_minus);

      // Inflow
      if (vn_l<0)
      {
        // HACK
        one_over_s = pow(rho_bound,gamma)/p_bound;

        // freestream total enthalpy
        v_sq = 0.;
        for (int i=0;i<n_dims;i++)
          v_sq += v_bound[i]*v_bound[i];
        h_free_stream = gamma/(gamma-1.)*p_bound/rho_bound + 0.5*v_sq;

        rho_r = pow(1./gamma*(one_over_s*c_star*c_star),1./(gamma-1.));

        // Compute velocity on the right side
        for (int i=0; i<n_dims; i++)
          v_r[i] = vn_star*norm[i] + (v_bound[i] - vn_bound*norm[i]);

        p_r = rho_r/gamma*c_star*c_star;
        e_r = rho_r*h_free_stream - p_r;

        // SA model
        if (turb_model == 1)
        {
          // set turbulent eddy viscosity
          double mu_tilde_inf = bdy_params[14];
          u_r[n_dims+2] = mu_tilde_inf;
        }
      }

      // Outflow
      else
      {
        one_over_s = pow(rho_l,gamma)/p_l;

        // freestream total enthalpy
        rho_r = pow(1./gamma*(one_over_s*c_star*c_star), 1./(gamma-1.));

        // Compute velocity on the right side
        for (int i=0; i<n_dims; i++)
          v_r[i] = vn_star*norm[i] + (v_l[i] - vn_l*norm[i]);

        p_r = rho_r/gamma*c_star*c_star;
        v_sq = 0.;
        for (int i=0; i<n_dims; i++)
          v_sq += (v_r[i]*v_r[i]);
        e_r = (p_r/(gamma-1.0)) + 0.5*rho_r*v_sq;

        // SA model
        if (turb_model == 1)
        {
          // extrapolate turbulent eddy viscosity
          u_r[n_dims+2] = u_l[n_dims+2];
        }
      }
    }

    // Dual consistent BC (see SD++ for more comments)
    else if (bdy_type==16)
    {
      // extrapolate density
      rho_r = rho_l;

      // Compute normal velocity on left side
      vn_l = 0.;
      for (int i=0; i<n_dims; i++)
        vn_l += v_l[i]*norm[i];

      // set u = u - (vn_l)nx
      // set v = v - (vn_l)ny
      // set w = w - (vn_l)nz
      for (int i=0; i<n_dims; i++)
        v_r[i] = v_l[i] - vn_l*norm[i];

      // extrapolate energy
      e_r = e_l;
    }

    // Boundary condition not implemented yet
    else
    {
      printf("bdy_type=%d\n",bdy_type);
      printf("Boundary conditions yet to be implemented");
    }

    // Conservative variables on right side
    u_r[0] = rho_r;
    for (int i=0; i<n_dims; i++)
      u_r[i+1] = rho_r*v_r[i];
    u_r[n_dims+1] = e_r;
  }

  // Advection, Advection-Diffusion Boundary Conditions
  if(equation==1)
  {
    // Trivial Dirichlet
    if(bdy_type==50)
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


template<int n_dims>
__device__ void inv_NS_flux(double* q, double* v_g, double *p, double* f, double gamma, int field, int turb_model)
{
  if(n_dims==2) {

    if (field==-1) {
      (*p) = (gamma-1.0)*(q[3]-0.5*(q[1]*q[1]+q[2]*q[2])/q[0]);
    }
    else if (field==0) {
      f[0] = q[1] - q[0]*v_g[0];
      f[1] = q[2] - q[0]*v_g[1];
    }
    else if (field==1) {
      f[0]  = (*p) + (q[1]*q[1]/q[0]) - q[1]*v_g[0];
      f[1]  = q[2]*q[1]/q[0]          - q[1]*v_g[1];
    }
    else if (field==2) {
      f[0]  = q[1]*q[2]/q[0]          - q[2]*v_g[0];
      f[1]  = (*p) + (q[2]*q[2]/q[0]) - q[2]*v_g[1];
    }
    else if (field==3) {
      f[0]  = q[1]/q[0]*(q[3]+(*p)) - q[3]*v_g[0];
      f[1]  = q[2]/q[0]*(q[3]+(*p)) - q[3]*v_g[1];
    }
    else if (field==4) {
      if (turb_model==1) {
        f[0] = q[4]*q[1]/q[0] - q[4]*v_g[0];
        f[1] = q[4]*q[2]/q[0] - q[4]*v_g[1];
      }
    }
  }
  else if(n_dims==3)
  {
    if (field==-1) {
      (*p) = (gamma-1.0)*(q[4]-0.5*(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/q[0]);
    }
    else if (field==0) {
      f[0] = q[1] - q[0]*v_g[0];
      f[1] = q[2] - q[0]*v_g[1];
      f[2] = q[3] - q[0]*v_g[2];
    }
    else if (field==1) {
      f[0] = (*p)+(q[1]*q[1]/q[0]) - q[1]*v_g[0];
      f[1] = q[2]*q[1]/q[0]        - q[1]*v_g[1];
      f[2] = q[3]*q[1]/q[0]        - q[1]*v_g[2];
    }
    else if (field==2) {
      f[0] = q[1]*q[2]/q[0]          - q[2]*v_g[0];
      f[1] = (*p) + (q[2]*q[2]/q[0]) - q[2]*v_g[1];
      f[2] = q[3]*q[2]/q[0]          - q[2]*v_g[2];
    }
    else if (field==3) {
      f[0] = q[1]*q[3]/q[0]          - q[3]*v_g[0];
      f[1] = q[2]*q[3]/q[0]          - q[3]*v_g[1];
      f[2] = (*p) + (q[3]*q[3]/q[0]) - q[3]*v_g[2];
    }
    else if (field==4) {
      f[0] = q[1]/q[0]*(q[4]+(*p)) - q[4]*v_g[0];
      f[1] = q[2]/q[0]*(q[4]+(*p)) - q[4]*v_g[1];
      f[2] = q[3]/q[0]*(q[4]+(*p)) - q[4]*v_g[2];
    }
    else if (field==5) {
      if (turb_model==1) {
        f[0] = q[5]*q[1]/q[0] - q[5]*v_g[0];
        f[1] = q[5]*q[2]/q[0] - q[5]*v_g[1];
        f[2] = q[5]*q[3]/q[0] - q[5]*v_g[2];
      }
    }
  }
}


template<int n_dims>
__device__ void vis_NS_flux(double* q, double* grad_q, double* grad_vel, double* grad_ene, double* stensor, double* f, double* inte, double* mu, double* mu_t, double prandtl, double gamma, double rt_inf, double mu_inf, double c_sth, double fix_vis, int field, int turb_model, double c_v1, double omega, double prandtl_t)
{
  double diag;
  double rt_ratio;

  if(n_dims==2) {

      if(field==-1) {

          // Internal energy
          (*inte) = (q[3]/q[0])-0.5*((q[1]*q[1]+q[2]*q[2])/(q[0]*q[0]));

          // Viscosity
          rt_ratio = (gamma-1.)*(*inte)/(rt_inf);
          (*mu) = mu_inf*pow(rt_ratio,1.5)*(1.+c_sth)/(rt_ratio+c_sth);
          (*mu) = (*mu) + fix_vis*(mu_inf - (*mu));

          // turbulent eddy viscosity
          if (turb_model==1) {
            double nu_tilde = q[4]/q[0];
            if (nu_tilde >= 0.0) {
              double f_v1 = pow(q[4]/(*mu), 3.0)/(pow(q[4]/(*mu), 3.0) + pow(c_v1, 3.0));
              (*mu_t) = q[4]*f_v1;
            }
            else {
              (*mu_t) = 0.0;
            }
          }
          else {
            (*mu_t) = 0.0;
          }

          // Velocity gradients
#pragma unroll
          for (int j=0;j<n_dims;j++)
            {
#pragma unroll
              for (int i=0;i<n_dims;i++)
                grad_vel[j*n_dims + i] = (grad_q[(j+1)*n_dims + i] - grad_q[0*n_dims + i]*q[j+1]/q[0])/q[0];
            }

          // Kinetic energy gradient
#pragma unroll
          for (int i=0;i<n_dims;i++)
            grad_ene[i] = 0.5*((q[1]*q[1]+q[2]*q[2])/(q[0]*q[0]))*grad_q[0*n_dims + i] + q[0]*((q[1]/q[0])*grad_vel[0*n_dims + i]+(q[2]/q[0])*grad_vel[1*n_dims + i]);

          // Total energy gradient
#pragma unroll
          for (int i=0;i<n_dims;i++)
            grad_ene[i] = (grad_q[3*n_dims + i] - grad_ene[i] - grad_q[0*n_dims + i]*(*inte))/q[0];

          diag = (grad_vel[0*n_dims + 0] + grad_vel[1*n_dims + 1])/3.0;

          // Stress tensor
#pragma unroll
          for (int i=0;i<n_dims;i++)
            stensor[i] = 2.0*((*mu)+(*mu_t))*(grad_vel[i*n_dims + i] - diag);

          stensor[2] = ((*mu)+(*mu_t))*(grad_vel[0*n_dims + 1] + grad_vel[1*n_dims + 0]);

        }
      else if (field==0) {
          f[0] = 0.0;
          f[1] = 0.0;
        }
      else if (field==1) {
          f[0]  = -stensor[0];
          f[1]  = -stensor[2];
        }
      else if (field==2) {
          f[0]  = -stensor[2];
          f[1]  = -stensor[1];
        }
      else if (field==3) {
          f[0]  = -((q[1]/q[0])*stensor[0] + (q[2]/q[0])*stensor[2] + ((*mu)/prandtl + (*mu_t)/prandtl_t)*gamma*grad_ene[0]);
          f[1]  = -((q[1]/q[0])*stensor[2] + (q[2]/q[0])*stensor[1] + ((*mu)/prandtl + (*mu_t)/prandtl_t)*gamma*grad_ene[1]);
        }
      else if (field==4) {
        if (turb_model==1) {
          double Chi, psi;
          Chi = q[4]/(*mu);
          if (Chi <= 10.0)
            psi = 0.05*log(1.0 + exp(20.0*Chi));
          else
            psi = Chi;

          double dnu_tilde_dx = (grad_q[4*n_dims + 0]-grad_q[0*n_dims + 0]*q[4]/q[0])/q[0];
          double dnu_tilde_dy = (grad_q[4*n_dims + 1]-grad_q[0*n_dims + 1]*q[4]/q[0])/q[0];

          f[0] = -(1.0/omega)*((*mu) + (*mu)*psi)*dnu_tilde_dx;
          f[1] = -(1.0/omega)*((*mu) + (*mu)*psi)*dnu_tilde_dy;
        }
      }
    }
  else if(n_dims==3)
    {
      if(field==-1) {

          // Internal energy
          (*inte) = (q[4]/q[0])-0.5*((q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/(q[0]*q[0]));

          // Viscosity
          rt_ratio = (gamma-1.)*(*inte)/(rt_inf);
          (*mu) = mu_inf*pow(rt_ratio,1.5)*(1.+c_sth)/(rt_ratio+c_sth);
          (*mu) = (*mu) + fix_vis*(mu_inf - (*mu));

          // turbulent eddy viscosity
          if (turb_model==1) {
            double nu_tilde = q[5]/q[0];
            if (nu_tilde >= 0.0) {
              double f_v1 = pow(q[5]/(*mu), 3.0)/(pow(q[5]/(*mu), 3.0) + pow(c_v1, 3.0));
              (*mu_t) = q[5]*f_v1;
            }
            else {
              (*mu_t) = 0.0;
            }
          }
          else {
            (*mu_t) = 0.0;
          }

          // Velocity gradients
#pragma unroll
          for (int j=0;j<n_dims;j++)
            {
#pragma unroll
              for (int i=0;i<n_dims;i++)
                grad_vel[j*n_dims + i] = (grad_q[(j+1)*n_dims + i] - grad_q[0*n_dims + i]*q[j+1]/q[0])/q[0];
            }

          // Kinetic energy gradient
#pragma unroll
          for (int i=0;i<n_dims;i++)
            grad_ene[i] = 0.5*((q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/(q[0]*q[0]))*grad_q[0*n_dims + i] + q[0]*((q[1]/q[0])*grad_vel[0*n_dims + i]+(q[2]/q[0])*grad_vel[1*n_dims + i]+(q[3]/q[0])*grad_vel[2*n_dims + i]);

          // Total energy gradient
#pragma unroll
          for (int i=0;i<n_dims;i++)
            grad_ene[i] = (grad_q[4*n_dims + i] - grad_ene[i] - grad_q[0*n_dims + i]*(*inte))/q[0];

          diag = (grad_vel[0*n_dims + 0] + grad_vel[1*n_dims + 1] + grad_vel[2*n_dims + 2])/3.0;

          // Stress tensor
#pragma unroll
          for (int i=0;i<n_dims;i++)
            stensor[i] = 2.0*((*mu)+(*mu_t))*(grad_vel[i*n_dims + i] - diag);

          stensor[3] = ((*mu)+(*mu_t))*(grad_vel[0*n_dims + 1] + grad_vel[1*n_dims + 0]);
          stensor[4] = ((*mu)+(*mu_t))*(grad_vel[0*n_dims + 2] + grad_vel[2*n_dims + 0]);
          stensor[5] = ((*mu)+(*mu_t))*(grad_vel[1*n_dims + 2] + grad_vel[2*n_dims + 1]);
        }
      else if (field==0) {
          f[0] = 0.0;
          f[1] = 0.0;
          f[2] = 0.0;
        }
      else if (field==1) {
          f[0]  = -stensor[0];
          f[1]  = -stensor[3];
          f[2]  = -stensor[4];
        }
      else if (field==2) {
          f[0] = -stensor[3];
          f[1] = -stensor[1];
          f[2] = -stensor[5];
        }
      else if (field==3) {
          f[0] = -stensor[4];
          f[1] = -stensor[5];
          f[2] = -stensor[2];
        }
      else if (field==4) {
          f[0] = -((q[1]/q[0])*stensor[0]+(q[2]/q[0])*stensor[3]+(q[3]/q[0])*stensor[4] + ((*mu)/prandtl + (*mu_t)/prandtl_t)*gamma*grad_ene[0]);
          f[1] = -((q[1]/q[0])*stensor[3]+(q[2]/q[0])*stensor[1]+(q[3]/q[0])*stensor[5] + ((*mu)/prandtl + (*mu_t)/prandtl_t)*gamma*grad_ene[1]);
          f[2] = -((q[1]/q[0])*stensor[4]+(q[2]/q[0])*stensor[5]+(q[3]/q[0])*stensor[2] + ((*mu)/prandtl + (*mu_t)/prandtl_t)*gamma*grad_ene[2]);
        }
      else if (field==5) {
        if (turb_model==1) {
          double Chi, psi;
          Chi = q[5]/(*mu);
          if (Chi <= 10.0)
            psi = 0.05*log(1.0 + exp(20.0*Chi));
          else
            psi = Chi;

          double dnu_tilde_dx = (grad_q[5*n_dims + 0]-grad_q[0*n_dims + 0]*q[5]/q[0])/q[0];
          double dnu_tilde_dy = (grad_q[5*n_dims + 1]-grad_q[0*n_dims + 1]*q[5]/q[0])/q[0];
          double dnu_tilde_dz = (grad_q[5*n_dims + 2]-grad_q[0*n_dims + 2]*q[5]/q[0])/q[0];

          f[0] = -(1.0/omega)*((*mu) + (*mu)*psi)*dnu_tilde_dx;
          f[1] = -(1.0/omega)*((*mu) + (*mu)*psi)*dnu_tilde_dy;
          f[2] = -(1.0/omega)*((*mu) + (*mu)*psi)*dnu_tilde_dz;
        }
      }
    }
}


template<int n_dims>
__device__ void calc_source_SA(double* in_u, double* in_grad_u, double* out_source, double d, double prandtl, double gamma, double rt_inf, double mu_inf, double c_sth, int fix_vis, double c_v1, double c_v2, double c_v3, double c_b1, double c_b2, double c_w2, double c_w3, double omega, double Kappa)
{
  if(n_dims == 2)
  {
    double rho, u, v, ene, nu_tilde;
    double dv_dx, du_dy, dnu_tilde_dx, dnu_tilde_dy;

    double inte, rt_ratio, mu;

    double nu_t_prod, nu_t_diff, nu_t_dest;

    double S, S_bar, S_tilde, Chi, psi, f_v1, f_v2;

    double c_w1, r, g, f_w;

    // primitive variables
    rho = in_u[0];
    u = in_u[1]/in_u[0];
    v = in_u[2]/in_u[0];
    ene = in_u[3];
    nu_tilde = in_u[4]/in_u[0];

    // gradients
    dv_dx = (in_grad_u[2*n_dims+0]-in_grad_u[0*n_dims+0]*v)/rho;
    du_dy = (in_grad_u[1*n_dims+1]-in_grad_u[0*n_dims+1]*u)/rho;

    dnu_tilde_dx = (in_grad_u[4*n_dims+0]-in_grad_u[0*n_dims+0]*nu_tilde)/rho;
    dnu_tilde_dy = (in_grad_u[4*n_dims+1]-in_grad_u[0*n_dims+1]*nu_tilde)/rho;

    // viscosity
    inte = ene/rho - 0.5*(u*u+v*v);
    rt_ratio = (gamma-1.0)*inte/(rt_inf);
    mu = (mu_inf)*pow(rt_ratio,1.5)*(1.+c_sth)/(rt_ratio+c_sth);
    mu = mu + fix_vis*(mu_inf - mu);

    // regulate eddy viscosity (must not become negative)
    Chi = in_u[4]/mu;
    if (Chi <= 10.0)
      psi = 0.05*log(1.0 + exp(20.0*Chi));
    else
      psi = Chi;

    // solve for production term for eddy viscosity
    // (solve for S = magnitude of vorticity)
    S = abs(dv_dx - du_dy);

    // (solve for S_bar)
    f_v1 = pow(in_u[4]/mu, 3.0)/(pow(in_u[4]/mu, 3.0) + pow(c_v1, 3.0));
    f_v2 = 1.0 - psi/(1.0 + psi*f_v1);
    S_bar = pow(mu*psi/rho, 2.0)*f_v2/(pow(Kappa, 2.0)*pow(d, 2.0));

    // (solve for S_tilde)
    if (S_bar >= -c_v2*S)
      S_tilde = S + S_bar;
    else
      S_tilde = S + S*(pow(c_v2, 2.0)*S + c_v3*S_bar)/((c_v3 - 2.0*c_v2)*S - S_bar);

    // (production term)
    nu_t_prod = c_b1*S_tilde*mu*psi;

    // solve for non-conservative diffusion term for eddy viscosity
    nu_t_diff = (1.0/omega)*(c_b2*rho*(pow(dnu_tilde_dx, 2.0)+pow(dnu_tilde_dy, 2.0)));

    // solve for destruction term for eddy viscosity
    c_w1 = c_b1/pow(Kappa, 2.0) + (1.0/omega)*(1.0 + c_b2);
    r = min((mu*psi/rho)/(S_tilde*pow(Kappa, 2.0)*pow(d, 2.0)), 10.0);
    g = r + c_w2*(pow(r, 6.0) - r);
    f_w = g*pow((1.0 + pow(c_w3, 6.0))/(pow(g, 6.0) + pow(c_w3, 6.0)), 1.0/6.0);

    // (destruction term)
    nu_t_dest = -c_w1*rho*f_w*pow((mu*psi/rho)/d, 2.0);

    // construct source term
    (*out_source) = nu_t_prod + nu_t_diff + nu_t_dest;
  }
  else if (n_dims == 3)
  {
    // 3d Source term not implemented yet!!!
  }
}


// Create rotation matrix from Cartesian to wall-aligned coords
template<int n_dims>
__device__ void rotation_matrix_kernel(double* norm, double* mrot)
{
  double nn;

  //cout << "norm "<< norm(0) << ", " << norm(1) << endl;

  if(n_dims==2) {
    if(abs(norm[1]) > 0.7) {
      mrot[0*n_dims+0] = norm[0];
      mrot[1*n_dims+0] = norm[1];
      mrot[0*n_dims+1] = norm[1];
      mrot[1*n_dims+1] = -norm[0];
    }
    else {
      mrot[0*n_dims+0] = -norm[0];
      mrot[1*n_dims+0] = -norm[1];
      mrot[0*n_dims+1] = norm[1];
      mrot[1*n_dims+1] = -norm[0];
    }
  }
  else if(n_dims==3) {
    if(abs(norm[2]) > 0.7) {
      nn = sqrt(norm[1]*norm[1]+norm[2]*norm[2]);

      mrot[0*n_dims+0] = norm[0]/nn;
      mrot[1*n_dims+0] = norm[1]/nn;
      mrot[2*n_dims+0] = norm[2]/nn;
      mrot[0*n_dims+1] = 0.0;
      mrot[1*n_dims+1] = -norm[2]/nn;
      mrot[2*n_dims+1] = norm[1]/nn;
      mrot[0*n_dims+2] = nn;
      mrot[1*n_dims+2] = -norm[0]*norm[1]/nn;
      mrot[2*n_dims+2] = -norm[0]*norm[2]/nn;
    }
    else {
      nn = sqrt(norm[0]*norm[0]+norm[1]*norm[1]);

      mrot[0*n_dims+0] = norm[0]/nn;
      mrot[1*n_dims+0] = norm[1]/nn;
      mrot[2*n_dims+0] = norm[2]/nn;
      mrot[0*n_dims+1] = norm[1]/nn;
      mrot[1*n_dims+1] = -norm[0]/nn;
      mrot[2*n_dims+1] = 0.0;
      mrot[0*n_dims+2] = norm[0]*norm[2]/nn;
      mrot[1*n_dims+2] = norm[1]*norm[2]/nn;
      mrot[2*n_dims+2] = -nn;
    }
  }
}

__device__ double wallfn_br(double yplus, double A, double B, double E, double kappa)
{
  double Rey;
  if     (yplus < 0.5)  Rey = yplus*yplus;
  else if(yplus > 30.0) Rey = yplus*log(E*yplus)/kappa;
  else                  Rey = yplus*(A*log(yplus)+B);

  return Rey;
}

__device__ double SGS_filter_width(double detjac, int ele_type, int n_dims, int order, double filter_ratio)
{
  // Define filter width by Deardorff's unstructured element method
  double delta, vol;

  if (ele_type==0) // triangle
  {
    vol = detjac*2.0;
  }
  else if (ele_type==1) // quads
  {
    vol = detjac*4.0;
  }
  else if (ele_type==2) // tets
  {
    vol = detjac*8.0/6.0;
  }
  else if (ele_type==4) // hexas
  {
    vol = detjac*8.0;
  }

  delta = filter_ratio*pow(vol,1./n_dims)/(order+1.);

  return delta;
}

/*! gpu kernel to calculate velocity and energy product terms for similarity model */
template<int n_fields>
__global__ void calc_similarity_terms_kernel(int n_upts_per_ele, int n_eles, int n_dims, double* disu_upts_ptr, double* uu_ptr, double* ue_ptr)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  int stride = n_upts_per_ele*n_eles;
  int i;
  double q[n_fields];
  double rsq;

   if(thread_id<n_upts_per_ele*n_eles) {

    // Solution
    #pragma unroll
    for (i=0;i<n_fields;i++) {
      q[i] = disu_upts_ptr[thread_id + i*stride];
    }

    rsq = q[0]*q[0];

    if(n_dims==2) {
      /*! velocity-velocity product */
      uu_ptr[thread_id + 0*stride] = q[1]*q[1]/rsq;
      uu_ptr[thread_id + 1*stride] = q[2]*q[2]/rsq;
      uu_ptr[thread_id + 2*stride] = q[1]*q[2]/rsq;

      /*! velocity-energy product */
      q[3] -= 0.5*(q[1]*q[1] + q[2]*q[2])/q[0]; // internal energy*rho

      ue_ptr[thread_id + 0*stride] = q[1]*q[3]/rsq; // subtract kinetic energy
      ue_ptr[thread_id + 1*stride] = q[2]*q[3]/rsq;
    }
    else if(n_dims==3) {
      /*! velocity-velocity product */
      uu_ptr[thread_id + 0*stride] = q[1]*q[1]/rsq;
      uu_ptr[thread_id + 1*stride] = q[2]*q[2]/rsq;
      uu_ptr[thread_id + 2*stride] = q[3]*q[3]/rsq;
      uu_ptr[thread_id + 3*stride] = q[1]*q[2]/rsq;
      uu_ptr[thread_id + 4*stride] = q[1]*q[3]/rsq;
      uu_ptr[thread_id + 5*stride] = q[2]*q[3]/rsq;

      /*! velocity-energy product */
      q[4] -= 0.5*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])/q[0]; // internal energy*rho

      ue_ptr[thread_id + 0*stride] = q[1]*q[4]/rsq; // subtract kinetic energy
      ue_ptr[thread_id + 1*stride] = q[2]*q[4]/rsq;
      ue_ptr[thread_id + 2*stride] = q[3]*q[4]/rsq;
    }
  }
}

/*! gpu kernel to calculate Leonard tensors for similarity model */
template<int n_fields>
__global__ void calc_Leonard_tensors_kernel(int n_upts_per_ele, int n_eles, int n_dims, double* disuf_upts_ptr, double* Lu_ptr, double* Le_ptr)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  int stride = n_upts_per_ele*n_eles;
  int i;
  double q[n_fields];
  double diag, rsq;

   if(thread_id<n_upts_per_ele*n_eles) {
    // filtered solution
    #pragma unroll
    for (i=0;i<n_fields;i++) {
      q[i] = disuf_upts_ptr[thread_id + i*stride];
    }

    rsq = q[0]*q[0];

    /*! subtract product of filtered solution terms from Leonard tensors */
    if(n_dims==2) {
      Lu_ptr[thread_id + 0*stride] = (Lu_ptr[thread_id + 0*stride] - q[1]*q[1])/rsq;
      Lu_ptr[thread_id + 1*stride] = (Lu_ptr[thread_id + 1*stride] - q[2]*q[2])/rsq;
      Lu_ptr[thread_id + 2*stride] = (Lu_ptr[thread_id + 2*stride] - q[1]*q[2])/rsq;

      diag = (Lu_ptr[thread_id + 0*stride] + Lu_ptr[thread_id + 1*stride])/3.0;

      q[3] -= 0.5*(q[1]*q[1] + q[2]*q[2])/q[0]; // internal energy*rho

      Le_ptr[thread_id + 0*stride] = (Le_ptr[thread_id + 0*stride] - q[1]*q[3])/rsq;
      Le_ptr[thread_id + 1*stride] = (Le_ptr[thread_id + 1*stride] - q[2]*q[3])/rsq;
    }
    else if(n_dims==3) {
      Lu_ptr[thread_id + 0*stride] = (Lu_ptr[thread_id + 0*stride] - q[1]*q[1])/rsq;
      Lu_ptr[thread_id + 1*stride] = (Lu_ptr[thread_id + 1*stride] - q[2]*q[2])/rsq;
      Lu_ptr[thread_id + 2*stride] = (Lu_ptr[thread_id + 2*stride] - q[3]*q[3])/rsq;
      Lu_ptr[thread_id + 3*stride] = (Lu_ptr[thread_id + 3*stride] - q[1]*q[2])/rsq;
      Lu_ptr[thread_id + 4*stride] = (Lu_ptr[thread_id + 4*stride] - q[1]*q[3])/rsq;
      Lu_ptr[thread_id + 5*stride] = (Lu_ptr[thread_id + 5*stride] - q[2]*q[3])/rsq;

      diag = (Lu_ptr[thread_id + 0*stride] + Lu_ptr[thread_id + 1*stride] + Lu_ptr[thread_id + 2*stride])/3.0;

      q[4] -= 0.5*(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])/q[0]; // internal energy*rho

      Le_ptr[thread_id + 0*stride] = (Le_ptr[thread_id + 0*stride] - q[1]*q[4])/rsq;
      Le_ptr[thread_id + 1*stride] = (Le_ptr[thread_id + 1*stride] - q[2]*q[4])/rsq;
      Le_ptr[thread_id + 2*stride] = (Le_ptr[thread_id + 2*stride] - q[3]*q[4])/rsq;
    }

    /*! subtract diagonal from Lu */
    #pragma unroll
    for (i=0;i<n_dims;++i) {
      Lu_ptr[thread_id + i*stride] -= diag;
    }
    // subtract diagonal from Le?
  }
}

template<int n_dims>
__device__ void wall_model_kernel(int wall_model, double rho, double* urot, double* inte, double* mu, double gamma, double prandtl, double y, double* tau_wall, double q_wall)
{
  double eps = 1.e-10;
  double Rey, Rey_c, u, uplus, utau, tw;
  double prandtl_t = 0.9;
  double ymatch = 11.8;
  int i;

  // Magnitude of surface velocity
  u = 0.0;
  #pragma unroll
  for(i=0;i<n_dims;++i) u += urot[i]*urot[i];

  u = sqrt(u);

  if(u > eps) {

    /*! Simple power-law wall model Werner and Wengle (1991)

              u+ = y+               for y+ < 11.8
              u+ = 8.3*(y+)^(1/7)   for y+ > 11.8
    */

    if(wall_model == 1) {

      Rey_c = ymatch*ymatch;
      Rey = rho*u*y/(*mu);

      if(Rey < Rey_c) uplus = sqrt(Rey);
      else            uplus = pow(8.3,0.875)*pow(Rey,0.125);

      utau = u/uplus;
      tw = rho*utau*utau;

      #pragma unroll
      for(i=0;i<n_dims;++i) tau_wall[i] = tw*urot[i]/u;

      // Wall heat flux
      if(Rey < Rey_c) q_wall = (*inte)*gamma*tw / (prandtl * u);
      else            q_wall = (*inte)*gamma*tw / (prandtl * (u + utau * sqrt(Rey_c) * (prandtl/prandtl_t-1.0)));
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
    else if(wall_model == 2) {

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
      phi = rho*y/(*mu);
      Rey0 = u*phi;
      utau = 0.0;

      #pragma unroll
      for (i=0;i<n_dims;i++)
        utau += tau_wall[i]*tau_wall[i];

      utau /= pow( (rho*rho), 0.25);
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

        #pragma unroll
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
      #pragma unroll
      for (i=0;i<n_dims;i++) tau_wall[i] = abs(tw*urot[i]/u);

      // Wall heat flux
      if(yplus <= ymatch) q_wall = (*inte)*gamma*tw / (prandtl * u);
      else                q_wall = (*inte)*gamma*tw / (prandtl * (u + utau * ymatch * (prandtl/prandtl_t-1.0)));
    }
  }

  // if velocity is 0
  else {
    #pragma unroll
    for (i=0;i<n_dims;i++) tau_wall[i] = 0.0;
    q_wall = 0.0;
  }
}

template<int n_dims>
__device__ void SGS_flux_kernel(double* q, double* grad_q, double* grad_vel, double* grad_ene, double* sdtensor, double* straintensor, double* Leonard_mom, double* Leonard_ene, double* f, int sgs_model, double delta, double gamma, int field)
{
  int i, j;
  int eddy, sim;
  double Cs, mu_t;
  double Smod=0.0;
  double prandtl_t=0.5; // turbulent Prandtl number
  double num=0.0;
  double denom=0.0;
  double diag=0.0;
  double eps=1.e-10;

  // Initialize SGS flux to 0
  #pragma unroll
  for (i=0;i<n_dims;i++)
    f[i] = 0.0;

  // Set flags depending on which SGS model we are using
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

  // Calculate eddy viscosity

  // Smagorinsky model
  if(sgs_model==0) {

    Cs=0.1;

    // Calculate modulus of strain rate tensor
    #pragma unroll
    for (i=0;i<n_dims;i++) {
      Smod += 2.0*straintensor[i]*straintensor[i];
    }

    // Now the off-diagonal components of strain tensor:
    if(n_dims==2) {
      Smod += 4.0*straintensor[2]*straintensor[2];
    }
    else if(n_dims==3) {
      Smod += 4.0*(straintensor[3]*straintensor[3] + straintensor[4]*straintensor[4] + straintensor[5]*straintensor[5]);
    }

    // Finally, the modulus of strain rate tensor
    Smod = sqrt(Smod);

    mu_t = q[0]*Cs*Cs*delta*delta*Smod;
  }

  // WALE or WSM model
  else if(sgs_model==1 || sgs_model==2) {

    Cs=0.5;

    // Square of velocity gradient tensor
    #pragma unroll
    for (i=0;i<n_dims;i++) {
      sdtensor[i] = 0.0;
      #pragma unroll
      for (j=0;j<n_dims;j++) {
        diag += grad_vel[i*n_dims + j]*grad_vel[j*n_dims + i]/3.0;
        sdtensor[i] += grad_vel[i*n_dims + j]*grad_vel[j*n_dims + i];
      }
    }

    // subtract trace from diagonal entries of tensor
    #pragma unroll
    for (i=0;i<n_dims;i++)
      sdtensor[i] -= diag;

    // off-diagonal terms of tensor
    if(n_dims==2) {
      sdtensor[2] = 0.0;
      #pragma unroll
      for (j=0;j<n_dims;j++) {
        sdtensor[2] += (grad_vel[0*n_dims + j]*grad_vel[j*n_dims + 1] + grad_vel[1*n_dims + j]*grad_vel[j*n_dims + 0])/2.0;
      }
    }
    else if(n_dims==3) {
      sdtensor[3] = 0.0;
      sdtensor[4] = 0.0;
      sdtensor[5] = 0.0;
      #pragma unroll
      for (j=0;j<n_dims;j++) {
        sdtensor[3] += (grad_vel[0*n_dims + j]*grad_vel[j*n_dims + 1] + grad_vel[1*n_dims + j]*grad_vel[j*n_dims + 0])/2.0;

        sdtensor[4] += (grad_vel[0*n_dims + j]*grad_vel[j*n_dims + 2] + grad_vel[2*n_dims + j]*grad_vel[j*n_dims + 0])/2.0;

        sdtensor[5] += (grad_vel[1*n_dims + j]*grad_vel[j*n_dims + 2] + grad_vel[2*n_dims + j]*grad_vel[j*n_dims + 1])/2.0;
      }
    }

    // numerator and denominator of eddy viscosity term
    #pragma unroll
    for (i=0;i<n_dims;i++) {
      num += sdtensor[i]*sdtensor[i];
      denom += straintensor[i]*straintensor[i];
    }

    if(n_dims==2) {
      num += 2.0*sdtensor[2]*sdtensor[2];
      denom += 2.0*straintensor[2]*straintensor[2];
    }
    else if(n_dims==3) {
      num += 2.0*(sdtensor[3]*sdtensor[3] + sdtensor[4]*sdtensor[4] + sdtensor[5]*sdtensor[5]);
      denom += 2.0*(straintensor[3]*straintensor[3] + straintensor[4]*straintensor[4] + straintensor[5]*straintensor[5]);
    }

    denom = pow(denom,2.5) + pow(num,1.25);
    num = pow(num,1.5);
    mu_t = q[0]*Cs*Cs*delta*delta*num/(denom+eps);
  }

  // Now set the flux values
  if (eddy==1) {
    if (n_dims==2) {

      // Density
      if (field==0) {
        f[0] = 0.0;
        f[1] = 0.0;
      }
      // u
      else if (field==1) {
        f[0] = -2.0*mu_t*straintensor[0];
        f[1] = -2.0*mu_t*straintensor[2];
      }
      // v
      else if (field==2) {
        f[0] = -2.0*mu_t*straintensor[2];
        f[1] = -2.0*mu_t*straintensor[1];
      }
      // Energy
      else if (field==3) {
        f[0] = -1.0*gamma*mu_t/prandtl_t*grad_ene[0];
        f[1] = -1.0*gamma*mu_t/prandtl_t*grad_ene[1];
      }
    }
    else if(n_dims==3) {

      // Density
      if (field==0) {
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
      }
      // u
      else if (field==1) {
        f[0] = -2.0*mu_t*straintensor[0];
        f[1] = -2.0*mu_t*straintensor[3];
        f[2] = -2.0*mu_t*straintensor[4];
      }
      // v
      else if (field==2) {
        f[0] = -2.0*mu_t*straintensor[3];
        f[1] = -2.0*mu_t*straintensor[1];
        f[2] = -2.0*mu_t*straintensor[5];
      }
      // w
      else if (field==3) {
        f[0] = -2.0*mu_t*straintensor[4];
        f[1] = -2.0*mu_t*straintensor[5];
        f[2] = -2.0*mu_t*straintensor[2];
      }
      // Energy
      else if (field==4) {
        f[0] = -1.0*gamma*mu_t/prandtl_t*grad_ene[0];
        f[1] = -1.0*gamma*mu_t/prandtl_t*grad_ene[1];
        f[2] = -1.0*gamma*mu_t/prandtl_t*grad_ene[2];
      }
    }
  }
  // Add similarity term to SGS fluxes if WSM or Similarity model
  if (sim==1)
  {
    if(n_dims==2) {

      // Density
      if (field==0) {
        f[0] += 0.0;
        f[1] += 0.0;
      }

      // u
      if (field==1) {
        f[0] += q[0]*Leonard_mom[0];
        f[1] += q[0]*Leonard_mom[2];
      }
      // v
      else if (field==2) {
        f[0] += q[0]*Leonard_mom[2];
        f[1] += q[0]*Leonard_mom[1];
      }
      // Energy
      else if (field==3) {
        f[0] += q[0]*gamma*Leonard_ene[0];
        f[1] += q[0]*gamma*Leonard_ene[1];
      }
    }
    else if(n_dims==3)
    {
      // u
      if (field==1) {
        f[0] += q[0]*Leonard_mom[0];
        f[1] += q[0]*Leonard_mom[3];
        f[2] += q[0]*Leonard_mom[4];
      }
      // v
      else if (field==2) {
        f[0] += q[0]*Leonard_mom[3];
        f[1] += q[0]*Leonard_mom[1];
        f[2] += q[0]*Leonard_mom[5];
      }
      // w
      else if (field==3) {
        f[0] += q[0]*Leonard_mom[4];
        f[1] += q[0]*Leonard_mom[5];
        f[2] += q[0]*Leonard_mom[2];
      }
      // Energy
      else if (field==4) {
        f[0] += q[0]*gamma*Leonard_ene[0];
        f[1] += q[0]*gamma*Leonard_ene[1];
        f[2] += q[0]*gamma*Leonard_ene[2];
      }
    }
  }
}

template<int n_dims>
__global__ void push_back_xv_kernel(int n_verts, double* xv_1, double* xv_2)
{
  const int iv = blockIdx.x*blockDim.x+threadIdx.x;
  int i;

  /// Taken from Kui, AIAA-2010-5031-661
#pragma unroll
  for(i=0;i<n_dims;i++) {
    xv_1[iv+i*n_verts] = xv_2[iv+i*n_verts];
  }
}

template<int n_dims>
__global__ void push_back_shape_dyn_kernel(int n_eles, int max_n_spts_per_ele, int n_levels, int* n_spts_per_ele, double* shape_dyn)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = n_eles*max_n_spts_per_ele;
  int ele = thread_id/max_n_spts_per_ele;
  int spt = thread_id%max_n_spts_per_ele;
  int n_spts;
  int i,k;

  if(thread_id<stride) {
    n_spts = n_spts_per_ele[ele];
    if (spt < n_spts) {
#pragma unroll
      for(i=n_levels-1; i>0; i--) {
        for(k=0; k<n_dims; k++) {
          shape_dyn[k+n_dims*(spt+max_n_spts_per_ele*(ele+n_eles*i))] = shape_dyn[k+n_dims*(spt+max_n_spts_per_ele*(ele+n_eles*(i-1)))];
        }
      }
    }
  }
}

template<int n_dims>
__global__ void calc_grid_vel_spts_kernel(int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* xv, double* grid_vel_spts, double dt)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = n_eles * max_n_spts_per_ele;
  int ele = thread_id/max_n_spts_per_ele;
  int spt = thread_id%max_n_spts_per_ele;
  int n_spts;
  int k;
  int i0, i1, i2, i3, i4;

  if (thread_id<stride) {
    n_spts = n_spts_per_ele[ele];
    if (spt<n_spts) {
#pragma unroll
      for(k=0; k<n_dims; k++) {
        i0 = k+n_dims*(spt+max_n_spts_per_ele*(ele+n_eles*0)); i1 = k+n_dims*(spt+max_n_spts_per_ele*(ele+n_eles*1));
        i2 = k+n_dims*(spt+max_n_spts_per_ele*(ele+n_eles*2)); i3 = k+n_dims*(spt+max_n_spts_per_ele*(ele+n_eles*3));
        i4 = k+n_dims*(spt+max_n_spts_per_ele*(ele+n_eles*4));
        grid_vel_spts[i0] = 25/12*xv[i0] - 4*xv[i1] + 3*xv[i2] - 4/3*xv[i3] + 1/4*xv[i4];
        grid_vel_spts[i0] /= dt;
      }
    }
  }
}

/* Interpolate the grid velocity at the shape points to the solution or flux points */
__global__ void eval_grid_vel_pts_kernel(int n_dims, int n_eles, int n_pts_per_ele, int max_n_spts_per_ele, int* n_spts_per_ele, double* nodal_s_basis_pts, double* grid_vel_spts, double* grid_vel_pts)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_eles*n_pts_per_ele;
  int ele = thread_id/n_pts_per_ele;
  int pt = thread_id%n_pts_per_ele;
  int n_spts;
  int j,k;

  if (thread_id<stride && pt<n_pts_per_ele) {
    n_spts = n_spts_per_ele[ele];
    for(k=0;k<n_dims;k++) {
      grid_vel_pts[thread_id+stride*k] = 0.0;
      for(j=0;j<n_spts;j++) {
        grid_vel_pts[thread_id+stride*k] += nodal_s_basis_pts[j+max_n_spts_per_ele*(pt+n_pts_per_ele*ele)]*grid_vel_spts[k+n_dims*(j+max_n_spts_per_ele*ele)];
      }
    }
  }
}

__global__ void rigid_motion_kernel(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* shape_dyn, double* motion_params, double rk_time)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = n_eles*max_n_spts_per_ele;
  int ele = thread_id/max_n_spts_per_ele;
  int spt = thread_id%max_n_spts_per_ele;
  int n_spts, j;
  double xv0;

  if (thread_id<stride) {
    n_spts = n_spts_per_ele[ele];
    if (spt<n_spts) {
      for (j=0; j<n_dims; j++) {
        xv0 = shape[j+(n_dims*(spt+max_n_spts_per_ele*ele))];

        shape_dyn[j+(n_dims*(spt+max_n_spts_per_ele*ele))] = xv0 + motion_params[2*j  ]*sin(motion_params[6+j]*rk_time);
        shape_dyn[j+(n_dims*(spt+max_n_spts_per_ele*ele))]+=       motion_params[2*j+1]*cos(motion_params[6+j]*rk_time);
      }
    }
  }
}

__global__ void rigid_motion_velocity_spts_kernel(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* motion_params, double* grid_vel, double rk_time)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = n_eles*max_n_spts_per_ele;
  int ele = thread_id/max_n_spts_per_ele;
  int spt = thread_id%max_n_spts_per_ele;
  int n_spts, j;

  if (thread_id<stride) {
    n_spts = n_spts_per_ele[ele];
    if (spt<n_spts) {
      for (j=0; j<n_dims; j++) {
        grid_vel[j+(n_dims*(spt+max_n_spts_per_ele*ele))] = motion_params[2*j  ]*motion_params[6+j]*cos(motion_params[6+j]*rk_time);
        grid_vel[j+(n_dims*(spt+max_n_spts_per_ele*ele))]+= motion_params[2*j+1]*motion_params[6+j]*sin(motion_params[6+j]*rk_time);
      }
    }
  }
}

__global__ void perturb_grid_velocity_spts_kernel(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* grid_vel, double rk_time)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = n_eles*max_n_spts_per_ele;
  int ele = thread_id/max_n_spts_per_ele;
  int spt = thread_id%max_n_spts_per_ele;
  int n_spts;
  double x0,y0,z0;

  if (thread_id<stride) {
    n_spts = n_spts_per_ele[ele];
    if (spt<n_spts) {
      if (n_dims==2) {
        x0 = shape[0+(n_dims*(spt+max_n_spts_per_ele*ele))];
        y0 = shape[1+(n_dims*(spt+max_n_spts_per_ele*ele))];
        grid_vel[0+(n_dims*(spt+max_n_spts_per_ele*ele))] = 4*PI/10*sin(PI*x0/10)*sin(PI*y0/10)*cos(2*PI*rk_time/10);
        grid_vel[1+(n_dims*(spt+max_n_spts_per_ele*ele))] = 4*PI/10*sin(PI*x0/10)*sin(PI*y0/10)*cos(2*PI*rk_time/10);
      }
      else if (n_dims==3) {
        x0 = shape[0+(n_dims*(spt+max_n_spts_per_ele*ele))];
        y0 = shape[1+(n_dims*(spt+max_n_spts_per_ele*ele))];
        z0 = shape[2+(n_dims*(spt+max_n_spts_per_ele*ele))];
        grid_vel[0+(n_dims*(spt+max_n_spts_per_ele*ele))] = 4*PI/10*sin(PI*x0/10)*sin(PI*y0/10)*sin(PI*z0/10)*cos(2*PI*rk_time/10);
        grid_vel[1+(n_dims*(spt+max_n_spts_per_ele*ele))] = 4*PI/10*sin(PI*x0/10)*sin(PI*y0/10)*sin(PI*z0/10)*cos(2*PI*rk_time/10);
        grid_vel[2+(n_dims*(spt+max_n_spts_per_ele*ele))] = 4*PI/10*sin(PI*x0/10)*sin(PI*y0/10)*sin(PI*z0/10)*cos(2*PI*rk_time/10);
      }
    }
  }
}

__global__ void perturb_shape_kernel(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* shape_dyn, double rk_time)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = n_eles*max_n_spts_per_ele;
  int ele = thread_id/max_n_spts_per_ele;
  int spt = thread_id%max_n_spts_per_ele;
  int n_spts;
  double x,y,z;

  if (thread_id<stride) {
    n_spts = n_spts_per_ele[ele];
    if (spt<n_spts) {
    /// Taken from Kui, AIAA-2010-5031-661
      if (n_dims==2) {
        x = shape[0+(n_dims*(spt+max_n_spts_per_ele*ele))];
        y = shape[1+(n_dims*(spt+max_n_spts_per_ele*ele))];
        shape_dyn[0+(n_dims*(spt+max_n_spts_per_ele*ele))] = x + 2*sin(PI*x/10)*sin(PI*y/10)*sin(2*PI*rk_time/10);
        shape_dyn[1+(n_dims*(spt+max_n_spts_per_ele*ele))] = y + 2*sin(PI*x/10)*sin(PI*y/10)*sin(2*PI*rk_time/10);
      }else if (n_dims==3) {
        x = shape[0+(n_dims*(spt+max_n_spts_per_ele*ele))];
        y = shape[1+(n_dims*(spt+max_n_spts_per_ele*ele))];
        z = shape[2+(n_dims*(spt+max_n_spts_per_ele*ele))];
        shape_dyn[0+(n_dims*(spt+max_n_spts_per_ele*ele))] = x + 2*sin(PI*x/10)*sin(PI*y/10)*sin(PI*z/10)*sin(2*PI*rk_time/10);
        shape_dyn[1+(n_dims*(spt+max_n_spts_per_ele*ele))] = y + 2*sin(PI*x/10)*sin(PI*y/10)*sin(PI*z/10)*sin(2*PI*rk_time/10);
        shape_dyn[2+(n_dims*(spt+max_n_spts_per_ele*ele))] = z + 2*sin(PI*x/10)*sin(PI*y/10)*sin(PI*z/10)*sin(2*PI*rk_time/10);
      }
    }
  }
}

//template<int n_dims>
//__global__ void perturb_shape_points_gpu_kernel(int n_verts, double* xv, double* xv_0, double rk_time)
//{
//  const int iv = blockIdx.x*blockDim.x+threadIdx.x;
//  int i;

//  /// Taken from Kui, AIAA-2010-5031-661
//  if (n_dims==2) {
//    for(i=0;i<2;i++) {
//      xv[i*n_verts+iv] = xv_0[i*n_verts+iv] + 2*sin(PI*xv_0[iv]/10)*sin(PI*xv_0[iv+n_verts]/10)*sin(2*PI*rk_time/10);
//    }
//  }else if (n_dims==3) {
//    for(i=0;i<3;i++) {
//      xv[iv+i*n_verts] = xv_0[iv+i*n_verts] + 2*sin(PI*xv_0[iv]/10)*sin(PI*xv_0[iv+n_verts]/10)*sin(PI*xv_0[iv+2*n_verts]/10)*sin(2*PI*rk_time/10);
//    }
//  }
//}

/**
 * GPU Kernel to calculate derivative of dynamic physical position wrt static/reference physical position at fpt
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
 */
template<int n_dims>
__device__ void calc_d_pos_dyn_kernel(int n_pts_per_ele, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* detjac_pts, double* JGinv_pts, double* d_nodal_s_basis_pts, double* shape_dyn, double *&out_d_pos)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  int stride = n_pts_per_ele*n_eles;
  int i,j,k;
  double dxdr[n_dims][n_dims];

  int ele = thread_id/n_pts_per_ele;
  int pt = thread_id%n_pts_per_ele;
  int n_spts;

  if(thread_id<stride) {
    n_spts = n_spts_per_ele[ele]; // access only after determined ele<n_eles
    // Calculate dx/dr
    #pragma unroll
    for(i=0; i<n_dims; i++) {
      #pragma unroll
      for(j=0; j<n_dims; j++) {
        dxdr[i][j] = 0.;
        #pragma unroll
        for(k=0; k<n_spts; k++) {
          dxdr[i][j] += shape_dyn[i+(n_dims*(k+max_n_spts_per_ele*ele))]*d_nodal_s_basis_pts[j+(n_dims*(k+max_n_spts_per_ele*(pt+n_pts_per_ele*ele)))];
        }
      }
    }

    // Calculate dx/dX using transformation matrix
    #pragma unroll
    for(i=0; i<n_dims; i++) {
      #pragma unroll
      for(j=0; j<n_dims; j++) {
        out_d_pos[i+j*n_dims] = 0.;
        #pragma unroll
        for(k=0; k<n_dims; k++) {
          out_d_pos[i+j*n_dims] += dxdr[i][k]*JGinv_pts[k+n_dims*(j+n_dims*(pt+n_pts_per_ele*ele))];
        }
      }
    }
  }
}

/*! gpu kernel to update coordiante transformation variables for moving grids */
template<int n_dims>
__global__ void set_transforms_dynamic_upts_kernel(int n_upts_per_ele, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_upts, double* J_dyn_upts, double* JGinv_upts, double* JGinv_dyn_upts, double* d_nodal_s_basis_upts, double* shape_dyn)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  int stride = n_upts_per_ele*n_eles;
  double d_pos[n_dims][n_dims];

  double xr, xs, xt;
  double yr, ys, yt;
  double zr, zs, zt;

  double dxdr[n_dims][n_dims];

  int i,j,k;
  int ele = thread_id/n_upts_per_ele;
  int upt = thread_id%n_upts_per_ele;
  int n_spts;

  if(thread_id<stride) {

    /**
    J_dyn_upts(n_upts_per_ele,n_eles): Determinant of the dynamic -> static reference transformation matrix ( |G| )
    JGinv_dyn_upts(n_dims,n_dims,n_upts_per_ele,n_eles): Total dynamic -> static reference transformation matrix ( |G|*G^{-1} )
    dyn_pos_upts(n_upts_per_ele,n_eles,n_dims): Physical position of solution points */

    n_spts = n_spts_per_ele[ele];

    // calculate first derivatives of shape functions at the solution point
    // First calculate with respect to computational coordinates
#pragma unroll
    for(i=0; i<n_dims; i++) {
#pragma unroll
      for(j=0; j<n_dims; j++) {
        dxdr[i][j] = 0.;
#pragma unroll
        for(k=0; k<n_spts; k++) {
          dxdr[i][j] += shape_dyn[i+(n_dims*(k+max_n_spts_per_ele*ele))]*d_nodal_s_basis_upts[j+(n_dims*(k+max_n_spts_per_ele*(upt+n_upts_per_ele*ele)))];
        }
      }
    }

    // Next transformat to static-phsycial derivatives (Calculate dx/dX) using transformation matrix
#pragma unroll
    for(i=0; i<n_dims; i++) {
#pragma unroll
      for(j=0; j<n_dims; j++) {
        d_pos[i][j] = 0.;
#pragma unroll
        for(k=0; k<n_dims; k++) {
          d_pos[i][j] += dxdr[i][k]*JGinv_upts[k+n_dims*(j+n_dims*(upt+n_upts_per_ele*ele))];
        }
      }
    }

    if(n_dims==2)
    {
      xr = d_pos[0][0]/J_upts[thread_id];
      xs = d_pos[0][1]/J_upts[thread_id];

      yr = d_pos[1][0]/J_upts[thread_id];
      ys = d_pos[1][1]/J_upts[thread_id];

      // store determinant of jacobian at solution point
      J_dyn_upts[thread_id]= xr*ys - xs*yr;

      //if (J_dyn_upts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_upts[0+n_dims*(0+n_dims*(upt+n_upts_per_ele*ele))] =  ys;
      JGinv_dyn_upts[0+n_dims*(1+n_dims*(upt+n_upts_per_ele*ele))] = -xs;
      JGinv_dyn_upts[1+n_dims*(0+n_dims*(upt+n_upts_per_ele*ele))] = -yr;
      JGinv_dyn_upts[1+n_dims*(1+n_dims*(upt+n_upts_per_ele*ele))] =  xr;
    }
    else if(n_dims==3)
    {
      xr = d_pos[0][0];
      xs = d_pos[0][1];
      xt = d_pos[0][2];

      yr = d_pos[1][0];
      ys = d_pos[1][1];
      yt = d_pos[1][2];

      zr = d_pos[2][0];
      zs = d_pos[2][1];
      zt = d_pos[2][2];

      // store determinant of jacobian at solution point
      J_dyn_upts[thread_id] = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

      //if (J_dyn_upts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_upts[0+n_dims*(0+n_dims*(upt+n_upts_per_ele*ele))] = (ys*zt - yt*zs);
      JGinv_dyn_upts[0+n_dims*(1+n_dims*(upt+n_upts_per_ele*ele))] = (xt*zs - xs*zt);
      JGinv_dyn_upts[0+n_dims*(2+n_dims*(upt+n_upts_per_ele*ele))] = (xs*yt - xt*ys);
      JGinv_dyn_upts[1+n_dims*(0+n_dims*(upt+n_upts_per_ele*ele))] = (yt*zr - yr*zt);
      JGinv_dyn_upts[1+n_dims*(1+n_dims*(upt+n_upts_per_ele*ele))] = (xr*zt - xt*zr);
      JGinv_dyn_upts[1+n_dims*(2+n_dims*(upt+n_upts_per_ele*ele))] = (xt*yr - xr*yt);
      JGinv_dyn_upts[2+n_dims*(0+n_dims*(upt+n_upts_per_ele*ele))] = (yr*zs - ys*zr);
      JGinv_dyn_upts[2+n_dims*(1+n_dims*(upt+n_upts_per_ele*ele))] = (xs*zr - xr*zs);
      JGinv_dyn_upts[2+n_dims*(2+n_dims*(upt+n_upts_per_ele*ele))] = (xr*ys - xs*yr);
    }
  }
}

/*! gpu kernel to update coordiante transformation variables for moving grids */
template<int n_dims>
__global__ void set_transforms_dynamic_fpts_kernel(int n_fpts_per_ele, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_fpts, double* J_dyn_fpts, double* JGinv_fpts, double* JGinv_dyn_fpts, double* tdA_dyn_fpts, double* norm_fpts, double* norm_dyn_fpts, double* d_nodal_s_basis_fpts, double* shape_dyn)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  int stride = n_fpts_per_ele*n_eles;
  double d_pos[n_dims][n_dims];
  double norm[n_dims];

  double xr, xs, xt;
  double yr, ys, yt;
  double zr, zs, zt;

  double dxdr[n_dims][n_dims];

  int i,j,k;
  int ele = thread_id/n_fpts_per_ele;
  int fpt = thread_id%n_fpts_per_ele;
  int n_spts;

  if(thread_id<stride) {

    // calculate first derivatives of shape functions at the solution point
    n_spts = n_spts_per_ele[ele];
#pragma unroll
    for(i=0; i<n_dims; i++) {
#pragma unroll
      for(j=0; j<n_dims; j++) {
        dxdr[i][j] = 0.;
#pragma unroll
        for(k=0; k<n_spts; k++) {
          dxdr[i][j] += shape_dyn[i+(n_dims*(k+max_n_spts_per_ele*ele))]*d_nodal_s_basis_fpts[j+(n_dims*(k+max_n_spts_per_ele*(fpt+n_fpts_per_ele*ele)))];
        }
      }
    }

    // Calculate dx/dX using transformation matrix
#pragma unroll
    for(i=0; i<n_dims; i++) {
#pragma unroll
      for(j=0; j<n_dims; j++) {
        d_pos[i][j] = 0.;
#pragma unroll
        for(k=0; k<n_dims; k++) {
          d_pos[i][j] += dxdr[i][k]*JGinv_fpts[k+n_dims*(j+n_dims*(fpt+n_fpts_per_ele*ele))];
        }
      }
    }

    if(n_dims==2)
    {

      xr = d_pos[0][0]/J_fpts[thread_id];
      xs = d_pos[0][1]/J_fpts[thread_id];

      yr = d_pos[1][0]/J_fpts[thread_id];
      ys = d_pos[1][1]/J_fpts[thread_id];

      // store determinant of jacobian at solution point
      J_dyn_fpts[thread_id]= xr*ys - xs*yr;

//      if (J_dyn_fpts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_fpts[0+n_dims*(0+n_dims*(fpt+n_fpts_per_ele*ele))] =  ys;
      JGinv_dyn_fpts[0+n_dims*(1+n_dims*(fpt+n_fpts_per_ele*ele))] = -xs;
      JGinv_dyn_fpts[1+n_dims*(0+n_dims*(fpt+n_fpts_per_ele*ele))] = -yr;
      JGinv_dyn_fpts[1+n_dims*(1+n_dims*(fpt+n_fpts_per_ele*ele))] =  xr;

      // temporarily store unnormalized transformed normal
      norm[0]= ( norm_fpts[thread_id]*ys -norm_fpts[thread_id+stride]*yr);
      norm[1]= (-norm_fpts[thread_id]*xs +norm_fpts[thread_id+stride]*xr);

      // store magnitude of transformed normal
      tdA_dyn_fpts[thread_id]=sqrt(norm[0]*norm[0]+norm[1]*norm[1]);

      // store normal at flux point
      norm_dyn_fpts[thread_id]       =norm[0]/tdA_dyn_fpts[thread_id];
      norm_dyn_fpts[thread_id+stride]=norm[1]/tdA_dyn_fpts[thread_id];
    }
    else if(n_dims==3)
    {
      xr = d_pos[0][0];
      xs = d_pos[0][1];
      xt = d_pos[0][2];

      yr = d_pos[1][0];
      ys = d_pos[1][1];
      yt = d_pos[1][2];

      zr = d_pos[2][0];
      zs = d_pos[2][1];
      zt = d_pos[2][2];

      // store determinant of jacobian at solution point
      J_dyn_fpts[thread_id] = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

      //if (J_dyn_fpts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_fpts[0+n_dims*(0+n_dims*(fpt+n_fpts_per_ele*ele))] = (ys*zt - yt*zs);
      JGinv_dyn_fpts[0+n_dims*(1+n_dims*(fpt+n_fpts_per_ele*ele))] = (xt*zs - xs*zt);
      JGinv_dyn_fpts[0+n_dims*(2+n_dims*(fpt+n_fpts_per_ele*ele))] = (xs*yt - xt*ys);
      JGinv_dyn_fpts[1+n_dims*(0+n_dims*(fpt+n_fpts_per_ele*ele))] = (yt*zr - yr*zt);
      JGinv_dyn_fpts[1+n_dims*(1+n_dims*(fpt+n_fpts_per_ele*ele))] = (xr*zt - xt*zr);
      JGinv_dyn_fpts[1+n_dims*(2+n_dims*(fpt+n_fpts_per_ele*ele))] = (xt*yr - xr*yt);
      JGinv_dyn_fpts[2+n_dims*(0+n_dims*(fpt+n_fpts_per_ele*ele))] = (yr*zs - ys*zr);
      JGinv_dyn_fpts[2+n_dims*(1+n_dims*(fpt+n_fpts_per_ele*ele))] = (xs*zr - xr*zs);
      JGinv_dyn_fpts[2+n_dims*(2+n_dims*(fpt+n_fpts_per_ele*ele))] = (xr*ys - xs*yr);

      // temporarily store moving-physical domain interface normal at the flux point
      norm[0]=((norm_fpts[thread_id]*(ys*zt-yt*zs))+(norm_fpts[thread_id+stride]*(yt*zr-yr*zt))+(norm_fpts[thread_id+2*stride]*(yr*zs-ys*zr)));
      norm[1]=((norm_fpts[thread_id]*(xt*zs-xs*zt))+(norm_fpts[thread_id+stride]*(xr*zt-xt*zr))+(norm_fpts[thread_id+2*stride]*(xs*zr-xr*zs)));
      norm[2]=((norm_fpts[thread_id]*(xs*yt-xt*ys))+(norm_fpts[thread_id+stride]*(xt*yr-xr*yt))+(norm_fpts[thread_id+2*stride]*(xr*ys-xs*yr)));

      // store magnitude of transformed normal
      tdA_dyn_fpts[thread_id]=sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2]);

      // store normal at flux point
      norm_dyn_fpts[thread_id         ]=norm[0]/tdA_dyn_fpts[thread_id];
      norm_dyn_fpts[thread_id  +stride]=norm[1]/tdA_dyn_fpts[thread_id];
      norm_dyn_fpts[thread_id+2*stride]=norm[2]/tdA_dyn_fpts[thread_id];
    }
  }
}

template<int n_fields, int n_dims>
__device__ __host__ void rusanov_flux(double* q_l, double *q_r, double* v_g, double *norm, double *fn, double gamma, int turb_model)
{
  double vn_l, vn_r;
  double vn_av_mag, c_av, vn_g;
  double p_l, p_r,f_l,f_r;

  double f[n_dims];

  // Compute normal velocity
  vn_l = 0.;
  vn_r = 0.;
  vn_g = 0.;
#pragma unroll
  for (int i=0;i<n_dims;i++) {
      vn_l += q_l[i+1]/q_l[0]*norm[i];
      vn_r += q_r[i+1]/q_r[0]*norm[i];
      vn_g += v_g[i]*norm[i];
    }

  // Flux prep
  inv_NS_flux<n_dims>(q_l,v_g,&p_l,f,gamma,-1,turb_model);
  inv_NS_flux<n_dims>(q_r,v_g,&p_r,f,gamma,-1,turb_model);

  vn_av_mag=0.5*fabs(vn_l+vn_r);
  c_av=sqrt((gamma*(p_l+p_r))/(q_l[0]+q_r[0]));

#pragma unroll
  for (int i=0;i<n_fields;i++)
    {
      // Left normal flux
      inv_NS_flux<n_dims>(q_l,v_g,&p_l,f,gamma,i,turb_model);

      f_l = f[0]*norm[0] + f[1]*norm[1];
      if(n_dims==3)
        f_l += f[2]*norm[2];

      // Right normal flux
      inv_NS_flux<n_dims>(q_r,v_g,&p_r,f,gamma,i,turb_model);

      f_r = f[0]*norm[0] + f[1]*norm[1];
      if(n_dims==3)
        f_r += f[2]*norm[2];

      // Common normal flux
      fn[i] = 0.5*(f_l+f_r) - 0.5*fabs(vn_av_mag-vn_g+c_av)*(q_r[i]-q_l[i]);
    }
}

template<int n_fields, int n_dims>
__device__ __host__ void convective_flux_boundary(double* q_l, double *q_r, double* v_g, double *norm, double *fn, double gamma, int turb_model)
{
  double vn_l, vn_r;
  double p_l, p_r,f_l,f_r;

  double f[n_dims];

  // Compute normal velocity
  vn_l = 0.;
  vn_r = 0.;
#pragma unroll
  for (int i=0;i<n_dims;i++) {
      vn_l += q_l[i+1]/q_l[0]*norm[i];
      vn_r += q_r[i+1]/q_r[0]*norm[i];
    }

  // Flux prep
  inv_NS_flux<n_dims>(q_l,v_g,&p_l,f,gamma,-1,turb_model);
  inv_NS_flux<n_dims>(q_r,v_g,&p_r,f,gamma,-1,turb_model);

#pragma unroll
  for (int i=0;i<n_fields;i++)
    {
      // Left normal flux
      inv_NS_flux<n_dims>(q_l,v_g,&p_l,f,gamma,i,turb_model);

      f_l = f[0]*norm[0] + f[1]*norm[1];
      if(n_dims==3)
        f_l += f[2]*norm[2];

      // Right normal flux
      inv_NS_flux<n_dims>(q_r,v_g,&p_r,f,gamma,i,turb_model);

      f_r = f[0]*norm[0] + f[1]*norm[1];
      if(n_dims==3)
        f_r += f[2]*norm[2];

      // Common normal flux
      fn[i] = 0.5*(f_l+f_r);  // Taking a purely convective flux without diffusive terms
    }
}



template<int n_fields, int n_dims>
__device__ __host__ void right_flux(double *q_r, double *norm, double *fn, double gamma, int turb_model)
{

  double p_r,f_r;
  double f[n_dims];
  double v_g[n_dims];

  // WARNING: right_flux never used, so not going to bother finishing this
#pragma unroll
  for (int i=0; i<n_dims; i++)
    v_g[i] = 0.;

  // Flux prep
  inv_NS_flux<n_dims>(q_r,v_g,&p_r,f,gamma,-1,turb_model);

#pragma unroll
  for (int i=0;i<n_fields;i++)
    {
      //Right normal flux
      inv_NS_flux<n_dims>(q_r,v_g,&p_r,f,gamma,i,turb_model);

      f_r = f[0]*norm[0] + f[1]*norm[1];
      if(n_dims==3)
        f_r += f[2]*norm[2];

      fn[i] = f_r;
    }
}


template<int n_fields, int n_dims>
__device__ __host__ void roe_flux(double* u_l, double* v_g, double *u_r, double *norm, double *fn, double gamma)
{
  double p_l,p_r;
  double h_l, h_r;
  double sq_rho,rrho,hm,usq,am,am_sq,unm,vgn;
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
      p_l=(gamma-1.0)*(u_l[3]-(0.5*u_l[0]*((v_l[0]*v_l[0])+(v_l[1]*v_l[1]))));
      p_r=(gamma-1.0)*(u_r[3]-(0.5*u_r[0]*((v_r[0]*v_r[0])+(v_r[1]*v_r[1]))));
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

  am_sq   = (gamma-1.)*(hm-usq);
  am  = sqrt(am_sq);

  unm = 0.;
  vgn = 0.;
#pragma unroll
  for (int i=0;i<n_dims;i++) {
    unm += um[i]*norm[i];
    vgn += v_g[i]*norm[i];
  }

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

  lambda0 = abs(unm-vgn);
  lambdaP = abs(unm-vgn+am);
  lambdaM = abs(unm-vgn-am);

  // Entropy fix
  eps = 0.5*(abs(rhoun_l/u_l[0]-rhoun_r/u_r[0])+ abs(sqrt(gamma*p_l/u_l[0])-sqrt(gamma*p_r/u_r[0])));
  if(lambda0 < 2.*eps)
    lambda0 = 0.25*lambda0*lambda0/eps + eps;
  if(lambdaP < 2.*eps)
    lambdaP = 0.25*lambdaP*lambdaP/eps + eps;
  if(lambdaM < 2.*eps)
    lambdaM = 0.25*lambdaM*lambdaM/eps + eps;


  a2 = 0.5*(lambdaP+lambdaM)-lambda0;
  a3 = 0.5*(lambdaP-lambdaM)/am;
  a1 = a2*(gamma-1.)/am_sq;
  a4 = a3*(gamma-1.);

  if (n_dims==2)
    {
      a5 = usq*du[0]-um[0]*du[1]-um[1]*du[2]+du[3];
      a6 = unm*du[0]-norm[0]*du[1]-norm[1]*du[2];
    }
  else if (n_dims==3)
    {
      a5 = usq*du[0]-um[0]*du[1]-um[1]*du[2]-um[2]*du[3]+du[4];
      a6 = unm*du[0]-norm[0]*du[1]-norm[1]*du[2]-norm[2]*du[3];
    }

  aL1 = a1*a5 - a3*a6;
  bL1 = a4*a5 - a2*a6;

  // Compute Euler flux (second part)
  if (n_dims==2)
    {
      fn[0] = fn[0] - (lambda0*du[0]+aL1);
      fn[1] = fn[1] - (lambda0*du[1]+aL1*um[0]+bL1*norm[0]);
      fn[2] = fn[2] - (lambda0*du[2]+aL1*um[1]+bL1*norm[1]);
      fn[3] = fn[3] - (lambda0*du[3]+aL1*hm   +bL1*unm);
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
      fn[i] =  0.5*(fn[i] - vgn*(u_l[i]+u_r[i]));

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


template<int n_dims, int n_fields, int flux_spec>
__device__ void ldg_solution(double* q_l, double* q_r, double* norm, double* q_c, double pen_fact)
{
  if(flux_spec==0) // Interior, mpi
    {
      // Choosing a unique direction for the switch

      if(n_dims==2)
        {
          if ((norm[0]+norm[1]) < 0.)
            pen_fact=-pen_fact;
        }
      if(n_dims==3)
        {
          if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
            pen_fact=-pen_fact;
        }

#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_c[i] = 0.5*(q_l[i]+q_r[i]) - pen_fact*(q_l[i]-q_r[i]);
    }
  else if(flux_spec==1) // Dirichlet
    {
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_c[i] = 0.5*(q_r[i] + q_l[i]);
    }
  else if(flux_spec==2) // von Neumann
    {
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_c[i] = 0.5*(q_l[i] + q_r[i]);
    }
}


template<int n_dims, int flux_spec>
__device__ __host__ void ldg_flux(double q_l, double q_r, double* f_l, double* f_r, double* f_c, double* norm, double pen_fact, double tau)
{
  if(flux_spec==0) //Interior, mpi
    {
      if(n_dims==2)
        {
          if ((norm[0]+norm[1]) < 0.)
            pen_fact=-pen_fact;
        }
      if(n_dims==3)
        {
          if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
            pen_fact=-pen_fact;
        }

      // Compute common interface flux
#pragma unroll
      for (int i=0;i<n_dims;i++)
        {
          f_c[i] = 0.5*(f_l[i] + f_r[i]) + tau*norm[i]*(q_l - q_r);
#pragma unroll
          for (int k=0;k<n_dims;k++)
            f_c[i] += pen_fact*norm[i]*norm[k]*(f_l[k] - f_r[k]);
        }
    }
  else if(flux_spec==1) // Dirichlet
    {
#pragma unroll
      for (int i=0;i<n_dims;i++)
        f_c[i] = f_l[i] + tau*norm[i]*(q_l - q_r);
    }
  else if(flux_spec==2) // von Neumann
    {
#pragma unroll
      for (int i=0;i<n_dims;i++)
        f_c[i] = f_r[i] + tau*norm[i]*(q_l - q_r); // Adding penalty factor term for this as well
    }
}


template<int n_dims>
__device__ void calc_dt_local(double* in_u, double* out_dt_local, double h_ref, double CFL, double gamma, double mu_inf, int order, int viscous)
{
  double lam_inv, lam_visc;
  double dt_inv, dt_visc;

  // 2-D Elements
  if (n_dims == 2)
  {
    double rho, u, v, ene, p, c;

    // primitive variables
    rho = in_u[0];
    u = in_u[1]/in_u[0];
    v = in_u[2]/in_u[0];
    ene = in_u[3];
    p = (gamma - 1.0) * (ene - 0.5*rho*(u*u+v*v));
    c = sqrt(gamma * p/rho);

    // Calculate internal wavespeed at each solution point
    lam_inv = sqrt(u*u + v*v) + c;
    lam_visc = 4.0/3.0*mu_inf/rho;

    if (viscous)
    {
      dt_visc = (CFL * 0.25 * h_ref * h_ref)/(lam_visc) * 1.0/(2.0*order+1.0);
      dt_inv = CFL*h_ref/lam_inv*1.0/(2.0*order + 1.0);
    }
    else
    {
      dt_visc = 1e16;
      dt_inv = CFL*h_ref/lam_inv*1.0/(2.0*order + 1.0);
    }
    (*out_dt_local) = min(dt_visc,dt_inv);
  }

  else if (n_dims == 3)
  {
    // Timestep type is not implemented in 3D yet!!!
  }
}


template<int n_dims, int n_fields>
__global__ void RK11_update_kernel(double* disu0_upts_ptr, double* div_tconf_upts_ptr, double* detjac_upts_ptr, double* src_upts, double* h_ref,
                                   int n_eles, int n_upts_per_ele, double dt, double const_src, double CFL, double gamma, double mu_inf, int order, int viscous, int dt_type)
{
  const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

  int ind;
  int stride = n_upts_per_ele*n_eles;

  double q[n_fields];

  if (thread_id<(n_upts_per_ele*n_eles))
  {
    // Compute local timestep
    if (dt_type == 2) {
      // Physical solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q[i] = disu0_upts_ptr[thread_id + i*stride];

      calc_dt_local<n_dims>(q,&dt,h_ref[thread_id],CFL,gamma,mu_inf,order,viscous);
    }

    // Update 5 fields
#pragma unroll
    for (int i=0;i<n_fields;i++)
    {
      ind = thread_id + i*stride;
      disu0_upts_ptr[ind] -= dt*(div_tconf_upts_ptr[ind]/detjac_upts_ptr[thread_id] - const_src - src_upts[ind]);
    }
  }
}


template<int n_dims, int n_fields>
__global__ void RK45_update_kernel(double *disu0_upts_ptr, double *div_tconf_upts_ptr, double *disu1_upts_ptr, double *detjac_upts_ptr, double *src_upts, double* h_ref,
                                   int n_eles, int n_upts_per_ele, double rk4a, double rk4b, double dt, double const_src, double CFL, double gamma, double mu_inf, int order, int viscous, int dt_type, int step)
{
  const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

  int ind;
  int stride = n_upts_per_ele*n_eles;

  double q[n_fields];
  double rhs,res;

  if (thread_id<(n_upts_per_ele*n_eles))
  {
    // Compute local timestep
    if (step == 0 && dt_type == 2) {
      // Physical solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q[i] = disu0_upts_ptr[thread_id + i*stride];

      calc_dt_local<n_dims>(q,&dt,h_ref[thread_id],CFL,gamma,mu_inf,order,viscous);
    }

    // Update 5 fields
#pragma unroll
    for (int i=0;i<n_fields;i++)
    {
      ind = thread_id + i*stride;
      rhs = -(div_tconf_upts_ptr[ind]/detjac_upts_ptr[thread_id] - const_src - src_upts[ind]);
      res = disu1_upts_ptr[ind];
      res = rk4a*res + dt*rhs;
      disu1_upts_ptr[ind] = res;
      disu0_upts_ptr[ind] += rk4b*res;
    }
  }
}


// gpu kernel to calculate transformed discontinuous inviscid flux at solution points for the wave equation
// otherwise, switch to one thread per output?
template<int n_dims>
__global__ void evaluate_invFlux_AD_gpu_kernel(int n_upts_per_ele, int n_eles, double* disu_upts_ptr, double* out_tdisf_upts_ptr, double* detjac_upts_ptr, double* JGinv_upts_ptr, double wave_speed_x, double wave_speed_y, double wave_speed_z)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int ele = thread_id/n_upts_per_ele;
  int upt = thread_id%n_upts_per_ele;
  double q;
  double f[n_dims];
  double met[n_dims][n_dims];

  int stride = n_upts_per_ele*n_eles;

  if(thread_id<(n_upts_per_ele*n_eles))
    {
      q = disu_upts_ptr[thread_id];

#pragma unroll
      for (int i=0;i<n_dims;i++)
#pragma unroll
        for (int j=0;j<n_dims;j++) {
          met[j][i] = JGinv_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];
        }

      int index;

      if (n_dims==2)
        {
          f[0] = wave_speed_x*q;
          f[1] = wave_speed_y*q;

          index = thread_id;
          out_tdisf_upts_ptr[index       ] = met[0][0]*f[0] + met[0][1]*f[1];
          out_tdisf_upts_ptr[index+stride] = met[1][0]*f[0] + met[1][1]*f[1];
        }
      else if (n_dims==3)
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


// gpu kernel to calculate transformed discontinuous inviscid flux at solution points for the Navier-Stokes equation
// otherwise, switch to one thread per output?
template<int n_dims, int n_fields>
__global__ void evaluate_invFlux_NS_gpu_kernel(int n_upts_per_ele, int n_eles, double* disu_upts_ptr, double* out_tdisf_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double* grid_vel_upts_ptr, double gamma, int motion, int turb_model)
{

  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int ele = thread_id/n_upts_per_ele;
  int upt = thread_id%n_upts_per_ele;
  double q[n_fields];
  double f[n_dims];
  double temp_f[n_dims];
  double met[n_dims][n_dims];
  double met_dyn[n_dims][n_dims];
  double v_g[n_dims];

  double p;
  int stride = n_upts_per_ele*n_eles;

  if(thread_id<(n_upts_per_ele*n_eles))
    {
      // Solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q[i] = disu_upts_ptr[thread_id + i*stride];


      // Metric terms
#pragma unroll
      for (int i=0;i<n_dims;i++)
#pragma unroll
        for (int j=0;j<n_dims;j++) {
          met[j][i] = JGinv_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];
        }

      if (motion) {
        // Transform to dynamic-physical domain
        for (int i=0;i<n_fields;i++)
          q[i] /= detjac_dyn_upts_ptr[thread_id];

        // Dynamic->static transformation matrix
#pragma unroll
        for (int i=0;i<n_dims;i++)
#pragma unroll
          for (int j=0;j<n_dims;j++)
            met_dyn[j][i] = JGinv_dyn_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];

        // Get grid velocity
#pragma unroll
        for (int i=0;i<n_dims;i++)
          v_g[i] = grid_vel_upts_ptr[thread_id + i*stride];
      }
      else
      {
        // Set grid velocity to 0
#pragma unroll
        for (int i=0;i<n_dims;i++)
          v_g[i] = 0.;
      }

      // Flux prep
      inv_NS_flux<n_dims>(q,v_g,&p,f,gamma,-1,turb_model);

      int index;

      // Flux computation
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
          inv_NS_flux<n_dims>(q,v_g,&p,f,gamma,i,turb_model);

          index = thread_id+i*stride;

          if (motion) {
            if (n_dims==2) {
              // Transform to static domain
              temp_f[0] = met_dyn[0][0]*f[0] + met_dyn[0][1]*f[1];
              temp_f[1] = met_dyn[1][0]*f[0] + met_dyn[1][1]*f[1];
              // copy back to f
              f[0] = temp_f[0];
              f[1] = temp_f[1];
            }
            else if(n_dims==3)
            {
              // Transform to static domain
              temp_f[0] = met_dyn[0][0]*f[0] + met_dyn[0][1]*f[1] + met_dyn[0][2]*f[2];
              temp_f[1] = met_dyn[1][0]*f[0] + met_dyn[1][1]*f[1] + met_dyn[1][2]*f[2];
              temp_f[2] = met_dyn[2][0]*f[0] + met_dyn[2][1]*f[1] + met_dyn[2][2]*f[2];
              // copy back to f
              f[0] = temp_f[0];
              f[1] = temp_f[1];
              f[2] = temp_f[2];
            }
          }
          
          // Transform back to computational domain
          if (n_dims==2) {
              out_tdisf_upts_ptr[index                    ] = met[0][0]*f[0] + met[0][1]*f[1];
              out_tdisf_upts_ptr[index+stride*n_fields ] = met[1][0]*f[0] + met[1][1]*f[1];
            }
          else if(n_dims==3)
            {
              out_tdisf_upts_ptr[index                      ] = met[0][0]*f[0] + met[0][1]*f[1] + met[0][2]*f[2];
              out_tdisf_upts_ptr[index+  stride*n_fields ] = met[1][0]*f[0] + met[1][1]*f[1] + met[1][2]*f[2];
              out_tdisf_upts_ptr[index+2*stride*n_fields ] = met[2][0]*f[0] + met[2][1]*f[1] + met[2][2]*f[2];
            }
        }

    }
}


// gpu kernel to calculate normal transformed continuous inviscid flux at the flux points
template <int n_dims, int n_fields, int riemann_solve_type, int vis_riemann_solve_type>
__global__ void calculate_common_invFlux_NS_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** norm_tconf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** tdA_dyn_fpts_l_ptr, double** tdA_dyn_fpts_r_ptr, double** detjac_dyn_fpts_l_ptr, double** detjac_dyn_fpts_r_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, double** delta_disu_fpts_l_ptr, double** delta_disu_fpts_r_ptr, double gamma, double pen_fact, int viscous, int motion, int turb_model)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l[n_fields];
  double q_r[n_fields];
  double fn[n_fields];
  double norm[n_dims];
  double v_g[n_dims];
  double q_c[n_fields];

  double jac;

  if(thread_id<stride)
  {
    if (motion) {
      // Compute left state solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]))/(*(detjac_dyn_fpts_l_ptr[thread_id]));

      // Compute right state solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_r[i]=(*(disu_fpts_r_ptr[thread_id+i*stride]))/(*(detjac_dyn_fpts_r_ptr[thread_id]));

      // Compute normal
#pragma unroll
      for (int i=0;i<n_dims;i++)
        norm[i]=*(norm_dyn_fpts_ptr[thread_id + i*stride]);

      // Get grid velocity
#pragma unroll
      for (int i=0;i<n_dims;i++)
        v_g[i]=*(grid_vel_fpts_ptr[thread_id + i*stride]);
    }
    else
    {
      // Compute left state solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]));

      // Compute right state solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_r[i]=(*(disu_fpts_r_ptr[thread_id+i*stride]));

      // Compute normal
#pragma unroll
      for (int i=0;i<n_dims;i++)
        norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);

      // Set grid velocity to 0
#pragma unroll
      for (int i=0;i<n_dims;i++)
        v_g[i]=0.;
    }

      if (riemann_solve_type==0)
        rusanov_flux<n_fields,n_dims> (q_l,q_r,v_g,norm,fn,gamma,turb_model);
      else if (riemann_solve_type==2)
        roe_flux<n_fields,n_dims> (q_l,q_r,v_g,norm,fn,gamma);

      // Store transformed flux (transform to computational domain)
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      if (motion)
        jac *= (*(tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_l_ptr[thread_id+i*stride]))=jac*fn[i];

      jac = (*(tdA_fpts_r_ptr[thread_id]));
      if (motion)
        jac *= (*(tdA_dyn_fpts_r_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_r_ptr[thread_id+i*stride]))=-jac*fn[i];

      // Viscous solution correction
      if(viscous)
        {
          if(vis_riemann_solve_type==0)
            ldg_solution<n_dims,n_fields,0> (q_l,q_r,norm,q_c,pen_fact);

          if (motion) {
            // Transform from dynamic-physical to static-physical domain
#pragma unroll
            for (int i=0;i<n_fields;i++)
              (*(delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i])*(*(detjac_dyn_fpts_l_ptr[thread_id]));

#pragma unroll
            for (int i=0;i<n_fields;i++)
              (*(delta_disu_fpts_r_ptr[thread_id+i*stride])) = (q_c[i]-q_r[i])*(*(detjac_dyn_fpts_r_ptr[thread_id]));
          }
          else
          {
#pragma unroll
            for (int i=0;i<n_fields;i++)
              (*(delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);

#pragma unroll
            for (int i=0;i<n_fields;i++)
              (*(delta_disu_fpts_r_ptr[thread_id+i*stride])) = (q_c[i]-q_r[i]);
          }
        }

    }
}


template <int n_dims, int vis_riemann_solve_type>
__global__ void calculate_common_invFlux_lax_friedrich_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** norm_tconf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** norm_fpts_ptr, double** delta_disu_fpts_l_ptr, double** delta_disu_fpts_r_ptr, double pen_fact, int viscous, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l;
  double q_r;
  double fn,u_av,u_diff;
  double norm_speed;
  double norm[n_dims];

  double q_c;
  double jac;

  if(thread_id<stride)
    {
      // Compute left state solution
      q_l=(*(disu_fpts_l_ptr[thread_id]));

      // Compute right state solution
      q_r=(*(disu_fpts_r_ptr[thread_id]));

      // Compute normal
#pragma unroll
      for (int i=0;i<n_dims;i++)
        norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);

      u_av = 0.5*(q_r+q_l);
      u_diff = q_l-q_r;

      norm_speed=0.;
      if (n_dims==2)
        norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1];
      else if (n_dims==3)
        norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2];

      // Compute common interface flux
      fn = 0.;
      if (n_dims==2)
        fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1])*u_av;
      else if (n_dims==3)
        fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2])*u_av;
      fn += 0.5*lambda*abs(norm_speed)*u_diff;

      // Store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      (*(norm_tconf_fpts_l_ptr[thread_id]))=jac*fn;

      jac = (*(tdA_fpts_r_ptr[thread_id]));
      (*(norm_tconf_fpts_r_ptr[thread_id]))=-jac*fn;

      // viscous solution correction
      if(viscous)
        {
          //if(vis_riemann_solve_type==0)
          //  ldg_solution<n_dims,1,0> (&q_l,&q_r,norm,&q_c,pen_fact);

          if(n_dims==2)
            {
              if ((norm[0]+norm[1]) < 0.)
                pen_fact=-pen_fact;
            }
          if(n_dims==3)
            {
              if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
                pen_fact=-pen_fact;
            }

          q_c = 0.5*(q_l+q_r) - pen_fact*(q_l-q_r);

          //printf("%4.2f \n", q_c);

          (*(delta_disu_fpts_l_ptr[thread_id])) = (q_c-q_l);

          (*(delta_disu_fpts_r_ptr[thread_id])) = (q_c-q_r);
        }
    }

}


// kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries
template<int n_dims, int n_fields, int riemann_solve_type, int vis_riemann_solve_type>
__global__ void evaluate_boundaryConditions_invFlux_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_l_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** loc_fpts_ptr, double** loc_dyn_fpts_ptr, double** grid_vel_fpts_ptr, int* boundary_type, double* bdy_params, double** delta_disu_fpts_l_ptr, double gamma, double R_ref, int viscous, int motion, double time_bound, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda, int equation, int turb_model)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  int bdy_spec;

  double q_l[n_fields];
  double q_r[n_fields];
  double fn[n_fields];
  double norm[n_dims];
  double loc[n_dims];
  double q_c[n_fields];
  double v_g[n_dims];

  double jac;

  if(thread_id<stride)
    {
      // Compute left solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]));

      if (motion>0) {
        // Tranform to dynamic-physical domain
#pragma unroll
        for (int i=0;i<n_fields;i++)
          q_l[i] /= *(detjac_dyn_fpts_l_ptr[thread_id]);
      }

      if (motion>0) {
        // Get normal & grid velocity in dynamic-physical domain
#pragma unroll
        for (int i=0;i<n_dims;i++) {
          norm[i]=*(norm_dyn_fpts_ptr[thread_id + i*stride]);
          v_g[i]=*(grid_vel_fpts_ptr[thread_id+i*stride]);
        }
      }
      else
      {
        // Get normal & grid velocity (0) in static-physical domain
#pragma unroll
        for (int i=0;i<n_dims;i++) {
          norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);
          v_g[i] = 0.;
        }
      }

      // Get physical position of flux points
      if (motion) {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          loc[i]=*(loc_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
        for (int i=0;i<n_dims;i++)
          loc[i]=*(loc_fpts_ptr[thread_id + i*stride]);
      }

      // Set boundary condition
      bdy_spec = boundary_type[thread_id/n_fpts_per_inter];
      set_inv_boundary_conditions_kernel<n_dims,n_fields>(bdy_spec,q_l,q_r,v_g,norm,loc,bdy_params,gamma, R_ref, time_bound, equation, turb_model);

      if (bdy_spec==16) // Dual consistent
        {
          //  right_flux<n_fields,n_dims> (q_r,norm,fn,gamma,turb_model);
          roe_flux<n_fields,n_dims> (q_l,q_r,v_g,norm,fn,gamma);
        }
      else
        {
          if (riemann_solve_type==0)
            convective_flux_boundary<n_fields,n_dims> (q_l,q_r,v_g,norm,fn,gamma,turb_model);
          else if (riemann_solve_type==1)
            lax_friedrichs_flux<n_dims> (q_l,q_r,norm,fn,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
          else if (riemann_solve_type==2)
            roe_flux<n_fields,n_dims> (q_l,q_r,v_g,norm,fn,gamma);
        }

      // Store transformed flux

      jac = (*(tdA_fpts_l_ptr[thread_id]));
      if (motion)
          jac *= (*(tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_l_ptr[thread_id+i*stride]))=jac*fn[i];

      // Viscous solution correction
      if(viscous)
        {
          if(bdy_spec == 12 || bdy_spec == 14) // Adiabatic
            {
              if (vis_riemann_solve_type==0)
                ldg_solution<n_dims,n_fields,2> (q_l,q_r,norm,q_c,0);
            }
          else
            {
              if (vis_riemann_solve_type==0)
                ldg_solution<n_dims,n_fields,1> (q_l,q_r,norm,q_c,0);
            }

          if(motion>0) {
            // Transform from dynamic back to static-physical domain
#pragma unroll
            for (int i=0;i<n_fields;i++)
              (*(delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i])*(*(detjac_dyn_fpts_l_ptr[thread_id]));
          }
          else
          {
#pragma unroll
            for (int i=0;i<n_fields;i++)
              (*(delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);
          }
        }

    }
}


// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int n_dims, int n_fields, int n_comp>
__global__ void evaluate_viscFlux_NS_gpu_kernel(int n_upts_per_ele, int n_eles, int ele_type, int order, double filter_ratio, int LES, int motion, int sgs_model, int wall_model, double wall_thickness, double* wall_dist_ptr, double* twall_ptr, double* Leonard_mom, double* Leonard_ene, double* disu_upts_ptr, double* out_tdisf_upts_ptr, double* out_sgsf_upts_ptr, double* grad_disu_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, int turb_model, double c_v1, double omega, double prandtl_t)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int ele = thread_id/n_upts_per_ele;
  int upt = thread_id%n_upts_per_ele;
  double q[n_fields];
  double f[n_dims];
  double temp_f[n_dims];
  double met[n_dims][n_dims];   // Static-Transformation Jacobian
  double met_dyn[n_dims][n_dims];   // Dynamic-Transformation Jacobian
  double stensor[n_comp];          // viscous stress tensor
  double grad_ene[n_dims];
  double grad_vel[n_dims*n_dims];
  double grad_q[n_fields*n_dims];
  double inte, mu, mu_t;

  // LES model variables
  double sgsf[n_fields*n_dims]; // SGS flux array
  double straintensor[n_comp];     // strain for SGS models
  double sdtensor[n_comp];         // for WALE SGS model
  double lmtensor[n_comp];         // local Leonard tensor for momentum
  double letensor[n_dims];         // local Leonard tensor for energy
  double jac, delta;

  // wall model variables
  double norm[n_dims];             // wall normal
  double tau[n_dims*n_dims];    // shear stress
  double mrot[n_dims*n_dims];   // rotation matrix
  double temp[n_dims*n_dims];   // array for matrix mult
  double urot[n_dims];             // rotated velocity components
  double tw[n_dims];               // wall shear stress components
  double qw;                          // wall heat flux
  double y;                           // wall distance
  int wall;                           // flag

  int i, j, k, index;
  int stride = n_upts_per_ele*n_eles;

   if(thread_id<(n_upts_per_ele*n_eles))
   {
    // Physical solution
    #pragma unroll
    for (i=0;i<n_fields;i++) {
      q[i] = disu_upts_ptr[thread_id + i*stride];
    }

    if (motion) {
#pragma unroll
      for (i=0;i<n_fields;i++) {
        q[i] /= detjac_dyn_upts_ptr[thread_id];
      }
    }

    if (motion) {
#pragma unroll
      for (i=0;i<n_dims;i++) {
#pragma unroll
        for (j=0;j<n_dims;j++) {
          met_dyn[j][i] = JGinv_dyn_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];
        }
      }
    }

    #pragma unroll
    for (i=0;i<n_dims;i++) {
      #pragma unroll
      for (j=0;j<n_dims;j++) {
        met[j][i] = JGinv_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];
      }
    }

    // Physical gradient
    #pragma unroll
    for (i=0;i<n_fields;i++)
    {
      index = thread_id + i*stride;
      grad_q[i*n_dims + 0] = grad_disu_upts_ptr[index];
      grad_q[i*n_dims + 1] = grad_disu_upts_ptr[index + stride*n_fields];

      if(n_dims==3)
        grad_q[i*n_dims + 2] = grad_disu_upts_ptr[index + 2*stride*n_fields];
    }

    // Flux prep
    vis_NS_flux<n_dims>(q, grad_q, grad_vel, grad_ene, stensor, f, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, -1, turb_model, c_v1, omega, prandtl_t);

    // Flux computation for each field
    #pragma unroll
    for (i=0;i<n_fields;i++) {

      index = thread_id + i*stride;

      vis_NS_flux<n_dims>(q, grad_q, grad_vel, grad_ene, stensor, f, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, i, turb_model, c_v1, omega, prandtl_t);

      if (motion) {
//#pragma unroll
//        for(j=0;j<n_dims;j++) {
//          temp_f[j] = 0.;
//#pragma unroll
//          for(k=0;k<n_dims;k++) {
//            temp_f[j] += met_dyn[j][k]*f[k];
//          }

        if(n_dims==2) {
          temp_f[0] = met_dyn[0][0]*f[0] + met_dyn[0][1]*f[1];
          temp_f[1] = met_dyn[1][0]*f[0] + met_dyn[1][1]*f[1];
        }
        else if(n_dims==3) {
          temp_f[0] = met_dyn[0][0]*f[0] + met_dyn[0][1]*f[1] + met_dyn[0][2]*f[2];
          temp_f[1] = met_dyn[1][0]*f[0] + met_dyn[1][1]*f[1] + met_dyn[1][2]*f[2];
          temp_f[2] = met_dyn[2][0]*f[0] + met_dyn[2][1]*f[1] + met_dyn[2][2]*f[2];
        }

        // Copy back into f
#pragma unroll
        for (j=0;j<n_dims;j++)
          f[j]=temp_f[j];
      }

      // Transform from static-physical to computational domain
//#pragma unroll
//      for(j=0;j<n_dims;j++) {
//#pragma unroll
//        for(k=0;k<n_dims;k++) {
//          out_tdisf_upts_ptr[index+i*stride*n_fields] += met[j][k]*f[k];
//        }
//      }
      if(n_dims==2) {
        out_tdisf_upts_ptr[index                   ] += met[0][0]*f[0] + met[0][1]*f[1];
        out_tdisf_upts_ptr[index+stride*n_fields] += met[1][0]*f[0] + met[1][1]*f[1];
      }
      else if(n_dims==3) {
        out_tdisf_upts_ptr[index                     ] += met[0][0]*f[0] + met[0][1]*f[1] + met[0][2]*f[2];
        out_tdisf_upts_ptr[index+  stride*n_fields] += met[1][0]*f[0] + met[1][1]*f[1] + met[1][2]*f[2];
        out_tdisf_upts_ptr[index+2*stride*n_fields] += met[2][0]*f[0] + met[2][1]*f[1] + met[2][2]*f[2];
      }
    }

    // wall flux prep.
    // If using a wall model, flag if upt is within wall distance threshold
    wall = 0;
    if(wall_model > 0) {

      // wall distance vector
      y = 0.0;
      #pragma unroll
      for (j=0;j<n_dims;j++)
        y += wall_dist_ptr[thread_id + j*stride]*wall_dist_ptr[thread_id + j*stride];

      y = sqrt(y);

      if(y < wall_thickness) wall = 1;

    }

    // if within near-wall region
    if (wall) {

      // get wall normal
      #pragma unroll
      for (j=0;j<n_dims;j++)
        norm[j] = wall_dist_ptr[thread_id + j*stride]/y;

      // calculate rotation matrix
      rotation_matrix_kernel<n_dims>(norm, mrot);

      // rotate velocity to surface
      if(n_dims==2) {
        urot[0] = q[1]*mrot[0*n_dims+1] + q[2]*mrot[1*n_dims+1];
        urot[1] = 0.0;
      }
      else {
        urot[0] = q[1]*mrot[0*n_dims+1] + q[2]*mrot[1*n_dims+1] + q[3]*mrot[2*n_dims+1];
        urot[1] = q[1]*mrot[0*n_dims+2] + q[2]*mrot[1*n_dims+2] + q[3]*mrot[2*n_dims+2];
        urot[2] = 0.0;
      }

      // get wall flux at previous timestep
      #pragma unroll
      for (j=0;j<n_dims;j++)
        tw[j] = twall_ptr[thread_id + (j+1)*stride];

      qw = twall_ptr[thread_id + (n_fields-1)*stride];

      // calculate wall flux
      wall_model_kernel<n_dims>( wall_model, q[0], urot, &inte, &mu, gamma, prandtl, y, tw, qw);

      // correct the sign of wall shear stress and wall heat flux? - see SD3D

      // Set arrays for next timestep
      #pragma unroll
      for (j=0;j<n_dims;j++)
        twall_ptr[thread_id + (j+1)*stride] = tw[j]; // momentum

      twall_ptr[thread_id] = 0.0; //density
      twall_ptr[thread_id + (n_fields-1)*stride] = qw; //energy

      // populate ndims*ndims rotated stress array
      if(n_dims==2) {
        tau[0] = 0.0;
        tau[1] = tw[0];
        tau[2] = tw[0];
        tau[3] = 0.0;
      }
      else {
        tau[0] = 0.0;
        tau[1] = tw[0];
        tau[2] = tw[1];
        tau[3] = tw[0];
        tau[4] = 0.0;
        tau[5] = 0.0;
        tau[6] = tw[1];
        tau[7] = 0.0;
        tau[8] = 0.0;
      }

      // rotate stress array back to Cartesian coordinates
      #pragma unroll
      for(i=0;i<n_dims;i++) {
        #pragma unroll
        for(j=0;j<n_dims;j++) {
          temp[i*n_dims + j] = 0.0;
          #pragma unroll
          for(k=0;k<n_dims;k++) {
            temp[i*n_dims + j] += tau[i*n_dims + k]*mrot[k*n_dims + j];
          }
        }
      }

      #pragma unroll
      for(i=0;i<n_dims;i++) {
        #pragma unroll
        for(j=0;j<n_dims;j++) {
          tau[i*n_dims + j] = 0.0;
          #pragma unroll
          for(k=0;k<n_dims;k++) {
            tau[i*n_dims + j] += mrot[k*n_dims + i]*temp[k*n_dims + j];
          }
        }
      }

      // set SGS fluxes
      #pragma unroll
      for(i=0;i<n_dims;i++) {

        // density
        sgsf[0*n_dims + i] = 0.0;

        // velocity
        #pragma unroll
        for(j=0;j<n_dims;j++) {
          sgsf[(j+1)*n_dims + i] = 0.5*(tau[j*n_dims+i]+tau[i*n_dims+j]);
        }

        // energy
        sgsf[(n_fields-1)*n_dims + i] = qw*norm[i];
      }

    }
    else {
      // if not near a wall and using LES, compute SGS flux
      if(LES) {

      // Calculate strain rate tensor from viscous stress tensor
      #pragma unroll
      for (j=0;j<n_comp;j++)
        straintensor[j] = stensor[j]/2.0/mu;

      // Calculate filter width
      jac = detjac_upts_ptr[thread_id];

      delta = SGS_filter_width(jac, ele_type, n_dims, order, filter_ratio);

      // momentum Leonard tensor
      #pragma unroll
      for (j=0;j<n_comp;j++)
        lmtensor[j] = Leonard_mom[thread_id + j*stride];

      // energy Leonard tensor - bugged or just sensitive to the filter?
      #pragma unroll
      for (j=0;j<n_dims;j++)
        letensor[j] = 0.0;
        //letensor[j] = Leonard_ene[thread_id + j*stride];

      //printf("Lu = %6.10f\n",lmtensor[0]);

      #pragma unroll
       for (i=0;i<n_fields;i++) {

        SGS_flux_kernel<n_dims>(q, grad_q, grad_vel, grad_ene, sdtensor, straintensor, lmtensor, letensor, f, sgs_model, delta, gamma, i);

        // set local SGS flux array
        #pragma unroll
        for(j=0;j<n_dims;j++)
          sgsf[i*n_dims + j] = f[j];

      }
    }
    }

    // add wall or SGS flux to output array
    if(LES || wall) {
      #pragma unroll
      for (i=0;i<n_fields;i++) {

        index = thread_id + i*stride;

        // Add in dynamic-static transformation here

        if(n_dims==2) {
          out_tdisf_upts_ptr[index                   ] += met[0][0]*sgsf[i*n_dims] + met[0][1]*sgsf[i*n_dims + 1];
          out_tdisf_upts_ptr[index+stride*n_fields] += met[1][0]*sgsf[i*n_dims] + met[1][1]*sgsf[i*n_dims + 1];
        }
        else if(n_dims==3) {
          out_tdisf_upts_ptr[index                     ] += met[0][0]*sgsf[i*n_dims] + met[0][1]*sgsf[i*n_dims + 1] + met[0][2]*sgsf[i*n_dims + 2];
          out_tdisf_upts_ptr[index+  stride*n_fields] += met[1][0]*sgsf[i*n_dims] + met[1][1]*sgsf[i*n_dims + 1] + met[1][2]*sgsf[i*n_dims + 2];
          out_tdisf_upts_ptr[index+2*stride*n_fields] += met[2][0]*sgsf[i*n_dims] + met[2][1]*sgsf[i*n_dims + 1] + met[2][2]*sgsf[i*n_dims + 2];
        }
      }
    }
  }
}


// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int n_dims>
__global__ void evaluate_viscFlux_AD_gpu_kernel(int n_upts_per_ele, int n_eles, double* disu_upts_ptr, double* out_tdisf_upts_ptr, double* grad_disu_upts_ptr, double* detjac_upts_ptr, double* JGinv_upts_ptr, double diff_coeff)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int ele = thread_id/n_upts_per_ele;
  int upt = thread_id%n_upts_per_ele;
  double f[n_dims];
  double met[n_dims][n_dims];
  double grad_q[n_dims];

  int ind;
  int index;
  int stride = n_upts_per_ele*n_eles;

  if(thread_id<(n_upts_per_ele*n_eles))
    {
      // Metric terms
#pragma unroll
      for (int i=0;i<n_dims;i++)
#pragma unroll
        for (int j=0;j<n_dims;j++)
          met[j][i] = JGinv_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];

      // Physical gradient
      ind = thread_id;
      grad_q[0] = grad_disu_upts_ptr[ind];
      grad_q[1] = grad_disu_upts_ptr[ind + stride];

      if(n_dims==3)
        grad_q[2] = grad_disu_upts_ptr[ind + 2*stride];


      // Flux computation
      f[0] = -diff_coeff*grad_q[0];
      f[1] = -diff_coeff*grad_q[1];

      if(n_dims==3)
        f[2] = -diff_coeff*grad_q[2];

      index = thread_id;

      if(n_dims==2) {
          out_tdisf_upts_ptr[index       ] += met[0][0]*f[0] + met[0][1]*f[1];
          out_tdisf_upts_ptr[index+stride] += met[1][0]*f[0] + met[1][1]*f[1];
        }
      else if(n_dims==3) {
          out_tdisf_upts_ptr[index         ] += met[0][0]*f[0] + met[0][1]*f[1] + met[0][2]*f[2];
          out_tdisf_upts_ptr[index+  stride] += met[1][0]*f[0] + met[1][1]*f[1] + met[1][2]*f[2];
          out_tdisf_upts_ptr[index+2*stride] += met[2][0]*f[0] + met[2][1]*f[1] + met[2][2]*f[2];
        }

    }
}

// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int n_dims, int n_fields>
__global__ void transform_grad_disu_upts_kernel(int n_upts_per_ele, int n_eles, double* grad_disu_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, int motion)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int ele = thread_id/n_upts_per_ele;
  int upt = thread_id%n_upts_per_ele;
  double dq[n_dims];
  double met[n_dims][n_dims];

  double jac;
  int ind;

  int stride = n_upts_per_ele*n_eles;

  if(thread_id<(n_upts_per_ele*n_eles))
    {
      // Compute physical gradient
      // First, transform to static-physical domain
      // Obtain metric terms
      jac = detjac_upts_ptr[thread_id];

#pragma unroll
      for (int i=0;i<n_dims;i++)
#pragma unroll
        for (int j=0;j<n_dims;j++) {
          met[j][i] = JGinv_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];
        }

      // Apply Transformation Metrics
#pragma unroll
      for (int i=0;i<n_fields;i++)
      {
        ind = thread_id + i*stride;
        dq[0] = grad_disu_upts_ptr[ind];
        dq[1] = grad_disu_upts_ptr[ind + stride*n_fields];

        if(n_dims==2)
        {
          grad_disu_upts_ptr[ind                   ] = (1./jac)*(dq[0]*met[0][0] + dq[1]*met[1][0]);
          grad_disu_upts_ptr[ind+stride*n_fields] = (1./jac)*(dq[0]*met[0][1] + dq[1]*met[1][1]);
        }
        if(n_dims==3)
        {
          dq[2] = grad_disu_upts_ptr[ind + 2*stride*n_fields];

          grad_disu_upts_ptr[ind                     ] = (1./jac)*(dq[0]*met[0][0] + dq[1]*met[1][0] + dq[2]*met[2][0]);
          grad_disu_upts_ptr[ind+stride*n_fields  ] = (1./jac)*(dq[0]*met[0][1] + dq[1]*met[1][1] + dq[2]*met[2][1]);
          grad_disu_upts_ptr[ind+2*stride*n_fields] = (1./jac)*(dq[0]*met[0][2] + dq[1]*met[1][2] + dq[2]*met[2][2]);
        }
      }

      // Lastly, transform to dynamic-physical domain
      if (motion) {
        // Obtain metric terms for 2nd transformation
        jac = detjac_dyn_upts_ptr[thread_id];

#pragma unroll
        for (int i=0;i<n_dims;i++)
#pragma unroll
          for (int j=0;j<n_dims;j++)
            met[j][i] = JGinv_dyn_upts_ptr[j+n_dims*(i+n_dims*(upt+n_upts_per_ele*ele))];

        // Next, transform to dynamic-physical domain
#pragma unroll
        for (int i=0;i<n_fields;i++)
        {
          ind = thread_id + i*stride;
          dq[0] = grad_disu_upts_ptr[ind];
          dq[1] = grad_disu_upts_ptr[ind + stride*n_fields];

          if(n_dims==2)
          {
            grad_disu_upts_ptr[ind                   ] = (1./jac)*(dq[0]*met[0][0] + dq[1]*met[1][0]);
            grad_disu_upts_ptr[ind+stride*n_fields] = (1./jac)*(dq[0]*met[0][1] + dq[1]*met[1][1]);
          }
          if(n_dims==3)
          {
            dq[2] = grad_disu_upts_ptr[ind + 2*stride*n_fields];

            grad_disu_upts_ptr[ind                     ] = (1./jac)*(dq[0]*met[0][0] + dq[1]*met[1][0] + dq[2]*met[2][0]);
            grad_disu_upts_ptr[ind+stride*n_fields  ] = (1./jac)*(dq[0]*met[0][1] + dq[1]*met[1][1] + dq[2]*met[2][1]);
            grad_disu_upts_ptr[ind+2*stride*n_fields] = (1./jac)*(dq[0]*met[0][2] + dq[1]*met[1][2] + dq[2]*met[2][2]);
          }
        }
      }
    }

}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int n_dims, int n_fields, int n_comp, int vis_riemann_solve_type>
__global__ void calculate_common_viscFlux_NS_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** norm_tconf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** tdA_dyn_fpts_l_ptr, double** tdA_dyn_fpts_r_ptr, double** detjac_dyn_fpts_l_ptr, double** detjac_dyn_fpts_r_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** sgsf_fpts_l_ptr, double** sgsf_fpts_r_ptr, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l[n_fields];
  double q_r[n_fields];
  double f_l[n_fields][n_dims];
  double f_r[n_fields][n_dims];
  double sgsf_l[n_fields][n_dims];
  double sgsf_r[n_fields][n_dims];
  double f_c[n_fields][n_dims];

  double fn[n_fields];
  double norm[n_dims];

  double grad_ene[n_dims];
  double grad_vel[n_dims*n_dims];
  double grad_q[n_fields*n_dims];

  double stensor[n_comp];

  double jac;
  double inte, mu, mu_t;

  if(thread_id<stride)
    {
      // Left solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]));

      if (motion) {
        // Transform to dynamic-physical domain
#pragma unroll
        for (int i=0;i<n_fields;i++)
          q_l[i] /= (*(detjac_dyn_fpts_l_ptr[thread_id]));
      }

      // Left solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<n_dims;j++)
            {
              grad_q[i*n_dims + j] = *(grad_disu_fpts_l_ptr[thread_id + (j*n_fields + i)*stride]);
            }
        }
      if(LES){
#pragma unroll
          for (int i=0;i<n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<n_dims;j++)
                {
                  sgsf_l[i][j] = *(sgsf_fpts_l_ptr[thread_id + (j*n_fields + i)*stride]);
                }
            }
        }

      // Normal vector
      if (motion) {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          norm[i]=*(norm_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);
      }

      // Left flux prep
      vis_NS_flux<n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, -1, turb_model, c_v1, omega, prandtl_t);

      // Left flux computation
#pragma unroll
      for (int i=0;i<n_fields;i++)
        vis_NS_flux<n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, f_l[i], &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, i, turb_model, c_v1, omega, prandtl_t);


      // Right solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_r[i]=(*(disu_fpts_r_ptr[thread_id+i*stride]));

      if (motion) {
        // Transform to dynamic-physical domain
#pragma unroll
        for (int i=0;i<n_fields;i++)
          q_r[i] /= (*(detjac_dyn_fpts_r_ptr[thread_id]));
      }

      // Right solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<n_dims;j++)
            {
              grad_q[i*n_dims + j] = *(grad_disu_fpts_r_ptr[thread_id + (j*n_fields + i)*stride]);
            }
        }
      if(LES){
#pragma unroll
          for (int i=0;i<n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<n_dims;j++)
                {
                  sgsf_r[i][j] = *(sgsf_fpts_r_ptr[thread_id + (j*n_fields + i)*stride]);
                }
            }
        }

      // Right flux prep
      vis_NS_flux<n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, -1, turb_model, c_v1, omega, prandtl_t);

      // Right flux computation
#pragma unroll
      for (int i=0;i<n_fields;i++)
        vis_NS_flux<n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f_r[i], &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, i, turb_model, c_v1, omega, prandtl_t);

      // If LES, add SGS fluxes to viscous fluxes
      if(LES)
        {
#pragma unroll
          for (int i=0;i<n_fields;i++)
            {
#pragma unroll
              for (int j=0;j<n_dims;j++)
                {
                  f_l[i][j] += sgsf_l[i][j];
                  f_r[i][j] += sgsf_r[i][j];
                }
            }
        }

      // Compute common flux
      if(vis_riemann_solve_type == 0)
        {
#pragma unroll
          for (int i=0;i<n_fields;i++)
            ldg_flux<n_dims,0>(q_l[i],q_r[i],f_l[i],f_r[i],f_c[i],norm,pen_fact,tau);
        }

      // Compute common normal flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
          fn[i] = f_c[i][0]*norm[0];
#pragma unroll
          for (int j=1;j<n_dims;j++)
            fn[i] += f_c[i][j]*norm[j];
        }

      // Store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      if (motion)
        jac *= (*(tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];

      jac = (*(tdA_fpts_r_ptr[thread_id]));
      if (motion)
        jac *= (*(tdA_dyn_fpts_r_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_r_ptr[thread_id+i*stride]))+=-jac*fn[i];
    }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int n_dims>
__global__ void calculate_common_viscFlux_AD_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** norm_tconf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** norm_fpts_ptr, double pen_fact, double tau, double diff_coeff)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l;
  double q_r;
  double f_l[n_dims];
  double f_r[n_dims];
  double f_c[n_dims];

  double fn;
  double norm[n_dims];

  double grad_q[n_dims];
  double jac;

  if(thread_id<stride)
    {
      // Left solution
      q_l=(*(disu_fpts_l_ptr[thread_id]));

      // Left solution gradient
#pragma unroll
      for(int j=0;j<n_dims;j++)
        grad_q[j] = *(grad_disu_fpts_l_ptr[thread_id + j*stride]);

      // Normal vector
#pragma unroll
      for (int i=0;i<n_dims;i++)
        norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);

      // Left flux computation
      f_l[0] = -diff_coeff*grad_q[0];
      f_l[1] = -diff_coeff*grad_q[1];

      if (n_dims==3)
        f_l[2] = -diff_coeff*grad_q[2];


      // Right solution
      q_r=(*(disu_fpts_r_ptr[thread_id]));

      // Right solution gradient
#pragma unroll
      for(int j=0;j<n_dims;j++)
        grad_q[j] = *(grad_disu_fpts_r_ptr[thread_id + j*stride]);

      // Right flux computation
      f_r[0] = -diff_coeff*grad_q[0];
      f_r[1] = -diff_coeff*grad_q[1];

      if (n_dims==3)
        f_r[2] = -diff_coeff*grad_q[2];

      // Compute common flux
      ldg_flux<n_dims,0>(q_l,q_r,f_l,f_r,f_c,norm,pen_fact,tau);

      // Compute common normal flux
      fn = f_c[0]*norm[0];
#pragma unroll
      for (int j=1;j<n_dims;j++)
        fn += f_c[j]*norm[j];

      // Store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      (*(norm_tconf_fpts_l_ptr[thread_id]))+=jac*fn;

      jac = (*(tdA_fpts_r_ptr[thread_id]));
      (*(norm_tconf_fpts_r_ptr[thread_id]))+=-jac*fn;

    }
}



// kernel to calculate normal transformed continuous viscous flux at the flux points at boundaries
template<int n_dims, int n_fields, int n_comp, int vis_riemann_solve_type>
__global__ void evaluate_boundaryConditions_viscFlux_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** grad_disu_fpts_l_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, double** loc_fpts_ptr, double** loc_dyn_fpts_ptr, double** sgsf_fpts_ptr, int* boundary_type, double* bdy_params, double** delta_disu_fpts_l_ptr, double R_ref, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, double time_bound, int equation, double diff_coeff, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  int bdy_spec;

  double q_l[n_fields];
  double q_r[n_fields];

  double f[n_fields][n_dims];
  double sgsf[n_fields][n_dims];
  double f_c[n_fields][n_dims];

  double fn[n_fields];
  double norm[n_dims];
  double loc[n_dims];
  double v_g[n_dims];

  double grad_ene[n_dims];
  double grad_vel[n_dims*n_dims];
  double grad_q[n_fields*n_dims];

  double stensor[n_comp];

  double jac;
  double inte, mu, mu_t;

  if(thread_id<stride)
    {
      // Left solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]));

      if (motion) {
 #pragma unroll
        for (int i=0;i<n_fields;i++)
          q_l[i]/=(*(detjac_dyn_fpts_ptr[thread_id]));
      }

     // Left solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<n_dims;j++)
            {
              grad_q[i*n_dims + j] = *(grad_disu_fpts_l_ptr[thread_id + (j*n_fields + i)*stride]);
            }
        }
      if(LES){
#pragma unroll
          for (int i=0;i<n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<n_dims;j++)
                {
                  sgsf[i][j] = *(sgsf_fpts_ptr[thread_id + (j*n_fields + i)*stride]);
                }
            }
        }

      // Normal vector
      if (motion) {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          norm[i]=*(norm_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);
      }

      // Get location
      if (motion) {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          loc[i]=*(loc_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          loc[i]=*(loc_fpts_ptr[thread_id + i*stride]);
      }

      if (motion) {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          v_g[i]=*(grid_vel_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          v_g[i]=0.;
      }

      // Right solution
      bdy_spec = boundary_type[thread_id/n_fpts_per_inter];
      set_inv_boundary_conditions_kernel<n_dims,n_fields>(bdy_spec,q_l,q_r,v_g,norm,loc,bdy_params,gamma,R_ref,time_bound,equation, turb_model);


      // Compute common flux
      if(bdy_spec == 12 || bdy_spec == 14)
        {
          // Right solution gradient
          set_vis_boundary_conditions_kernel<n_dims,n_fields>(bdy_spec,q_l,q_r,grad_q,norm,loc,bdy_params,gamma,R_ref,time_bound,equation);

          if(equation==0)
            {
              // Right flux prep
              vis_NS_flux<n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, -1, turb_model, c_v1, omega, prandtl_t);

              // Right flux computation
#pragma unroll
              for (int i=0;i<n_fields;i++)
                vis_NS_flux<n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f[i], &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, i, turb_model, c_v1, omega, prandtl_t);
            }
          if(equation==1)
            {
              f[0][0] = -diff_coeff*grad_q[0];
              f[0][1] = -diff_coeff*grad_q[1];

              if(n_dims==3)
                f[0][2] = -diff_coeff*grad_q[2];
            }

          if (vis_riemann_solve_type==0)
            {
#pragma unroll
              for (int i=0;i<n_fields;i++)
                ldg_flux<n_dims,2>(q_l[i],q_r[i],NULL,f[i],f_c[i],norm,pen_fact,tau); // von Neumann
            }
        }
      else
        {
          if(equation==0)
            {
              // Left flux prep
              vis_NS_flux<n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, -1, turb_model, c_v1, omega, prandtl_t);

              // Left flux computation
#pragma unroll
              for (int i=0;i<n_fields;i++)
                vis_NS_flux<n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, f[i], &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, i, turb_model, c_v1, omega, prandtl_t);

              // If LES (but no wall model?), add SGS flux to viscous flux
              if(LES)
                {
#pragma unroll
                  for (int i=0;i<n_fields;i++)
                    {
#pragma unroll
                      for (int j=0;j<n_dims;j++)
                        {
                          f[i][j] += sgsf[i][j];
                        }
                    }
                }
            }
          if(equation==1)
            {
              f[0][0] = -diff_coeff*grad_q[0];
              f[0][1] = -diff_coeff*grad_q[1];

              if(n_dims==3)
                f[0][2] = -diff_coeff*grad_q[2];
            }

          if (vis_riemann_solve_type==0)
            {
#pragma unroll
              for (int i=0;i<n_fields;i++)
                ldg_flux<n_dims,1>(q_l[i],q_r[i],f[i],NULL,f_c[i],norm,pen_fact,tau); // Dirichlet
            }
        }

      // compute common normal flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
          fn[i] = f_c[i][0]*norm[0];
#pragma unroll
          for (int j=1;j<n_dims;j++)
            fn[i] += f_c[i][j]*norm[j];
        }

      // store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      if (motion)
        jac *= (*(tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];
    }
}


template<int n_dims, int n_fields>
__global__ void calc_src_upts_gpu_kernel(int n_upts_per_ele,int n_eles,double* disu_upts_ptr,double* grad_disu_upts_ptr,double* wall_distance_mag_ptr,double* out_src_upts_ptr,double gamma,double prandtl,double rt_inf,double mu_inf,double c_sth,int fix_vis,double c_v1,double c_v2,double c_v3,double c_b1,double c_b2,double c_w2,double c_w3,double omega,double Kappa)
{

  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double q[n_fields];
  double src_term;

  double grad_q[n_fields*n_dims];
  double wall_distance_mag;

  int ind;
  int stride = n_upts_per_ele*n_eles;

  if(thread_id<(n_upts_per_ele*n_eles))
  {
    // Physical solution
#pragma unroll
    for (int i=0;i<n_fields;i++)
      q[i] = disu_upts_ptr[thread_id + i*stride];

    // Physical gradient
#pragma unroll
    for (int i=0;i<n_fields;i++)
    {
      ind = thread_id + i*stride;
      grad_q[i*n_dims + 0] = grad_disu_upts_ptr[ind];
      grad_q[i*n_dims + 1] = grad_disu_upts_ptr[ind + stride*n_fields];

      if(n_dims==3)
        grad_q[i*n_dims + 2] = grad_disu_upts_ptr[ind + 2*stride*n_fields];
    }

    wall_distance_mag = wall_distance_mag_ptr[thread_id];

    calc_source_SA<n_dims>(q,grad_q,&src_term,wall_distance_mag,prandtl,gamma,rt_inf,mu_inf,c_sth,fix_vis,c_v1,c_v2,c_v3,c_b1,c_b2,c_w2,c_w3,omega,Kappa);

    out_src_upts_ptr[thread_id + (n_fields-1)*stride] = src_term;
  }
}

__global__ void shock_capture_concentration_gpu_kernel(int in_n_eles, int in_n_upts_per_ele, int in_n_fields, int in_order, int in_ele_type, int in_artif_type, double s0, double kappa, double* in_disu_upts_ptr, double* in_inv_vandermonde_ptr, double* in_inv_vandermonde2D_ptr, double* in_vandermonde2D_ptr, double* concentration_array_ptr, double* out_sensor, double* sigma)
{
    const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    if(thread_id < in_n_eles)
    {
        int stride = in_n_upts_per_ele*in_n_eles;
        double sensor = 0;

        double nodal_rho[8];  // Array allocated so that it can handle upto p=7
        double modal_rho[8];
        double uE[8];
        double temp;
        double p = 3;	// exponent in concentration method
        double J = 0.15;
        int shock_found = 0;

        // X-slices
        for(int i=0; i<in_order+1; i++)
        {
            for(int j=0; j<in_order+1; j++){
                nodal_rho[j] = in_disu_upts_ptr[thread_id*in_n_upts_per_ele + i*(in_order+1) + j];
            }

            for(int j=0; j<in_order+1; j++){
                modal_rho[j] = 0;
                for(int k=0; k<in_order+1; k++){
                    modal_rho[j] += in_inv_vandermonde_ptr[j + k*(in_order+1)]*nodal_rho[k];
                }
            }

            for(int j=0; j<in_order+1; j++){
                uE[j] = 0;
                for(int k=0; k<in_order+1; k++)
                    uE[j] += modal_rho[k]*concentration_array_ptr[j*(in_order+1) + k];

                uE[j] = abs((3.1415/(in_order+1))*uE[j]);
                temp = 0.0;//pow(uE[j],p)*pow(in_order+1,p/2);

                if(temp >= J)
                    shock_found++;

                if(temp > sensor)
                    sensor = temp;
            }

        }

        // Y-slices
        for(int i=0; i<in_order+1; i++)
        {
            for(int j=0; j<in_order+1; j++){
                nodal_rho[j] = in_disu_upts_ptr[thread_id*in_n_upts_per_ele + j*(in_order+1) + i];
            }

            for(int j=0; j<in_order+1; j++){
                modal_rho[j] = 0;
                for(int k=0; k<in_order+1; k++)
                    modal_rho[j] += in_inv_vandermonde_ptr[j + k*(in_order+1)]*nodal_rho[k];
            }

            for(int j=0; j<in_order+1; j++){
                uE[j] = 0;
                for(int k=0; k<in_order+1; k++)
                    uE[j] += modal_rho[k]*concentration_array_ptr[j*(in_order+1) + k];

                uE[j] = (3.1415/(in_order+1))*uE[j];
                temp = 0.0;//pow(abs(uE[j]),p)*pow(in_order+1,p/2);

                if(temp >= J)
                    shock_found++;

                if(temp > sensor)
                    sensor = temp;
            }
        }

        out_sensor[thread_id] = sensor;

    /* -------------------------------------------------------------------------------------- */
                    /* Modal Order Reduction */

    //      if(sensor > s0 + kappa && in_artif_type == 1) {
    //            double nodal_sol[36];
    //            double modal_sol[36];
    //            int n_trun_modes = 1;
    //            //printf("Sensor value is %f in thread %d \n",sensor, thread_id);

    //            for(int k=0; k<in_n_fields; k++) {

    //                for(int i=0; i<in_n_upts_per_ele; i++){
    //                    nodal_sol[i] = in_disu_upts_ptr[thread_id*in_n_upts_per_ele + k*stride + i];
    //                }
    //                // Nodal to modal only upto 1st order
    //                for(int i=0; i<in_n_upts_per_ele; i++){
    //                    modal_sol[i] = 0;
    //                    if(i < n_trun_modes){
    //                        for(int j=0; j<in_n_upts_per_ele; j++)
    //                            modal_sol[i] += in_inv_vandermonde2D_ptr[i + j*in_n_upts_per_ele]*nodal_sol[j];
    //                    }
    //                }

    //                // Change back to nodal
    //                for(int i=0; i<in_n_upts_per_ele; i++){
    //                    nodal_sol[i] = 0;
    //                    for(int j=0; j<in_n_upts_per_ele; j++)
    //                        nodal_sol[i] += in_vandermonde2D_ptr[i + j*in_n_upts_per_ele]*modal_sol[j];

    //                    in_disu_upts_ptr[thread_id*in_n_upts_per_ele + k*stride + i] = nodal_sol[i];
    //                }
    //            }
    //        }

    //        if(sensor <= s0 && sensor > s0 - kappa){
    //            double nodal_sol[36];
    //            double modal_sol[36];
    //            int n_trun_modes = 3;
    //            //printf("Sensor value is %f in thread %d \n",sensor, thread_id);

    //            for(int k=0; k<in_n_fields; k++) {

    //                for(int i=0; i<in_n_upts_per_ele; i++){
    //                    nodal_sol[i] = in_disu_upts_ptr[thread_id*in_n_upts_per_ele + k*stride + i];
    //                }
    //                // Nodal to modal only upto 1st order
    //                for(int i=0; i<in_n_upts_per_ele; i++){
    //                    modal_sol[i] = 0;
    //                    if(i < n_trun_modes){
    //                        for(int j=0; j<in_n_upts_per_ele; j++)
    //                            modal_sol[i] += in_inv_vandermonde2D_ptr[i + j*in_n_upts_per_ele]*nodal_sol[j];
    //                    }
    //                }

    //                // Change back to nodal
    //                for(int i=0; i<in_n_upts_per_ele; i++){
    //                    nodal_sol[i] = 0;
    //                    for(int j=0; j<in_n_upts_per_ele; j++)
    //                        nodal_sol[i] += in_vandermonde2D_ptr[i + j*in_n_upts_per_ele]*modal_sol[j];

    //                    in_disu_upts_ptr[thread_id*in_n_upts_per_ele + k*stride + i] = nodal_sol[i];
    //                }
    //            }

    //            out_epsilon_ptr[thread_id] = out_epsilon_ptr[thread_id] + 1;
    //            epsilon_global_eles[global_ele_num] = out_epsilon_ptr[thread_id];
    //        }


/* -------------------------------------------------------------------------------------- */
            /* Exponential modal filter */

        if(sensor > s0 + kappa && in_artif_type == 1) {
            double nodal_sol[36];
            double modal_sol[36];

            for(int k=0; k<in_n_fields; k++) {

                for(int i=0; i<in_n_upts_per_ele; i++){
                    nodal_sol[i] = in_disu_upts_ptr[thread_id*in_n_upts_per_ele + k*stride + i];
                }

                // Nodal to modal only upto 1st order
                for(int i=0; i<in_n_upts_per_ele; i++){
                    modal_sol[i] = 0;
                    for(int j=0; j<in_n_upts_per_ele; j++)
                        modal_sol[i] += in_inv_vandermonde2D_ptr[i + j*in_n_upts_per_ele]*nodal_sol[j];

                    modal_sol[i] = modal_sol[i]*sigma[i];
                    //printf("The exp filter values are %f \n",modal_sol[i]);
                }

                // Change back to nodal
                for(int i=0; i<in_n_upts_per_ele; i++){
                    nodal_sol[i] = 0;
                    for(int j=0; j<in_n_upts_per_ele; j++)
                        nodal_sol[i] += in_vandermonde2D_ptr[i + j*in_n_upts_per_ele]*modal_sol[j];

                    in_disu_upts_ptr[thread_id*in_n_upts_per_ele + k*stride + i] = nodal_sol[i];
                }
            }
        }
  }
}

// kernel to add body force to viscous flux
//TODO: implement body force calculation from eles.cpp
template<int n_dims, int n_fields>
__global__ void evaluate_body_force_gpu_kernel(int n_upts_per_ele, int n_eles, double* src_upts_ptr, double* body_force_ptr)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int i;

  int stride = n_upts_per_ele*n_eles;

  if(thread_id<(n_upts_per_ele*n_eles)) {

#pragma unroll
    for (i=0;i<n_fields;i++) {
      src_upts_ptr[thread_id + i*stride] += body_force_ptr[i];
    }
  }
}

#ifdef _MPI

// gpu kernel to calculate normal transformed continuous inviscid flux at the flux points for mpi faces
template <int n_dims, int n_fields, int riemann_solve_type, int vis_riemann_solve_type>
__global__ void calculate_common_invFlux_NS_mpi_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, double** delta_disu_fpts_l_ptr, double gamma, double pen_fact, int viscous, int motion, int turb_model)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l[n_fields];
  double q_r[n_fields];
  double fn[n_fields];
  double norm[n_dims];
  double v_g[n_dims];

  double q_c[n_fields];

  double jac;

  if(thread_id<stride)
    {
      // Compute left state solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]));

      // Compute right state solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_r[i]=*(disu_fpts_r_ptr[thread_id+i*stride]);

      // Transform to dynamic-physical domain
      if (motion) {
#pragma unroll
        for (int i=0;i<n_fields;i++) {
          q_l[i] /= *(detjac_dyn_fpts_ptr[thread_id]);
          q_r[i] /= *(detjac_dyn_fpts_ptr[thread_id]);
        }
      }

      // Compute normal
      if (motion>0) {
#pragma unroll
        for (int i=0;i<n_dims;i++) {
          norm[i]=*(norm_dyn_fpts_ptr[thread_id + i*stride]);
          v_g[i] =*(grid_vel_fpts_ptr[thread_id + i*stride]);
        }
      }
      else
      {
#pragma unroll
        for (int i=0;i<n_dims;i++) {
          norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);
          v_g[i] = 0.;
        }
      }

      if (riemann_solve_type==0)
        rusanov_flux<n_fields,n_dims> (q_l,q_r,v_g,norm,fn,gamma,turb_model);
      else if (riemann_solve_type==2)
        roe_flux<n_fields,n_dims> (q_l,q_r,v_g,norm,fn,gamma);

      // Store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      if (motion>0)
        jac *= (*(tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_l_ptr[thread_id+i*stride]))=jac*fn[i];

      // viscous solution correction
      if(viscous)
        {
          if(vis_riemann_solve_type==0)
            ldg_solution<n_dims,n_fields,0> (q_l,q_r,norm,q_c,pen_fact);

#pragma unroll
          for (int i=0;i<n_fields;i++)
            (*(delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);

          // Tranform back to static-reference domain
          if (motion>0) {
#pragma unroll
            for (int i=0;i<n_fields;i++)
              (*(delta_disu_fpts_l_ptr[thread_id+i*stride])) *= (*(detjac_dyn_fpts_ptr[thread_id]));
          }
        }
    }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int n_dims, int n_fields, int n_comp, int vis_riemann_solve_type>
__global__ void calculate_common_viscFlux_NS_mpi_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** sgsf_fpts_l_ptr, double** sgsf_fpts_r_ptr, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l[n_fields];
  double q_r[n_fields];
  double f_l[n_fields][n_dims];
  double f_r[n_fields][n_dims];
  double sgsf_l[n_fields][n_dims];
  double sgsf_r[n_fields][n_dims];
  double f_c[n_fields][n_dims];

  double fn[n_fields];
  double norm[n_dims];

  double grad_ene[n_dims];
  double grad_vel[n_dims*n_dims];
  double grad_q[n_fields*n_dims];

  double stensor[n_comp];

  double jac;
  double inte, mu, mu_t;

  if(thread_id<stride)
    {
      // Left solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]));

      if (motion) {
#pragma unroll
        for (int i=0;i<n_fields;i++)
          q_l[i] /= (*(detjac_dyn_fpts_ptr[thread_id]));
      }

      // Left solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<n_dims;j++)
            {
              grad_q[i*n_dims + j] = *(grad_disu_fpts_l_ptr[thread_id + (j*n_fields + i)*stride]);
            }
        }
      if(LES){
#pragma unroll
          for (int i=0;i<n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<n_dims;j++)
                {
                  sgsf_l[i][j] = *(sgsf_fpts_l_ptr[thread_id + (j*n_fields + i)*stride]);
                }
            }
        }


      // Normal vector
      if (motion) {
#pragma unroll
        for (int i=0;i<n_dims;i++)
          norm[i]=*(norm_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
 #pragma unroll
        for (int i=0;i<n_dims;i++)
          norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);
      }

      // Left flux prep
      vis_NS_flux<n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, -1, turb_model, c_v1, omega, prandtl_t);

      // Left flux computation
#pragma unroll
      for (int i=0;i<n_fields;i++)
        vis_NS_flux<n_dims>(q_l, grad_q, grad_vel, grad_ene, stensor, f_l[i], &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, i, turb_model, c_v1, omega, prandtl_t);


      // Right solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_r[i]=(*(disu_fpts_r_ptr[thread_id+i*stride]));// don't divide by jac, since points to buffer

      // Transform to dynamic-physical domain
      if (motion) {
#pragma unroll
        for (int i=0;i<n_fields;i++)
          q_r[i] /= (*(detjac_dyn_fpts_ptr[thread_id]));
      }

      // Right solution gradientand SGS flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<n_dims;j++)
            {
              grad_q[i*n_dims + j] = *(grad_disu_fpts_r_ptr[thread_id + (j*n_fields + i)*stride]);
            }
        }
      if(LES){
#pragma unroll
          for (int i=0;i<n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<n_dims;j++)
                {
                  sgsf_r[i][j] = *(sgsf_fpts_r_ptr[thread_id + (j*n_fields + i)*stride]);
                }
            }
        }

      // Right flux prep
      vis_NS_flux<n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, -1, turb_model, c_v1, omega, prandtl_t);

      // Right flux computation
#pragma unroll
      for (int i=0;i<n_fields;i++)
        vis_NS_flux<n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f_r[i], &inte, &mu, &mu_t, prandtl, gamma, rt_inf, mu_inf, c_sth, fix_vis, i, turb_model, c_v1, omega, prandtl_t);

      // If LES, add SGS fluxes to viscous fluxes
      if(LES)
        {
#pragma unroll
          for (int i=0;i<n_fields;i++)
            {
#pragma unroll
              for (int j=0;j<n_dims;j++)
                {
                  f_l[i][j] += sgsf_l[i][j];
                  f_r[i][j] += sgsf_r[i][j];
                }
            }
        }

      // Compute common flux
      if(vis_riemann_solve_type == 0)
        {
#pragma unroll
          for (int i=0;i<n_fields;i++)
            ldg_flux<n_dims,0>(q_l[i],q_r[i],f_l[i],f_r[i],f_c[i],norm,pen_fact,tau);
        }

      // Compute common normal flux
#pragma unroll
      for (int i=0;i<n_fields;i++)
        {
          fn[i] = f_c[i][0]*norm[0];
#pragma unroll
          for (int j=1;j<n_dims;j++)
            fn[i] += f_c[i][j]*norm[j];
        }

      // Store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      if (motion)
        jac *= (*(tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<n_fields;i++)
        (*(norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];
    }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int n_dims>
__global__ void calculate_common_viscFlux_AD_mpi_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** norm_fpts_ptr, double pen_fact, double tau, double diff_coeff)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l;
  double q_r;
  double f_l[n_dims];
  double f_r[n_dims];
  double f_c[n_dims];

  double fn;
  double norm[n_dims];

  double grad_q[n_dims];
  double jac;

  if(thread_id<stride)
    {
      // Left solution
      q_l=(*(disu_fpts_l_ptr[thread_id]));

      // Left solution gradient
#pragma unroll
      for(int j=0;j<n_dims;j++)
        grad_q[j] = *(grad_disu_fpts_l_ptr[thread_id + j*stride]);

      // Normal vector
#pragma unroll
      for (int i=0;i<n_dims;i++)
        norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);

      // Left flux computation
      f_l[0] = -diff_coeff*grad_q[0];
      f_l[1] = -diff_coeff*grad_q[1];

      if (n_dims==3)
        f_l[2] = -diff_coeff*grad_q[2];


      // Right solution
      q_r=(*(disu_fpts_r_ptr[thread_id]));

      // Right solution gradient
#pragma unroll
      for(int j=0;j<n_dims;j++)
        grad_q[j] = *(grad_disu_fpts_r_ptr[thread_id + j*stride]);

      // Right flux computation
      f_r[0] = -diff_coeff*grad_q[0];
      f_r[1] = -diff_coeff*grad_q[1];

      if (n_dims==3)
        f_r[2] = -diff_coeff*grad_q[2];

      // Compute common flux
      ldg_flux<n_dims,0>(q_l,q_r,f_l,f_r,f_c,norm,pen_fact,tau);

      // Compute common normal flux
      fn = f_c[0]*norm[0];
#pragma unroll
      for (int j=1;j<n_dims;j++)
        fn += f_c[j]*norm[j];

      // Store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      (*(norm_tconf_fpts_l_ptr[thread_id]))+=jac*fn;

    }
}


template <int n_dims, int vis_riemann_solve_type>
__global__ void calculate_common_invFlux_lax_friedrich_mpi_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** norm_fpts_ptr, double** delta_disu_fpts_l_ptr, double pen_fact, int viscous, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = n_fpts_per_inter*n_inters;

  double q_l;
  double q_r;
  double fn,u_av,u_diff;
  double norm_speed;
  double norm[n_dims];

  double q_c;
  double jac;

  if(thread_id<stride)
    {

      // Compute left state solution
      q_l=(*(disu_fpts_l_ptr[thread_id]));

      // Compute right state solution
      q_r=(*(disu_fpts_r_ptr[thread_id]));

      // Compute normal
#pragma unroll
      for (int i=0;i<n_dims;i++)
        norm[i]=*(norm_fpts_ptr[thread_id + i*stride]);

      u_av = 0.5*(q_r+q_l);
      u_diff = q_l-q_r;

      norm_speed=0.;
      if (n_dims==2)
        norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1];
      else if (n_dims==3)
        norm_speed += wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2];

      // Compute common interface flux
      fn = 0.;
      if (n_dims==2)
        fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1])*u_av;
      else if (n_dims==3)
        fn += (wave_speed_x*norm[0] + wave_speed_y*norm[1] + wave_speed_z*norm[2])*u_av;
      fn += 0.5*lambda*abs(norm_speed)*u_diff;

      // Store transformed flux
      jac = (*(tdA_fpts_l_ptr[thread_id]));
      (*(norm_tconf_fpts_l_ptr[thread_id]))=jac*fn;

      // viscous solution correction
      if(viscous)
        {
          if(n_dims==2)
            {
              if ((norm[0]+norm[1]) < 0.)
                pen_fact=-pen_fact;
            }
          if(n_dims==3)
            {
              if ((norm[0]+norm[1]+sqrt(2.)*norm[2]) < 0.)
                pen_fact=-pen_fact;
            }

          q_c = 0.5*(q_l+q_r) - pen_fact*(q_l-q_r);

          /*
      if(vis_riemann_solve_type==0)
        ldg_solution<n_dims,1,0> (&q_l,&q_r,norm,&q_c,pen_fact);
      */

          (*(delta_disu_fpts_l_ptr[thread_id])) = (q_c-q_l);
        }
    }
}


template <int n_fields>
__global__ void  pack_out_buffer_disu_gpu_kernel(int n_fpts_per_inter, int n_inters, double** disu_fpts_l_ptr, double* out_buffer_disu_ptr)
{

  double q_l[n_fields];

  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int inter = thread_id/(n_fpts_per_inter);
  const int fpt = thread_id - inter*n_fpts_per_inter;
  const int stride=n_fpts_per_inter*n_inters;

  if (thread_id < stride)
    {
      // Compute left state solution
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_l[i]=(*(disu_fpts_l_ptr[thread_id+i*stride]));

#pragma unroll
      for (int i=0;i<n_fields;i++)
        out_buffer_disu_ptr[inter*n_fpts_per_inter*n_fields+i*n_fpts_per_inter+fpt]=q_l[i];

    }

}


template <int n_fields, int n_dims>
__global__ void  pack_out_buffer_grad_disu_gpu_kernel(int n_fpts_per_inter, int n_inters, double** grad_disu_fpts_l_ptr, double* out_buffer_grad_disu_ptr)
{

  double dq[n_fields][n_dims];

  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int inter = thread_id/(n_fpts_per_inter);
  const int fpt = thread_id - inter*n_fpts_per_inter;
  const int stride=n_fpts_per_inter*n_inters;

  if (thread_id < stride)
    {
      // Compute left state solution
#pragma unroll
      for (int j=0;j<n_dims;j++)
#pragma unroll
        for (int i=0;i<n_fields;i++)
          dq[i][j]=(*(grad_disu_fpts_l_ptr[thread_id+(j*n_fields+i)*stride]));

#pragma unroll
      for (int j=0;j<n_dims;j++)
#pragma unroll
        for (int i=0;i<n_fields;i++)
          out_buffer_grad_disu_ptr[inter*n_fpts_per_inter*n_fields*n_dims+j*n_fpts_per_inter*n_fields+i*n_fpts_per_inter+fpt]=dq[i][j];

    }

}

template <int n_fields, int n_dims>
__global__ void  pack_out_buffer_sgsf_gpu_kernel(int n_fpts_per_inter, int n_inters, double** sgsf_fpts_l_ptr, double* out_buffer_sgsf_ptr)
{

  double dq[n_fields][n_dims];

  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int inter = thread_id/(n_fpts_per_inter);
  const int fpt = thread_id - inter*n_fpts_per_inter;
  const int stride=n_fpts_per_inter*n_inters;

  if (thread_id < stride)
    {
      // Compute left state solution
#pragma unroll
      for (int j=0;j<n_dims;j++)
#pragma unroll
        for (int i=0;i<n_fields;i++)
          dq[i][j]=(*(sgsf_fpts_l_ptr[thread_id+(j*n_fields+i)*stride]));

#pragma unroll
      for (int j=0;j<n_dims;j++)
#pragma unroll
        for (int i=0;i<n_fields;i++)
          out_buffer_sgsf_ptr[inter*n_fpts_per_inter*n_fields*n_dims+j*n_fpts_per_inter*n_fields+i*n_fpts_per_inter+fpt]=dq[i][j];

    }

}

#endif

void RK45_update_kernel_wrapper(int n_upts_per_ele,int n_dims,int n_fields,int n_eles,double* disu0_upts_ptr,double* disu1_upts_ptr,double* div_tconf_upts_ptr, double* detjac_upts_ptr, double* src_upts_ptr, double* h_ref, double rk4a, double rk4b, double dt, double const_src, double CFL, double gamma, double mu_inf, int order, int viscous, int dt_type, int step)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_eles*n_upts_per_ele-1)/256)+1;

  if (n_dims == 2)
  {
    if (n_fields==1)
    {
      RK45_update_kernel <2,1> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, disu1_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, rk4a, rk4b, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type, step);
    }
    else if (n_fields==4)
    {
      RK45_update_kernel <2,4> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, disu1_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, rk4a, rk4b, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type, step);
    }
    else if (n_fields==5)
    {
      RK45_update_kernel <2,5> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, disu1_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, rk4a, rk4b, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type, step);
    }
    else
      FatalError("ERROR: Invalid number of fields for this dimension ... ")
  }
  else if (n_dims == 3)
  {
    if (n_fields==1)
    {
      RK45_update_kernel <3,1> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, disu1_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, rk4a, rk4b, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type, step);
    }
    else if (n_fields==5)
    {
      RK45_update_kernel <3,5> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, disu1_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, rk4a, rk4b, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type, step);
    }
    else if (n_fields==6)
    {
      RK45_update_kernel <3,6> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, disu1_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, rk4a, rk4b, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type, step);
    }
    else
      FatalError("ERROR: Invalid number of fields for this dimension ... ")
  }
  else
    FatalError("ERROR: Invalid number of dimensions ... ");
}

void RK11_update_kernel_wrapper(int n_upts_per_ele,int n_dims,int n_fields,int n_eles,double* disu0_upts_ptr,double* div_tconf_upts_ptr, double* detjac_upts_ptr, double* src_upts_ptr, double* h_ref, double dt, double const_src, double CFL, double gamma, double mu_inf, int order, int viscous, int dt_type)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_eles*n_upts_per_ele-1)/256)+1;

  if (n_dims == 2)
  {
    if (n_fields==1)
    {
      RK11_update_kernel <2,1> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type);
    }
    else if (n_fields==4)
    {
      RK11_update_kernel <2,4> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type);
    }
    else if (n_fields==5)
    {
      RK11_update_kernel <2,5> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type);
    }
    else
      FatalError("ERROR: Invalid number of fields for this dimension ... ")
  }
  else if (n_dims==3)
  {
    if (n_fields==1)
    {
      RK11_update_kernel <3,1> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type);
    }
    else if (n_fields==5)
    {
      RK11_update_kernel <3,5> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type);
    }
    else if (n_fields==6)
    {
      RK11_update_kernel <3,6> <<< n_blocks,256>>> (disu0_upts_ptr, div_tconf_upts_ptr, detjac_upts_ptr, src_upts_ptr, h_ref, n_eles, n_upts_per_ele, dt, const_src, CFL, gamma, mu_inf, order, viscous, dt_type);
    }
    else
      FatalError("ERROR: Invalid number of fields for this dimension ... ")
  }
  else
    FatalError("ERROR: Invalid number of dimensions ... ");
}


// wrapper for gpu kernel to calculate transformed discontinuous inviscid flux at solution points
void evaluate_invFlux_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* disu_upts_ptr, double* out_tdisf_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double* grid_vel_upts_ptr, double gamma, int motion, int equation, double wave_speed_x, double wave_speed_y, double wave_speed_z, int turb_model)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_eles*n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (equation==0)
  {
    if (n_dims==2)
    {
      if (n_fields==4)
      {
        evaluate_invFlux_NS_gpu_kernel<2,4> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,out_tdisf_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,grid_vel_upts_ptr,gamma,motion,turb_model);
      }
      else if (n_fields==5)
      {
        evaluate_invFlux_NS_gpu_kernel<2,5> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,out_tdisf_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,grid_vel_upts_ptr,gamma,motion,turb_model);
      }
      else
        FatalError("ERROR: Invalid number of fields for this dimension ... ")
    }
    else if (n_dims==3)
    {
      if (n_fields==5)
      {
        evaluate_invFlux_NS_gpu_kernel<3,5> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,out_tdisf_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,grid_vel_upts_ptr,gamma,motion,turb_model);
      }
      else if (n_fields==6)
      {
        evaluate_invFlux_NS_gpu_kernel<3,6> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,out_tdisf_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,grid_vel_upts_ptr,gamma,motion,turb_model);
      }
      else
        FatalError("ERROR: Invalid number of fields for this dimension ... ")
    }
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else if (equation==1)
  {
    if (n_dims==2)
      evaluate_invFlux_AD_gpu_kernel<2> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,out_tdisf_upts_ptr,detjac_upts_ptr,JGinv_upts_ptr,wave_speed_x,wave_speed_y,wave_speed_z);
    else if (n_dims==3)
      evaluate_invFlux_AD_gpu_kernel<3> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,out_tdisf_upts_ptr,detjac_upts_ptr,JGinv_upts_ptr,wave_speed_x,wave_speed_y,wave_speed_z);
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else
    FatalError("equation not recognized");

  check_cuda_error("After",__FILE__, __LINE__);
}



// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points
void calculate_common_invFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconinvf_fpts_l_ptr, double** norm_tconinvf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** tdA_dyn_fpts_l_ptr, double **tdA_dyn_fpts_r_ptr, double** detjac_dyn_fpts_l_ptr, double** detjac_dyn_fpts_r_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, int riemann_solve_type, double **delta_disu_fpts_l_ptr, double **delta_disu_fpts_r_ptr, double gamma, double pen_fact, int viscous, int motion, int vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda, int turb_model)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_inters*n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);
  
  if (riemann_solve_type==0) // Rusanov
  {
    if(vis_riemann_solve_type==0) //LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          calculate_common_invFlux_NS_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==5)
        {
          calculate_common_invFlux_NS_gpu_kernel<2,5,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          calculate_common_invFlux_NS_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==6)
        {
          calculate_common_invFlux_NS_gpu_kernel<3,6,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if ( riemann_solve_type==2) // Roe
  {
    if(vis_riemann_solve_type==0) //LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          calculate_common_invFlux_NS_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==5)
        {
          calculate_common_invFlux_NS_gpu_kernel<2,5,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          calculate_common_invFlux_NS_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==6)
        {
          calculate_common_invFlux_NS_gpu_kernel<3,6,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,tdA_dyn_fpts_l_ptr,tdA_dyn_fpts_r_ptr,detjac_dyn_fpts_l_ptr,detjac_dyn_fpts_r_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if (riemann_solve_type==1) // Lax-Friedrich
  {
    if(vis_riemann_solve_type==0) //LDG
    {
      if (n_dims==2)
        calculate_common_invFlux_lax_friedrich_gpu_kernel<2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,norm_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,pen_fact,viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
      else if (n_dims==3)
        calculate_common_invFlux_lax_friedrich_gpu_kernel<3,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconinvf_fpts_l_ptr,norm_tconinvf_fpts_r_ptr,tdA_fpts_l_ptr,tdA_fpts_r_ptr,norm_fpts_ptr,delta_disu_fpts_l_ptr,delta_disu_fpts_r_ptr,pen_fact,viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else
    FatalError("ERROR: Riemann solver type not recognized ... ");

  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries
void evaluate_boundaryConditions_invFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_l_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** loc_fpts_ptr, double** loc_dyn_fpts_ptr, double** grid_vel_fpts_ptr, int* boundary_type, double* bdy_params, int riemann_solve_type, double** delta_disu_fpts_l_ptr, double gamma, double R_ref, int viscous, int motion, int vis_riemann_solve_type, double time_bound, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda, int equation, int turb_model)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_inters*n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (riemann_solve_type==0)  // Rusanov
  {
    if (vis_riemann_solve_type==0) // LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else if (n_fields==5)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<2,5,0,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else if (n_fields==6)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<3,6,0,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
  }
  else if (riemann_solve_type==1)  // Lax-Friedrichs
  {
    if (vis_riemann_solve_type==0) // LDG
    {
      if (n_dims==2)
        evaluate_boundaryConditions_invFlux_gpu_kernel<2,1,1,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation, turb_model);
      else if (n_dims==3)
        evaluate_boundaryConditions_invFlux_gpu_kernel<3,1,1,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation, turb_model);
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
  }
  else if (riemann_solve_type==2) // Roe
  {
    if (vis_riemann_solve_type==0) // LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else if (n_fields==5)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<2,5,2,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else if (n_fields==6)
        {
          evaluate_boundaryConditions_invFlux_gpu_kernel<3,6,2,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, grid_vel_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, gamma, R_ref, viscous, motion, time_bound, wave_speed_x, wave_speed_y, wave_speed_z, lambda, equation,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
  }
  else
    FatalError("ERROR: Riemann solver type not recognized in bdy riemann solver");

  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate transformed discontinuous viscous flux at solution points
void evaluate_viscFlux_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, int ele_type, int order, double filter_ratio, int LES, int motion, int sgs_model, int wall_model, double wall_thickness, double* wall_dist_ptr, double* twall_ptr, double* Lu_ptr, double* Le_ptr, double* disu_upts_ptr, double* out_tdisf_upts_ptr, double* out_sgsf_upts_ptr, double* grad_disu_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, int equation, double diff_coeff, int turb_model, double c_v1, double omega, double prandtl_t)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_eles*n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (equation==0)
  {
    if (n_dims==2)
    {
      if (n_fields==4)
      {
        evaluate_viscFlux_NS_gpu_kernel<2,4,3> <<<n_blocks,256>>>(n_upts_per_ele, n_eles, ele_type, order, filter_ratio, LES, motion, sgs_model, wall_model, wall_thickness, wall_dist_ptr, twall_ptr, Lu_ptr, Le_ptr, disu_upts_ptr, out_tdisf_upts_ptr, out_sgsf_upts_ptr, grad_disu_upts_ptr, detjac_upts_ptr, detjac_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, turb_model, c_v1, omega, prandtl_t);
      }
      else if (n_fields==5)
      {
        evaluate_viscFlux_NS_gpu_kernel<2,5,3> <<<n_blocks,256>>>(n_upts_per_ele, n_eles, ele_type, order, filter_ratio, LES, motion, sgs_model, wall_model, wall_thickness, wall_dist_ptr, twall_ptr, Lu_ptr, Le_ptr, disu_upts_ptr, out_tdisf_upts_ptr, out_sgsf_upts_ptr, grad_disu_upts_ptr, detjac_upts_ptr, detjac_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, turb_model, c_v1, omega, prandtl_t);
      }
      else
        FatalError("ERROR: Invalid number of fields for this dimension ... ")
    }
    else if (n_dims==3)
    {
      if (n_fields==5)
      {
        evaluate_viscFlux_NS_gpu_kernel<3,5,6> <<<n_blocks,256>>>(n_upts_per_ele, n_eles, ele_type, order, filter_ratio, LES, motion, sgs_model, wall_model, wall_thickness, wall_dist_ptr, twall_ptr, Lu_ptr, Le_ptr, disu_upts_ptr, out_tdisf_upts_ptr, out_sgsf_upts_ptr, grad_disu_upts_ptr, detjac_upts_ptr, detjac_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, turb_model, c_v1, omega, prandtl_t);
      }
      else if (n_fields==6)
      {
        evaluate_viscFlux_NS_gpu_kernel<3,6,6> <<<n_blocks,256>>>(n_upts_per_ele, n_eles, ele_type, order, filter_ratio, LES, motion, sgs_model, wall_model, wall_thickness, wall_dist_ptr, twall_ptr, Lu_ptr, Le_ptr, disu_upts_ptr, out_tdisf_upts_ptr, out_sgsf_upts_ptr, grad_disu_upts_ptr, detjac_upts_ptr, detjac_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, turb_model, c_v1, omega, prandtl_t);
      }
      else
        FatalError("ERROR: Invalid number of fields for this dimension ... ")
    }
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else if (equation==1)
  {
    if (n_dims==2)
      evaluate_viscFlux_AD_gpu_kernel<2> <<<n_blocks,256>>>(n_upts_per_ele, n_eles, disu_upts_ptr, out_tdisf_upts_ptr, grad_disu_upts_ptr, detjac_upts_ptr, JGinv_upts_ptr, diff_coeff);
    else if (n_dims==3)
      evaluate_viscFlux_AD_gpu_kernel<3> <<<n_blocks,256>>>(n_upts_per_ele, n_eles, disu_upts_ptr, out_tdisf_upts_ptr, grad_disu_upts_ptr, detjac_upts_ptr, JGinv_upts_ptr, diff_coeff);
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else
    FatalError("equation not recognized");

  check_cuda_error("After",__FILE__, __LINE__);
}

// wrapper for gpu kernel to transform gradient at sol points to physical gradient
void transform_grad_disu_upts_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* grad_disu_upts_ptr, double* detjac_upts_ptr, double* detjac_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, int equation, int motion)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_eles*n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(equation == 0) {
    if (n_dims==2)
    {
      if (n_fields==4)
      {
        transform_grad_disu_upts_kernel<2,4> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,grad_disu_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,motion);
      }
      else if (n_fields==5)
      {
        transform_grad_disu_upts_kernel<2,5> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,grad_disu_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,motion);
      }
      else
        FatalError("ERROR: Invalid number of fields for this dimension ... ")
    }
    else if (n_dims==3)
    {
      if (n_fields==5)
      {
        transform_grad_disu_upts_kernel<3,5> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,grad_disu_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,motion);
      }
      else if (n_fields==6)
      {
        transform_grad_disu_upts_kernel<3,6> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,grad_disu_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,motion);
      }
      else
        FatalError("ERROR: Invalid number of fields for this dimension ... ")
    }
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else if(equation == 1) {
    if (n_dims==2)
      transform_grad_disu_upts_kernel<2,1> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,grad_disu_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,motion);
    else if (n_dims==3)
      transform_grad_disu_upts_kernel<3,1> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,grad_disu_upts_ptr,detjac_upts_ptr,detjac_dyn_upts_ptr,JGinv_upts_ptr,JGinv_dyn_upts_ptr,motion);
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else
    FatalError("equation not recognized");

  check_cuda_error("After",__FILE__, __LINE__);
}


// wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points
void calculate_common_viscFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** norm_tconf_fpts_r_ptr, double** tdA_fpts_l_ptr, double** tdA_fpts_r_ptr, double** tdA_dyn_fpts_l_ptr, double** tdA_dyn_fpts_r_ptr, double** detjac_dyn_fpts_l_ptr, double** detjac_dyn_fpts_r_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** sgsf_fpts_l_ptr, double** sgsf_fpts_r_ptr, int riemann_solve_type, int vis_riemann_solve_type, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, int equation, double diff_coeff, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_inters*n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(equation==0)
  {
    if (vis_riemann_solve_type==0) // LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          calculate_common_viscFlux_NS_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, norm_tconf_fpts_r_ptr, tdA_fpts_l_ptr, tdA_fpts_r_ptr, tdA_dyn_fpts_l_ptr, tdA_dyn_fpts_r_ptr, detjac_dyn_fpts_l_ptr, detjac_dyn_fpts_r_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else if (n_fields==5)
        {
          calculate_common_viscFlux_NS_gpu_kernel<2,5,3,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, norm_tconf_fpts_r_ptr, tdA_fpts_l_ptr, tdA_fpts_r_ptr, tdA_dyn_fpts_l_ptr, tdA_dyn_fpts_r_ptr, detjac_dyn_fpts_l_ptr, detjac_dyn_fpts_r_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          calculate_common_viscFlux_NS_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, norm_tconf_fpts_r_ptr, tdA_fpts_l_ptr, tdA_fpts_r_ptr, tdA_dyn_fpts_l_ptr, tdA_dyn_fpts_r_ptr, detjac_dyn_fpts_l_ptr, detjac_dyn_fpts_r_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else if (n_fields==6)
        {
          calculate_common_viscFlux_NS_gpu_kernel<3,6,6,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, norm_tconf_fpts_r_ptr, tdA_fpts_l_ptr, tdA_fpts_r_ptr, tdA_dyn_fpts_l_ptr, tdA_dyn_fpts_r_ptr, detjac_dyn_fpts_l_ptr, detjac_dyn_fpts_r_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if(equation==1)
  {
    if (vis_riemann_solve_type==0) // LDG
    {
      if (n_dims==2)
        calculate_common_viscFlux_AD_gpu_kernel<2> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, norm_tconf_fpts_r_ptr, tdA_fpts_l_ptr, tdA_fpts_r_ptr, norm_fpts_ptr, pen_fact, tau, diff_coeff);
      else if (n_dims==3)
        calculate_common_viscFlux_AD_gpu_kernel<3> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, norm_tconf_fpts_r_ptr, tdA_fpts_l_ptr, tdA_fpts_r_ptr, norm_fpts_ptr, pen_fact, tau, diff_coeff);
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else
    FatalError("equation not recognized");

  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points at boundaries
void evaluate_boundaryConditions_viscFlux_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** grad_disu_fpts_l_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_l_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, double** loc_fpts_ptr, double** loc_dyn_fpts_ptr, double** sgsf_fpts_ptr, int* boundary_type, double* bdy_params, double** delta_disu_fpts_l_ptr, int riemann_solve_type, int vis_riemann_solve_type, double R_ref, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, double time_bound, int equation, double diff_coeff, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_inters*n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (vis_riemann_solve_type==0) // LDG
  {
    if(equation==0)
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          evaluate_boundaryConditions_viscFlux_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, grad_disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, grid_vel_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, sgsf_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, R_ref, pen_fact, tau, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, time_bound, equation, diff_coeff, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else if (n_fields==5)
        {
          evaluate_boundaryConditions_viscFlux_gpu_kernel<2,5,3,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, grad_disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, grid_vel_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, sgsf_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, R_ref, pen_fact, tau, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, time_bound, equation, diff_coeff, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          evaluate_boundaryConditions_viscFlux_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, grad_disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, grid_vel_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, sgsf_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, R_ref, pen_fact, tau, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, time_bound, equation, diff_coeff, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else if (n_fields==6)
        {
          evaluate_boundaryConditions_viscFlux_gpu_kernel<3,6,6,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, grad_disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, grid_vel_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, sgsf_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, R_ref, pen_fact, tau, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, time_bound, equation, diff_coeff, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else if(equation==1)
    {
      if (n_dims==2)
        evaluate_boundaryConditions_viscFlux_gpu_kernel<2,1,1,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, grad_disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, grid_vel_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, sgsf_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, R_ref, pen_fact, tau, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, time_bound, equation, diff_coeff, LES, motion, turb_model, c_v1, omega, prandtl_t);
      else if (n_dims==3)
        evaluate_boundaryConditions_viscFlux_gpu_kernel<3,1,1,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, grad_disu_fpts_l_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_l_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, grid_vel_fpts_ptr, loc_fpts_ptr, loc_dyn_fpts_ptr, sgsf_fpts_ptr, boundary_type, bdy_params, delta_disu_fpts_l_ptr, R_ref, pen_fact, tau, gamma, prandtl, rt_inf, mu_inf, c_sth, fix_vis, time_bound, equation, diff_coeff, LES, motion, turb_model, c_v1, omega, prandtl_t);
    }
  }
  else
    FatalError("ERROR: Viscous riemann solver type not recognized ... ");

  check_cuda_error("After", __FILE__, __LINE__);
}


// gpu kernel wrapper to calculate source term for SA turbulence model
void calc_src_upts_SA_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* disu_upts_ptr, double* grad_disu_upts_ptr, double* wall_distance_mag_ptr, double* src_upts_ptr, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, int fix_vis, double c_v1, double c_v2, double c_v3, double c_b1, double c_b2, double c_w2, double c_w3, double omega, double Kappa)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_eles*n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(n_dims == 2)
    calc_src_upts_gpu_kernel<2,5> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,grad_disu_upts_ptr,wall_distance_mag_ptr,src_upts_ptr,gamma,prandtl,rt_inf,mu_inf,c_sth,fix_vis,c_v1,c_v2,c_v3,c_b1,c_b2,c_w2,c_w3,omega,Kappa);
  else if(n_dims == 3)
    calc_src_upts_gpu_kernel<3,6> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,disu_upts_ptr,grad_disu_upts_ptr,wall_distance_mag_ptr,src_upts_ptr,gamma,prandtl,rt_inf,mu_inf,c_sth,fix_vis,c_v1,c_v2,c_v3,c_b1,c_b2,c_w2,c_w3,omega,Kappa);
  else
    FatalError("ERROR: Invalid number of dimensions ... ");

  check_cuda_error("After",__FILE__, __LINE__);
}


/*! wrapper for gpu kernel to calculate terms for similarity model */
void calc_similarity_model_kernel_wrapper(int flag, int n_fields, int n_upts_per_ele, int n_eles, int n_dims, double* disu_upts_ptr, double* disuf_upts_ptr, double* uu_ptr, double* ue_ptr, double* Lu_ptr, double* Le_ptr)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((n_eles*n_upts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  /*! Calculate product terms uu, ue */
  if (flag==0) {
    // fixed n_fields at 4 for 2d and 5 for 3d
    if(n_dims==2) {
      calc_similarity_terms_kernel<4> <<< n_blocks,block_size >>> (n_upts_per_ele, n_eles, n_dims, disu_upts_ptr, uu_ptr, ue_ptr);
    }
    else if(n_dims==3) {
      calc_similarity_terms_kernel<5> <<< n_blocks,block_size >>> (n_upts_per_ele, n_eles, n_dims, disu_upts_ptr, uu_ptr, ue_ptr);
    }
  }
  /*! Calculate Leonard tensors Lu, Le */
  else if (flag==1) {
    // fixed n_fields at 4 for 2d and 5 for 3d
    if(n_dims==2) {
      calc_Leonard_tensors_kernel<4> <<< n_blocks,block_size >>> (n_upts_per_ele, n_eles, n_dims, disuf_upts_ptr, Lu_ptr, Le_ptr);
    }
    else if(n_dims==3) {
      calc_Leonard_tensors_kernel<5> <<< n_blocks,block_size >>> (n_upts_per_ele, n_eles, n_dims, disuf_upts_ptr, Lu_ptr, Le_ptr);
    }
  }

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void rigid_motion_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* shape_dyn, double* motion_params, double rk_time)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((n_eles*max_n_spts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  rigid_motion_kernel <<< n_blocks,block_size >>> (n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele,shape,shape_dyn,motion_params,rk_time);

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void perturb_shape_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* shape_dyn, double rk_time)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((n_eles*max_n_spts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  perturb_shape_kernel <<< n_blocks,block_size >>> (n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele,shape,shape_dyn,rk_time);

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void push_back_xv_kernel_wrapper(int n_dims, int n_verts, double* xv_1, double* xv_2)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((n_verts-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(n_dims==2) {
    push_back_xv_kernel<2> <<< n_blocks,block_size >>> (n_verts,xv_1,xv_2);
  }
  else if(n_dims==3) {
    push_back_xv_kernel<3> <<< n_blocks,block_size >>> (n_verts,xv_1,xv_2);
  }

  check_cuda_error("After",__FILE__, __LINE__);
}

void push_back_shape_dyn_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int n_levels, int* n_spts_per_ele, double* shape_dyn)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((n_eles*max_n_spts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(n_dims==2) {
    push_back_shape_dyn_kernel<2> <<< n_blocks,block_size >>> (n_eles,max_n_spts_per_ele,n_levels,n_spts_per_ele,shape_dyn);
  }
  else if(n_dims==3) {
    push_back_shape_dyn_kernel<3> <<< n_blocks,block_size >>> (n_eles,max_n_spts_per_ele,n_levels,n_spts_per_ele,shape_dyn);
  }

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! Wrapper for gpu kernel to calculate the grid velocity at the shape points using backward-difference formula */
void calc_grid_vel_spts_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape_dyn, double* grid_vel, double dt)
{
  int block_size=256;
  int n_blocks=((n_eles*max_n_spts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(n_dims==2) {
    calc_grid_vel_spts_kernel<2> <<< n_blocks,block_size >>> (n_eles,max_n_spts_per_ele,n_spts_per_ele,shape_dyn,grid_vel,dt);
  }
  else if(n_dims==3) {
    calc_grid_vel_spts_kernel<3> <<< n_blocks,block_size >>> (n_eles,max_n_spts_per_ele,n_spts_per_ele,shape_dyn,grid_vel,dt);
  }

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! Wrapper for gpu kernel to calculate the grid velocity at the shape points using backward-difference formula */
void calc_rigid_grid_vel_spts_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* motion_params, double* grid_vel, double rk_time)
{
  int block_size=256;
  int n_blocks=((n_eles*max_n_spts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  rigid_motion_velocity_spts_kernel <<< n_blocks,block_size >>> (n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele,motion_params,grid_vel,rk_time);

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! Wrapper for gpu kernel to calculate the grid velocity at the shape points using backward-difference formula */
void calc_perturb_grid_vel_spts_kernel_wrapper(int n_dims, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* shape, double* grid_vel, double rk_time)
{
  int block_size=256;
  int n_blocks=((n_eles*max_n_spts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  perturb_grid_velocity_spts_kernel <<< n_blocks,block_size >>> (n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele,shape,grid_vel,rk_time);

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! Wrapper for gpu kernel to interpolate the grid veloicty at the shape points to either the solution or flux points */
void eval_grid_vel_pts_kernel_wrapper(int n_dims, int n_eles, int n_pts_per_ele, int max_n_spts_per_ele, int* n_spts_per_ele, double* nodal_s_basis_pts, double* grid_vel_spts, double* grid_vel_pts)
{
  int block_size=256;
  int n_blocks=((n_eles*n_pts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  eval_grid_vel_pts_kernel <<< n_blocks,block_size >>> (n_dims, n_eles, n_pts_per_ele, max_n_spts_per_ele, n_spts_per_ele, nodal_s_basis_pts, grid_vel_spts, grid_vel_pts);

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void set_transforms_dynamic_upts_kernel_wrapper(int n_upts_per_ele, int n_eles, int n_dims, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_upts_ptr, double* J_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double* d_nodal_s_basis_upts, double* shape_dyn)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((n_eles*n_upts_per_ele-1)/block_size)+1;
  int err = 0;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(n_dims==2) {
    set_transforms_dynamic_upts_kernel<2> <<< n_blocks,block_size >>> (n_upts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_upts_ptr, J_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, d_nodal_s_basis_upts, shape_dyn);
  }
  else if(n_dims==3) {
    set_transforms_dynamic_upts_kernel<3> <<< n_blocks,block_size >>> (n_upts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_upts_ptr, J_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, d_nodal_s_basis_upts, shape_dyn);
  }

  if (err)
    cout << "ERROR: Negative Jacobian found at solution point!" << endl;

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void set_transforms_dynamic_fpts_kernel_wrapper(int n_fpts_per_ele, int n_eles, int n_dims, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_fpts_ptr, double* J_dyn_fpts_ptr, double* JGinv_fpts_ptr, double* JGinv_dyn_fpts_ptr, double* tdA_dyn_fpts_ptr, double* norm_fpts_ptr, double* norm_dyn_fpts_ptr, double* d_nodal_s_basis_fpts, double* shape_dyn)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((n_eles*n_fpts_per_ele-1)/block_size)+1;
  //int *err = new int[1];

  check_cuda_error("Before", __FILE__, __LINE__);

  if(n_dims==2) {
    set_transforms_dynamic_fpts_kernel<2> <<< n_blocks,block_size >>> (n_fpts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_fpts_ptr, J_dyn_fpts_ptr, JGinv_fpts_ptr, JGinv_dyn_fpts_ptr, tdA_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, d_nodal_s_basis_fpts, shape_dyn);
  }
  else if(n_dims==3) {
    set_transforms_dynamic_fpts_kernel<3> <<< n_blocks,block_size >>> (n_fpts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_fpts_ptr, J_dyn_fpts_ptr, JGinv_fpts_ptr, JGinv_dyn_fpts_ptr, tdA_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, d_nodal_s_basis_fpts, shape_dyn);
  }

  /*if (*err)
    cout << "ERROR: Negative Jacobian found at flux point!" << endl;*/

  check_cuda_error("After",__FILE__, __LINE__);
}

// Wrapper for gpu kernel for shock capturing using artificial viscosity
void shock_capture_concentration_gpu_kernel_wrapper(int in_n_eles, int in_n_upts_per_ele, int in_n_fields, int in_order, int in_ele_type, int in_artif_type, double s0, double kappa, double* in_disu_upts_ptr, double* in_inv_vandermonde_ptr, double* in_inv_vandermonde2D_ptr, double* in_vandermonde2D_ptr, double* concentration_array_ptr, double* out_sensor, double* sigma)
{
    cudaError_t err;

    // HACK: fix 256 threads per block
    int n_blocks=((in_n_eles-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

    shock_capture_concentration_gpu_kernel<<<n_blocks,256>>>(in_n_eles, in_n_upts_per_ele, in_n_fields, in_order, in_ele_type, in_artif_type, s0, kappa, in_disu_upts_ptr, in_inv_vandermonde_ptr, in_inv_vandermonde2D_ptr, in_vandermonde2D_ptr, concentration_array_ptr, out_sensor, sigma);

    // This thread synchronize may not be necessary
    err=cudaThreadSynchronize();
    if( err != cudaSuccess)
            printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));

      check_cuda_error("After",__FILE__, __LINE__);
}

// wrapper for gpu kernel to add body force to viscous flux
void evaluate_body_force_gpu_kernel_wrapper(int n_upts_per_ele, int n_dims, int n_fields, int n_eles, double* src_upts_ptr, double* body_force_ptr)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_eles*n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (n_dims==2) {
    if (n_fields==4) {
      evaluate_body_force_gpu_kernel<2,4> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,src_upts_ptr,body_force_ptr);
    }
    else if (n_fields==5) {
      evaluate_body_force_gpu_kernel<2,5> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,src_upts_ptr,body_force_ptr);
    }
    else
      FatalError("ERROR: Invalid number of fields for this dimension ... ")
  }
  else if (n_dims==3) {
    if (n_fields==5) {
      evaluate_body_force_gpu_kernel<3,5> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,src_upts_ptr,body_force_ptr);
    }
    else if (n_fields==6) {
      evaluate_body_force_gpu_kernel<3,6> <<<n_blocks,256>>>(n_upts_per_ele,n_eles,src_upts_ptr,body_force_ptr);
    }
    else
      FatalError("ERROR: Invalid number of fields for this dimension ... ")
  }
  else
    FatalError("ERROR: Invalid number of dimensions ... ");

  check_cuda_error("After",__FILE__, __LINE__);
}

#ifdef _MPI

void pack_out_buffer_disu_gpu_kernel_wrapper(int n_fpts_per_inter,int n_inters,int n_fields,double** disu_fpts_l_ptr, double* out_buffer_disu_ptr)
{
  int block_size=256;
  int n_blocks=((n_inters*n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (n_fields==1)
    pack_out_buffer_disu_gpu_kernel<1> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,disu_fpts_l_ptr,out_buffer_disu_ptr);
  else if (n_fields==4)
    pack_out_buffer_disu_gpu_kernel<4> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,disu_fpts_l_ptr,out_buffer_disu_ptr);
  else if (n_fields==5)
    pack_out_buffer_disu_gpu_kernel<5> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,disu_fpts_l_ptr,out_buffer_disu_ptr);
  else if (n_fields==6)
    pack_out_buffer_disu_gpu_kernel<6> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,disu_fpts_l_ptr,out_buffer_disu_ptr);
  else
    FatalError("Number of fields not supported in pack_out_buffer");

  check_cuda_error("After", __FILE__, __LINE__);

}

void pack_out_buffer_grad_disu_gpu_kernel_wrapper(int n_fpts_per_inter,int n_inters,int n_fields,int n_dims, double** grad_disu_fpts_l_ptr, double* out_buffer_grad_disu_ptr)
{
  int block_size=256;
  int n_blocks=((n_inters*n_fpts_per_inter*n_dims-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (n_dims==2)
  {
    if (n_fields==1)
    {
      pack_out_buffer_grad_disu_gpu_kernel<1,2> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,grad_disu_fpts_l_ptr,out_buffer_grad_disu_ptr);
    }
    else if (n_fields==4)
    {
      pack_out_buffer_grad_disu_gpu_kernel<4,2> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,grad_disu_fpts_l_ptr,out_buffer_grad_disu_ptr);
    }
    else if (n_fields==5)
    {
      pack_out_buffer_grad_disu_gpu_kernel<5,2> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,grad_disu_fpts_l_ptr,out_buffer_grad_disu_ptr);
    }
    else
      FatalError("Number of fields not supported for this dimension in pack_out_buffer");
  }
  else if (n_dims==3)
  {
    if (n_fields==1)
    {
      pack_out_buffer_grad_disu_gpu_kernel<1,3> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,grad_disu_fpts_l_ptr,out_buffer_grad_disu_ptr);
    }
    else if (n_fields==5)
    {
      pack_out_buffer_grad_disu_gpu_kernel<5,3> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,grad_disu_fpts_l_ptr,out_buffer_grad_disu_ptr);
    }
    else if (n_fields==6)
    {
      pack_out_buffer_grad_disu_gpu_kernel<6,3> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,grad_disu_fpts_l_ptr,out_buffer_grad_disu_ptr);
    }
    else
      FatalError("Number of fields not supported for this dimension in pack_out_buffer");
  }
  else
    FatalError("Number of dimensions not supported in pack_out_buffer");

  check_cuda_error("After", __FILE__, __LINE__);
}

void pack_out_buffer_sgsf_gpu_kernel_wrapper(int n_fpts_per_inter,int n_inters,int n_fields,int n_dims, double** sgsf_fpts_l_ptr, double* out_buffer_sgsf_ptr)
{
  int block_size=256;
  int n_blocks=((n_inters*n_fpts_per_inter*n_dims-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (n_dims==2)
  {
    if (n_fields==1)
    {
      pack_out_buffer_sgsf_gpu_kernel<1,2> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,sgsf_fpts_l_ptr,out_buffer_sgsf_ptr);
    }
    else if (n_fields==4)
    {
      pack_out_buffer_sgsf_gpu_kernel<4,2> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,sgsf_fpts_l_ptr,out_buffer_sgsf_ptr);
    }
    else if (n_fields==5)
    {
      pack_out_buffer_sgsf_gpu_kernel<5,2> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,sgsf_fpts_l_ptr,out_buffer_sgsf_ptr);
    }
    else
      FatalError("Number of fields not supported for this dimension in pack_out_buffer");
  }
  else if (n_dims==3)
  {
    if (n_fields==1)
    {
      pack_out_buffer_sgsf_gpu_kernel<1,3> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,sgsf_fpts_l_ptr,out_buffer_sgsf_ptr);
    }
    else if (n_fields==5)
    {
      pack_out_buffer_sgsf_gpu_kernel<5,3> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,sgsf_fpts_l_ptr,out_buffer_sgsf_ptr);
    }
    else if (n_fields==6)
    {
      pack_out_buffer_sgsf_gpu_kernel<6,3> <<< n_blocks,block_size >>> (n_fpts_per_inter,n_inters,sgsf_fpts_l_ptr,out_buffer_sgsf_ptr);
    }
    else
      FatalError("Number of fields not supported for this dimension in pack_out_buffer");
  }
  else
    FatalError("Number of dimensions not supported in pack_out_buffer");

  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points
void calculate_common_invFlux_mpi_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** grid_vel_fpts_ptr, int riemann_solve_type, double** delta_disu_fpts_l_ptr, double gamma, double pen_fact,  int viscous, int motion, int vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda, int turb_model)
{
  int block_size=256;
  int n_blocks=((n_inters*n_fpts_per_inter-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (riemann_solve_type==0 ) // Rusanov
  {
    if (vis_riemann_solve_type==0 ) //LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==5)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<2,5,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==6)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<3,6,0,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if (riemann_solve_type==2 ) // Roe
  {
    if (vis_riemann_solve_type==0 ) //LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==5)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<2,5,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else if (n_fields==6)
        {
          calculate_common_invFlux_NS_mpi_gpu_kernel<3,6,2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,tdA_dyn_fpts_l_ptr,detjac_dyn_fpts_ptr,norm_fpts_ptr,norm_dyn_fpts_ptr,grid_vel_fpts_ptr,delta_disu_fpts_l_ptr,gamma,pen_fact,viscous,motion,turb_model);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if (riemann_solve_type==1) // Lax-Friedrich
  {
    if(vis_riemann_solve_type==0) //LDG
    {
      if (n_dims==2)
        calculate_common_invFlux_lax_friedrich_mpi_gpu_kernel<2,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,norm_fpts_ptr,delta_disu_fpts_l_ptr,pen_fact,viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
      else if (n_dims==3)
        calculate_common_invFlux_lax_friedrich_mpi_gpu_kernel<3,0> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,norm_fpts_ptr,delta_disu_fpts_l_ptr,pen_fact,viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
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
void calculate_common_viscFlux_mpi_gpu_kernel_wrapper(int n_fpts_per_inter, int n_dims, int n_fields, int n_inters, double** disu_fpts_l_ptr, double** disu_fpts_r_ptr, double** grad_disu_fpts_l_ptr, double** grad_disu_fpts_r_ptr, double** norm_tconf_fpts_l_ptr, double** tdA_fpts_l_ptr, double** tdA_dyn_fpts_l_ptr, double** detjac_dyn_fpts_ptr, double** norm_fpts_ptr, double** norm_dyn_fpts_ptr, double** sgsf_fpts_l_ptr, double** sgsf_fpts_r_ptr, int riemann_solve_type, int vis_riemann_solve_type, double pen_fact, double tau, double gamma, double prandtl, double rt_inf, double mu_inf, double c_sth, double fix_vis, double diff_coeff, int LES, int motion, int turb_model, double c_v1, double omega, double prandtl_t)
{

  // HACK: fix 256 threads per block
  int n_blocks=((n_inters*n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (riemann_solve_type==0 ) // Rusanov
  {
    if (vis_riemann_solve_type==0) // LDG
    {
      if (n_dims==2)
      {
        if (n_fields==4)
        {
          calculate_common_viscFlux_NS_mpi_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else if (n_fields==5)
        {
          calculate_common_viscFlux_NS_mpi_gpu_kernel<2,5,3,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else if (n_dims==3)
      {
        if (n_fields==5)
        {
          calculate_common_viscFlux_NS_mpi_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else if (n_fields==6)
        {
          calculate_common_viscFlux_NS_mpi_gpu_kernel<3,6,6,0> <<<n_blocks,256>>>(n_fpts_per_inter, n_inters, disu_fpts_l_ptr, disu_fpts_r_ptr, grad_disu_fpts_l_ptr, grad_disu_fpts_r_ptr, norm_tconf_fpts_l_ptr, tdA_fpts_l_ptr, tdA_dyn_fpts_l_ptr, detjac_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, sgsf_fpts_l_ptr, sgsf_fpts_r_ptr, pen_fact, tau, gamma, prandtl, rt_inf,  mu_inf, c_sth, fix_vis, LES, motion, turb_model, c_v1, omega, prandtl_t);
        }
        else
          FatalError("ERROR: Invalid number of fields for this dimension ... ")
      }
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else if (riemann_solve_type==1) // Lax-Friedrich
  {
    if (vis_riemann_solve_type==0) // LDG
    {
      if (n_dims==2)
        calculate_common_viscFlux_AD_mpi_gpu_kernel<2> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,grad_disu_fpts_l_ptr,grad_disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,norm_fpts_ptr,pen_fact,tau,diff_coeff);
      else if (n_dims==3)
        calculate_common_viscFlux_AD_mpi_gpu_kernel<3> <<<n_blocks,256>>>(n_fpts_per_inter,n_inters,disu_fpts_l_ptr,disu_fpts_r_ptr,grad_disu_fpts_l_ptr,grad_disu_fpts_r_ptr,norm_tconf_fpts_l_ptr,tdA_fpts_l_ptr,norm_fpts_ptr,pen_fact,tau,diff_coeff);
    }
    else
      FatalError("ERROR: Viscous riemann solver type not recognized ... ");
  }
  else
    FatalError("ERROR: Riemann solver type not recognized ... ");

  check_cuda_error("After", __FILE__, __LINE__);
}

#endif

void bespoke_SPMV(int m, int n, int n_fields, int n_eles, double* opp_ell_data_ptr, int* opp_ell_indices_ptr, int nnz_per_row, double* b_ptr, double *c_ptr, int cell_type, int order, int add_flag)
{
  int eles_per_block=2;
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
  else if (n_fields==6)
  {
    bespoke_SPMV_kernel<6> <<<grid_size, block_size, shared_mem*sizeof(double) >>> (c_ptr, b_ptr, opp_ell_data_ptr, opp_ell_indices_ptr, nnz_per_row, n_eles, n, m, eles_per_block,n_eles*n,n_eles*m,add_flag);
  }
}

