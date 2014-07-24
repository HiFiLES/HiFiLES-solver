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
__device__ void set_inv_boundary_conditions_kernel(int bdy_type, double* u_l, double* u_r, double* v_g, double* norm, double* loc, double *bdy_params, double gamma, double R_ref, double time_bound, int equation)
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

      // Subsonic inflow simple (free pressure)
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
        }

      // Subsonic outflow simple (fixed pressure)
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
        }

      // Adiabatic, no-slip wall (fixed)
      else if(bdy_type == 12)
        {
          // extrapolate density
          rho_r = rho_l;

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


template<int in_n_dims>
__device__ void inv_NS_flux(double* q, double* v_g, double *p, double* f, double in_gamma, int in_field)
{
  if(in_n_dims==2) {

    if (in_field==-1) {
      (*p) = (in_gamma-1.0)*(q[3]-0.5*(q[1]*q[1]+q[2]*q[2])/q[0]);
    }
    else if (in_field==0) {
      f[0] = q[1] - q[0]*v_g[0];
      f[1] = q[2] - q[0]*v_g[1];
    }
    else if (in_field==1) {
      f[0]  = (*p) + (q[1]*q[1]/q[0]) - q[1]*v_g[0];
      f[1]  = q[2]*q[1]/q[0]          - q[1]*v_g[1];
    }
    else if (in_field==2) {
      f[0]  = q[1]*q[2]/q[0]          - q[2]*v_g[0];
      f[1]  = (*p) + (q[2]*q[2]/q[0]) - q[2]*v_g[1];
    }
    else if (in_field==3) {
      f[0]  = q[1]/q[0]*(q[3]+(*p)) - q[3]*v_g[0];
      f[1]  = q[2]/q[0]*(q[3]+(*p)) - q[3]*v_g[1];
    }
  }
  else if(in_n_dims==3)
  {
    if (in_field==-1) {
      (*p) = (in_gamma-1.0)*(q[4]-0.5*(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/q[0]);
    }
    else if (in_field==0) {
      f[0] = q[1] - q[0]*v_g[0];
      f[1] = q[2] - q[0]*v_g[1];
      f[2] = q[3] - q[0]*v_g[2];
    }
    else if (in_field==1) {
      f[0] = (*p)+(q[1]*q[1]/q[0]) - q[1]*v_g[0];
      f[1] = q[2]*q[1]/q[0]        - q[1]*v_g[1];
      f[2] = q[3]*q[1]/q[0]        - q[1]*v_g[2];
    }
    else if (in_field==2) {
      f[0] = q[1]*q[2]/q[0]          - q[2]*v_g[0];
      f[1] = (*p) + (q[2]*q[2]/q[0]) - q[2]*v_g[1];
      f[2] = q[3]*q[2]/q[0]          - q[2]*v_g[2];
    }
    else if (in_field==3) {
      f[0] = q[1]*q[3]/q[0]          - q[3]*v_g[0];
      f[1] = q[2]*q[3]/q[0]          - q[3]*v_g[1];
      f[2] = (*p) + (q[3]*q[3]/q[0]) - q[3]*v_g[2];
    }
    else if (in_field==4) {
      f[0] = q[1]/q[0]*(q[4]+(*p)) - q[4]*v_g[0];
      f[1] = q[2]/q[0]*(q[4]+(*p)) - q[4]*v_g[1];
      f[2] = q[3]/q[0]*(q[4]+(*p)) - q[4]*v_g[2];
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

// Create rotation matrix from Cartesian to wall-aligned coords
template<int in_n_dims>
__device__ void rotation_matrix_kernel(double* norm, double* mrot)
{
  double nn;

  if(in_n_dims==2) {
    if(abs(norm[1]) > 0.7) {
      mrot[0*in_n_dims+0] = norm[0];
      mrot[1*in_n_dims+0] = norm[1];
      mrot[0*in_n_dims+1] = norm[1];
      mrot[1*in_n_dims+1] = -norm[0];
    }
    else {
      mrot[0*in_n_dims+0] = -norm[0];
      mrot[1*in_n_dims+0] = -norm[1];
      mrot[0*in_n_dims+1] = norm[1];
      mrot[1*in_n_dims+1] = -norm[0];
    }
  }
  else if(in_n_dims==3) {
    if(abs(norm[2]) > 0.7) {
      nn = sqrt(norm[1]*norm[1]+norm[2]*norm[2]);

      mrot[0*in_n_dims+0] = norm[0]/nn;
      mrot[1*in_n_dims+0] = norm[1]/nn;
      mrot[2*in_n_dims+0] = norm[2]/nn;
      mrot[0*in_n_dims+1] = 0.0;
      mrot[1*in_n_dims+1] = -norm[2]/nn;
      mrot[2*in_n_dims+1] = norm[1]/nn;
      mrot[0*in_n_dims+2] = nn;
      mrot[1*in_n_dims+2] = -norm[0]*norm[1]/nn;
      mrot[2*in_n_dims+2] = -norm[0]*norm[2]/nn;
    }
    else {
      nn = sqrt(norm[0]*norm[0]+norm[1]*norm[1]);

      mrot[0*in_n_dims+0] = norm[0]/nn;
      mrot[1*in_n_dims+0] = norm[1]/nn;
      mrot[2*in_n_dims+0] = norm[2]/nn;
      mrot[0*in_n_dims+1] = norm[1]/nn;
      mrot[1*in_n_dims+1] = -norm[0]/nn;
      mrot[2*in_n_dims+1] = 0.0;
      mrot[0*in_n_dims+2] = norm[0]*norm[2]/nn;
      mrot[1*in_n_dims+2] = norm[1]*norm[2]/nn;
      mrot[2*in_n_dims+2] = -nn;
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

__device__ double SGS_filter_width(double in_detjac, int in_ele_type, int in_n_dims, int in_order, double in_filter_ratio)
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

  delta = in_filter_ratio*pow(vol,1./in_n_dims)/(in_order+1.);

  return delta;
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
__global__ void calc_Leonard_tensors_kernel(int in_n_upts_per_ele, int in_n_eles, int in_n_dims, double* in_disuf_upts_ptr, double* in_Lu_ptr, double* in_Le_ptr)
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

template<int in_n_dims>
__device__ void wall_model_kernel(int wall_model, double rho, double* urot, double* inte, double* mu, double in_gamma, double in_prandtl, double y, double* tau_wall, double q_wall)
{
  double eps = 1.e-10;
  double Rey, Rey_c, u, uplus, utau, tw;
  double prandtl_t = 0.9;
  double ymatch = 11.8;
  int i;

  // Magnitude of surface velocity
  u = 0.0;
  #pragma unroll
  for(i=0;i<in_n_dims;++i) u += urot[i]*urot[i];

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
      for(i=0;i<in_n_dims;++i) tau_wall[i] = tw*urot[i]/u;

      // Wall heat flux
      if(Rey < Rey_c) q_wall = (*inte)*in_gamma*tw / (in_prandtl * u);
      else            q_wall = (*inte)*in_gamma*tw / (in_prandtl * (u + utau * sqrt(Rey_c) * (in_prandtl/prandtl_t-1.0)));
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
      for (i=0;i<in_n_dims;i++)
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
      for (i=0;i<in_n_dims;i++) tau_wall[i] = abs(tw*urot[i]/u);

      // Wall heat flux
      if(yplus <= ymatch) q_wall = (*inte)*in_gamma*tw / (in_prandtl * u);
      else                q_wall = (*inte)*in_gamma*tw / (in_prandtl * (u + utau * ymatch * (in_prandtl/prandtl_t-1.0)));
    }
  }

  // if velocity is 0
  else {
    #pragma unroll
    for (i=0;i<in_n_dims;i++) tau_wall[i] = 0.0;
    q_wall = 0.0;
  }
}

template<int in_n_dims, int in_n_comp>
__device__ void SGS_flux_kernel(double* q, double* qf, double* grad_vel, double* grad_velf, double* grad_ene, double* grad_enef, double* gsq, double* sd, double* strain, double* strainf, double* Lm, double* Le, double* mu_t, double* Cs, double* sgsf, int sgs_model, double delta, double in_gamma)
{
  int i,j,k;
  int eddy, sim;
  double deltaf;
  double Smod, Sfmod;
  double prandtl_t=0.5; // turbulent Prandtl number
  double num, denom;
  double diag;
  double eps=1.e-12;
  double M[in_n_comp];          // M tensor for dynamic model

  // Set flags depending on which SGS model we are using
  // 0: Smagorinsky, 1: WALE, 2: WALE-similarity, 3: SVV, 4: Similarity, 5: dynamic
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
  else if(sgs_model==5) {
    eddy = 1;
    sim = 0;
  }

  // Calculate eddy viscosity

  // Smagorinsky model
  if(sgs_model==0) {

    *Cs=0.1;

    // Calculate modulus of strain rate tensor
    Smod = 0.0;
    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      Smod += 2.0*strain[i]*strain[i];
    }

    // Now the off-diagonal components of strain tensor:
    if(in_n_dims==2) {
      Smod += 4.0*strain[2]*strain[2];
    }
    else if(in_n_dims==3) {
      Smod += 4.0*(strain[3]*strain[3] + strain[4]*strain[4] + strain[5]*strain[5]);
    }

    // modulus of strain rate tensor
    Smod = sqrt(Smod);

    // eddy viscosity
    *mu_t = q[0]*(*Cs)*(*Cs)*delta*delta*Smod;
  }

  // WALE or WSM model
  else if(sgs_model==1 || sgs_model==2) {

    *Cs=0.5;

    // squared gradient tensor - NOT symmetric!
    diag = 0.0;
    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      #pragma unroll
      for (j=0;j<in_n_dims;j++) {
        gsq[i*in_n_dims+j] = 0.0;
        #pragma unroll
        for (k=0;k<in_n_dims;k++) {
          gsq[i*in_n_dims+j] += grad_vel[i*in_n_dims+j]*grad_vel[j*in_n_dims+i];
        }
      }
      diag += gsq[i*in_n_dims+i]/3.0;
    }

    // construct sd tensor - NOT symmetric!
    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      #pragma unroll
      for (j=0;j<in_n_dims;j++) {
        sd[i*in_n_dims+j] = 0.5*(gsq[i*in_n_dims+j] + gsq[j*in_n_dims+i]);
      }
      sd[i*in_n_dims+i] -= diag;
    }

    // numerator
    num = 0.0;
    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      #pragma unroll
      for (j=0;j<in_n_dims;j++) {
        num += sd[i*in_n_dims+j]*sd[i*in_n_dims+j];
      }
    }

    // Denominator
    denom = 0.0;
    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      denom += strain[i]*strain[i];
    }

    if(in_n_dims==2) {
      denom += 2.0*strain[2]*strain[2];
    }
    else if(in_n_dims==3) {
      denom += 2.0*(strain[3]*strain[3] + strain[4]*strain[4] + strain[5]*strain[5]);
    }

    denom = pow(denom,2.5) + pow(num,1.25);
    num = pow(num,1.5);

    // eddy viscosity
    *mu_t = q[0]*(*Cs)*(*Cs)*delta*delta*num/(denom+eps);

    // HACK: prevent negative mu_t
    *mu_t = max(*mu_t,0.0);
  }
  // Dynamic model
  else if(sgs_model==5) {

    // test filter width
    deltaf = 2.0*delta;

    // Calculate modulus of strain rate tensor
    Sfmod = 0.0;
    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      Sfmod += 2.0*strainf[i]*strainf[i];
    }

    if(in_n_dims==2) {
      Sfmod += 4.0*strainf[2]*strainf[2];
    }
    else if(in_n_dims==3) {
      Sfmod += 4.0*(strainf[3]*strainf[3] + strainf[4]*strainf[4] + strainf[5]*strainf[5]);
    }

    Sfmod = sqrt(Sfmod);
    
    // M tensor
    // initial simple version: filtered product of S and Smod is equal to
    // product of filtered S and filtered Smod

    #pragma unroll
    for (i=0;i<in_n_comp;i++) {
      M[i] = (deltaf*deltaf - delta*delta)*Sfmod*strainf[i];
    }

    // compute Smagorinsky coefficient

    num = 0.0; denom = 0.0;

    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      denom += M[i]*M[i];
      num += Lm[i]*M[i];
    }

    if(in_n_dims==2) {
      denom += 2.0*(M[2]*M[2]);
      num += 2.0*Lm[2]*M[2];
    }
    else {
      denom += 2.0*(M[3]*M[3]+M[4]*M[4]+M[5]*M[5]);
      num += 2.0*(Lm[3]*M[3]+Lm[4]*M[4]+Lm[5]*M[5]);
    }

    // prevent division by zero
    *Cs = 0.5*num/(denom+eps);
        
    // limit value to prevent instability
    *Cs=min(max((*Cs),0.0),0.04);
        
    // eddy viscosity
    *mu_t = qf[0]*(*Cs)*delta*delta*Smod;

  }

  // Now set the flux values
  if (eddy==1) {

    if (in_n_dims==2) {

      // Density
      sgsf[0*in_n_dims + 0] = 0.0;
      sgsf[0*in_n_dims + 1] = 0.0;

      // u
      sgsf[1*in_n_dims + 0] = -2.0*(*mu_t)*strain[0];
      sgsf[1*in_n_dims + 1] = -2.0*(*mu_t)*strain[2];

      // v
      sgsf[2*in_n_dims + 0] = -2.0*(*mu_t)*strain[2];
      sgsf[2*in_n_dims + 1] = -2.0*(*mu_t)*strain[1];

      // energy
      sgsf[3*in_n_dims + 0] = -1.0*in_gamma*(*mu_t)/prandtl_t*grad_ene[0];
      sgsf[3*in_n_dims + 1] = -1.0*in_gamma*(*mu_t)/prandtl_t*grad_ene[1];

    }
    else if(in_n_dims==3) {

      // Density
      sgsf[0*in_n_dims + 0] = 0.0;
      sgsf[0*in_n_dims + 1] = 0.0;
      sgsf[0*in_n_dims + 2] = 0.0;

      // u
      sgsf[1*in_n_dims + 0] = -2.0*(*mu_t)*strain[0];
      sgsf[1*in_n_dims + 1] = -2.0*(*mu_t)*strain[3];
      sgsf[1*in_n_dims + 2] = -2.0*(*mu_t)*strain[4];

      // v
      sgsf[2*in_n_dims + 0] = -2.0*(*mu_t)*strain[3];
      sgsf[2*in_n_dims + 1] = -2.0*(*mu_t)*strain[1];
      sgsf[2*in_n_dims + 2] = -2.0*(*mu_t)*strain[5];

      // w
      sgsf[3*in_n_dims + 0] = -2.0*(*mu_t)*strain[4];
      sgsf[3*in_n_dims + 1] = -2.0*(*mu_t)*strain[5];
      sgsf[3*in_n_dims + 2] = -2.0*(*mu_t)*strain[2];

      // energy
      sgsf[4*in_n_dims + 0] = -1.0*in_gamma*(*mu_t)/prandtl_t*grad_ene[0];
      sgsf[4*in_n_dims + 1] = -1.0*in_gamma*(*mu_t)/prandtl_t*grad_ene[1];
      sgsf[4*in_n_dims + 2] = -1.0*in_gamma*(*mu_t)/prandtl_t*grad_ene[2];

    }
  }
  // Add similarity term to SGS fluxes if WSM or Similarity model
  if (sim==1) {
    if(in_n_dims==2) {

      // u
      sgsf[1*in_n_dims + 0] += q[0]*Lm[0];
      sgsf[1*in_n_dims + 1] += q[0]*Lm[2];

      // v
      sgsf[2*in_n_dims + 0] += q[0]*Lm[2];
      sgsf[2*in_n_dims + 1] += q[0]*Lm[1];

      // energy
      sgsf[3*in_n_dims + 0] += q[0]*in_gamma*Le[0];
      sgsf[3*in_n_dims + 1] += q[0]*in_gamma*Le[1];

    }
    else if(in_n_dims==3) {

      // u
      sgsf[1*in_n_dims + 0] += q[0]*Lm[0];
      sgsf[1*in_n_dims + 1] += q[0]*Lm[3];
      sgsf[1*in_n_dims + 2] += q[0]*Lm[4];

      // v
      sgsf[2*in_n_dims + 0] += q[0]*Lm[3];
      sgsf[2*in_n_dims + 1] += q[0]*Lm[1];
      sgsf[2*in_n_dims + 2] += q[0]*Lm[5];

      // w
      sgsf[3*in_n_dims + 0] += q[0]*Lm[4];
      sgsf[3*in_n_dims + 1] += q[0]*Lm[5];
      sgsf[3*in_n_dims + 2] += q[0]*Lm[2];

      // energy
      sgsf[4*in_n_dims + 0] += q[0]*in_gamma*Le[0];
      sgsf[4*in_n_dims + 1] += q[0]*in_gamma*Le[1];
      sgsf[4*in_n_dims + 2] += q[0]*in_gamma*Le[2];

    }
  }
}

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
  int n_spts = n_spts_per_ele[ele];

  if(thread_id<n_pts_per_ele*n_eles) {
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
          out_d_pos[i+j*n_dims] += dxdr[i][k]*JGinv_pts[thread_id+(j*n_dims+k)*stride]/detjac_pts[thread_id];
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
  double* d_pos = new double[n_dims*n_dims];

  double xr, xs, xt;
  double yr, ys, yt;
  double zr, zs, zt;

  if(thread_id<n_upts_per_ele*n_eles) {

    /**
    J_dyn_upts(n_upts_per_ele,n_eles): Determinant of the dynamic -> static reference transformation matrix ( |G| )
    JGinv_dyn_upts(n_upts_per_ele,n_eles,n_dims,n_dims): Total dynamic -> static reference transformation matrix ( |G|*G^{-1} )
    dyn_pos_upts(n_upts_per_ele,n_eles,n_dims): Physical position of solution points */

    if(n_dims==2)
    {
      // calculate first derivatives of shape functions at the solution point
      calc_d_pos_dyn_kernel<2>(n_upts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_upts, JGinv_upts, d_nodal_s_basis_upts, shape_dyn, d_pos);

      xr = d_pos[0+0*n_dims];
      xs = d_pos[0+1*n_dims];

      yr = d_pos[1+0*n_dims];
      ys = d_pos[1+1*n_dims];

      // store determinant of jacobian at solution point
      J_dyn_upts[thread_id]= xr*ys - xs*yr;

      //if (J_dyn_upts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_upts[thread_id + (0*n_dims+0)*stride] =  ys;
      JGinv_dyn_upts[thread_id + (1*n_dims+0)*stride] = -xs;
      JGinv_dyn_upts[thread_id + (0*n_dims+1)*stride] = -yr;
      JGinv_dyn_upts[thread_id + (1*n_dims+1)*stride] =  xr;
    }
    else if(n_dims==3)
    {
      calc_d_pos_dyn_kernel<3>(n_upts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_upts, JGinv_upts, d_nodal_s_basis_upts, shape_dyn, d_pos);

      xr = d_pos[0+0*n_dims];
      xs = d_pos[0+1*n_dims];
      xt = d_pos[0+2*n_dims];

      yr = d_pos[1+0*n_dims];
      ys = d_pos[1+1*n_dims];
      yt = d_pos[1+2*n_dims];

      zr = d_pos[2+0*n_dims];
      zs = d_pos[2+1*n_dims];
      zt = d_pos[2+2*n_dims];

      // store determinant of jacobian at solution point
      J_dyn_upts[thread_id] = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

      //if (J_dyn_upts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_upts[thread_id + (0*n_dims+0)*stride] = (ys*zt - yt*zs);
      JGinv_dyn_upts[thread_id + (1*n_dims+0)*stride] = (xt*zs - xs*zt);
      JGinv_dyn_upts[thread_id + (2*n_dims+0)*stride] = (xs*yt - xt*ys);
      JGinv_dyn_upts[thread_id + (0*n_dims+1)*stride] = (yt*zr - yr*zt);
      JGinv_dyn_upts[thread_id + (1*n_dims+1)*stride] = (xr*zt - xt*zr);
      JGinv_dyn_upts[thread_id + (2*n_dims+1)*stride] = (xt*yr - xr*yt);
      JGinv_dyn_upts[thread_id + (0*n_dims+2)*stride] = (yr*zs - ys*zr);
      JGinv_dyn_upts[thread_id + (1*n_dims+2)*stride] = (xs*zr - xr*zs);
      JGinv_dyn_upts[thread_id + (2*n_dims+2)*stride] = (xr*ys - xs*yr);
    }
  }
  delete[] d_pos;
}

/*! gpu kernel to update coordiante transformation variables for moving grids */
template<int n_dims>
__global__ void set_transforms_dynamic_fpts_kernel(int n_fpts_per_ele, int n_eles, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_fpts, double* J_dyn_fpts, double* JGinv_fpts, double* JGinv_dyn_fpts, double* tdA_dyn_fpts, double* norm_fpts, double* norm_dyn_fpts, double* d_nodal_s_basis_fpts, double* shape_dyn)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  int stride = n_fpts_per_ele*n_eles;
  double* d_pos = new double[n_dims*n_dims];
  double norm[n_dims];

  double xr, xs, xt;
  double yr, ys, yt;
  double zr, zs, zt;

  if(thread_id<n_fpts_per_ele*n_eles) {

    if(n_dims==2)
    {
      // calculate first derivatives of shape functions at the solution point
      calc_d_pos_dyn_kernel<2>(n_fpts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_fpts, JGinv_fpts, d_nodal_s_basis_fpts, shape_dyn, d_pos);

      xr = d_pos[0+0*n_dims];
      xs = d_pos[0+1*n_dims];

      yr = d_pos[1+0*n_dims];
      ys = d_pos[1+1*n_dims];

      // store determinant of jacobian at solution point
      J_dyn_fpts[thread_id]= xr*ys - xs*yr;

//      if (J_dyn_fpts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_fpts[thread_id + (0*n_dims+0)*stride] =  ys;
      JGinv_dyn_fpts[thread_id + (1*n_dims+0)*stride] = -xs;
      JGinv_dyn_fpts[thread_id + (0*n_dims+1)*stride] = -yr;
      JGinv_dyn_fpts[thread_id + (1*n_dims+1)*stride] =  xr;

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
      calc_d_pos_dyn_kernel<3>(n_fpts_per_ele, n_eles, max_n_spts_per_ele, n_spts_per_ele, J_fpts, JGinv_fpts, d_nodal_s_basis_fpts, shape_dyn, d_pos);

      xr = d_pos[0+0*n_dims];
      xs = d_pos[0+1*n_dims];
      xt = d_pos[0+2*n_dims];

      yr = d_pos[1+0*n_dims];
      ys = d_pos[1+1*n_dims];
      yt = d_pos[1+2*n_dims];

      zr = d_pos[2+0*n_dims];
      zs = d_pos[2+1*n_dims];
      zt = d_pos[2+2*n_dims];

      // store determinant of jacobian at solution point
      J_dyn_fpts[thread_id] = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

      //if (J_dyn_fpts[thread_id] < 0) *err = 1;

      // store determinant of jacobian multiplied by inverse of jacobian at the solution point
      JGinv_dyn_fpts[thread_id + (0*n_dims+0)*stride] = (ys*zt - yt*zs);
      JGinv_dyn_fpts[thread_id + (1*n_dims+0)*stride] = (xt*zs - xs*zt);
      JGinv_dyn_fpts[thread_id + (2*n_dims+0)*stride] = (xs*yt - xt*ys);
      JGinv_dyn_fpts[thread_id + (0*n_dims+1)*stride] = (yt*zr - yr*zt);
      JGinv_dyn_fpts[thread_id + (1*n_dims+1)*stride] = (xr*zt - xt*zr);
      JGinv_dyn_fpts[thread_id + (2*n_dims+1)*stride] = (xt*yr - xr*yt);
      JGinv_dyn_fpts[thread_id + (0*n_dims+2)*stride] = (yr*zs - ys*zr);
      JGinv_dyn_fpts[thread_id + (1*n_dims+2)*stride] = (xs*zr - xr*zs);
      JGinv_dyn_fpts[thread_id + (2*n_dims+2)*stride] = (xr*ys - xs*yr);

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
  delete[] d_pos;
}

template<int in_n_fields, int in_n_dims>
__device__ __host__ void rusanov_flux(double* q_l, double *q_r, double* v_g, double *norm, double *fn, double in_gamma)
{
  double vn_l, vn_r;
  double vn_av_mag, c_av, vn_g;
  double p_l, p_r,f_l,f_r;

  double f[in_n_dims];

  // Compute normal velocity
  vn_l = 0.;
  vn_r = 0.;
  vn_g = 0.;
#pragma unroll
  for (int i=0;i<in_n_dims;i++) {
      vn_l += q_l[i+1]/q_l[0]*norm[i];
      vn_r += q_r[i+1]/q_r[0]*norm[i];
      vn_g += v_g[i]*norm[i];
    }

  // Flux prep
  inv_NS_flux<in_n_dims>(q_l,v_g,&p_l,f,in_gamma,-1);
  inv_NS_flux<in_n_dims>(q_r,v_g,&p_r,f,in_gamma,-1);

  vn_av_mag=0.5*fabs(vn_l+vn_r);
  c_av=sqrt((in_gamma*(p_l+p_r))/(q_l[0]+q_r[0]));

#pragma unroll
  for (int i=0;i<in_n_fields;i++)
    {
      // Left normal flux
      inv_NS_flux<in_n_dims>(q_l,v_g,&p_l,f,in_gamma,i);

      f_l = f[0]*norm[0] + f[1]*norm[1];
      if(in_n_dims==3)
        f_l += f[2]*norm[2];

      // Right normal flux
      inv_NS_flux<in_n_dims>(q_r,v_g,&p_r,f,in_gamma,i);

      f_r = f[0]*norm[0] + f[1]*norm[1];
      if(in_n_dims==3)
        f_r += f[2]*norm[2];

      // Common normal flux
      fn[i] = 0.5*(f_l+f_r) - 0.5*fabs(vn_av_mag-vn_g+c_av)*(q_r[i]-q_l[i]);
    }
}

template<int in_n_fields, int in_n_dims>
__device__ __host__ void convective_flux_boundary(double* q_l, double *q_r, double* v_g, double *norm, double *fn, double in_gamma)
{
  double vn_l, vn_r;
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
  inv_NS_flux<in_n_dims>(q_l,v_g,&p_l,f,in_gamma,-1);
  inv_NS_flux<in_n_dims>(q_r,v_g,&p_r,f,in_gamma,-1);

#pragma unroll
  for (int i=0;i<in_n_fields;i++)
    {
      // Left normal flux
      inv_NS_flux<in_n_dims>(q_l,v_g,&p_l,f,in_gamma,i);

      f_l = f[0]*norm[0] + f[1]*norm[1];
      if(in_n_dims==3)
        f_l += f[2]*norm[2];

      // Right normal flux
      inv_NS_flux<in_n_dims>(q_r,v_g,&p_r,f,in_gamma,i);

      f_r = f[0]*norm[0] + f[1]*norm[1];
      if(in_n_dims==3)
        f_r += f[2]*norm[2];

      // Common normal flux
      fn[i] = 0.5*(f_l+f_r);  // Taking a purely convective flux without diffusive terms
    }
}



template<int in_n_fields, int in_n_dims>
__device__ __host__ void right_flux(double *q_r, double *norm, double *fn, double in_gamma)
{

  double p_r,f_r;
  double f[in_n_dims];
  double v_g[in_n_dims]; 

  // WARNING: right_flux never used, so not going to bother finishing this
#pragma unroll
  for (int i=0; i<in_n_dims; i++)
    v_g[i] = 0.;

  // Flux prep
  inv_NS_flux<in_n_dims>(q_r,v_g,&p_r,f,in_gamma,-1);

#pragma unroll
  for (int i=0;i<in_n_fields;i++)
    {
      //Right normal flux
      inv_NS_flux<in_n_dims>(q_r,v_g,&p_r,f,in_gamma,i);

      f_r = f[0]*norm[0] + f[1]*norm[1];
      if(in_n_dims==3)
        f_r += f[2]*norm[2];

      fn[i] = f_r;
    }
}


template<int n_fields, int n_dims>
__device__ __host__ void roe_flux(double* u_l, double* v_g, double *u_r, double *norm, double *fn, double in_gamma)
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
        q_c[i] = 0.5*(q_r[i] + q_l[i]);
    }
  else if(flux_spec==2) // von Neumann
    {
#pragma unroll
      for (int i=0;i<n_fields;i++)
        q_c[i] = 0.5*(q_l[i] + q_r[i]);
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
        f_c[i] = f_r[i] + in_tau*norm[i]*(q_l - q_r); // Adding penalty factor term for this as well
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
__global__ void evaluate_invFlux_AD_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_detjac_upts_ptr, double* in_JGinv_upts_ptr, double wave_speed_x, double wave_speed_y, double wave_speed_z)
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
          met[j][i] = in_JGinv_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

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


// gpu kernel to calculate transformed discontinuous inviscid flux at solution points for the Navier-Stokes equation
// otherwise, switch to one thread per output?
template<int in_n_dims, int in_n_fields>
__global__ void evaluate_invFlux_NS_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, double* in_grid_vel_upts_ptr, double in_gamma, int in_motion)
{

  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double q[in_n_fields];
  double f[in_n_dims];
  double temp_f[in_n_dims];
  double met[in_n_dims][in_n_dims];
  double met_dyn[in_n_dims][in_n_dims];
  double v_g[in_n_dims];

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
          met[j][i] = in_JGinv_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

      if (in_motion) {
        // Transform to dynamic-physical domain
        for (int i=0;i<in_n_fields;i++)
          q[i] /= in_detjac_dyn_upts_ptr[thread_id];

        // Dynamic->static transformation matrix
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
#pragma unroll
          for (int j=0;j<in_n_dims;j++)
            met_dyn[j][i] = in_JGinv_dyn_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

        // Get grid velocity
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          v_g[i] = in_grid_vel_upts_ptr[thread_id + i*stride];
      }
      else
      {
        // Set grid velocity to 0
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          v_g[i] = 0.;
      }

      // Flux prep
      inv_NS_flux<in_n_dims>(q,v_g,&p,f,in_gamma,-1);

      int index;

      // Flux computation
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        {
          inv_NS_flux<in_n_dims>(q,v_g,&p,f,in_gamma,i);

          index = thread_id+i*stride;

          if (in_motion) {
            if (in_n_dims==2) {
              // Transform to static domain
              temp_f[0] = met_dyn[0][0]*f[0] + met_dyn[0][1]*f[1];
              temp_f[1] = met_dyn[1][0]*f[0] + met_dyn[1][1]*f[1];
              // copy back to f
              f[0] = temp_f[0];
              f[1] = temp_f[1];
            }
            else if(in_n_dims==3)
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
__global__ void calculate_common_invFlux_NS_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_tdA_dyn_fpts_r_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double** in_delta_disu_fpts_r_ptr, double in_gamma, double in_pen_fact, int in_viscous, int in_motion)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;

  double q_l[in_n_fields];
  double q_r[in_n_fields];
  double fn[in_n_fields];
  double norm[in_n_dims];
  double v_g[in_n_dims];
  double q_c[in_n_fields];

  double jac;

  if(thread_id<stride)
  {
    if (in_motion) {
      // Compute left state solution
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride]))/(*(in_detjac_dyn_fpts_l_ptr[thread_id]));

      // Compute right state solution
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        q_r[i]=(*(in_disu_fpts_r_ptr[thread_id+i*stride]))/(*(in_detjac_dyn_fpts_r_ptr[thread_id]));

      // Compute normal
#pragma unroll
      for (int i=0;i<in_n_dims;i++)
        norm[i]=*(in_norm_dyn_fpts_ptr[thread_id + i*stride]);

      // Get grid velocity
#pragma unroll
      for (int i=0;i<in_n_dims;i++)
        v_g[i]=*(in_grid_vel_fpts_ptr[thread_id + i*stride]);
    }
    else
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

      // Set grid velocity to 0
#pragma unroll
      for (int i=0;i<in_n_dims;i++)
        v_g[i]=0.;
    }

      if (in_riemann_solve_type==0)
        rusanov_flux<in_n_fields,in_n_dims> (q_l,q_r,v_g,norm,fn,in_gamma);
      else if (in_riemann_solve_type==2)
        roe_flux<in_n_fields,in_n_dims> (q_l,q_r,v_g,norm,fn,in_gamma);

      // Store transformed flux (transform to computational domain)
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      if (in_motion)
        jac *= (*(in_tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))=jac*fn[i];

      jac = (*(in_tdA_fpts_r_ptr[thread_id]));
      if (in_motion)
        jac *= (*(in_tdA_dyn_fpts_r_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        (*(in_norm_tconf_fpts_r_ptr[thread_id+i*stride]))=-jac*fn[i];

      // Viscous solution correction
      if(in_viscous)
        {
          if(in_vis_riemann_solve_type==0)
            ldg_solution<in_n_dims,in_n_fields,0> (q_l,q_r,norm,q_c,in_pen_fact);

          if (in_motion) {
            // Transform from dynamic-physical to static-physical domain
#pragma unroll
            for (int i=0;i<in_n_fields;i++)
              (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i])*(*(in_detjac_dyn_fpts_l_ptr[thread_id]));

#pragma unroll
            for (int i=0;i<in_n_fields;i++)
              (*(in_delta_disu_fpts_r_ptr[thread_id+i*stride])) = (q_c[i]-q_r[i])*(*(in_detjac_dyn_fpts_r_ptr[thread_id]));
          }
          else
          {
#pragma unroll
            for (int i=0;i<in_n_fields;i++)
              (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);

#pragma unroll
            for (int i=0;i<in_n_fields;i++)
              (*(in_delta_disu_fpts_r_ptr[thread_id+i*stride])) = (q_c[i]-q_r[i]);
          }
        }

    }
}


template <int in_n_dims, int in_vis_riemann_solve_type>
__global__ void calculate_common_invFlux_lax_friedrich_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double** in_delta_disu_fpts_r_ptr, double in_pen_fact, int in_viscous, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
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
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      (*(in_norm_tconf_fpts_l_ptr[thread_id]))=jac*fn;

      jac = (*(in_tdA_fpts_r_ptr[thread_id]));
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
__global__ void evaluate_boundaryConditions_invFlux_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_loc_fpts_ptr, double** in_loc_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_R_ref, int in_viscous, int in_motion, double in_time_bound, double in_wave_speed_x, double in_wave_speed_y, double in_wave_speed_z, double in_lambda, int in_equation)
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
  double v_g[in_n_dims];

  double jac;

  if(thread_id<stride)
    {
      // Compute left solution
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        q_l[i]=(*(in_disu_fpts_l_ptr[thread_id+i*stride]));

      if (in_motion>0) {
        // Tranform to dynamic-physical domain
#pragma unroll
        for (int i=0;i<in_n_fields;i++)
          q_l[i] /= *(in_detjac_dyn_fpts_l_ptr[thread_id]);
      }

      if (in_motion>0) {
        // Get normal & grid velocity in dynamic-physical domain
#pragma unroll
        for (int i=0;i<in_n_dims;i++) {
          norm[i]=*(in_norm_dyn_fpts_ptr[thread_id + i*stride]);
          v_g[i]=*(in_grid_vel_fpts_ptr[thread_id+i*stride]);
        }
      }
      else
      {
        // Get normal & grid velocity (0) in static-physical domain
#pragma unroll
        for (int i=0;i<in_n_dims;i++) {
          norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
          v_g[i] = 0.;
        }
      }

      // Get physical position of flux points
      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          loc[i]=*(in_loc_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
        for (int i=0;i<in_n_dims;i++)
          loc[i]=*(in_loc_fpts_ptr[thread_id + i*stride]);
      }

      // Set boundary condition
      bdy_spec = in_boundary_type[thread_id/in_n_fpts_per_inter];
      set_inv_boundary_conditions_kernel<in_n_dims,in_n_fields>(bdy_spec,q_l,q_r,v_g,norm,loc,in_bdy_params,in_gamma, in_R_ref, in_time_bound, in_equation);

      if (bdy_spec==16) // Dual consistent
        {
          //  right_flux<in_n_fields,in_n_dims> (q_r,norm,fn,in_gamma);
          roe_flux<in_n_fields,in_n_dims> (q_l,q_r,v_g,norm,fn,in_gamma);
        }
      else
        {
          if (in_riemann_solve_type==0)
            convective_flux_boundary<in_n_fields,in_n_dims> (q_l,q_r,v_g,norm,fn,in_gamma);
          else if (in_riemann_solve_type==1)
            lax_friedrichs_flux<in_n_dims> (q_l,q_r,norm,fn,in_wave_speed_x,in_wave_speed_y,in_wave_speed_z,in_lambda);
          else if (in_riemann_solve_type==2)
            roe_flux<in_n_fields,in_n_dims> (q_l,q_r,v_g,norm,fn,in_gamma);
        }

      // Store transformed flux
      if (in_motion>0)
      {
        jac = (*(in_tdA_fpts_l_ptr[thread_id]))*(*(in_tdA_dyn_fpts_l_ptr[thread_id]));
      }
      else
      {
        jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      }
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

          if(in_motion>0) {
            // Transform from dynamic back to static-physical domain
#pragma unroll
            for (int i=0;i<in_n_fields;i++)
              (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i])*(*(in_detjac_dyn_fpts_l_ptr[thread_id]));
          }
          else
          {
#pragma unroll
            for (int i=0;i<in_n_fields;i++)
              (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);
          }
        }

    }
}


// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int in_n_dims, int in_n_fields, int in_n_comp>
__global__ void evaluate_viscFlux_NS_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, int in_ele_type, int in_order, double in_filter_ratio, int in_LES, int in_motion, int sgs_model, int wall_model, double in_wall_thickness, double* in_wall_dist_ptr, double* in_twall_ptr, double* Leonard_mom, double* Leonard_ene, double* in_turb_visc_ptr, double* in_dynamic_coeff_ptr, double* in_disu_upts_ptr, double* in_disuf_upts_ptr, double* out_tdisf_upts_ptr, double* out_sgsf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_grad_disuf_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double q[in_n_fields];
  double f[in_n_dims];
  double temp_f[in_n_dims];
  double met[in_n_dims][in_n_dims];   // Static-Transformation Jacobian
  double met_dyn[in_n_dims][in_n_dims];   // Dynamic-Transformation Jacobian
  double stensor[in_n_comp];          // viscous stress tensor
  double grad_ene[in_n_dims];
  double grad_vel[in_n_dims*in_n_dims];
  double grad_q[in_n_fields*in_n_dims];
  double inte, mu;
  double eps=1.e-12;

  // LES model variables
  double sgsf[in_n_fields*in_n_dims]; // SGS flux array
  double strain[in_n_comp];           // strain for SGS models
  double gsq[in_n_dims*in_n_dims];    // for WALE SGS model
  double sd[in_n_dims*in_n_dims];     // for WALE SGS model
  double lm[in_n_comp];               // local Leonard tensor for momentum
  double le[in_n_dims];               // local Leonard tensor for energy
  double jac, delta, Cs, mu_t;
  // dynamic LES variables
  double qf[in_n_fields];             // filtered solution
  double grad_enef[in_n_dims];
  double grad_velf[in_n_dims*in_n_dims];
  double grad_qf[in_n_fields*in_n_dims]; // gradient of filtered solution
  double strainf[in_n_comp];
  double sftensor[in_n_comp];

  // wall model variables
  double norm[in_n_dims];             // wall normal
  double tau[in_n_dims*in_n_dims];    // shear stress
  double mrot[in_n_dims*in_n_dims];   // rotation matrix
  double temp[in_n_dims*in_n_dims];   // array for matrix mult
  double urot[in_n_dims];             // rotated velocity components
  double tw[in_n_dims];               // wall shear stress components
  double qw;                          // wall heat flux
  double y;                           // wall distance
  int wall;                           // flag

  int i, j, k, index;
  int stride = in_n_upts_per_ele*in_n_eles;

  if(thread_id<(in_n_upts_per_ele*in_n_eles)) {

    // Physical solution
    #pragma unroll
    for (i=0;i<in_n_fields;i++) {
      q[i] = in_disu_upts_ptr[thread_id + i*stride];
    }

    if (in_motion) {
#pragma unroll
      for (i=0;i<in_n_fields;i++) {
        q[i] /= in_detjac_dyn_upts_ptr[thread_id];
      }
    }

    if (in_motion) {
#pragma unroll
      for (i=0;i<in_n_dims;i++) {
#pragma unroll
        for (j=0;j<in_n_dims;j++) {
          met_dyn[j][i] = in_JGinv_dyn_upts_ptr[thread_id + (i*in_n_dims+j)*stride];
        }
      }
    }

    #pragma unroll
    for (i=0;i<in_n_dims;i++) {
      #pragma unroll
      for (j=0;j<in_n_dims;j++) {
        met[j][i] = in_JGinv_upts_ptr[thread_id + (i*in_n_dims+j)*stride];
      }
    }

    // Physical gradient
    #pragma unroll
    for (i=0;i<in_n_fields;i++)
    {
      index = thread_id + i*stride;
      grad_q[i*in_n_dims + 0] = in_grad_disu_upts_ptr[index];
      grad_q[i*in_n_dims + 1] = in_grad_disu_upts_ptr[index + stride*in_n_fields];

      if(in_n_dims==3)
        grad_q[i*in_n_dims + 2] = in_grad_disu_upts_ptr[index + 2*stride*in_n_fields];
    }

    // Get gradient of filtered solution if using dynamic LES model
    if(sgs_model==5) {

      // filtered solution
      #pragma unroll
      for (i=0;i<in_n_fields;i++) {
        qf[i] = in_disuf_upts_ptr[thread_id + i*stride];
      }

      // gradient of filtered solution
      #pragma unroll
      for (i=0;i<in_n_fields;i++)
      {
        index = thread_id + i*stride;
        grad_qf[i*in_n_dims + 0] = in_grad_disuf_upts_ptr[index];
        grad_qf[i*in_n_dims + 1] = in_grad_disuf_upts_ptr[index + stride*in_n_fields];

        if(in_n_dims==3)
          grad_qf[i*in_n_dims + 2] = in_grad_disuf_upts_ptr[index + 2*stride*in_n_fields];
      }

      // dynamic LES prep
      vis_NS_flux<in_n_dims>(qf, grad_qf, grad_velf, grad_enef, sftensor, f, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

    }

    // viscous flux prep
    vis_NS_flux<in_n_dims>(q, grad_q, grad_vel, grad_ene, stensor, f, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

    // Flux computation for each field
    #pragma unroll
    for (i=0;i<in_n_fields;i++) {

      index = thread_id + i*stride;

      vis_NS_flux<in_n_dims>(q, grad_q, grad_vel, grad_ene, stensor, f, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);

      if (in_motion) {
//#pragma unroll
//        for(j=0;j<in_n_dims;j++) {
//          temp_f[j] = 0.;
//#pragma unroll
//          for(k=0;k<in_n_dims;k++) {
//            temp_f[j] += met_dyn[j][k]*f[k];
//          }

        if(in_n_dims==2) {
          temp_f[0] = met_dyn[0][0]*f[0] + met_dyn[0][1]*f[1];
          temp_f[1] = met_dyn[1][0]*f[0] + met_dyn[1][1]*f[1];
        }
        else if(in_n_dims==3) {
          temp_f[0] = met_dyn[0][0]*f[0] + met_dyn[0][1]*f[1] + met_dyn[0][2]*f[2];
          temp_f[1] = met_dyn[1][0]*f[0] + met_dyn[1][1]*f[1] + met_dyn[1][2]*f[2];
          temp_f[2] = met_dyn[2][0]*f[0] + met_dyn[2][1]*f[1] + met_dyn[2][2]*f[2];
        }

        // Copy back into f
#pragma unroll
        for (j=0;j<in_n_dims;j++)
          f[j]=temp_f[j];
      }

      // Transform from static-physical to computational domain
//#pragma unroll
//      for(j=0;j<in_n_dims;j++) {
//#pragma unroll
//        for(k=0;k<in_n_dims;k++) {
//          out_tdisf_upts_ptr[index+i*stride*in_n_fields] += met[j][k]*f[k];
//        }
//      }
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

    // wall flux prep.
    // If using a wall model, flag if upt is within wall distance threshold
    wall = 0;
    if(wall_model > 0) {

      // wall distance vector
      y = 0.0;
      #pragma unroll
      for (j=0;j<in_n_dims;j++)
        y += in_wall_dist_ptr[thread_id + j*stride]*in_wall_dist_ptr[thread_id + j*stride];

      y = sqrt(y);

      if(y < in_wall_thickness) wall = 1;

    }

    // if within near-wall region
    if (wall) {

      // get wall normal
      #pragma unroll
      for (j=0;j<in_n_dims;j++)
        norm[j] = in_wall_dist_ptr[thread_id + j*stride]/y;

      // calculate rotation matrix
      rotation_matrix_kernel<in_n_dims>(norm, mrot);

      // rotate velocity to surface
      if(in_n_dims==2) {
        urot[0] = q[1]*mrot[0*in_n_dims+1] + q[2]*mrot[1*in_n_dims+1];
        urot[1] = 0.0;
      }
      else {
        urot[0] = q[1]*mrot[0*in_n_dims+1] + q[2]*mrot[1*in_n_dims+1] + q[3]*mrot[2*in_n_dims+1];
        urot[1] = q[1]*mrot[0*in_n_dims+2] + q[2]*mrot[1*in_n_dims+2] + q[3]*mrot[2*in_n_dims+2];
        urot[2] = 0.0;
      }

      // get wall flux at previous timestep
      #pragma unroll
      for (j=0;j<in_n_dims;j++)
        tw[j] = in_twall_ptr[thread_id + (j+1)*stride];

      qw = in_twall_ptr[thread_id + (in_n_fields-1)*stride];

      // calculate wall flux
      wall_model_kernel<in_n_dims>( wall_model, q[0], urot, &inte, &mu, in_gamma, in_prandtl, y, tw, qw);

      // correct the sign of wall shear stress and wall heat flux? - see SD3D

      // Set arrays for next timestep
      #pragma unroll
      for (j=0;j<in_n_dims;j++)
        in_twall_ptr[thread_id + (j+1)*stride] = tw[j]; // momentum

      in_twall_ptr[thread_id] = 0.0; //density
      in_twall_ptr[thread_id + (in_n_fields-1)*stride] = qw; //energy

      // populate ndims*ndims rotated stress array
      if(in_n_dims==2) {
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
      for(i=0;i<in_n_dims;i++) {
        #pragma unroll
        for(j=0;j<in_n_dims;j++) {
          temp[i*in_n_dims + j] = 0.0;
          #pragma unroll
          for(k=0;k<in_n_dims;k++) {
            temp[i*in_n_dims + j] += tau[i*in_n_dims + k]*mrot[k*in_n_dims + j];
          }
        }
      }

      #pragma unroll
      for(i=0;i<in_n_dims;i++) {
        #pragma unroll
        for(j=0;j<in_n_dims;j++) {
          tau[i*in_n_dims + j] = 0.0;
          #pragma unroll
          for(k=0;k<in_n_dims;k++) {
            tau[i*in_n_dims + j] += mrot[k*in_n_dims + i]*temp[k*in_n_dims + j];
          }
        }
      }

      // set SGS fluxes
      #pragma unroll
      for(i=0;i<in_n_dims;i++) {

        // density
        sgsf[0*in_n_dims + i] = 0.0;

        // velocity
        #pragma unroll
        for(j=0;j<in_n_dims;j++) {
          sgsf[(j+1)*in_n_dims + i] = 0.5*(tau[j*in_n_dims+i]+tau[i*in_n_dims+j]);
        }

        // energy
        sgsf[(in_n_fields-1)*in_n_dims + i] = qw*norm[i];
      }

    }
    else {
      // if not near a wall and using LES, compute SGS flux
      if(in_LES) {

      // Calculate strain rate tensor from viscous stress tensor
      #pragma unroll
      for (j=0;j<in_n_comp;j++)
        strain[j] = stensor[j]/2.0/mu;

      // If dynamic model, get filtered strain rate
      if(sgs_model==5) {
        #pragma unroll
        for (j=0;j<in_n_comp;j++)
          strainf[j] = sftensor[j]/2.0/mu;
      }

      // Calculate filter width
      jac = in_detjac_upts_ptr[thread_id];

      delta = SGS_filter_width(jac, in_ele_type, in_n_dims, in_order, in_filter_ratio);

      // momentum Leonard tensor
      #pragma unroll
      for (j=0;j<in_n_comp;j++)
        lm[j] = Leonard_mom[thread_id + j*stride];

      // energy Leonard tensor - bugged or just sensitive to the filter?
      #pragma unroll
      for (j=0;j<in_n_dims;j++)
        le[j] = 0.0;
        //le[j] = Leonard_ene[thread_id + j*stride];

      //printf("Lu = %6.10f\n",lm[0]);
      mu_t = 0.0;

      SGS_flux_kernel<in_n_dims,in_n_comp>(q, qf, grad_vel, grad_velf, grad_ene, grad_enef, gsq, sd, strain, strainf, lm, le, &mu_t, &Cs, sgsf, sgs_model, delta, in_gamma);

      // if eddy visc model, set eddy viscosity field for output to Paraview
      if(sgs_model==0 || sgs_model==1 || sgs_model==2 || sgs_model==5) {
        in_turb_visc_ptr[thread_id] = mu_t;
      }

      // if dynamic LES, set coeff field for output to Paraview
      if(sgs_model==5) {
        in_dynamic_coeff_ptr[thread_id] = Cs;
      }

      }
    }

    // add wall or SGS flux to output array
    if(in_LES || wall) {
      #pragma unroll
      for (i=0;i<in_n_fields;i++) {

        index = thread_id + i*stride;

        // Add in dynamic-static transformation here

        if(in_n_dims==2) {
          out_tdisf_upts_ptr[index                   ] += met[0][0]*sgsf[i*in_n_dims] + met[0][1]*sgsf[i*in_n_dims + 1];
          out_tdisf_upts_ptr[index+stride*in_n_fields] += met[1][0]*sgsf[i*in_n_dims] + met[1][1]*sgsf[i*in_n_dims + 1];
        }
        else if(in_n_dims==3) {
          out_tdisf_upts_ptr[index                     ] += met[0][0]*sgsf[i*in_n_dims] + met[0][1]*sgsf[i*in_n_dims + 1] + met[0][2]*sgsf[i*in_n_dims + 2];
          out_tdisf_upts_ptr[index+  stride*in_n_fields] += met[1][0]*sgsf[i*in_n_dims] + met[1][1]*sgsf[i*in_n_dims + 1] + met[1][2]*sgsf[i*in_n_dims + 2];
          out_tdisf_upts_ptr[index+2*stride*in_n_fields] += met[2][0]*sgsf[i*in_n_dims] + met[2][1]*sgsf[i*in_n_dims + 1] + met[2][2]*sgsf[i*in_n_dims + 2];
        }
      }
    }
  }
}


// gpu kernel to calculate transformed discontinuous viscous flux at solution points
template<int in_n_dims>
__global__ void evaluate_viscFlux_AD_gpu_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_JGinv_upts_ptr, double diff_coeff)
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
          met[j][i] = in_JGinv_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

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
__global__ void transform_grad_disu_upts_kernel(int in_n_upts_per_ele, int in_n_eles, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, int in_motion)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;

  double dq[in_n_dims];
  double met[in_n_dims][in_n_dims];

  double jac;
  int ind;

  int stride = in_n_upts_per_ele*in_n_eles;

  if(thread_id<(in_n_upts_per_ele*in_n_eles))
    {
      // Compute physical gradient
      // First, transform to static-physical domain
      // Obtain metric terms
      jac = in_detjac_upts_ptr[thread_id];

#pragma unroll
      for (int i=0;i<in_n_dims;i++)
#pragma unroll
        for (int j=0;j<in_n_dims;j++)
          met[j][i] = in_JGinv_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

      // Apply Transformation Metrics
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

      // Lastly, transform to dynamic-physical domain
      if (in_motion) {
        // Obtain metric terms for 2nd transformation
        jac = in_detjac_dyn_upts_ptr[thread_id];

#pragma unroll
        for (int i=0;i<in_n_dims;i++)
#pragma unroll
          for (int j=0;j<in_n_dims;j++)
            met[j][i] = in_JGinv_dyn_upts_ptr[thread_id + (i*in_n_dims+j)*stride];

        // Next, transform to dynamic-physical domain
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

}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims, int in_n_fields, int in_n_comp, int in_vis_riemann_solve_type>
__global__ void calculate_common_viscFlux_NS_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_tdA_dyn_fpts_r_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_sgsf_fpts_l_ptr, double** in_sgsf_fpts_r_ptr, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int in_LES, int in_motion)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;

  double q_l[in_n_fields];
  double q_r[in_n_fields];
  double f_l[in_n_fields][in_n_dims];
  double f_r[in_n_fields][in_n_dims];
  double sgsf_l[in_n_fields][in_n_dims];
  double sgsf_r[in_n_fields][in_n_dims];
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

      if (in_motion) {
        // Transform to dynamic-physical domain
#pragma unroll
        for (int i=0;i<in_n_fields;i++)
          q_l[i] /= (*(in_detjac_dyn_fpts_l_ptr[thread_id]));
      }

      // Left solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<in_n_dims;j++)
            {
              grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
            }
        }
      if(in_LES){
#pragma unroll
          for (int i=0;i<in_n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<in_n_dims;j++)
                {
                  sgsf_l[i][j] = *(in_sgsf_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
                }
            }
        }

      // Normal vector
      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          norm[i]=*(in_norm_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
      }

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

      if (in_motion) {
        // Transform to dynamic-physical domain
#pragma unroll
        for (int i=0;i<in_n_fields;i++)
          q_r[i] /= (*(in_detjac_dyn_fpts_r_ptr[thread_id]));
      }

      // Right solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<in_n_dims;j++)
            {
              grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_r_ptr[thread_id + (j*in_n_fields + i)*stride]);
            }
        }
      if(in_LES){
#pragma unroll
          for (int i=0;i<in_n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<in_n_dims;j++)
                {
                  sgsf_r[i][j] = *(in_sgsf_fpts_r_ptr[thread_id + (j*in_n_fields + i)*stride]);
                }
            }
        }

      // Right flux prep
      vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

      // Right flux computation
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f_r[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);

      // If LES, add SGS fluxes to viscous fluxes
      if(in_LES)
        {
#pragma unroll
          for (int i=0;i<in_n_fields;i++)
            {
#pragma unroll
              for (int j=0;j<in_n_dims;j++)
                {
                  f_l[i][j] += sgsf_l[i][j];
                  f_r[i][j] += sgsf_r[i][j];
                }
            }
        }

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
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      if (in_motion)
        jac *= (*(in_tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];

      jac = (*(in_tdA_fpts_r_ptr[thread_id]));
      if (in_motion)
        jac *= (*(in_tdA_dyn_fpts_r_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        (*(in_norm_tconf_fpts_r_ptr[thread_id+i*stride]))+=-jac*fn[i];
    }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims>
__global__ void calculate_common_viscFlux_AD_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_norm_fpts_ptr, double in_pen_fact, double in_tau, double diff_coeff)
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
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      (*(in_norm_tconf_fpts_l_ptr[thread_id]))+=jac*fn;

      jac = (*(in_tdA_fpts_r_ptr[thread_id]));
      (*(in_norm_tconf_fpts_r_ptr[thread_id]))+=-jac*fn;

    }
}



// kernel to calculate normal transformed continuous viscous flux at the flux points at boundaries
template<int in_n_dims, int in_n_fields, int in_n_comp, int in_vis_riemann_solve_type>
__global__ void evaluate_boundaryConditions_viscFlux_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_grad_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, double** in_loc_fpts_ptr, double** in_loc_dyn_fpts_ptr, double** in_sgsf_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, double in_R_ref, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_time_bound, int in_equation, double diff_coeff, int in_LES, int in_motion)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;

  int bdy_spec;

  double q_l[in_n_fields];
  double q_r[in_n_fields];

  double f[in_n_fields][in_n_dims];
  double sgsf[in_n_fields][in_n_dims];
  double f_c[in_n_fields][in_n_dims];

  double fn[in_n_fields];
  double norm[in_n_dims];
  double loc[in_n_dims];
  double v_g[in_n_dims];

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

      if (in_motion) {
 #pragma unroll
        for (int i=0;i<in_n_fields;i++)
          q_l[i]/=(*(in_detjac_dyn_fpts_ptr[thread_id]));
      }

     // Left solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<in_n_dims;j++)
            {
              grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
            }
        }
      if(in_LES){
#pragma unroll
          for (int i=0;i<in_n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<in_n_dims;j++)
                {
                  sgsf[i][j] = *(in_sgsf_fpts_ptr[thread_id + (j*in_n_fields + i)*stride]);
                }
            }
        }

      // Normal vector
      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          norm[i]=*(in_norm_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
      }

      // Get location
      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          loc[i]=*(in_loc_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          loc[i]=*(in_loc_fpts_ptr[thread_id + i*stride]);
      }

      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          v_g[i]=*(in_grid_vel_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          v_g[i]=0.;
      }

      // Right solution
      bdy_spec = in_boundary_type[thread_id/in_n_fpts_per_inter];
      set_inv_boundary_conditions_kernel<in_n_dims,in_n_fields>(bdy_spec,q_l,q_r,v_g,norm,loc,in_bdy_params,in_gamma,in_R_ref,in_time_bound,in_equation);


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

              // If LES (but no wall model?), add SGS flux to viscous flux
              if(in_LES)
                {
#pragma unroll
                  for (int i=0;i<in_n_fields;i++)
                    {
#pragma unroll
                      for (int j=0;j<in_n_dims;j++)
                        {
                          f[i][j] += sgsf[i][j];
                        }
                    }
                }
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
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      if (in_motion)
        jac *= (*(in_tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];
    }
}


#ifdef _MPI

// gpu kernel to calculate normal transformed continuous inviscid flux at the flux points for mpi faces
template <int in_n_dims, int in_n_fields, int in_riemann_solve_type, int in_vis_riemann_solve_type>
__global__ void calculate_common_invFlux_NS_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_pen_fact, int in_viscous, int in_motion)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;

  double q_l[in_n_fields];
  double q_r[in_n_fields];
  double fn[in_n_fields];
  double norm[in_n_dims];
  double v_g[in_n_dims];

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

      // Transform to dynamic-physical domain
      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_fields;i++) {
          q_l[i] /= *(in_detjac_dyn_fpts_ptr[thread_id]);
          q_r[i] /= *(in_detjac_dyn_fpts_ptr[thread_id]);
        }
      }

      // Compute normal
      if (in_motion>0) {
#pragma unroll
        for (int i=0;i<in_n_dims;i++) {
          norm[i]=*(in_norm_dyn_fpts_ptr[thread_id + i*stride]);
          v_g[i] =*(in_grid_vel_fpts_ptr[thread_id + i*stride]);
        }
      }
      else
      {
#pragma unroll
        for (int i=0;i<in_n_dims;i++) {
          norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
          v_g[i] = 0.;
        }
      }

      if (in_riemann_solve_type==0)
        rusanov_flux<in_n_fields,in_n_dims> (q_l,q_r,v_g,norm,fn,in_gamma);
      else if (in_riemann_solve_type==2)
        roe_flux<in_n_fields,in_n_dims> (q_l,q_r,v_g,norm,fn,in_gamma);

      // Store transformed flux
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      if (in_motion>0)
        jac *= (*(in_tdA_dyn_fpts_l_ptr[thread_id]));
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
            (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) = (q_c[i]-q_l[i]);

          // Tranform back to static-reference domain
          if (in_motion>0) {
#pragma unroll
            for (int i=0;i<in_n_fields;i++)
              (*(in_delta_disu_fpts_l_ptr[thread_id+i*stride])) *= (*(in_detjac_dyn_fpts_ptr[thread_id]));
          }
        }
    }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims, int in_n_fields, int in_n_comp, int in_vis_riemann_solve_type>
__global__ void calculate_common_viscFlux_NS_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_sgsf_fpts_l_ptr, double** in_sgsf_fpts_r_ptr, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int in_LES, int in_motion)
{
  const int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  const int stride = in_n_fpts_per_inter*in_n_inters;

  double q_l[in_n_fields];
  double q_r[in_n_fields];
  double f_l[in_n_fields][in_n_dims];
  double f_r[in_n_fields][in_n_dims];
  double sgsf_l[in_n_fields][in_n_dims];
  double sgsf_r[in_n_fields][in_n_dims];
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

      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_fields;i++)
          q_l[i] /= (*(in_detjac_dyn_fpts_ptr[thread_id]));
      }

      // Left solution gradient and SGS flux
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<in_n_dims;j++)
            {
              grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
            }
        }
      if(in_LES){
#pragma unroll
          for (int i=0;i<in_n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<in_n_dims;j++)
                {
                  sgsf_l[i][j] = *(in_sgsf_fpts_l_ptr[thread_id + (j*in_n_fields + i)*stride]);
                }
            }
        }


      // Normal vector
      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_dims;i++)
          norm[i]=*(in_norm_dyn_fpts_ptr[thread_id + i*stride]);
      }
      else
      {
 #pragma unroll
        for (int i=0;i<in_n_dims;i++)
          norm[i]=*(in_norm_fpts_ptr[thread_id + i*stride]);
      }

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

      // Transform to dynamic-physical domain
      if (in_motion) {
#pragma unroll
        for (int i=0;i<in_n_fields;i++)
          q_r[i] /= (*(in_detjac_dyn_fpts_ptr[thread_id]));
      }

      // Right solution gradientand SGS flux
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        {
#pragma unroll
          for(int j=0;j<in_n_dims;j++)
            {
              grad_q[i*in_n_dims + j] = *(in_grad_disu_fpts_r_ptr[thread_id + (j*in_n_fields + i)*stride]);
            }
        }
      if(in_LES){
#pragma unroll
          for (int i=0;i<in_n_fields;i++)
            {
#pragma unroll
              for(int j=0;j<in_n_dims;j++)
                {
                  sgsf_r[i][j] = *(in_sgsf_fpts_r_ptr[thread_id + (j*in_n_fields + i)*stride]);
                }
            }
        }

      // Right flux prep
      vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, NULL, &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, -1);

      // Right flux computation
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        vis_NS_flux<in_n_dims>(q_r, grad_q, grad_vel, grad_ene, stensor, f_r[i], &inte, &mu, in_prandtl, in_gamma, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, i);

      // If LES, add SGS fluxes to viscous fluxes
      if(in_LES)
        {
#pragma unroll
          for (int i=0;i<in_n_fields;i++)
            {
#pragma unroll
              for (int j=0;j<in_n_dims;j++)
                {
                  f_l[i][j] += sgsf_l[i][j];
                  f_r[i][j] += sgsf_r[i][j];
                }
            }
        }

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
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      if (in_motion)
        jac *= (*(in_tdA_dyn_fpts_l_ptr[thread_id]));
#pragma unroll
      for (int i=0;i<in_n_fields;i++)
        (*(in_norm_tconf_fpts_l_ptr[thread_id+i*stride]))+=jac*fn[i];
    }
}


// gpu kernel to calculate normal transformed continuous viscous flux at the flux points
template <int in_n_dims>
__global__ void calculate_common_viscFlux_AD_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_norm_fpts_ptr, double in_pen_fact, double in_tau, double diff_coeff)
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
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
      (*(in_norm_tconf_fpts_l_ptr[thread_id]))+=jac*fn;

    }
}


template <int in_n_dims, int in_vis_riemann_solve_type>
__global__ void calculate_common_invFlux_lax_friedrich_mpi_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_delta_disu_fpts_l_ptr, double in_pen_fact, int in_viscous, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
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
      jac = (*(in_tdA_fpts_l_ptr[thread_id]));
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

template <int in_n_fields, int in_n_dims>
__global__ void  pack_out_buffer_sgsf_gpu_kernel(int in_n_fpts_per_inter, int in_n_inters, double** in_sgsf_fpts_l_ptr, double* in_out_buffer_sgsf_ptr)
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
          dq[i][j]=(*(in_sgsf_fpts_l_ptr[thread_id+(j*in_n_fields+i)*stride]));

#pragma unroll
      for (int j=0;j<in_n_dims;j++)
#pragma unroll
        for (int i=0;i<in_n_fields;i++)
          in_out_buffer_sgsf_ptr[inter*in_n_fpts_per_inter*in_n_fields*in_n_dims+j*in_n_fpts_per_inter*in_n_fields+i*in_n_fpts_per_inter+fpt]=dq[i][j];

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
void evaluate_invFlux_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_disu_upts_ptr, double* out_tdisf_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, double* in_grid_vel_upts_ptr, double in_gamma, int in_motion, int equation, double wave_speed_x, double wave_speed_y, double wave_speed_z)
{
  // HACK: fix 256 threads per block
  int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (equation==0)
    {
      if (in_n_dims==2)
        evaluate_invFlux_NS_gpu_kernel<2,4> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_detjac_dyn_upts_ptr,in_JGinv_upts_ptr,in_JGinv_dyn_upts_ptr,in_grid_vel_upts_ptr,in_gamma,in_motion);
      else if (in_n_dims==3)
        evaluate_invFlux_NS_gpu_kernel<3,5> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_detjac_dyn_upts_ptr,in_JGinv_upts_ptr,in_JGinv_dyn_upts_ptr,in_grid_vel_upts_ptr,in_gamma,in_motion);
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
  else if (equation==1)
    {
      if (in_n_dims==2)
        evaluate_invFlux_AD_gpu_kernel<2> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_JGinv_upts_ptr,wave_speed_x,wave_speed_y,wave_speed_z);
      else if (in_n_dims==3)
        evaluate_invFlux_AD_gpu_kernel<3> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_disu_upts_ptr,out_tdisf_upts_ptr,in_detjac_upts_ptr,in_JGinv_upts_ptr,wave_speed_x,wave_speed_y,wave_speed_z);
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
void calculate_common_invFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconinvf_fpts_l_ptr, double** in_norm_tconinvf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_tdA_dyn_fpts_l_ptr, double **in_tdA_dyn_fpts_r_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, int in_riemann_solve_type, double **in_delta_disu_fpts_l_ptr, double **in_delta_disu_fpts_r_ptr, double in_gamma, double in_pen_fact, int in_viscous, int in_motion, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{
  // HACK: fix 256 threads per block
  int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);
  
  if (in_riemann_solve_type==0) // Rusanov
    {
      if(in_vis_riemann_solve_type==0) //LDG
        {
          if (in_n_dims==2)
            calculate_common_invFlux_NS_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_tdA_fpts_l_ptr,in_tdA_fpts_r_ptr,in_tdA_dyn_fpts_l_ptr,in_tdA_dyn_fpts_r_ptr,in_detjac_dyn_fpts_l_ptr,in_detjac_dyn_fpts_r_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
          else if (in_n_dims==3)
            calculate_common_invFlux_NS_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_tdA_fpts_l_ptr,in_tdA_fpts_r_ptr,in_tdA_dyn_fpts_l_ptr,in_tdA_dyn_fpts_r_ptr,in_detjac_dyn_fpts_l_ptr,in_detjac_dyn_fpts_r_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else if ( in_riemann_solve_type==2) // Roe
    {
      if(in_vis_riemann_solve_type==0) //LDG
        {
          if (in_n_dims==2)
            calculate_common_invFlux_NS_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_tdA_fpts_l_ptr,in_tdA_fpts_r_ptr,in_tdA_dyn_fpts_l_ptr,in_tdA_dyn_fpts_r_ptr,in_detjac_dyn_fpts_l_ptr,in_detjac_dyn_fpts_r_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
          else if (in_n_dims==3)
            calculate_common_invFlux_NS_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_tdA_fpts_l_ptr,in_tdA_fpts_r_ptr,in_tdA_dyn_fpts_l_ptr,in_tdA_dyn_fpts_r_ptr,in_detjac_dyn_fpts_l_ptr,in_detjac_dyn_fpts_r_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else if (in_riemann_solve_type==1) // Lax-Friedrich
    {
      if(in_vis_riemann_solve_type==0) //LDG
        {
          if (in_n_dims==2)
            calculate_common_invFlux_lax_friedrich_gpu_kernel<2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_tdA_fpts_l_ptr,in_tdA_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
          else if (in_n_dims==3)
            calculate_common_invFlux_lax_friedrich_gpu_kernel<3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconinvf_fpts_l_ptr,in_norm_tconinvf_fpts_r_ptr,in_tdA_fpts_l_ptr,in_tdA_fpts_r_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_delta_disu_fpts_r_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else
    FatalError("ERROR: Riemann solver type not recognized ... ");

  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points at boundaries
void evaluate_boundaryConditions_invFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_loc_fpts_ptr, double** in_loc_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, int* in_boundary_type, double* in_bdy_params, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_R_ref, int in_viscous, int in_motion, int in_vis_riemann_solve_type, double in_time_bound, double in_wave_speed_x, double in_wave_speed_y, double in_wave_speed_z, double in_lambda, int in_equation)
{

  check_cuda_error("Before", __FILE__, __LINE__);
  // HACK: fix 256 threads per block
  int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  if (in_riemann_solve_type==0)  // Rusanov
    {
      if (in_vis_riemann_solve_type==0) // LDG
        {
          if (in_n_dims==2)
            evaluate_boundaryConditions_invFlux_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_motion, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
          else if (in_n_dims==3)
            evaluate_boundaryConditions_invFlux_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_motion, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
    }
  else if (in_riemann_solve_type==1)  // Lax-Friedrichs
    {
      if (in_vis_riemann_solve_type==0) // LDG
        {
          if (in_n_dims==2)
            evaluate_boundaryConditions_invFlux_gpu_kernel<2,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_motion, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
          else if (in_n_dims==3)
            evaluate_boundaryConditions_invFlux_gpu_kernel<3,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_motion, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized in bdy riemann solver");
    }
  else if (in_riemann_solve_type==2) // Roe
    {
      if (in_vis_riemann_solve_type==0) // LDG
        {
          if (in_n_dims==2)
            evaluate_boundaryConditions_invFlux_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_motion, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
          else if (in_n_dims==3)
            evaluate_boundaryConditions_invFlux_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_gamma, in_R_ref, in_viscous, in_motion, in_time_bound, in_wave_speed_x, in_wave_speed_y, in_wave_speed_z, in_lambda, in_equation);
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
void evaluate_viscFlux_gpu_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, int in_ele_type, int in_order, double in_filter_ratio, int LES, int in_motion, int sgs_model, int wall_model, double in_wall_thickness, double* in_wall_dist_ptr, double* in_twall_ptr, double* in_Lu_ptr, double* in_Le_ptr, double* in_turb_visc_ptr, double* in_dynamic_coeff_ptr, double* in_disu_upts_ptr, double* in_disuf_upts_ptr, double* out_tdisf_upts_ptr, double* out_sgsf_upts_ptr, double* in_grad_disu_upts_ptr, double* in_grad_disuf_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double in_diff_coeff)
{
  // HACK: fix 256 threads per block
  int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (equation==0)
  {
    if (in_n_dims==2)
      evaluate_viscFlux_NS_gpu_kernel<2,4,3> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_ele_type, in_order, in_filter_ratio, LES, in_motion, sgs_model, wall_model, in_wall_thickness, in_wall_dist_ptr, in_twall_ptr, in_Lu_ptr, in_Le_ptr, in_turb_visc_ptr, in_dynamic_coeff_ptr, in_disu_upts_ptr, in_disuf_upts_ptr, out_tdisf_upts_ptr, out_sgsf_upts_ptr, in_grad_disu_upts_ptr, in_grad_disuf_upts_ptr, in_detjac_upts_ptr, in_detjac_dyn_upts_ptr, in_JGinv_upts_ptr, in_JGinv_dyn_upts_ptr, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis);
    else if (in_n_dims==3)
      evaluate_viscFlux_NS_gpu_kernel<3,5,6> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_ele_type, in_order, in_filter_ratio, LES, in_motion, sgs_model, wall_model, in_wall_thickness, in_wall_dist_ptr, in_twall_ptr, in_Lu_ptr, in_Le_ptr, in_turb_visc_ptr, in_dynamic_coeff_ptr, in_disu_upts_ptr, in_disuf_upts_ptr, out_tdisf_upts_ptr, out_sgsf_upts_ptr, in_grad_disu_upts_ptr, in_grad_disuf_upts_ptr, in_detjac_upts_ptr, in_detjac_dyn_upts_ptr, in_JGinv_upts_ptr, in_JGinv_dyn_upts_ptr, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis);
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else if (equation==1)
  {
    if (in_n_dims==2)
      evaluate_viscFlux_AD_gpu_kernel<2> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_disu_upts_ptr, out_tdisf_upts_ptr, in_grad_disu_upts_ptr, in_detjac_upts_ptr, in_JGinv_upts_ptr, in_diff_coeff);
    else if (in_n_dims==3)
      evaluate_viscFlux_AD_gpu_kernel<3> <<<n_blocks,256>>>(in_n_upts_per_ele, in_n_eles, in_disu_upts_ptr, out_tdisf_upts_ptr, in_grad_disu_upts_ptr, in_detjac_upts_ptr, in_JGinv_upts_ptr, in_diff_coeff);
    else
      FatalError("ERROR: Invalid number of dimensions ... ");
  }
  else
    FatalError("equation not recognized");

  check_cuda_error("After",__FILE__, __LINE__);
}

// wrapper for gpu kernel to transform gradient at sol points to physical gradient
void transform_grad_disu_upts_kernel_wrapper(int in_n_upts_per_ele, int in_n_dims, int in_n_fields, int in_n_eles, double* in_grad_disu_upts_ptr, double* in_detjac_upts_ptr, double* in_detjac_dyn_upts_ptr, double* in_JGinv_upts_ptr, double* in_JGinv_dyn_upts_ptr, int equation, int in_motion)
{
  // HACK: fix 256 threads per block
  int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(equation == 0) {
      if (in_n_dims==2)
        transform_grad_disu_upts_kernel<2,4> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_detjac_dyn_upts_ptr,in_JGinv_upts_ptr,in_JGinv_dyn_upts_ptr,in_motion);
      else if (in_n_dims==3)
        transform_grad_disu_upts_kernel<3,5> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_detjac_dyn_upts_ptr,in_JGinv_upts_ptr,in_JGinv_dyn_upts_ptr,in_motion);
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
  else if(equation == 1) {
      if (in_n_dims==2)
        transform_grad_disu_upts_kernel<2,1> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_detjac_dyn_upts_ptr,in_JGinv_upts_ptr,in_JGinv_dyn_upts_ptr,in_motion);
      else if (in_n_dims==3)
        transform_grad_disu_upts_kernel<3,1> <<<n_blocks,256>>>(in_n_upts_per_ele,in_n_eles,in_grad_disu_upts_ptr,in_detjac_upts_ptr,in_detjac_dyn_upts_ptr,in_JGinv_upts_ptr,in_JGinv_dyn_upts_ptr,in_motion);
      else
        FatalError("ERROR: Invalid number of dimensions ... ");
    }
  else
    FatalError("equation not recognized");

  check_cuda_error("After",__FILE__, __LINE__);
}


// wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points
void calculate_common_viscFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_norm_tconf_fpts_r_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_fpts_r_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_tdA_dyn_fpts_r_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_r_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_sgsf_fpts_l_ptr, double** in_sgsf_fpts_r_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, int equation, double in_diff_coeff, int in_LES, int in_motion)
{
  // HACK: fix 256 threads per block
  int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(equation==0)
    {
      if (in_vis_riemann_solve_type==0) // LDG
        {
          if (in_n_dims==2)
            calculate_common_viscFlux_NS_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_disu_fpts_r_ptr, in_grad_disu_fpts_l_ptr, in_grad_disu_fpts_r_ptr, in_norm_tconf_fpts_l_ptr, in_norm_tconf_fpts_r_ptr, in_tdA_fpts_l_ptr, in_tdA_fpts_r_ptr, in_tdA_dyn_fpts_l_ptr, in_tdA_dyn_fpts_r_ptr, in_detjac_dyn_fpts_l_ptr, in_detjac_dyn_fpts_r_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_sgsf_fpts_l_ptr, in_sgsf_fpts_r_ptr, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf,  in_mu_inf, in_c_sth, in_fix_vis, in_LES, in_motion);
          else if (in_n_dims==3)
            calculate_common_viscFlux_NS_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_disu_fpts_r_ptr, in_grad_disu_fpts_l_ptr, in_grad_disu_fpts_r_ptr, in_norm_tconf_fpts_l_ptr, in_norm_tconf_fpts_r_ptr, in_tdA_fpts_l_ptr, in_tdA_fpts_r_ptr, in_tdA_dyn_fpts_l_ptr, in_tdA_dyn_fpts_r_ptr, in_detjac_dyn_fpts_l_ptr, in_detjac_dyn_fpts_r_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_sgsf_fpts_l_ptr, in_sgsf_fpts_r_ptr, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf,  in_mu_inf, in_c_sth, in_fix_vis, in_LES, in_motion);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else if(equation==1)
    {
      if (in_vis_riemann_solve_type==0) // LDG
        {
          if (in_n_dims==2)
            calculate_common_viscFlux_AD_gpu_kernel<2> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_disu_fpts_r_ptr, in_grad_disu_fpts_l_ptr, in_grad_disu_fpts_r_ptr, in_norm_tconf_fpts_l_ptr, in_norm_tconf_fpts_r_ptr, in_tdA_fpts_l_ptr, in_tdA_fpts_r_ptr, in_norm_fpts_ptr, in_pen_fact, in_tau, in_diff_coeff);
          else if (in_n_dims==3)
            calculate_common_viscFlux_AD_gpu_kernel<3> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_disu_fpts_r_ptr, in_grad_disu_fpts_l_ptr, in_grad_disu_fpts_r_ptr, in_norm_tconf_fpts_l_ptr, in_norm_tconf_fpts_r_ptr, in_tdA_fpts_l_ptr, in_tdA_fpts_r_ptr, in_norm_fpts_ptr, in_pen_fact, in_tau, in_diff_coeff);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else
    FatalError("equation not recognized");


  check_cuda_error("After", __FILE__, __LINE__);
}

// wrapper for gpu kernel to calculate normal transformed continuous viscous flux at the flux points at boundaries
void evaluate_boundaryConditions_viscFlux_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_grad_disu_fpts_l_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_l_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, double** in_loc_fpts_ptr, double** in_loc_dyn_fpts_ptr, double** in_sgsf_fpts_ptr, int* in_boundary_type, double* in_bdy_params, double** in_delta_disu_fpts_l_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_R_ref, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_time_bound, int in_equation, double in_diff_coeff, int in_LES, int in_motion)
{

  // HACK: fix 256 threads per block
  int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_vis_riemann_solve_type==0) // LDG
    {
      if(in_equation==0)
        {
          if (in_n_dims==2)
            evaluate_boundaryConditions_viscFlux_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_grad_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_sgsf_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_R_ref, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, in_time_bound, in_equation, in_diff_coeff, in_LES, in_motion);
          else if (in_n_dims==3)
            evaluate_boundaryConditions_viscFlux_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_grad_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_sgsf_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_R_ref, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, in_time_bound, in_equation, in_diff_coeff, in_LES, in_motion);
        }
      else if(in_equation==1)
        {
          if (in_n_dims==2)
            evaluate_boundaryConditions_viscFlux_gpu_kernel<2,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_grad_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_sgsf_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_R_ref, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, in_time_bound, in_equation, in_diff_coeff, in_LES, in_motion);
          else if (in_n_dims==3)
            evaluate_boundaryConditions_viscFlux_gpu_kernel<3,1,1,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_grad_disu_fpts_l_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_l_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_grid_vel_fpts_ptr, in_loc_fpts_ptr, in_loc_dyn_fpts_ptr, in_sgsf_fpts_ptr, in_boundary_type, in_bdy_params, in_delta_disu_fpts_l_ptr, in_R_ref, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf, in_mu_inf, in_c_sth, in_fix_vis, in_time_bound, in_equation, in_diff_coeff, in_LES, in_motion);
        }
    }
  else
    FatalError("ERROR: Viscous riemann solver type not recognized ... ");

  check_cuda_error("After", __FILE__, __LINE__);
}

/*! wrapper for gpu kernel to calculate terms for similarity model */
void calc_similarity_model_kernel_wrapper(int flag, int in_n_fields, int in_n_upts_per_ele, int in_n_eles, int in_n_dims, double* in_disu_upts_ptr, double* in_disuf_upts_ptr, double* in_uu_ptr, double* in_ue_ptr, double* in_Lu_ptr, double* in_Le_ptr)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  /*! Calculate product terms uu, ue */
  if (flag==0) {
    // fixed n_fields at 4 for 2d and 5 for 3d
    if(in_n_dims==2) {
      calc_similarity_terms_kernel<4> <<< n_blocks,block_size >>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disu_upts_ptr, in_uu_ptr, in_ue_ptr);
    }
    else if(in_n_dims==3) {
      calc_similarity_terms_kernel<5> <<< n_blocks,block_size >>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disu_upts_ptr, in_uu_ptr, in_ue_ptr);
    }
  }
  /*! Calculate Leonard tensors Lu, Le */
  else if (flag==1) {
    // fixed n_fields at 4 for 2d and 5 for 3d
    if(in_n_dims==2) {
      calc_Leonard_tensors_kernel<4> <<< n_blocks,block_size >>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disuf_upts_ptr, in_Lu_ptr, in_Le_ptr);
    }
    else if(in_n_dims==3) {
      calc_Leonard_tensors_kernel<5> <<< n_blocks,block_size >>> (in_n_upts_per_ele, in_n_eles, in_n_dims, in_disuf_upts_ptr, in_Lu_ptr, in_Le_ptr);
    }
  }

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void set_transforms_dynamic_upts_kernel_wrapper(int in_n_upts_per_ele, int in_n_eles, int in_n_dims, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_upts_ptr, double* J_dyn_upts_ptr, double* JGinv_upts_ptr, double* JGinv_dyn_upts_ptr, double* d_nodal_s_basis_upts, double* shape_dyn)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((in_n_eles*in_n_upts_per_ele-1)/block_size)+1;
  int err = 0;

  check_cuda_error("Before", __FILE__, __LINE__);

  if(in_n_dims==2) {
    set_transforms_dynamic_upts_kernel<2> <<< n_blocks,block_size >>> (in_n_upts_per_ele, in_n_eles, max_n_spts_per_ele, n_spts_per_ele, J_upts_ptr, J_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, d_nodal_s_basis_upts, shape_dyn);
  }
  else if(in_n_dims==3) {
    set_transforms_dynamic_upts_kernel<3> <<< n_blocks,block_size >>> (in_n_upts_per_ele, in_n_eles, max_n_spts_per_ele, n_spts_per_ele, J_upts_ptr, J_dyn_upts_ptr, JGinv_upts_ptr, JGinv_dyn_upts_ptr, d_nodal_s_basis_upts, shape_dyn);
  }

  if (err)
    cout << "ERROR: Negative Jacobian found at solution point!" << endl;

  check_cuda_error("After",__FILE__, __LINE__);
}

/*! wrapper for gpu kernel to update coordinate transformations for moving grids */
void set_transforms_dynamic_fpts_kernel_wrapper(int in_n_fpts_per_ele, int in_n_eles, int in_n_dims, int max_n_spts_per_ele, int* n_spts_per_ele, double* J_fpts_ptr, double* J_dyn_fpts_ptr, double* JGinv_fpts_ptr, double* JGinv_dyn_fpts_ptr, double* tdA_dyn_fpts_ptr, double* norm_fpts_ptr, double* norm_dyn_fpts_ptr, double* d_nodal_s_basis_fpts, double* shape_dyn)
{
  // HACK: fix 256 threads per block
  int block_size=256;
  int n_blocks=((in_n_eles*in_n_fpts_per_ele-1)/block_size)+1;
  //int *err = new int[1];

  check_cuda_error("Before", __FILE__, __LINE__);

  if(in_n_dims==2) {
    set_transforms_dynamic_fpts_kernel<2> <<< n_blocks,block_size >>> (in_n_fpts_per_ele, in_n_eles, max_n_spts_per_ele, n_spts_per_ele, J_fpts_ptr, J_dyn_fpts_ptr, JGinv_fpts_ptr, JGinv_dyn_fpts_ptr, tdA_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, d_nodal_s_basis_fpts, shape_dyn);
  }
  else if(in_n_dims==3) {
    set_transforms_dynamic_fpts_kernel<3> <<< n_blocks,block_size >>> (in_n_fpts_per_ele, in_n_eles, max_n_spts_per_ele, n_spts_per_ele, J_fpts_ptr, J_dyn_fpts_ptr, JGinv_fpts_ptr, JGinv_dyn_fpts_ptr, tdA_dyn_fpts_ptr, norm_fpts_ptr, norm_dyn_fpts_ptr, d_nodal_s_basis_fpts, shape_dyn);
  }

  /*if (*err)
    cout << "ERROR: Negative Jacobian found at flux point!" << endl;*/

  check_cuda_error("After",__FILE__, __LINE__);
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

void pack_out_buffer_sgsf_gpu_kernel_wrapper(int in_n_fpts_per_inter,int in_n_inters,int in_n_fields,int in_n_dims, double** in_sgsf_fpts_l_ptr, double* in_out_buffer_sgsf_ptr)
{
  int block_size=256;
  int n_blocks=((in_n_inters*in_n_fpts_per_inter*in_n_dims-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_n_fields==1)
    {
      if (in_n_dims==2) {
          pack_out_buffer_sgsf_gpu_kernel<1,2> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_sgsf_fpts_l_ptr,in_out_buffer_sgsf_ptr);
        }
      else if (in_n_dims==3) {
          pack_out_buffer_sgsf_gpu_kernel<1,3> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_sgsf_fpts_l_ptr,in_out_buffer_sgsf_ptr);
        }

    }
  else if (in_n_fields==4)
    {
      pack_out_buffer_sgsf_gpu_kernel<4,2> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_sgsf_fpts_l_ptr,in_out_buffer_sgsf_ptr);
    }
  else if (in_n_fields==5)
    {
      pack_out_buffer_sgsf_gpu_kernel<5,3> <<< n_blocks,block_size >>> (in_n_fpts_per_inter,in_n_inters,in_sgsf_fpts_l_ptr,in_out_buffer_sgsf_ptr);
    }
  else
    FatalError("Number of fields not supported in pack_out_buffer");

  check_cuda_error("After", __FILE__, __LINE__);

}

// wrapper for gpu kernel to calculate normal transformed continuous inviscid flux at the flux points
void calculate_common_invFlux_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_grid_vel_fpts_ptr, int in_riemann_solve_type, double** in_delta_disu_fpts_l_ptr, double in_gamma, double in_pen_fact,  int in_viscous, int in_motion, int in_vis_riemann_solve_type, double wave_speed_x, double wave_speed_y, double wave_speed_z, double lambda)
{

  int block_size=256;
  int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/block_size)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_riemann_solve_type==0 ) // Rusanov
    {
      if (in_vis_riemann_solve_type==0 ) //LDG
        {
          if (in_n_dims==2)
            calculate_common_invFlux_NS_mpi_gpu_kernel<2,4,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_tdA_dyn_fpts_l_ptr,in_detjac_dyn_fpts_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
          else if (in_n_dims==3)
            calculate_common_invFlux_NS_mpi_gpu_kernel<3,5,0,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_tdA_dyn_fpts_l_ptr,in_detjac_dyn_fpts_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else if (in_riemann_solve_type==2 ) // Roe
    {
      if (in_vis_riemann_solve_type==0 ) //LDG
        {
          if (in_n_dims==2)
            calculate_common_invFlux_NS_mpi_gpu_kernel<2,4,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_tdA_dyn_fpts_l_ptr,in_detjac_dyn_fpts_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
          else if (in_n_dims==3)
            calculate_common_invFlux_NS_mpi_gpu_kernel<3,5,2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_tdA_dyn_fpts_l_ptr,in_detjac_dyn_fpts_ptr,in_norm_fpts_ptr,in_norm_dyn_fpts_ptr,in_grid_vel_fpts_ptr,in_delta_disu_fpts_l_ptr,in_gamma,in_pen_fact,in_viscous,in_motion);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else if (in_riemann_solve_type==1) // Lax-Friedrich
    {
      if(in_vis_riemann_solve_type==0) //LDG
        {
          if (in_n_dims==2)
            calculate_common_invFlux_lax_friedrich_mpi_gpu_kernel<2,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
          else if (in_n_dims==3)
            calculate_common_invFlux_lax_friedrich_mpi_gpu_kernel<3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_norm_fpts_ptr,in_delta_disu_fpts_l_ptr,in_pen_fact,in_viscous,wave_speed_x,wave_speed_y,wave_speed_z,lambda);
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
void calculate_common_viscFlux_mpi_gpu_kernel_wrapper(int in_n_fpts_per_inter, int in_n_dims, int in_n_fields, int in_n_inters, double** in_disu_fpts_l_ptr, double** in_disu_fpts_r_ptr, double** in_grad_disu_fpts_l_ptr, double** in_grad_disu_fpts_r_ptr, double** in_norm_tconf_fpts_l_ptr, double** in_tdA_fpts_l_ptr, double** in_tdA_dyn_fpts_l_ptr, double** in_detjac_dyn_fpts_ptr, double** in_norm_fpts_ptr, double** in_norm_dyn_fpts_ptr, double** in_sgsf_fpts_l_ptr, double** in_sgsf_fpts_r_ptr, int in_riemann_solve_type, int in_vis_riemann_solve_type, double in_pen_fact, double in_tau, double in_gamma, double in_prandtl, double in_rt_inf, double in_mu_inf, double in_c_sth, double in_fix_vis, double in_diff_coeff, int in_LES, int in_motion)
{
  // HACK: fix 256 threads per block
  int n_blocks=((in_n_inters*in_n_fpts_per_inter-1)/256)+1;

  check_cuda_error("Before", __FILE__, __LINE__);

  if (in_riemann_solve_type==0 ) // Rusanov
    {
      if (in_vis_riemann_solve_type==0) // LDG
        {
          if (in_n_dims==2)
            calculate_common_viscFlux_NS_mpi_gpu_kernel<2,4,3,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_disu_fpts_r_ptr, in_grad_disu_fpts_l_ptr, in_grad_disu_fpts_r_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_sgsf_fpts_l_ptr, in_sgsf_fpts_r_ptr, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf,  in_mu_inf, in_c_sth, in_fix_vis, in_LES, in_motion);
          else if (in_n_dims==3)
            calculate_common_viscFlux_NS_mpi_gpu_kernel<3,5,6,0> <<<n_blocks,256>>>(in_n_fpts_per_inter, in_n_inters, in_disu_fpts_l_ptr, in_disu_fpts_r_ptr, in_grad_disu_fpts_l_ptr, in_grad_disu_fpts_r_ptr, in_norm_tconf_fpts_l_ptr, in_tdA_fpts_l_ptr, in_tdA_dyn_fpts_l_ptr, in_detjac_dyn_fpts_ptr, in_norm_fpts_ptr, in_norm_dyn_fpts_ptr, in_sgsf_fpts_l_ptr, in_sgsf_fpts_r_ptr, in_pen_fact, in_tau, in_gamma, in_prandtl, in_rt_inf,  in_mu_inf, in_c_sth, in_fix_vis, in_LES, in_motion);
        }
      else
        FatalError("ERROR: Viscous riemann solver type not recognized ... ");
    }
  else if (in_riemann_solve_type==1) // Lax-Friedrich
    {
      if (in_vis_riemann_solve_type==0) // LDG
        {
          if (in_n_dims==2)
            calculate_common_viscFlux_AD_mpi_gpu_kernel<2> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_diff_coeff);
          else if (in_n_dims==3)
            calculate_common_viscFlux_AD_mpi_gpu_kernel<3> <<<n_blocks,256>>>(in_n_fpts_per_inter,in_n_inters,in_disu_fpts_l_ptr,in_disu_fpts_r_ptr,in_grad_disu_fpts_l_ptr,in_grad_disu_fpts_r_ptr,in_norm_tconf_fpts_l_ptr,in_tdA_fpts_l_ptr,in_norm_fpts_ptr,in_pen_fact,in_tau,in_diff_coeff);
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

}

