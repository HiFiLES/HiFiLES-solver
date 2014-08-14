/*!
 * \file inters.cpp
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
#include <cmath>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/inters.h"
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

inters::inters()
{
  order=run_input.order;
  viscous=run_input.viscous;
  LES = run_input.LES;
  motion = run_input.motion;
}

inters::~inters() { }

// #### methods ####

void inters::setup_inters(int in_n_inters, int in_inters_type)
{
  n_inters    = in_n_inters;
  inters_type = in_inters_type;

  if(inters_type==0) // segs
    {
      n_dims=2;

      if (run_input.equation==0)
        n_fields=4;
      else if (run_input.equation==1)
        n_fields=1;
      else
        FatalError("Equation not supported");

      n_fpts_per_inter=order+1;
    }
  else if(inters_type==1) // tris
    {
      n_dims=3;

      if (run_input.equation==0)
        n_fields=5;
      else if (run_input.equation==1)
        n_fields=1;
      else
        FatalError("Equation not supported");

      n_fpts_per_inter=(order+2)*(order+1)/2;
    }
  else if(inters_type==2) // quads
    {
      n_dims=3;
      if (run_input.equation==0)
        n_fields=5;
      else if (run_input.equation==1)
        n_fields=1;
      else
        FatalError("Equation not supported");

      n_fpts_per_inter=(order+1)*(order+1);
    }
  else
    {
      FatalError("ERROR: Invalid interface type ... ");
    }

  if (run_input.turb_model==1)
    n_fields++;

      disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);
      norm_tconf_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);
      detjac_fpts_l.setup(n_fpts_per_inter,n_inters);
      tdA_fpts_l.setup(n_fpts_per_inter,n_inters);
      norm_fpts.setup(n_fpts_per_inter,n_inters,n_dims);
      pos_fpts.setup(n_fpts_per_inter,n_inters,n_dims);

      if (motion)
      {
        if (run_input.GCL) {
          disu_GCL_fpts_l.setup(n_fpts_per_inter,n_inters);
          norm_tconf_GCL_fpts_l.setup(n_fpts_per_inter,n_inters);
        }
        grid_vel_fpts.setup(n_fpts_per_inter,n_inters,n_dims);
        ndA_dyn_fpts_l.setup(n_fpts_per_inter,n_inters);
        norm_dyn_fpts.setup(n_fpts_per_inter,n_inters,n_dims);
        J_dyn_fpts_l.setup(n_fpts_per_inter,n_inters);        
        pos_dyn_fpts.setup(n_fpts_per_inter,n_inters,n_dims);
      }

      delta_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);

      if(viscous)
        {
          grad_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
          normal_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields);
          pos_disu_fpts_l.setup(n_fpts_per_inter,n_inters,n_dims);
        }

      if(LES) {
        sgsf_fpts_l.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
        sgsf_fpts_r.setup(n_fpts_per_inter,n_inters,n_fields,n_dims);
        temp_sgsf_l.setup(n_fields,n_dims);
        temp_sgsf_r.setup(n_fields,n_dims);
      }
      else {
        sgsf_fpts_l.setup(1);
        sgsf_fpts_r.setup(1);
      }

      temp_u_l.setup(n_fields);
      temp_u_r.setup(n_fields);

      temp_v.setup(n_dims);

      temp_grad_u_l.setup(n_fields,n_dims);
      temp_grad_u_r.setup(n_fields,n_dims);

      temp_normal_u_l.setup(n_fields);

      temp_pos_u_l.setup(n_dims);

      temp_f_l.setup(n_fields,n_dims);
      temp_f_r.setup(n_fields,n_dims);

      temp_f.setup(n_fields,n_dims);

      temp_fn_l.setup(n_fields);
      temp_fn_r.setup(n_fields);

      temp_loc.setup(n_dims);

      lut.setup(n_fpts_per_inter);

      // For Roe flux computation
      v_l.setup(n_dims);
      v_r.setup(n_dims);
      um.setup(n_dims);
      du.setup(n_fields);
}

// get look up table for flux point connectivity based on rotation tag
void inters::get_lut(int in_rot_tag)
{
  int i,j;

  if(inters_type==0) // segment
    {
      for(i=0;i<n_fpts_per_inter;i++)
        {
          lut(i)=n_fpts_per_inter-i-1;
        }
    }
  else if(inters_type==1) // triangle face
    {
      int index0,index1;
      if(in_rot_tag==0) // Example face 0 with 1
        {
          for(j=0;j<order+1;j++)
            {
              for (i=0;i<order-j+1;i++)
                {
                  index0 = j*(order+1) - (j-1)*j/2 + i;
                  index1 = i*(order+1) - (i-1)*i/2 + j;
                  lut(index0) = index1;

                }
            }
        }
      else if(in_rot_tag==1) // Example face 0 with 3
        {
          for(j=0;j<order+1;j++)
            {
              for (i=0;i<order+1-j;i++)
                {
                  index0 = j*(order+1) - (j-1)*j/2 + i;
                  index1 = (order+1)*(order+2)/2 -1 -(i+j)*(i+j+1)/2 -j;
                  lut(index0) = index1;

                }
            }
        }
      else if(in_rot_tag==2) // Example face 0 with 2
        {

          for(j=0;j<order+1;j++)
            {
              for (i=0;i<order+1-j;i++)
                {
                  index0 = j*(order+1) - (j-1)*j/2 + i;
                  index1 = j*(order+1) - (j-1)*j/2 + (order-j-i);
                  lut(index0) = index1;
                }
            }
        }
      else
        {
          cout << "ERROR: Unknown rotation of triangular face..." << endl;
        }
    }
  else if(inters_type==2) // quad face
    {
      if(in_rot_tag==0)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=((order+1)-1-j)+((order+1)*i);
                }
            }
        }
      else if(in_rot_tag==1)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=n_fpts_per_inter-((order+1)-1-j)-((order+1)*i)-1;
                }
            }
        }
      else if(in_rot_tag==2)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=((order+1)*j)+i;
                }
            }
        }
      else if(in_rot_tag==3)
        {
          for(i=0;i<(order+1);i++)
            {
              for(j=0;j<(order+1);j++)
                {
                  lut((i*(order+1))+j)=n_fpts_per_inter-((order+1)*j)-i-1;
                }
            }
        }
      else
        {
          cout << "ERROR: Unknown rotation tag ... " << endl;
        }
    }
  else
    {
      FatalError("ERROR: Invalid interface type ... ");
    }
}

// Rusanov inviscid numerical flux
void inters::right_flux(array<double> &f_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double gamma)
{
  // calculate normal flux from discontinuous solution at flux points
  for(int k=0;k<n_fields;k++) {
      fn(k)=0.;
      for(int l=0;l<n_dims;l++) {
          fn(k)+=f_r(k,l)*norm(l);
        }
    }
}

// Rusanov inviscid numerical flux
void inters::rusanov_flux(array<double> &u_l, array<double> &u_r, array<double> &v_g, array<double> &f_l, array<double> &f_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double gamma)
{
  double vx_l,vy_l,vx_r,vy_r,vz_l,vz_r,vn_l,vn_r,p_l,p_r,vn_g,vn_av_mag,c_av,eig;
  array<double> fn_l(n_fields),fn_r(n_fields);

  // calculate normal flux from discontinuous solution at flux points
  for(int k=0;k<n_fields;k++) {

      fn_l(k)=0.;
      fn_r(k)=0.;

      for(int l=0;l<n_dims;l++) {
          fn_l(k)+=f_l(k,l)*norm(l);
          fn_r(k)+=f_r(k,l)*norm(l);
        }
    }

  // calculate wave speeds
  vx_l=u_l(1)/u_l(0);
  vx_r=u_r(1)/u_r(0);

  vy_l=u_l(2)/u_l(0);
  vy_r=u_r(2)/u_r(0);

  if(n_dims==2) {
      vn_l=vx_l*norm(0)+vy_l*norm(1);
      vn_r=vx_r*norm(0)+vy_r*norm(1);
      vn_g = v_g(0)*norm(0) + v_g(1)*norm(1);

      p_l=(gamma-1.0)*(u_l(3)-(0.5*u_l(0)*((vx_l*vx_l)+(vy_l*vy_l))));
      p_r=(gamma-1.0)*(u_r(3)-(0.5*u_r(0)*((vx_r*vx_r)+(vy_r*vy_r))));
    }
  else if(n_dims==3) {
      vz_l=u_l(3)/u_l(0);
      vz_r=u_r(3)/u_r(0);

      vn_l=vx_l*norm(0)+vy_l*norm(1)+vz_l*norm(2);
      vn_r=vx_r*norm(0)+vy_r*norm(1)+vz_r*norm(2);
      vn_g = v_g(0)*norm(0) + v_g(1)*norm(1) + v_g(2)*norm(2);

      p_l=(gamma-1.0)*(u_l(4)-(0.5*u_l(0)*((vx_l*vx_l)+(vy_l*vy_l)+(vz_l*vz_l))));
      p_r=(gamma-1.0)*(u_r(4)-(0.5*u_r(0)*((vx_r*vx_r)+(vy_r*vy_r)+(vz_r*vz_r))));
    }
  else
    FatalError("ERROR: Invalid number of dimensions ... ");

  vn_av_mag=sqrt(0.25*(vn_l+vn_r)*(vn_l+vn_r));
  c_av=sqrt((gamma*(p_l+p_r))/(u_l(0)+u_r(0)));
  eig = fabs(vn_av_mag - vn_g + c_av);

  // calculate the normal continuous flux at the flux points

  for(int k=0;k<n_fields;k++)
    fn(k) = 0.5*( (fn_l(k)+fn_r(k)) - eig*(u_r(k)-u_l(k)) );
}

// Central-difference inviscid numerical flux at the boundaries
void inters::convective_flux_boundary( array<double> &f_l, array<double> &f_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields)
{
  array<double> fn_l(n_fields),fn_r(n_fields);

  // calculate normal flux from total discontinuous flux at flux points
  for(int k=0;k<n_fields;k++) {

      fn_l(k)=0.;
      fn_r(k)=0.;

      for(int l=0;l<n_dims;l++) {
          fn_l(k)+=f_l(k,l)*norm(l);
          fn_r(k)+=f_r(k,l)*norm(l);
        }
    }

  // calculate the normal transformed continuous flux at the flux points
  for(int k=0;k<n_fields;k++)
    fn(k)=0.5*(fn_l(k)+fn_r(k));
}

// Roe inviscid numerical flux
void inters::roe_flux(array<double> &u_l, array<double> &u_r, array<double> &v_g, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double gamma)
{
  double p_l,p_r;
  double h_l, h_r;
  double sq_rho,rrho,hm,usq,am,am_sq,unm,vgn;
  double lambda0,lambdaP,lambdaM;
  double rhoun_l, rhoun_r,eps;
  double a1,a2,a3,a4,a5,a6,aL1,bL1;
  //array<double> um(n_dims);

  // velocities
  for (int i=0;i<n_dims;i++)  {
      v_l(i) = u_l(i+1)/u_l(0);
      v_r(i) = u_r(i+1)/u_r(0);
    }

  if (n_dims==2) {
      p_l=(gamma-1.0)*(u_l(3)-(0.5*u_l(0)*((v_l(0)*v_l(0))+(v_l(1)*v_l(1)))));
      p_r=(gamma-1.0)*(u_r(3)-(0.5*u_r(0)*((v_r(0)*v_r(0))+(v_r(1)*v_r(1)))));
    }
  else
    FatalError("Roe not implemented in 3D");

  h_l = (u_l(n_dims+1)+p_l)/u_l(0);
  h_r = (u_r(n_dims+1)+p_r)/u_r(0);

  sq_rho = sqrt(u_r(0)/u_l(0));
  rrho = 1./(sq_rho+1.);

  for (int i=0;i<n_dims;i++)
    um(i) = rrho*(v_l(i)+sq_rho*v_r(i));

  hm      = rrho*(h_l     +sq_rho*h_r);


  usq=0.;
  for (int i=0;i<n_dims;i++)
    usq += 0.5*um(i)*um(i);

  am_sq   = (gamma-1.)*(hm-usq);
  am  = sqrt(am_sq);
  unm = 0.;
  vgn = 0.;
  for (int i=0;i<n_dims;i++) {
    unm += um(i)*norm(i);
    vgn += v_g(i)*norm(i);
  }

  // Compute Euler flux (first part)
  rhoun_l = 0.;
  rhoun_r = 0.;
  for (int i=0;i<n_dims;i++)
    {
      rhoun_l += u_l(i+1)*norm(i);
      rhoun_r += u_r(i+1)*norm(i);
    }

  if (n_dims==2)
    {
      fn(0) = rhoun_l + rhoun_r;
      fn(1) = rhoun_l*v_l(0) + rhoun_r*v_r(0) + (p_l+p_r)*norm(0);
      fn(2) = rhoun_l*v_l(1) + rhoun_r*v_r(1) + (p_l+p_r)*norm(1);
      fn(3) = rhoun_l*h_l   +rhoun_r*h_r;

    }
  else
    FatalError("Roe not implemented in 3D");

  for (int i=0;i<n_fields;i++)
    {
      du(i) = u_r(i)-u_l(i);
    }

  lambda0 = abs(unm-vgn);
  lambdaP = abs(unm-vgn+am);
  lambdaM = abs(unm-vgn-am);

  // Entropy fix
  eps = 0.5*(abs(rhoun_l/u_l(0)-rhoun_r/u_r(0))+ abs(sqrt(gamma*p_l/u_l(0))-sqrt(gamma*p_r/u_r(0))));
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

      a5 = usq*du(0)-um(0)*du(1)-um(1)*du(2)+du(3);
      a6 = unm*du(0)-norm(0)*du(1)-norm(1)*du(2);
    }
  else if (n_dims==3)
    {
      a5 = usq*du(0)-um(0)*du(1)-um(1)*du(2)-um(2)*du(3)+du(4);
      a6 = unm*du(0)-norm(0)*du(1)-norm(1)*du(2)-norm(2)*du(3);
    }


  aL1 = a1*a5 - a3*a6;
  bL1 = a4*a5 - a2*a6;

  // Compute Euler flux (second part)
  if (n_dims==2)
    {
      fn(0) = fn(0) - (lambda0*du(0)+aL1);
      fn(1) = fn(1) - (lambda0*du(1)+aL1*um(0)+bL1*norm(0));
      fn(2) = fn(2) - (lambda0*du(2)+aL1*um(1)+bL1*norm(1));
      fn(3) = fn(3) - (lambda0*du(3)+aL1*hm   +bL1*unm);

    }
  else if (n_dims==3)
    {
      fn(0) = fn(0) - (lambda0*du(0)+aL1);
      fn(1) = fn(1) - (lambda0*du(1)+aL1*um(0)+bL1*norm(0));
      fn(2) = fn(2) - (lambda0*du(2)+aL1*um(1)+bL1*norm(1));
      fn(3) = fn(3) - (lambda0*du(3)+aL1*um(2)+bL1*norm(2));
      fn(4) = fn(4) - (lambda0*du(4)+aL1*hm   +bL1*unm);
    }

  for (int i=0;i<n_fields;i++)
  {
    fn(i) =  0.5*fn(i) - 0.5*vgn*(u_r(i)+u_l(i));
  }

}


// Rusanov inviscid numerical flux
void inters::lax_friedrich(array<double> &u_l, array<double> &u_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double lambda, array<double>& wave_speed)
{

  double u_av;
  double u_diff;

  u_av = 0.5*(u_l(0)+u_r(0));
  u_diff = (u_l(0)-u_r(0));

  double norm_speed = 0;
  for (int i=0;i<n_dims;i++)
    {
      norm_speed += wave_speed(i)*norm(i);
    }

  fn(0) = 0.;
  for (int i=0;i<n_dims;i++)
    {
      fn(0) += wave_speed(i)*norm(i)*u_av;
    }

  fn(0) += 0.5*lambda*abs(norm_speed)*u_diff;
}


// LDG viscous numerical flux
void inters::ldg_flux(int flux_spec, array<double> &u_l, array<double> &u_r, array<double> &f_l, array<double> &f_r, array<double> &norm, array<double> &fn, int n_dims, int n_fields, double tau, double pen_fact)
{
  array<double> f_c(n_fields,n_dims);
  double norm_x, norm_y, norm_z;

  if(n_dims==2) // needs to be reviewed and understood
    {
      if ((norm(0)+norm(1)) <0.)
        pen_fact = -pen_fact;
    }
  if(n_dims==3)
    {
      if ((norm(0)+norm(1)+sqrt(2.)*norm(2)) <0.)
        pen_fact = -pen_fact;
    }

  norm_x = norm(0);
  norm_y = norm(1);

  if(n_dims == 3)
    norm_z = norm(2);


  if(flux_spec == 0) //Interior and mpi
    {
      for(int k=0;k<n_fields;k++)
        {
          if(n_dims == 2)
            {
              f_c(k,0) = 0.5*(f_l(k,0) + f_r(k,0)) + pen_fact*norm_x*( norm_x*(f_l(k,0) - f_r(k,0)) + norm_y*(f_l(k,1) - f_r(k,1)) ) + tau*norm_x*(u_l(k) - u_r(k));
              f_c(k,1) = 0.5*(f_l(k,1) + f_r(k,1)) + pen_fact*norm_y*( norm_x*(f_l(k,0) - f_r(k,0)) + norm_y*(f_l(k,1) - f_r(k,1)) ) + tau*norm_y*(u_l(k) - u_r(k));
            }

          if(n_dims == 3)
            {
              f_c(k,0) = 0.5*(f_l(k,0) + f_r(k,0)) + pen_fact*norm_x*( norm_x*(f_l(k,0) - f_r(k,0)) + norm_y*(f_l(k,1) - f_r(k,1)) + norm_z*(f_l(k,2) - f_r(k,2)) ) + tau*norm_x*(u_l(k) - u_r(k));
              f_c(k,1) = 0.5*(f_l(k,1) + f_r(k,1)) + pen_fact*norm_y*( norm_x*(f_l(k,0) - f_r(k,0)) + norm_y*(f_l(k,1) - f_r(k,1)) + norm_z*(f_l(k,2) - f_r(k,2)) ) + tau*norm_y*(u_l(k) - u_r(k));
              f_c(k,2) = 0.5*(f_l(k,2) + f_r(k,2)) + pen_fact*norm_z*( norm_x*(f_l(k,0) - f_r(k,0)) + norm_y*(f_l(k,1) - f_r(k,1)) + norm_z*(f_l(k,2) - f_r(k,2)) ) + tau*norm_z*(u_l(k) - u_r(k));
            }
        }
    }
  else if(flux_spec == 1) //Dirichlet
    {
      for(int k=0;k<n_fields;k++)
        {
          if(n_dims == 2)
            {
              f_c(k,0) = f_l(k,0) + tau*norm_x*(u_l(k) - u_r(k));
              f_c(k,1) = f_l(k,1) + tau*norm_y*(u_l(k) - u_r(k));
            }

          if(n_dims == 3)
            {
              f_c(k,0) = f_l(k,0) + tau*norm_x*(u_l(k) - u_r(k));
              f_c(k,1) = f_l(k,1) + tau*norm_y*(u_l(k) - u_r(k));
              f_c(k,2) = f_l(k,2) + tau*norm_z*(u_l(k) - u_r(k));
            }
        }
    }
  else if(flux_spec == 2) //von Neumann
    {
      for(int k=0;k<n_fields;k++)
        {
          if(n_dims == 2)
            {
              f_c(k,0) = f_r(k,0) + tau*norm_x*(u_l(k) - u_r(k));
              f_c(k,1) = f_r(k,1) + tau*norm_y*(u_l(k) - u_r(k));
            }

          if(n_dims == 3)
            {
              f_c(k,0) = f_r(k,0) + tau*norm_x*(u_l(k) - u_r(k));
              f_c(k,1) = f_r(k,1) + tau*norm_y*(u_l(k) - u_r(k));
              f_c(k,2) = f_r(k,2) + tau*norm_z*(u_l(k) - u_r(k));
            }
        }
    }
  else
    FatalError("This variant of the LDG flux has not been implemented");


  // calculate normal flux from discontinuous solution at flux points
  for(int k=0;k<n_fields;k++)
    {
      fn(k) = f_c(k,0)*norm(0);

      for(int l=1;l<n_dims;l++)
        {
          fn(k) += f_c(k,l)*norm(l);
        }
    }
}


// LDG common solution
void inters::ldg_solution(int flux_spec, array<double> &u_l, array<double> &u_r, array<double> &u_c, double pen_fact, array<double>& norm)
{

  if(flux_spec == 0) // Interior and mpi
    {
      // Choosing a unique direction for the switch
      if(n_dims==2)
        {
          if ((norm(0)+norm(1)) <0.)
            pen_fact = -pen_fact;
        }
      if(n_dims==3)
        {
          if ((norm(0)+norm(1)+sqrt(2.)*norm(2)) <0.)
            pen_fact = -pen_fact;
        }

      for(int k=0;k<n_fields;k++)
        u_c(k) = 0.5*(u_l(k) + u_r(k)) - pen_fact*(u_l(k) - u_r(k));
    }
  else if(flux_spec == 1) //Dirichlet
    {
      for(int k=0;k<n_fields;k++)
        u_c(k) = 0.5 * ( u_r(k) + u_l(k) );
    }
  else if(flux_spec == 2) //von Neumann
    {
      for(int k=0;k<n_fields;k++)
        u_c(k) = 0.5 * ( u_r(k) + u_l(k) );
    }
  else
    FatalError("This variant of the LDG flux has not been implemented");

}

