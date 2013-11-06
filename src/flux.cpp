/*!
 * \file flux.cpp
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

#include <cmath>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/flux.h"

using namespace std;

// calculate inviscid flux in 2D

void calc_invf_2d(array<double>& in_u, array<double>& out_f)
{	
  if (run_input.equation==0) // Euler equation
  {
	  double vx;
	  double vy;
	  double p;
	  
	  vx=in_u(1)/in_u(0);
	  vy=in_u(2)/in_u(0);
	  p=(run_input.gamma-1.0)*(in_u(3)-(0.5*in_u(0)*((vx*vx)+(vy*vy))));
	  		
	  out_f(0,0)=in_u(1);
	  out_f(1,0)=p+(in_u(1)*vx);
	  out_f(2,0)=in_u(2)*vx;
	  out_f(3,0)=vx*(in_u(3)+p);
	  			
	  out_f(0,1)=in_u(2);
	  out_f(1,1)=in_u(1)*vy;
	  out_f(2,1)=p+(in_u(2)*vy);
	  out_f(3,1)=vy*(in_u(3)+p);
  }
  else if (run_input.equation==1) // Advection-diffusion equation
  {
    out_f(0,0) = run_input.wave_speed(0)*in_u(0);
    out_f(0,1) = run_input.wave_speed(1)*in_u(0);
  }
  else
  {
    FatalError("equation not recognized");
  }
}

// calculate inviscid flux in 3D

void calc_invf_3d(array<double>& in_u, array<double>& out_f)
{

  if (run_input.equation==0) // Euler Equation
  {
    double vx;
	  double vy;
	  double vz;
	  double p;
	  
	  vx=in_u(1)/in_u(0);
	  vy=in_u(2)/in_u(0);
	  vz=in_u(3)/in_u(0);
	  p=(run_input.gamma-1.0)*(in_u(4)-(0.5*in_u(0)*((vx*vx)+(vy*vy)+(vz*vz))));
	  
	  out_f(0,0)=in_u(1);
	  out_f(1,0)=p+(in_u(1)*vx);
	  out_f(2,0)=in_u(2)*vx;
	  out_f(3,0)=in_u(3)*vx;
	  out_f(4,0)=vx*(in_u(4)+p);
	  			
	  out_f(0,1)=in_u(2);
	  out_f(1,1)=in_u(1)*vy;
	  out_f(2,1)=p+(in_u(2)*vy);
	  out_f(3,1)=in_u(3)*vy;
	  out_f(4,1)=vy*(in_u(4)+p);
	  
	  out_f(0,2)=in_u(3);
	  out_f(1,2)=in_u(1)*vz;
	  out_f(2,2)=in_u(2)*vz;
	  out_f(3,2)=p+(in_u(3)*vz);
	  out_f(4,2)=vz*(in_u(4)+p);
  }
  else if (run_input.equation==1) // Advection-diffusion equation
  {
	  out_f(0,0) = run_input.wave_speed(0)*in_u(0);
    out_f(0,1) = run_input.wave_speed(1)*in_u(0);
    out_f(0,2) = run_input.wave_speed(2)*in_u(0);
  }
  else
  {
    FatalError("equation not recognized");
  }

}

// calculate viscous flux in 2D

void calc_visf_2d(array<double>& in_u, array<double>& in_grad_u, array<double>& out_f)
{
  if (run_input.equation==0) // Navier-Stokes equations
  {
    double rho, mom_x, mom_y, ene;

	  double rho_dx, mom_x_dx, mom_y_dx, ene_dx;
	  double rho_dy, mom_x_dy, mom_y_dy, ene_dy;

	  double inte,u,v;
	  double du_dx,du_dy,dv_dx,dv_dy,dke_dx,dke_dy;
	  double de_dx, de_dy;
	  double diag,tauxx,tauxy,tauyy;
	  double rt_ratio;
	  
	  double mu;
	  double p,T,R;
	  double inv_Re_c, Mach_c;
	  double T_gas_non, S_gas_non;	
	  
	  // states
	  
	  rho   = in_u(0);
	  mom_x = in_u(1);
	  mom_y = in_u(2);
	  ene   = in_u(3);

	  // gradients
	  
	  rho_dx	 = in_grad_u(0,0);
	  mom_x_dx = in_grad_u(1,0);
	  mom_y_dx = in_grad_u(2,0);
	  ene_dx	 = in_grad_u(3,0);

	  rho_dy	 = in_grad_u(0,1);
	  mom_x_dy = in_grad_u(1,1);
	  mom_y_dy = in_grad_u(2,1);
	  ene_dy	 = in_grad_u(3,1);

	  // states
	  
	  u = mom_x/rho;
	  v = mom_y/rho;
	  inte = ene/rho - 0.5*(u*u+v*v);

	  // viscosity
	  rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
	  mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1.+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
    mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

	  // gradients
	  
	  du_dx = (mom_x_dx-rho_dx*u)/rho;
	  du_dy = (mom_x_dy-rho_dy*u)/rho;

	  dv_dx = (mom_y_dx-rho_dx*v)/rho;
	  dv_dy = (mom_y_dy-rho_dy*v)/rho;

	  dke_dx = 0.5*(u*u+v*v)*rho_dx+rho*(u*du_dx+v*dv_dx);
	  dke_dy = 0.5*(u*u+v*v)*rho_dy+rho*(u*du_dy+v*dv_dy);

	  de_dx = (ene_dx-dke_dx-rho_dx*inte)/rho;
	  de_dy = (ene_dy-dke_dy-rho_dy*inte)/rho;

	  diag = (du_dx + dv_dy)/3.0;

	  tauxx = 2.0*mu*(du_dx-diag);
	  tauxy = mu*(du_dy + dv_dx);
	  tauyy = 2.0*mu*(dv_dy-diag);	

	  // construct flux
	  
	  out_f(0,0) = 0.0;
	  out_f(1,0) = -tauxx;
	  out_f(2,0) = -tauxy;
	  out_f(3,0) = -(u*tauxx+v*tauxy+mu*(run_input.gamma)*de_dx/(run_input.prandtl));
	  
	  out_f(0,1) = 0.0;
	  out_f(1,1) = -tauxy;
	  out_f(2,1) = -tauyy;
	  out_f(3,1) = -(u*tauxy+v*tauyy+mu*(run_input.gamma)*de_dy/(run_input.prandtl));
  }
  else if (run_input.equation==1) // Advection-diffusion equation
  {
    out_f(0,0) = -run_input.diff_coeff*in_grad_u(0,0);
    out_f(0,1) = -run_input.diff_coeff*in_grad_u(0,1);
  }
  else
  {
    FatalError("equation not recognized");
  }
}


// calculate viscous flux in 3D

void calc_visf_3d(array<double>& in_u, array<double>& in_grad_u, array<double>& out_f)
{
  if (run_input.equation==0) // Navier-Stokes equations
  {
	  double rho, mom_x, mom_y, mom_z, ene;

	  double rho_dx, mom_x_dx, mom_y_dx, mom_z_dx, ene_dx;
	  double rho_dy, mom_x_dy, mom_y_dy, mom_z_dy, ene_dy;
	  double rho_dz, mom_x_dz, mom_y_dz, mom_z_dz, ene_dz;

	  double inte,u,v,w;
	  double du_dx, du_dy, du_dz;
	  double dv_dx, dv_dy, dv_dz;
	  double dw_dx, dw_dy, dw_dz;

	  double dke_dx, dke_dy, dke_dz;
	  double de_dx, de_dy, de_dz;

	  double diag;
	  double tauxx, tauyy, tauzz;
	  double tauxy, tauxz, tauyz;
	  double rt_ratio;
	  
	  double mu;
	  double p,T,R;
	  double inv_Re_c, Mach_c;
	  double T_gas_non, S_gas_non;	
	  
	  // states
	  
	  rho   = in_u(0);
	  mom_x = in_u(1);
	  mom_y = in_u(2);
	  mom_z = in_u(3);
	  ene   = in_u(4);

	  // gradients
	  
	  rho_dx	 = in_grad_u(0,0);
	  mom_x_dx = in_grad_u(1,0);
	  mom_y_dx = in_grad_u(2,0);
	  mom_z_dx = in_grad_u(3,0);
	  ene_dx	 = in_grad_u(4,0);

	  rho_dy	 = in_grad_u(0,1);
	  mom_x_dy = in_grad_u(1,1);
	  mom_y_dy = in_grad_u(2,1);
	  mom_z_dy = in_grad_u(3,1);
	  ene_dy	 = in_grad_u(4,1);

	  rho_dz	 = in_grad_u(0,2);
	  mom_x_dz = in_grad_u(1,2);
	  mom_y_dz = in_grad_u(2,2);
	  mom_z_dz = in_grad_u(3,2);
	  ene_dz	 = in_grad_u(4,2);

	  // states
	  
	  u = mom_x/rho;
	  v = mom_y/rho;
	  w = mom_z/rho;

	  inte = ene/rho - 0.5*(u*u+v*v+w*w);

	  // viscosity
	  rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
    mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
    mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

	  //gradients
	  
	  du_dx = (mom_x_dx-rho_dx*u)/rho;
	  du_dy = (mom_x_dy-rho_dy*u)/rho;
	  du_dz = (mom_x_dz-rho_dz*u)/rho;

	  dv_dx = (mom_y_dx-rho_dx*v)/rho;
	  dv_dy = (mom_y_dy-rho_dy*v)/rho;
	  dv_dz = (mom_y_dz-rho_dz*v)/rho;

	  dw_dx = (mom_z_dx-rho_dx*w)/rho;
	  dw_dy = (mom_z_dy-rho_dy*w)/rho;
	  dw_dz = (mom_z_dz-rho_dz*w)/rho;

	  dke_dx = 0.5*(u*u+v*v+w*w)*rho_dx + rho*(u*du_dx+v*dv_dx+w*dw_dx);
	  dke_dy = 0.5*(u*u+v*v+w*w)*rho_dy + rho*(u*du_dy+v*dv_dy+w*dw_dy);
	  dke_dz = 0.5*(u*u+v*v+w*w)*rho_dz + rho*(u*du_dz+v*dv_dz+w*dw_dz);

	  de_dx = (ene_dx-dke_dx-rho_dx*inte)/rho;
	  de_dy = (ene_dy-dke_dy-rho_dy*inte)/rho;
	  de_dz = (ene_dz-dke_dz-rho_dz*inte)/rho;

	  diag = (du_dx + dv_dy + dw_dz)/3.0;

	  tauxx = 2.0*mu*(du_dx-diag);
	  tauyy = 2.0*mu*(dv_dy-diag);	
	  tauzz = 2.0*mu*(dw_dz-diag);
	  
	  tauxy = mu*(du_dy + dv_dx);
	  tauxz = mu*(du_dz + dw_dx);
	  tauyz = mu*(dv_dz + dw_dy);

	  // construct flux
	  
	  out_f(0,0) = 0.0;
	  out_f(1,0) = -tauxx;
	  out_f(2,0) = -tauxy;
	  out_f(3,0) = -tauxz;
	  out_f(4,0) = -(u*tauxx+v*tauxy+w*tauxz+mu*(run_input.gamma)*de_dx/(run_input.prandtl));
	  
	  out_f(0,1) = 0.0;
	  out_f(1,1) = -tauxy;
	  out_f(2,1) = -tauyy;
	  out_f(3,1) = -tauyz;
	  out_f(4,1) = -(u*tauxy+v*tauyy+w*tauyz+mu*(run_input.gamma)*de_dy/(run_input.prandtl));

	  out_f(0,2) = 0.0;
	  out_f(1,2) = -tauxz;
	  out_f(2,2) = -tauyz;
	  out_f(3,2) = -tauzz;
	  out_f(4,2) = -(u*tauxz+v*tauyz+w*tauzz+mu*(run_input.gamma)*de_dz/(run_input.prandtl));
  }
  else if (run_input.equation==1) // Advection-diffusion equation
  {
    out_f(0,0) = -run_input.diff_coeff*in_grad_u(0,0);
    out_f(0,1) = -run_input.diff_coeff*in_grad_u(0,1);
    out_f(0,2) = -run_input.diff_coeff*in_grad_u(0,2);
  }
  else
  {
    FatalError("equation not recognized");
  }
}
