/*!
 * \file source.cpp
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

#include <cmath>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/source.h"

using namespace std;

// calculate source term for Spalart-Allmaras turbulence model in 2D
void calc_source_SA_2d(array<double>& in_u, array<double>& in_grad_u, double& d, double& out_source)
{
  double rho, u, v, ene, nu_tilde;
  double dv_dx, du_dy, dnu_tilde_dx, dnu_tilde_dy;

  double inte, rt_ratio, mu;

  double nu_t_prod, nu_t_diff, nu_t_dest;

  double S, S_bar, S_tilde, Chi, psi, f_v1, f_v2;

  double c_w1, r, g, f_w;

  // primitive variables
  rho = in_u(0);
  u = in_u(1)/in_u(0);
  v = in_u(2)/in_u(0);
  ene = in_u(3);
  nu_tilde = in_u(4)/in_u(0);

  // gradients
  dv_dx = (in_grad_u(2,0)-in_grad_u(0,0)*v)/rho;
  du_dy = (in_grad_u(1,1)-in_grad_u(0,1)*u)/rho;

  dnu_tilde_dx = (in_grad_u(4,0)-in_grad_u(0,0)*nu_tilde)/rho;
  dnu_tilde_dy = (in_grad_u(4,1)-in_grad_u(0,1)*nu_tilde)/rho;

  // viscosity
  inte = ene/rho - 0.5*(u*u+v*v);
  rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
  mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1.+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
  mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

  // regulate eddy viscosity (must not become negative)
  Chi = in_u(4)/mu;
  if (Chi <= 10.0)
    psi = 0.05*log(1.0 + exp(20.0*Chi));
  else
    psi = Chi;

  // solve for production term for eddy viscosity
  // (solve for S = magnitude of vorticity)
  S = abs(dv_dx - du_dy);

  // (solve for S_bar)
  f_v1 = pow(in_u(4)/mu, 3.0)/(pow(in_u(4)/mu, 3.0) + pow(run_input.c_v1, 3.0));
  f_v2 = 1.0 - psi/(1.0 + psi*f_v1);
  S_bar = pow(mu*psi/rho, 2.0)*f_v2/(pow(run_input.Kappa, 2.0)*pow(d, 2.0));

  // (solve for S_tilde)
  if (S_bar >= -run_input.c_v2*S)
    S_tilde = S + S_bar;
  else
    S_tilde = S + S*(pow(run_input.c_v2, 2.0)*S + run_input.c_v3*S_bar)/((run_input.c_v3 - 2.0*run_input.c_v2)*S - S_bar);

  // (production term)
  nu_t_prod = run_input.c_b1*S_tilde*mu*psi;

  // solve for non-conservative diffusion term for eddy viscosity
  nu_t_diff = (1.0/run_input.omega)*(run_input.c_b2*rho*(pow(dnu_tilde_dx, 2.0)+pow(dnu_tilde_dy, 2.0)));

  // solve for destruction term for eddy viscosity
  c_w1 = run_input.c_b1/pow(run_input.Kappa, 2.0) + (1.0/run_input.omega)*(1.0 + run_input.c_b2);
  r = min((mu*psi/rho)/(S_tilde*pow(run_input.Kappa, 2.0)*pow(d, 2.0)), 10.0);
  g = r + run_input.c_w2*(pow(r, 6.0) - r);
  f_w = g*pow((1.0 + pow(run_input.c_w3, 6.0))/(pow(g, 6.0) + pow(run_input.c_w3, 6.0)), 1.0/6.0);

  // (destruction term)
  nu_t_dest = -c_w1*rho*f_w*pow((mu*psi/rho)/d, 2.0);

  // construct source term
  out_source = nu_t_prod + nu_t_diff + nu_t_dest;
}

// calculate source term for Spalart-Allmaras turbulence model in 3D
// NOTE:: I STILL NEED TO WRITE THIS
void calc_source_SA_3d(array<double>& in_u, array<double>& in_grad_u, double& d, double& out_source)
{
  cout << "3D source term not implemented yet" << endl;
}
