/*!
 * \file source.cpp
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

using namespace std;

#include "global.h"
#include "array.h"
#include "source.h"

// calculate source term for Spalart-Allmaras turbulence model in 2D
void calc_source_SA_2d(array<double>& in_u, array<double>& in_grad_u, double& d, double& out_source)
{
    double rho, u, v, ene, nu_tilda;
    double dv_dx, du_dy, dnu_tilda_dx, dnu_tilda_dy;

    double inte, rt_ratio, mu;

    double nu_t_prod, nu_t_diff, nu_t_dest;

    double S, S_bar, S_tilda, Chi, psi, f_v1, f_v2;

    double c_w1, r, g, f_w;

    // primitive variables
    rho = in_u(0);
    u = in_u(1)/in_u(0);
    v = in_u(2)/in_u(0);
    ene = in_u(3);
    nu_tilda = in_u(4)/in_u(0);

    // gradients
    dv_dx = (in_grad_u(2,0)-in_grad_u(0,0)*v)/rho;
    du_dy = (in_grad_u(1,1)-in_grad_u(0,1)*u)/rho;

    dnu_tilda_dx = (in_grad_u(4,0)-in_grad_u(0,0)*nu_tilda)/rho;
    dnu_tilda_dy = (in_grad_u(4,1)-in_grad_u(0,1)*nu_tilda)/rho;

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

    // (solve for S_tilda)
    if (S_bar >= -run_input.c_v2*S)
        S_tilda = S + S_bar;
    else
        S_tilda = S + S*(pow(run_input.c_v2, 2.0)*S + run_input.c_v3*S_bar)/((run_input.c_v3 - 2.0*run_input.c_v2)*S - S_bar);

    // (production term)
    nu_t_prod = run_input.c_b1*S_tilda*mu*psi;

    // solve for non-conservative diffusion term for eddy viscosity
    nu_t_diff = (1.0/run_input.omega)*(run_input.c_b2*rho*(pow(dnu_tilda_dx, 2.0)+pow(dnu_tilda_dy, 2.0)));

    // solve for destruction term for eddy viscosity
    c_w1 = run_input.c_b1/pow(run_input.Kappa, 2.0) + (1.0/run_input.omega)*(1.0 + run_input.c_b2);
    r = (mu*psi/rho)/(S_tilda*pow(run_input.Kappa, 2.0)*pow(d, 2.0));
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

}
