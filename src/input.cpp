/*!
 * \file input.cpp
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
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>

#include "../include/input.h"
#include "../include/Array.h"
#include "../include/funcs.h"
#include "../include/global.h"

using namespace std;

// #### constructors ####

// default constructor

input::input()
{
  // ** Set default values for commonly-used parameters **
  // -- Simulation Parameters
  equation = 0;            // Euler / Navier-Stokes
  viscous = 0;             // Inviscid
  riemann_solve_type = 0;  // Rusanov
  ic_form = 1;             // Uniform flow
  test_case = 0;           // Not a test case
  dt_type = 0;             // User-supplied constant time step
  tau = 0;                 // Used in LDG flux
  pen_fact = 0.5;          // Used in LDG flux
  // -- Restart Options
  restart_flag = 0;        // Not restarting
  // -- Mesh Options
  dx_cyclic = INFINITY;    // Not periodic in x
  dy_cyclic = INFINITY;    // Not periodic in y
  dz_cyclic = INFINITY;    // Not periodic in z
  // -- Data Output Options
  monitor_cp_freq = 0;     // No output
  res_norm_type = 2;       // Use L2 norm on residual
  error_norm_type = 2;     // Use L2 norm on residual
  // -- Fluid Properties
  R_gas = 286.9;           // Typical value of ideal gas constant for air (J/kg-K)
  gamma = 1.4;             // Typical value of gamma for air
  prandtl = .72;           // Typical value of Prandtl number for air
  S_gas = 120;             // Typical value of S from Sutherland's Law
  T_gas = 291.15;          // Typical temperature for Sutherland's Law (Kelvin)
  mu_gas = 1.827E-5;       // Typical value of mu for air (kg/m-s)
  // -- Boundary Conditions
  L_free_stream = 1;       // Reference length of 1
  T_free_stream = 300;     // Typical temperature (Kelvin)
  Mach_wall = 0;           // Mach number of "moving wall" b.c.
  T_wall = 300;
  // -- Element Parameters
  upts_type_tri = 0;
  fpts_type_tri = 0;
  vcjh_scheme_tri = 1;
  c_tri = 0.0;
  sparse_tri = 0;
  upts_type_quad = 0;
  vcjh_scheme_quad = 0;
  eta_quad = 0.0;
  sparse_quad = 0;
  upts_type_hexa = 0;
  vcjh_scheme_hexa = 0;
  eta_hexa = 0;
  sparse_hexa = 0;
  upts_type_tet = 1;
  fpts_type_tet = 0;
  vcjh_scheme_tet = 0;
  eta_tet =  0;
  sparse_tet = 0;
  upts_type_pri_tri = 0;
  upts_type_pri_1d = 0;
  vcjh_scheme_pri_1d = 0;
  eta_pri = 0;
  sparse_pri = 0;

  // Set default values for optional parameters
  turb_model = 0;
  motion = 0;
  GCL = 0;
  n_deform_iters = 1;
  restart_mesh_out = 0;
  mesh_output_freq = INFINITY;
  mesh_output_format = 1;
  spinup_time = 0.0;

  // Set shock capturing parameters to 0 in case they are not used
  ArtifOn = 0;
  artif_only = 0;
  artif_type = 0;
  epsilon0 = 0;
  s0 = 0;
  kappa = 0;
  shock_vortex_restart = 0;
  p_bound_out = 0;

  // Initialize initial-condition values
  // Viscous Params
  Mach_c_ic = INFINITY;
  nx_c_ic = INFINITY;
  ny_c_ic = INFINITY;
  nz_c_ic = INFINITY;
  Re_c_ic = INFINITY;
  T_c_ic = INFINITY;

  // Inviscid Params
  rho_c_ic = INFINITY;
  u_c_ic = INFINITY;
  v_c_ic = INFINITY;
  w_c_ic = INFINITY;
  p_c_ic = INFINITY;
}

input::~input()
{
}

void input::set_order(int in_order)
{
  order=in_order;
}

void input::set_dt(double in_dt)
{
  dt=in_dt;
}

void input::set_c(double in_c_tri, double in_c_quad)
{
  c_tri = in_c_tri;
  c_quad = in_c_quad;
  
  double a_k = eval_gamma(2*order+1)/( pow(2.,order)*pow(eval_gamma(order+1),2) );
  eta_quad=in_c_quad*0.5*(2.*order+1.)*a_k*eval_gamma(order+1)*a_k*eval_gamma(order+1);
}

void input::set_vcjh_scheme_tri(int in_vcjh_scheme_tri)
{
  vcjh_scheme_tri = in_vcjh_scheme_tri;
}
void input::set_vcjh_scheme_hexa(int in_vcjh_scheme_hexa)
{
  vcjh_scheme_hexa= in_vcjh_scheme_hexa;
}
void input::set_vcjh_scheme_pri_1d(int in_vcjh_scheme_pri_1d)
{
  vcjh_scheme_pri_1d= in_vcjh_scheme_pri_1d;
}

void input::setup(ifstream& in_run_input_file, int rank)
{
  v_bound.setup(3);
  wave_speed.setup(3);
  v_wall.setup(3);
  diff_coeff = 0.;
  
  char buf[BUFSIZ]={""};
  char section_TXT[100];
  string dummy, param_name;
  bool blank;

  // First loop over the input file and print content to output
  if (rank==0)
  {
    while(!in_run_input_file.eof())
    {
      in_run_input_file.getline(buf,BUFSIZ);
    }
    // Rewind
    in_run_input_file.clear();
    in_run_input_file.seekg(0, ios::beg);
  }

  // Now read in parameters
  while(!in_run_input_file.eof() )
  {
    // Read section name
    in_run_input_file.getline(buf,BUFSIZ);
    sscanf(buf,"%s",section_TXT);
    param_name.assign(section_TXT,0,99);
    blank = false;

    if (in_run_input_file.peek()=='\n') {
      // Blank line. Funky, but it's the only way
      while(in_run_input_file.peek()=='\n')
        in_run_input_file.get();
      blank = true;
    }
    else if (!param_name.compare(0,5,"-----"))
    {
      // Section header, ignore next two lines
      in_run_input_file.getline(buf,BUFSIZ);
    }
    else if (!param_name.compare(0,2,"//"))
    {
      // Skip comment line
    }
    else if (!param_name.compare("equation"))
    {
      in_run_input_file >> equation;
    }
    else if (!param_name.compare("order"))
    {
      in_run_input_file >> order;
    }
    else if (!param_name.compare("viscous"))
    {
      in_run_input_file >> viscous;
    }
    else if (!param_name.compare("riemann_solve_type"))
    {
      in_run_input_file >> riemann_solve_type;
    }
    else if (!param_name.compare("vis_riemann_solve_type"))
    {
      in_run_input_file >> vis_riemann_solve_type;
    }
    else if (!param_name.compare("ic_form"))
    {
      in_run_input_file >> ic_form;
    }
    else if (!param_name.compare("test_case"))
    {
      in_run_input_file >> test_case;
    }
    else if (!param_name.compare("n_diagnostic_fields"))
    {
      in_run_input_file >> n_diagnostic_fields;
      diagnostic_fields.setup(n_diagnostic_fields);
      for (int i=0;i<n_diagnostic_fields;i++)
        in_run_input_file >> diagnostic_fields(i);
    }
    else if (!param_name.compare("n_average_fields"))
    {
      in_run_input_file >> n_average_fields;
      average_fields.setup(n_average_fields);
      for (int i=0;i<n_average_fields;i++)
        in_run_input_file >> average_fields(i);
    }
    else if (!param_name.compare("spinup_time"))
    {
      in_run_input_file >> spinup_time;
    }
    else if (!param_name.compare("inters_cub_order"))
    {
      in_run_input_file >> inters_cub_order;
    }
    else if (!param_name.compare("volume_cub_order"))
    {
      in_run_input_file >> volume_cub_order;
    }
    else if (!param_name.compare("dt_type"))
    {
      in_run_input_file >> dt_type;

      if (dt_type == 2 && rank == 0)
      {
        cout << "Note: Local timestepping is still in an experimental phase," 
          << " especially for viscous simulations." ;
      }
    }
    else if (!param_name.compare("dt"))
    {
      in_run_input_file >> dt;
    }
    else if (!param_name.compare("CFL"))
    {
      in_run_input_file >> CFL;
    }
    else if (!param_name.compare("n_steps"))
    {
      in_run_input_file >> n_steps;
    }
    else if (!param_name.compare("turb_model"))
    {
      in_run_input_file >> turb_model;
    }
    else if (!param_name.compare("LES"))
    {
      in_run_input_file >> LES;
    }
    else if (!param_name.compare("filter_type"))
    {
      in_run_input_file >> filter_type;
    }
    else if (!param_name.compare("filter_ratio"))
    {
      in_run_input_file >> filter_ratio;
    }
    else if (!param_name.compare("SGS_model"))
    {
      in_run_input_file >> SGS_model;
    }
    else if (!param_name.compare("wall_model"))
    {
      in_run_input_file >> wall_model;
    }
    else if (!param_name.compare("wall_layer_thickness"))
    {
      in_run_input_file >> wall_layer_t;
    }
    else if (!param_name.compare("plot_freq"))
    {
      in_run_input_file >> plot_freq;
    }
    else if (!param_name.compare("restart_dump_freq"))
    {
      in_run_input_file >> restart_dump_freq;
    }
    else if (!param_name.compare("restart_mesh_out"))
    {
      in_run_input_file >> restart_mesh_out;
    }
    else if (!param_name.compare("adv_type"))
    {
      in_run_input_file >> adv_type;
    }
    else if (!param_name.compare("const_src"))
    {
      in_run_input_file >> const_src;
    }
    else if (!param_name.compare("monitor_res_freq"))
    {
      in_run_input_file >> monitor_res_freq;
    }
    else if (!param_name.compare("monitor_cp_freq"))
    {
      in_run_input_file >> monitor_cp_freq;
    }
    else if (!param_name.compare("monitor_integrals_freq"))
    {
      in_run_input_file >> monitor_integrals_freq;
    }
    else if (!param_name.compare("n_integral_quantities"))
    {
      in_run_input_file >> n_integral_quantities;
      integral_quantities.setup(n_integral_quantities);
      for (int i=0;i<n_integral_quantities;i++)
        in_run_input_file >> integral_quantities(i) ;
    }
    else if (!param_name.compare("res_norm_type"))
    {
      in_run_input_file >> res_norm_type;
    }
    else if (!param_name.compare("error_norm_type"))
    {
      in_run_input_file >> error_norm_type;
    }
    else if (!param_name.compare("res_norm_field"))
    {
      in_run_input_file >> res_norm_field;
    }
    else if (!param_name.compare("restart_flag"))
    {
      in_run_input_file >> restart_flag;
    }
    else if (!param_name.compare("restart_iter"))
    {
      in_run_input_file >> restart_iter;
    }
    else if (!param_name.compare("n_restart_files"))
    {
      in_run_input_file >> n_restart_files;
    }
    else if (!param_name.compare("rho_c_ic"))
    {
      in_run_input_file >> rho_c_ic;
    }
    else if (!param_name.compare("u_c_ic"))
    {
      in_run_input_file >> u_c_ic;
    }
    else if (!param_name.compare("v_c_ic"))
    {
      in_run_input_file >> v_c_ic;
    }
    else if (!param_name.compare("w_c_ic"))
    {
      in_run_input_file >> w_c_ic;
    }
    else if (!param_name.compare("p_c_ic"))
    {
      in_run_input_file >> p_c_ic;
    }
    else if (!param_name.compare("rho_bound"))
    {
      in_run_input_file >> rho_bound;
    }
    else if (!param_name.compare("u_bound"))
    {
      in_run_input_file >> v_bound(0);
    }
    else if (!param_name.compare("v_bound"))
    {
      in_run_input_file >> v_bound(1);
    }
    else if (!param_name.compare("w_bound"))
    {
      in_run_input_file >> v_bound(2);
    }
    else if (!param_name.compare("p_bound"))
    {
      in_run_input_file >> p_bound;
    }
    else if (!param_name.compare("wave_speed_x"))
    {
      in_run_input_file >> wave_speed(0);
    }
    else if (!param_name.compare("wave_speed_y"))
    {
      in_run_input_file >> wave_speed(1);
    }
    else if (!param_name.compare("wave_speed_z"))
    {
      in_run_input_file >> wave_speed(2);
    }
    else if (!param_name.compare("diff_coeff"))
    {
      in_run_input_file >> diff_coeff;
    }
    else if (!param_name.compare("lambda"))
    {
      in_run_input_file >> lambda;
    }
    else if (!param_name.compare("mesh_file"))
    {
      in_run_input_file >> mesh_file;
    }
    else if (!param_name.compare("motion_flag"))
    {
      in_run_input_file >> motion;
    }
    else if (!param_name.compare("GCL_flag"))
    {
      in_run_input_file >> GCL;
    }
    else if (!param_name.compare("moving_boundaries"))
    {
      in_run_input_file >> n_moving_bnds;
      motion_type.setup(n_moving_bnds);
      bound_vel_simple.setup(n_moving_bnds);
      boundary_flags.setup(n_moving_bnds);
      for (int i=0; i<n_moving_bnds; i++) {
        in_run_input_file.getline(buf,BUFSIZ);
        in_run_input_file >> boundary_flags(i) >> motion_type(i);
        bound_vel_simple(i).setup(9);
        for (int j=0; j<9; j++) {
          in_run_input_file >> bound_vel_simple(i)(j);
          //cout << bound_vel_simple(i)(j) << " ";
        }
      }
    }
    else if (!param_name.compare("n_deform_iters"))
    {
      in_run_input_file >> n_deform_iters;
    }
    else if (!param_name.compare("simple_bound_velocity"))
    {
//      bound_vel_simple.setup(3);
//      for (int i=0; i<3; i++)
//        in_run_input_file >> bound_vel_simple(i);
    }
    else if (!param_name.compare("mesh_output_freq"))
    {
      in_run_input_file >> mesh_output_freq;
    }
    else if (!param_name.compare("mesh_output_format"))
    {
      in_run_input_file >> mesh_output_format;
    }
    else if (!param_name.compare("upts_type_tri"))
    {
      in_run_input_file >> upts_type_tri;
    }
    else if (!param_name.compare("fpts_type_tri"))
    {
      in_run_input_file >> fpts_type_tri;
    }
    else if (!param_name.compare("vcjh_scheme_tri"))
    {
      in_run_input_file >> vcjh_scheme_tri;
    }
    else if (!param_name.compare("c_tri"))
    {
      in_run_input_file >> c_tri;
    }
    else if (!param_name.compare("sparse_tri"))
    {
      in_run_input_file >> sparse_tri;
    }
    else if (!param_name.compare("upts_type_quad"))
    {
      in_run_input_file >> upts_type_quad;
    }
    else if (!param_name.compare("vcjh_scheme_quad"))
    {
      in_run_input_file >> vcjh_scheme_quad;
    }
    else if (!param_name.compare("eta_quad"))
    {
      in_run_input_file >> eta_quad;
    }
    else if (!param_name.compare("sparse_quad"))
    {
      in_run_input_file >> sparse_quad;
    }
    else if (!param_name.compare("upts_type_hexa"))
    {
      in_run_input_file >> upts_type_hexa;
    }
    else if (!param_name.compare("vcjh_scheme_hexa"))
    {
      in_run_input_file >> vcjh_scheme_hexa;
    }
    else if (!param_name.compare("eta_hexa"))
    {
      in_run_input_file >> eta_hexa;
    }
    else if (!param_name.compare("sparse_hexa"))
    {
      in_run_input_file >> sparse_hexa;
    }
    else if (!param_name.compare("upts_type_tet"))
    {
      in_run_input_file >> upts_type_tet;
    }
    else if (!param_name.compare("fpts_type_tet"))
    {
      in_run_input_file >> fpts_type_tet;
    }
    else if (!param_name.compare("vcjh_scheme_tet"))
    {
      in_run_input_file >> vcjh_scheme_tet;
    }
    else if (!param_name.compare("c_tet"))
    {
      in_run_input_file >> c_tet;
    }
    else if (!param_name.compare("eta_tet"))
    {
      in_run_input_file >> eta_tet;
    }
    else if (!param_name.compare("sparse_tet"))
    {
      in_run_input_file >> sparse_tet;
    }
    else if (!param_name.compare("upts_type_pri_tri"))
    {
      in_run_input_file >> upts_type_pri_tri;
    }
    else if (!param_name.compare("upts_type_pri_1d"))
    {
      in_run_input_file >> upts_type_pri_1d;
    }
    else if (!param_name.compare("vcjh_scheme_pri_1d"))
    {
      in_run_input_file >> vcjh_scheme_pri_1d;
    }
    else if (!param_name.compare("eta_pri"))
    {
      in_run_input_file >> eta_pri;
    }
    else if (!param_name.compare("sparse_pri"))
    {
      in_run_input_file >> sparse_pri;
    }
    else if (!param_name.compare("dx_cyclic"))
    {
      in_run_input_file >> dx_cyclic;
    }
    else if (!param_name.compare("dy_cyclic"))
    {
      in_run_input_file >> dy_cyclic;
    }
    else if (!param_name.compare("dz_cyclic"))
    {
      in_run_input_file >> dz_cyclic;
    }
    else if (!param_name.compare("p_res"))
    {
      in_run_input_file >> p_res;
    }
    else if (!param_name.compare("write_type"))
    {
      in_run_input_file >> write_type;
    }
    else if (!param_name.compare("tau"))
    {
      in_run_input_file >> tau;
    }
    else if (!param_name.compare("pen_fact"))
    {
      in_run_input_file >> pen_fact;
    }
    else if (!param_name.compare("gamma"))
    {
      in_run_input_file >> gamma;
    }
    else if (!param_name.compare("prandtl"))
    {
      in_run_input_file >> prandtl;
    }
    else if (!param_name.compare("S_gas"))
    {
      in_run_input_file >> S_gas;
    }
    else if (!param_name.compare("T_gas"))
    {
      in_run_input_file >> T_gas;
    }
    else if (!param_name.compare("R_gas"))
    {
      in_run_input_file >> R_gas;
    }
    else if (!param_name.compare("mu_gas"))
    {
      in_run_input_file >> mu_gas;
    }
    else if (!param_name.compare("Mach_free_stream"))
    {
      in_run_input_file >> Mach_free_stream;
    }
    else if (!param_name.compare("nx_free_stream"))
    {
      in_run_input_file >> nx_free_stream;
    }
    else if (!param_name.compare("ny_free_stream"))
    {
      in_run_input_file >> ny_free_stream;
    }
    else if (!param_name.compare("nz_free_stream"))
    {
      in_run_input_file >> nz_free_stream;
    }
    else if (!param_name.compare("Re_free_stream"))
    {
      in_run_input_file >> Re_free_stream;
    }
    else if (!param_name.compare("L_free_stream"))
    {
      in_run_input_file >> L_free_stream;
    }
    else if (!param_name.compare("T_free_stream"))
    {
      in_run_input_file >> T_free_stream;
    }
    else if (!param_name.compare("fix_vis"))
    {
      in_run_input_file >> fix_vis;
    }
    else if (!param_name.compare("Mach_wall"))
    {
      in_run_input_file >> Mach_wall;
    }
    else if (!param_name.compare("nx_wall"))
    {
      in_run_input_file >> nx_wall;
    }
    else if (!param_name.compare("ny_wall"))
    {
      in_run_input_file >> ny_wall;
    }
    else if (!param_name.compare("nz_wall"))
    {
      in_run_input_file >> nz_wall;
    }
    else if (!param_name.compare("T_wall"))
    {
      in_run_input_file >> T_wall;
    }
    else if (!param_name.compare("Mach_c_ic"))
    {
      in_run_input_file >> Mach_c_ic;
    }
    else if (!param_name.compare("nx_c_ic"))
    {
      in_run_input_file >> nx_c_ic;
    }
    else if (!param_name.compare("ny_c_ic"))
    {
      in_run_input_file >> ny_c_ic;
    }
    else if (!param_name.compare("nz_c_ic"))
    {
      in_run_input_file >> nz_c_ic;
    }
    else if (!param_name.compare("Re_c_ic"))
    {
      in_run_input_file >> Re_c_ic;
    }
    else if (!param_name.compare("T_c_ic"))
    {
      in_run_input_file >> T_c_ic;
    }
    else if (!param_name.compare("body_forcing"))
    {
      in_run_input_file >> forcing;
    }
    else if (!param_name.compare("x_coeffs"))
    {
      x_coeffs.setup(13);
      for (int i=0;i<13;i++)
        in_run_input_file >> x_coeffs(i) ;
    }
    else if (!param_name.compare("y_coeffs"))
    {
      y_coeffs.setup(13);
      for (int i=0;i<13;i++)
        in_run_input_file >> y_coeffs(i) ;
    }
    else if (!param_name.compare("z_coeffs"))
    {
      z_coeffs.setup(13);
      for (int i=0;i<13;i++)
        in_run_input_file >> z_coeffs(i) ;
    }
    else if (!param_name.compare("perturb_ic"))
    {
      in_run_input_file >> perturb_ic;
    }
    else if (!param_name.compare("ArtifOn"))
    {
      in_run_input_file >> ArtifOn;
    }
    else if (!param_name.compare("artif_only"))
    {
      in_run_input_file >> artif_only;
    }
    else if (!param_name.compare("artif_type"))
    {
      in_run_input_file >> artif_type;
    }
    else if (!param_name.compare("epsilon0"))
    {
      in_run_input_file >> epsilon0;
    }
    else if (!param_name.compare("s0"))
    {
      in_run_input_file >> s0;
    }
    else if (!param_name.compare("kappa"))
    {
      in_run_input_file >> kappa;
    }
    else if (!param_name.compare("shock_vortex_restart"))
    {
      in_run_input_file >> shock_vortex_restart;
    }
    else if (!param_name.compare("filter_frequency"))
    {
      in_run_input_file >> filter_frequency;
    }
    else if (!param_name.compare("filter_width"))
    {
      in_run_input_file >> filter_width;
    }
    else if (!param_name.compare("filter_alpha"))
    {
      in_run_input_file >> filter_alpha;
    }
    else
    {
      cout << "input parameter =" << param_name << endl;
      FatalError("input parameter not recognized");
    }
    
    // Read end of line, if NOT a comment line or blank line
    if (param_name.compare(0,2,"//") && !blank)
      in_run_input_file.getline(buf,BUFSIZ);
  }

  // -------------------------------------------
  // SETUP INITIAL CONDITIONS IF NOT SPECIFIED
  // -------------------------------------------

  if (Mach_c_ic == INFINITY) {
    Mach_c_ic = Mach_free_stream;
  }
  if (nx_c_ic == INFINITY) {
    nx_c_ic = nx_free_stream;
  }
  if (ny_c_ic == INFINITY) {
    ny_c_ic = ny_free_stream;
  }
  if (nz_c_ic == INFINITY) {
    nz_c_ic = nz_free_stream;
  }
  if (Re_c_ic == INFINITY) {
    Re_c_ic = Re_free_stream;
  }
  if (T_c_ic == INFINITY) {
    T_c_ic = T_free_stream;
  }
  if (u_c_ic == INFINITY) {
    u_c_ic = v_bound(0);
  }
  if (v_c_ic == INFINITY) {
    v_c_ic = v_bound(1);
  }
  if (w_c_ic == INFINITY) {
    w_c_ic = v_bound(2);
  }
  if (p_c_ic == INFINITY) {
    p_c_ic = p_bound;
  }
  if (rho_c_ic == INFINITY) {
    rho_c_ic = rho_bound;
  }

  // --------------------
  // ERROR CHECKING
  // --------------------
  
  if (monitor_res_freq == 0) monitor_res_freq = 100000000;
  if (monitor_cp_freq == 0) monitor_cp_freq = 100000000;
  if (monitor_integrals_freq == 0) monitor_integrals_freq = 100000000;
  
  if (!mesh_file.compare(mesh_file.size()-3,3,"neu"))
    mesh_format=0;
  else if (!mesh_file.compare(mesh_file.size()-3,3,"msh"))
    mesh_format=1;
  else
    FatalError("Mesh format not recognized");
  
  if (equation==0)
  {
    if (riemann_solve_type==1)
      FatalError("Lax-Friedrich flux not supported with NS/RANS equation");
    if (ic_form==2 || ic_form==3 || ic_form==4)
      FatalError("Initial condition not supported with NS/RANS equation");
  }
  else if (equation==1)
  {
    if (riemann_solve_type==0)
      FatalError("Rusanov flux not supported with Advection-Diffusion equation");
    if (ic_form==0 || ic_form==1)
      FatalError("Initial condition not supported with Advection-Diffusion equation");
  }

  if (turb_model>0)
  {
    if (riemann_solve_type==2)
      FatalError("Roe flux not supported with RANS equation");
  }
  
  
  if (rank==0)
    cout << endl << "---------------------- Non-dimensionalization ---------------------" << endl;
  
  
  if(viscous) {
    
    // If we have chosen an isentropic vortex case as the initial condition
    
    if(ic_form == 0 || artif_only || ic_form == 8)   {
      
      fix_vis  = 1.;
      R_ref     = 1.;
      c_sth     = 1.;
      rt_inf    = 1.;
      mu_inf    = 0.1;
    
    } else { // Any other type of initial condition
      
      // Dimensional reference quantities for temperature and length
      
      T_ref = T_free_stream;
      L_ref = L_free_stream;
      
      // Compute the freestream velocity from the Mach number and direction
      
      uvw_ref = Mach_free_stream*sqrt(gamma*R_gas*T_free_stream);

      u_free_stream   = uvw_ref*nx_free_stream;
      v_free_stream   = uvw_ref*ny_free_stream;
      w_free_stream   = uvw_ref*nz_free_stream;
      
      // Set either a fixed value for the viscosity or a value from Sutherland's law
      
      if(fix_vis)
      {
        mu_free_stream = mu_gas;
      }
      else
      {
        mu_free_stream = mu_gas*pow(T_free_stream/T_gas, 1.5)*( (T_gas + S_gas)/(T_free_stream + S_gas));
      }
      
      // Compute the corresponding density from the definition of the Reynolds number
      // Re and the Re length are specified in the input file.
      
      rho_free_stream   = (mu_free_stream*Re_free_stream)/(uvw_ref*L_free_stream);
      
      // Compute the freestream pressure from the gas law
      
      p_free_stream = rho_free_stream*R_gas*T_free_stream;
      
      // Choose the following consistent reference quantities for other variables
      
      rho_ref   = rho_free_stream;
      p_ref     = rho_ref*uvw_ref*uvw_ref;
      mu_ref    = rho_ref*uvw_ref*L_ref;
      time_ref  = L_ref/uvw_ref;
      R_ref     = (R_gas*T_ref)/(uvw_ref*uvw_ref);
      
      // ?
      c_sth     = S_gas/T_gas;
      
      mu_inf    = mu_gas/mu_ref;
      rt_inf    = T_gas*R_gas/(uvw_ref*uvw_ref);
      
      // Set up the dimensionless conditions @ free-stream boundaries
      
      rho_bound = 1.; // Note that we have chosen our non-dim. such that rho_ref = rho_free_stream
      v_bound(0) = u_free_stream/uvw_ref;
      v_bound(1) = v_free_stream/uvw_ref;
      v_bound(2) = w_free_stream/uvw_ref;
      p_bound = p_free_stream/p_ref;
      T_total_bound = (T_free_stream/T_ref)*(1.0 + 0.5*(gamma-1.0)*Mach_free_stream*Mach_free_stream);
      p_total_bound = p_bound*pow(1.0 + 0.5*(gamma-1.0)*Mach_free_stream*Mach_free_stream, gamma/(gamma-1.0));
      
      // Set up the dimensionless conditions @ moving boundaries

      uvw_wall  = Mach_wall*sqrt(gamma*R_gas*T_wall);
      v_wall(0) = (uvw_wall*nx_wall)/uvw_ref;
      v_wall(1) = (uvw_wall*ny_wall)/uvw_ref;
      v_wall(2) = (uvw_wall*nz_wall)/uvw_ref;
      T_wall    = T_wall/T_ref;
      
      // Set up the dimensionless initial conditions (repeat process above for freestream)

      uvw_c_ic  = Mach_c_ic*sqrt(gamma*R_gas*T_c_ic);
      u_c_ic   = (uvw_c_ic*nx_c_ic)/uvw_ref;
      v_c_ic   = (uvw_c_ic*ny_c_ic)/uvw_ref;
      w_c_ic   = (uvw_c_ic*nz_c_ic)/uvw_ref;
      if(fix_vis) {
        mu_c_ic = mu_gas;
      } else {
        mu_c_ic = mu_gas*pow(T_c_ic/T_gas, 1.5)*( (T_gas + S_gas)/(T_c_ic + S_gas));
      }
      rho_c_ic = (mu_c_ic*Re_c_ic)/(uvw_c_ic*L_ref);
      p_c_ic   = rho_c_ic*R_gas*T_c_ic;
      mu_c_ic  = mu_c_ic/mu_ref;
      rho_c_ic = rho_c_ic/rho_ref;
      p_c_ic   = p_c_ic/p_ref;
      T_c_ic   = T_c_ic/T_ref;
      
      // SA turblence model parameters
      prandtl_t = 0.9;
      if (turb_model == 1)
      {
          c_v1 = 7.1;
          c_v2 = 0.7;
          c_v3 = 0.9;
          c_b1 = 0.1355;
          c_b2 = 0.622;
          c_w2 = 0.3;
          c_w3 = 2.0;
          omega = 2.0/3.0;
          Kappa = 0.41;
          mu_tilde_c_ic = 5.0*mu_c_ic;
          mu_tilde_inf = 5.0*mu_inf;
      }

      // Master node outputs information about the I.C.s to the console
      if (rank==0)
      {
        cout << "uvw_ref: " << uvw_ref << endl;
        cout << "rho_free_stream: " << rho_free_stream << endl;
        cout << "rho_c_ic=" << rho_c_ic << endl;
        cout << "u_c_ic=" << u_c_ic << endl;
        cout << "v_c_ic=" << v_c_ic << endl;
        cout << "w_c_ic=" << w_c_ic << endl;
        cout << "mu_c_ic=" << mu_c_ic << endl;
      }
    }
  }
}
