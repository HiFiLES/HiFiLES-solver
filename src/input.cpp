/*!
 * \file input.cpp
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
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>

#include "../include/input.h"
#include "../include/array.h"
#include "../include/funcs.h"
#include "../include/global.h"

using namespace std;

// #### constructors ####

// default constructor

input::input()
{
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

	double a_k = eval_gamma(2.*order+1)/( pow(2.,order)*pow(eval_gamma(order+1),2) );
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
	char section_TXT[100], param_TXT[100];
	string dummy, param_name;

  // First loop over the input file and print content to output
    if (rank==0)
    {
	    while(!in_run_input_file.eof())
      {
	    	in_run_input_file.getline(buf,BUFSIZ);
        cout << buf << endl;
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
		sscanf(buf,"%s",&section_TXT);
	  param_name.assign(section_TXT,0,99);

		if (!param_name.compare(0,5,"-----"))
    {
      // Section header, ignore next two lines
		  in_run_input_file.getline(buf,BUFSIZ);
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
		else if (!param_name.compare("run_type"))
		{
			in_run_input_file >> run_type;
		}
		else if (!param_name.compare("n_plot_quantities"))
		{
			in_run_input_file >> n_plot_quantities;
      plot_quantities.setup(n_plot_quantities);
      for (int i=0;i<n_plot_quantities;i++)
        in_run_input_file >> plot_quantities(i) ;
		}
		else if (!param_name.compare("inters_cub_order"))
		{
			in_run_input_file >> inters_cub_order;
		}
		else if (!param_name.compare("volume_cub_order"))
		{
			in_run_input_file >> volume_cub_order;
		}
		else if (!param_name.compare("dt"))
		{
			in_run_input_file >> dt;
    }
		else if (!param_name.compare("n_steps"))
		{
			in_run_input_file >> n_steps;
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
		else if (!param_name.compare("adv_type"))
		{
			in_run_input_file >> adv_type;
    }
		else if (!param_name.compare("const_src_term"))
		{
			in_run_input_file >> const_src_term;
    }
		else if (!param_name.compare("monitor_res_freq"))
		{
			in_run_input_file >> monitor_res_freq;
    }
		else if (!param_name.compare("monitor_force_freq"))
		{
			in_run_input_file >> monitor_force_freq;
    }
		else if (!param_name.compare("diagnostics_freq"))
		{
			in_run_input_file >> diagnostics_freq;
    }
		else if (!param_name.compare("n_diagnostics"))
		{
			in_run_input_file >> n_diagnostics;
      diagnostics.setup(n_diagnostics);
      for (int i=0;i<n_diagnostics;i++)
        in_run_input_file >> diagnostics(i) ;
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
    else if (!param_name.compare("rans_model"))
    {
        in_run_input_file >> rans_model;
    }
    else
    {
      cout << "input parameter =" << param_name << endl;
      FatalError("input parameter not recognized");
    }

	  // Read end of line
	  in_run_input_file.getline(buf,BUFSIZ);
  }

  // --------------------
  // ERROR CHECKING
  // --------------------

  if (monitor_res_freq == 0) monitor_res_freq = 1000;
  if (monitor_force_freq == 0) monitor_force_freq = 1000;
  if (diagnostics_freq == 0) diagnostics_freq = 1000;

  if (!mesh_file.compare(mesh_file.size()-3,3,"neu"))
    mesh_format=0;
  else if (!mesh_file.compare(mesh_file.size()-3,3,"msh"))
    mesh_format=1;
  else
    FatalError("Mesh format not recognized");

  if (equation==0 || equation==2)
  {
    const_src_term = 0.0;
    if (riemann_solve_type==1)
      FatalError("Lax-Friedrich flux not supported with NS equation");
    if (ic_form==2 || ic_form==3 || ic_form==4)
      FatalError("Initial condition not supported with NS equation");
  }
  else if (equation==1)
  {
    if (riemann_solve_type==0)
      FatalError("Rusanov flux not supported with Advection-Diffusion equation");
    if (ic_form==0 || ic_form==1)
      FatalError("Initial condition not supported with Advection-Diffusion equation");
    if (run_type==1)
      FatalError("Run type not supported with Advection-Diffusion equation");
  }

  if(viscous)
  {
    if(ic_form == 0)  {

      fix_vis  = 1.;
      R_ref     = 1.;
      c_sth     = 1.;
      rt_inf    = 1.;
	    mu_inf    = 0.1;
    }
    else {

      //Dimensional reference
      T_ref = T_free_stream;
      L_ref = L_free_stream;

      uvw_ref = Mach_free_stream*sqrt(gamma*R_gas*T_free_stream);
      cout << "uvw_ref: " << uvw_ref << endl;
      u_free_stream   = uvw_ref*nx_free_stream;
      v_free_stream   = uvw_ref*ny_free_stream;
      w_free_stream   = uvw_ref*nz_free_stream;

      if(fix_vis)
      {
        mu_free_stream = mu_gas;
      }
      else
      {
        mu_free_stream = mu_gas*pow(T_free_stream/T_gas, 1.5)*( (T_gas + S_gas)/(T_free_stream + S_gas));
      }

      rho_free_stream   = (mu_free_stream*Re_free_stream)/(uvw_ref*L_free_stream);
      cout << "rho_free_stream: " << rho_free_stream << endl;
      p_free_stream = rho_free_stream*R_gas*T_free_stream;

      rho_ref   = rho_free_stream;
      p_ref     = rho_ref*uvw_ref*uvw_ref;
      mu_ref    = rho_ref*uvw_ref*L_ref;
      time_ref  = L_ref/uvw_ref;
      R_ref     = (R_gas*T_ref)/(uvw_ref*uvw_ref);

      c_sth     = S_gas/T_gas;
	    mu_inf    = mu_gas/mu_ref;
	    rt_inf    = T_gas*R_gas/(uvw_ref*uvw_ref);

      //Dimensionless free-stream boundary
      rho_bound = 1.;
      v_bound(0) = u_free_stream/uvw_ref;
      v_bound(1) = v_free_stream/uvw_ref;
      v_bound(2) = w_free_stream/uvw_ref;
      p_bound = p_free_stream/p_ref;
      T_total_bound = (T_free_stream/T_ref)*(1.0 + 0.5*(gamma-1.0)*Mach_free_stream*Mach_free_stream);
      p_total_bound = p_bound*pow(1.0 + 0.5*(gamma-1.0)*Mach_free_stream*Mach_free_stream, gamma/(gamma-1.0));

      uvw_wall  = Mach_wall*sqrt(gamma*R_gas*T_wall);
      v_wall(0) = (uvw_wall*nx_wall)/uvw_ref;
      v_wall(1) = (uvw_wall*ny_wall)/uvw_ref;
      v_wall(2) = (uvw_wall*nz_wall)/uvw_ref;
      T_wall    = T_wall/T_ref;

      uvw_c_ic  = Mach_c_ic*sqrt(gamma*R_gas*T_c_ic);
      u_c_ic   = (uvw_c_ic*nx_c_ic)/uvw_ref;
      v_c_ic   = (uvw_c_ic*ny_c_ic)/uvw_ref;
      w_c_ic   = (uvw_c_ic*nz_c_ic)/uvw_ref;

      if(fix_vis)
      {
        mu_c_ic = mu_gas;
      }
      else
      {
        mu_c_ic = mu_gas*pow(T_c_ic/T_gas, 1.5)*( (T_gas + S_gas)/(T_c_ic + S_gas));
      }

      rho_c_ic   = (mu_c_ic*Re_c_ic)/(uvw_c_ic*L_ref);
      p_c_ic = rho_c_ic*R_gas*T_c_ic;

      mu_c_ic = mu_c_ic/mu_ref;

      rho_c_ic = rho_c_ic/rho_ref;
      p_c_ic = p_c_ic/p_ref;
      T_c_ic = T_c_ic/T_ref;

      // SA turblence model parameters
      prandtl_t = 0.9;
      if (rans_model == 1)
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

				// HACK: set to zero if using RANS as a wall model
				if(wall_model != 3) {
          mu_tilde_c_ic = 5.0*mu_c_ic;
          mu_tilde_inf = 5.0*mu_inf;
				}
				else {
          mu_tilde_c_ic = 0.0;
          mu_tilde_inf = 0.0;
				}
      }

      if (rank==0)
      {
        cout << "rho_c_ic=" << rho_c_ic << endl;
        cout << "u_c_ic=" << u_c_ic << endl;
        cout << "v_c_ic=" << v_c_ic << endl;
        cout << "w_c_ic=" << w_c_ic << endl;
        cout << "mu_c_ic=" << mu_c_ic << endl;
	      if (rans_model == 1)
	      {
					cout << "mu_tilde_c_ic=" << mu_tilde_c_ic << endl;
				}
      }
    }
  }
}


void input::reset(int c_ind, int p_ind, int grid_ind, int vis_ind, int tau_ind, int dev_ind, int dim_ind)
{
	int n_freq;
	int t_flag = 1;  //0 period based, 1 runs indefinitely
	int d_flag = dim_ind;

  //Set scheme
  vcjh_scheme_tet = c_ind;
  vcjh_scheme_tri = c_ind;

  //Set order
  order = p_ind;

  //set device
  device_num = dev_ind;

  //Set Tau
  if(tau_ind == 1)
  {
    tau = 0.001;
  }
  else if(tau_ind == 2)
  {
    tau = 0.01;
  }
  else if(tau_ind == 3)
  {
    tau = 0.1;
  }
  else if(tau_ind == 4)
  {
    tau = 1.0;
  }
  else
  {
    cout << "Invalid value of tau" << endl;
    exit(1);
  }


  //Grid
  if(grid_ind == 1)
  {
		if(d_flag == 2)
		{
			mesh_file = "tri_2_1.neu";
		}
		if(d_flag == 3)
		{
			mesh_file = "sd7003_711K.neu";
		}
  }
  else if(grid_ind == 2)
  {
		if(d_flag == 2)
		{
    	mesh_file = "tri_4_2.neu";
		}
		if(d_flag == 3)
		{
    	mesh_file = "sd7003_390K.neu";
    }
  }
  else if(grid_ind == 3)
  {
		if(d_flag == 2)
		{
    	mesh_file = "tri_6_3.neu";
		}
		if(d_flag == 3)
		{
    	mesh_file = "sd7003_450K.neu";
    }
  }
  else if(grid_ind == 4)
  {
		if(d_flag == 2)
		{
    	mesh_file = "tri_8_4.neu";
		}
		if(d_flag == 3)
		{
    	mesh_file = "sd7003_430K.neu";
    }
  }
  else
  {
    cout << "Invalid grid" << endl;
    exit(1);
  }

  mesh_format = 0;
  //mesh_format = 1;


  // Time
  //dt = 1.0e-5;
  dt = 1.25e-5;
  //dt = 1.0e-4;

  //dt = dt*(1.0/(((double) (order*order))/4.0));

	if(t_flag==0)
	{
  	double tperiod = 1.0;
  	n_steps = (int) ceil(tperiod/dt);
  	dt = tperiod/n_steps;

		plot_freq = n_steps;
		monitor_res_freq = n_steps;
	}
	if(t_flag==1)
	{
  	n_steps = 50000000;
		n_freq = 1000;

		plot_freq = 100000;
		monitor_res_freq = n_freq;
  	monitor_force_freq = n_freq;
	}

  cout << endl;
  cout <<"dt " << dt << endl;
  cout <<"steps " << n_steps << endl;
  cout << endl;

}

