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

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

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

void input::read_input_file(string fileName, int rank)
{
  fileReader opts(fileName);

  v_bound.setup(3);
  wave_speed.setup(3);
  v_wall.setup(3);

  /*
   * HiFiLES Developers - Please keep this organized!  There are
   * many parameters, so organization is the key to clarity.
   */

  /* ---- Basic Simulation Parameters ---- */

  opts.getScalarValue("equation",equation);
  opts.getScalarValue("order",order);
  opts.getScalarValue("viscous",viscous,0);
  opts.getScalarValue("mesh_file",mesh_file);
  opts.getScalarValue("ic_form",ic_form,1);
  opts.getScalarValue("test_case",test_case,0);
  opts.getScalarValue("n_steps",n_steps);
  opts.getScalarValue("restart_flag",restart_flag,0);
  if (restart_flag == 1) {
    opts.getScalarValue("restart_iter",restart_iter);
    opts.getScalarValue("n_restart_files",n_restart_files);
  }

  /* ---- Visualization / Monitoring / Output Parameters ---- */

  opts.getScalarValue("plot_freq",plot_freq,500);
  opts.getScalarValue("data_file_name",data_file_name,string("Mesh"));
  opts.getScalarValue("restart_dump_freq",restart_dump_freq,0);
  opts.getScalarValue("monitor_res_freq",monitor_res_freq,100);
  opts.getScalarValue("monitor_cp_freq",monitor_cp_freq,0);
  opts.getScalarValue("monitor_integrals_freq",monitor_integrals_freq,0);
  opts.getScalarValue("res_norm_type",res_norm_type,2);
  opts.getScalarValue("error_norm_type",error_norm_type,2);
  opts.getScalarValue("res_norm_field",res_norm_field,0);
  opts.getScalarValue("p_res",p_res,3);
  opts.getScalarValue("write_type",write_type,1);
  opts.getScalarValue("inters_cub_order",inters_cub_order,3);
  opts.getScalarValue("volume_cub_order", volume_cub_order,3);

  opts.getVectorValueOptional("integral_quantities",integral_quantities);
  opts.getVectorValueOptional("diagnostic_fields",diagnostic_fields);
  opts.getVectorValueOptional("average_fields",average_fields);
  n_integral_quantities = integral_quantities.get_dim(0);
  n_diagnostic_fields = diagnostic_fields.get_dim(0);
  n_average_fields = average_fields.get_dim(0);

  for (int i=0; i<n_integral_quantities; i++) {
    std::transform(integral_quantities(i).begin(), integral_quantities(i).end(),
                   integral_quantities(i).begin(), ::tolower);
  }
  for (int i=0; i<n_diagnostic_fields; i++) {
    std::transform(diagnostic_fields(i).begin(), diagnostic_fields(i).end(),
                   diagnostic_fields(i).begin(), ::tolower);
  }
  for (int i=0; i<n_average_fields; i++) {
    std::transform(average_fields(i).begin(), average_fields(i).end(),
                   average_fields(i).begin(), ::tolower);
  }

  /* ---- Basic Solver Parameters ---- */

  opts.getScalarValue("riemann_solve_type",riemann_solve_type);
  opts.getScalarValue("vis_riemann_solve_type",vis_riemann_solve_type);
  opts.getScalarValue("adv_type",adv_type);
  opts.getScalarValue("dt_type",dt_type);
  if (dt_type == 2 && rank == 0) {
    cout << "!!!!!!" << endl;
    cout << "  Note: Local timestepping is still in an experimental phase,";
    cout << "  especially for viscous simulations." << endl;
    cout << "!!!!!!" << endl;
  }

  if (dt_type == 0) {
    opts.getScalarValue("dt",dt);
  }
  else {
    opts.getScalarValue("CFL",CFL);
  }

  /* ---- Turbulence Modeling Parameters ---- */

  opts.getScalarValue("turb_model",turb_model,0);
  opts.getScalarValue("LES",LES,0);
  if (LES) {
    opts.getScalarValue("filter_type",filter_type);
    opts.getScalarValue("filter_ratio",filter_ratio);
    opts.getScalarValue("SGS_model",SGS_model);
    opts.getScalarValue("wall_model",wall_model);
    opts.getScalarValue("wall_layer_thickness",wall_layer_t);
  }

  /* ---- Mesh Motion Parameters ---- */

  opts.getScalarValue("motion_flag",motion,0);
  if (motion != STATIC_MESH) {
    opts.getScalarValue("GCL_flag",GCL,0);
    opts.getVectorValueOptional("moving_boundaries",motion_type);

    bound_vel_simple.setup(1);
    opts.getVectorValueOptional("simple_bound_velocity",bound_vel_simple(0));
    //opts.getVectorValueOptional("bound_vel_simple",bound_vel_simple);
    //      in_run_input_file >> n_moving_bnds;
    //      motion_type.setup(n_moving_bnds);
    //      bound_vel_simple.setup(n_moving_bnds);
    //      boundary_flags.setup(n_moving_bnds);
    //      for (int i=0; i<n_moving_bnds; i++) {
    //        in_run_input_file.getline(buf,BUFSIZ);
    //        in_run_input_file >> boundary_flags(i) >> motion_type(i);
    //        bound_vel_simple(i).setup(9);
    //        for (int j=0; j<9; j++) {
    //          in_run_input_file >> bound_vel_simple(i)(j);
    //          //cout << bound_vel_simple(i)(j) << " ";
    //        }
    //      }
    opts.getScalarValue("n_deform_iters",n_deform_iters);
    opts.getScalarValue("mesh_output_freq",mesh_output_freq,0);
    opts.getScalarValue("mesh_output_format",mesh_output_format,1);
    opts.getScalarValue("restart_mesh_out",restart_mesh_out,0);
  }

  /* ---- Gas Parameters ---- */

  opts.getScalarValue("gamma",gamma,1.4);
  opts.getScalarValue("prandtl",prandtl,.72);
  opts.getScalarValue("S_gas",S_gas,120.);
  opts.getScalarValue("T_gas",T_gas,291.15);
  opts.getScalarValue("R_gas",R_gas,286.9);
  opts.getScalarValue("mu_gas",mu_gas,1.827E-5);

  /* ---- Boundary Conditions ---- */

  opts.getScalarValue("dx_cyclic",dx_cyclic,(double)INFINITY);
  opts.getScalarValue("dy_cyclic",dy_cyclic,(double)INFINITY);
  opts.getScalarValue("dz_cyclic",dz_cyclic,(double)INFINITY);

  opts.getScalarValue("Mach_free_stream",Mach_free_stream);
  opts.getScalarValue("nx_free_stream",nx_free_stream);
  opts.getScalarValue("ny_free_stream",ny_free_stream);
  opts.getScalarValue("nz_free_stream",nz_free_stream);
  opts.getScalarValue("Re_free_stream",Re_free_stream);
  opts.getScalarValue("L_free_stream",L_free_stream);
  opts.getScalarValue("T_free_stream",T_free_stream);

  opts.getScalarValue("rho_bound",rho_bound);
  opts.getScalarValue("u_bound",v_bound(0));
  opts.getScalarValue("v_bound",v_bound(1));
  opts.getScalarValue("w_bound",v_bound(2));
  opts.getScalarValue("p_bound",p_bound);

  opts.getScalarValue("Mach_wall",Mach_wall,0.);
  opts.getScalarValue("nx_wall",nx_wall,0.);
  opts.getScalarValue("ny_wall",ny_wall,0.);
  opts.getScalarValue("nz_wall",nz_wall,0.);
  opts.getScalarValue("T_wall",T_wall,300.);

  opts.getScalarValue("fix_vis",fix_vis,1.);
  opts.getScalarValue("tau",tau,0.);
  opts.getScalarValue("pen_fact",pen_fact,0.5);

  /* ---- Initial Conditions (use BC's by default) ---- */

  opts.getScalarValue("Mach_c_ic",Mach_c_ic,Mach_free_stream);
  opts.getScalarValue("nx_c_ic",nx_c_ic,nx_free_stream);
  opts.getScalarValue("ny_c_ic",ny_c_ic,ny_free_stream);
  opts.getScalarValue("nz_c_ic",nz_c_ic,nz_free_stream);
  opts.getScalarValue("Re_c_ic",Re_c_ic,Re_free_stream);
  opts.getScalarValue("T_c_ic",T_c_ic,T_free_stream);

  opts.getScalarValue("rho_c_ic",rho_c_ic,rho_bound);
  opts.getScalarValue("u_c_ic",u_c_ic,v_bound(0));
  opts.getScalarValue("v_c_ic",v_c_ic,v_bound(1));
  opts.getScalarValue("w_c_ic",w_c_ic,v_bound(2));
  opts.getScalarValue("p_c_ic",p_c_ic,p_bound);

  /* ---- Shock Capturing / Filtering ---- */

  opts.getScalarValue("ArtifOn",ArtifOn,0);
  if (ArtifOn) {
    opts.getScalarValue("artif_only",artif_only);
    opts.getScalarValue("artif_type",artif_type);
    opts.getScalarValue("epsilon0",epsilon0);
    opts.getScalarValue("s0",s0);
    opts.getScalarValue("kappa",kappa);
  }
  opts.getScalarValue("shock_vortex_restart",shock_vortex_restart,0);

  /* ---- FR Element Solution Point / Correction Function Parameters ---- */
  // Tris
  opts.getScalarValue("upts_type_tri",upts_type_tri,0);
  opts.getScalarValue(" fpts_type_tri",fpts_type_tri,0);
  opts.getScalarValue("vcjh_scheme_tri",vcjh_scheme_tri,0);
  opts.getScalarValue("c_tri",c_tri,0.);
  opts.getScalarValue("sparse_tri",sparse_tri,0);
  // Quads
  opts.getScalarValue("upts_type_quad",upts_type_quad,0);
  opts.getScalarValue("vcjh_scheme_quad",vcjh_scheme_quad,0);
  opts.getScalarValue("eta_quad",eta_quad,0.);
  opts.getScalarValue("sparse_quad",sparse_quad,0);
  // Hexs
  opts.getScalarValue("upts_type_hexa",upts_type_hexa,0);
  opts.getScalarValue("vcjh_scheme_hexa",vcjh_scheme_hexa,0);
  opts.getScalarValue("eta_hexa",eta_hexa,0.);
  opts.getScalarValue("sparse_hexa",sparse_hexa,0);
  // Tets
  opts.getScalarValue("upts_type_tet",upts_type_tet,0);
  opts.getScalarValue("fpts_type_tet",fpts_type_tet,0);
  opts.getScalarValue("vcjh_scheme_tet",vcjh_scheme_tet,0);
  opts.getScalarValue("c_tet",c_tet,0.);
  opts.getScalarValue("eta_tet",eta_tet,0.);
  opts.getScalarValue("sparse_tet",sparse_tet,0);
  // Prisms
  opts.getScalarValue("upts_type_pri_tri",upts_type_pri_tri,0);
  opts.getScalarValue("upts_type_pri_1d",upts_type_pri_1d,0);
  opts.getScalarValue("vcjh_scheme_pri_1d",vcjh_scheme_pri_1d,0);
  opts.getScalarValue("eta_pri",eta_pri,0.);
  opts.getScalarValue("sparse_pri",sparse_pri);

  /* ---- Advection-Diffusion Parameters ---- */
  if (equation == 1) {
  opts.getScalarValue("wave_speed_x",wave_speed(0));
  opts.getScalarValue("wave_speed_y",wave_speed(1));
  opts.getScalarValue( "wave_speed_z",wave_speed(2));
  opts.getScalarValue("diff_coeff",diff_coeff,0.);
  opts.getScalarValue("lambda",lambda);
  }

  /* ---- Uncategorized / Other ---- */

  opts.getScalarValue("spinup_time",spinup_time,0.);
  opts.getScalarValue("const_src",const_src,0.);
  opts.getScalarValue("body_forcing",forcing,0);
  opts.getScalarValue("perturb_ic",perturb_ic,0);

  // NOTE: the input file line must look like "x_coeffs <# coeffs> x1 x2 x3..."
  opts.getVectorValueOptional("x_coeffs",x_coeffs);
  opts.getVectorValueOptional("y_coeffs",y_coeffs);
  opts.getVectorValueOptional("z_coeffs",z_coeffs);
}

void input::setup(char* fileNameC, int rank)
{
  string fileNameS;
  fileNameS.assign(fileNameC);

  /* ---- Read necessary parameters from the input file ---- */
  read_input_file(fileNameS,rank);

  /* ---- Non-Dimensionalization and other setup ---- */
  setup_params(rank);
}

void input::setup_params(int rank)
{
  // --------------------
  // ERROR CHECKING
  // --------------------
  
  if (monitor_res_freq == 0) monitor_res_freq = INFINITY;
  if (monitor_cp_freq == 0) monitor_cp_freq = INFINITY;
  if (monitor_integrals_freq == 0) monitor_integrals_freq = INFINITY;
  
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

fileReader::fileReader()
{

}

fileReader::fileReader(string fileName)
{
  this->fileName = fileName;
}

fileReader::~fileReader()
{
  if (optFile.is_open()) optFile.close();
}

void fileReader::setFile(string fileName)
{
  this->fileName = fileName;
}

void fileReader::openFile(void)
{
  optFile.open(fileName.c_str(), ifstream::in);
}

void fileReader::closeFile()
{
  optFile.close();
}

template<typename T>
void fileReader::getScalarValue(string optName, T &opt, T defaultVal)
{
  string str, optKey;

  openFile();

  if (!optFile.is_open() || !getline(optFile,str)) {
    optFile.open(fileName.c_str());
    if (!optFile.is_open())
      FatalError("Cannont open input file for reading.");
  }

  // Rewind to the start of the file
  optFile.clear();
  optFile.seekg(0,optFile.beg);

  // Search for the given option string
  while (getline(optFile,str)) {
    // Remove any leading whitespace & see if first word is the input option
    stringstream ss;
    ss.str(str);
    ss >> optKey;
    if (optKey.compare(optName)==0) {
      if (!(ss >> opt)) {
        // This could happen if, for example, trying to assign a string to a double
        cout << "WARNING: Unable to assign value to option " << optName << endl;
        cout << "Using default value of " << defaultVal << " instead." << endl;
        opt = defaultVal;
      }

      closeFile();
      return;
    }
  }

  opt = defaultVal;
  closeFile();
}

template<typename T>
void fileReader::getScalarValue(string optName, T &opt)
{
  string str, optKey;

  openFile();

  if (!optFile.is_open()) {
    optFile.open(fileName.c_str());
    if (!optFile.is_open())
      FatalError("Cannont open input file for reading.");
  }

  // Rewind to the start of the file
  optFile.clear();
  optFile.seekg(0,optFile.beg);

  // Search for the given option string
  while (getline(optFile,str)) {
    // Remove any leading whitespace & see if first word is the input option
    stringstream ss;
    ss.str(str);
    ss >> optKey;

    if (optKey.compare(optName)==0) {
      if (!(ss >> opt)) {
        // This could happen if, for example, trying to assign a string to a double
        cerr << "WARNING: Unable to assign value to option " << optName << endl;
        string errMsg = "Required option not set: " + optName;
        FatalError(errMsg.c_str())
      }

      closeFile();
      return;
    }
  }

  // Option was not found; throw error & exit
  string errMsg = "Required option not found: " + optName;
  FatalError(errMsg.c_str())
}

template<typename T, typename U>
void fileReader::getMap(string optName, map<T,U> &opt) {
  string str, optKey;
  T tmpT;
  U tmpU;
  bool found;

  openFile();

  if (!optFile.is_open()) {
    optFile.open(fileName.c_str());
    if (!optFile.is_open())
      FatalError("Cannont open input file for reading.");
  }

  // Rewind to the start of the file
  optFile.clear();
  optFile.seekg(0,optFile.beg);

  // Search for the given option string
  while (getline(optFile,str)) {
    // Remove any leading whitespace & see if first word is the input option
    stringstream ss;
    ss.str(str);
    ss >> optKey;
    if (optKey.compare(optName)==0) {
      found = true;
      if (!(ss >> tmpT >> tmpU)) {
        // This could happen if, for example, trying to assign a string to a double
        cerr << "WARNING: Unable to assign value to option " << optName << endl;
        string errMsg = "Required option not set: " + optName;
        FatalError(errMsg.c_str())
      }

      opt[tmpT] = tmpU;
      optKey = "";
    }
  }

  if (!found) {
    // Option was not found; throw error & exit
    string errMsg = "Required option not found: " + optName;
    FatalError(errMsg.c_str())
  }

  closeFile();
}

template<typename T>
void fileReader::getVectorValue(string optName, vector<T> &opt)
{
  string str, optKey;

  openFile();

  if (!optFile.is_open()) {
    optFile.open(fileName.c_str());
    if (!optFile.is_open())
      FatalError("Cannont open input file for reading.");
  }

  // Rewind to the start of the file
  optFile.clear();
  optFile.seekg(0,optFile.beg);

  // Search for the given option string
  while (getline(optFile,str)) {
    // Remove any leading whitespace & see if first word is the input option
    stringstream ss;
    ss.str(str);
    ss >> optKey;
    if (optKey.compare(optName)==0) {
      int nVals;
      if (!(ss >> nVals)) {
        // This could happen if, for example, trying to assign a string to a double
        cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
        string errMsg = "Required option not set: " + optName;
        FatalError(errMsg.c_str());
      }

      opt.resize(nVals);
      for (int i=0; i<nVals; i++) {
        if (!ss >> opt[i]) {
          cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
          string errMsg = "Required option not set: " + optName;
          FatalError(errMsg.c_str())
        }
      }

      closeFile();
      return;
    }
  }

  // Option was not found; throw error & exit
  string errMsg = "Required option not found: " + optName;
  FatalError(errMsg.c_str())
}

template<typename T>
void fileReader::getVectorValue(string optName, array<T> &opt)
{
  string str, optKey;

  openFile();

  if (!optFile.is_open()) {
    optFile.open(fileName.c_str());
    if (!optFile.is_open())
      FatalError("Cannont open input file for reading.");
  }

  // Rewind to the start of the file
  optFile.clear();
  optFile.seekg(0,optFile.beg);

  // Search for the given option string
  while (getline(optFile,str)) {
    // Remove any leading whitespace & see if first word is the input option
    stringstream ss;
    ss.str(str);
    ss >> optKey;
    if (optKey.compare(optName)==0) {
      int nVals;
      if (!(ss >> nVals)) {
        // This could happen if, for example, trying to assign a string to a double
        cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
        string errMsg = "Required option not set: " + optName;
        FatalError(errMsg.c_str());
      }

      opt.setup(nVals);
      for (int i=0; i<nVals; i++) {
        if (!(ss >> opt(i))) {
          cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
          string errMsg = "Required option not set: " + optName;
          FatalError(errMsg.c_str());
        }
      }

      closeFile();
      return;
    }
  }

  // Option was not found; throw error & exit
  string errMsg = "Required option not found: " + optName;
  FatalError(errMsg.c_str())
}

template<typename T>
void fileReader::getVectorValueOptional(string optName, array<T> &opt)
{
  string str, optKey;

  openFile();

  if (!optFile.is_open()) {
    optFile.open(fileName.c_str());
    if (!optFile.is_open())
      FatalError("Cannont open input file for reading.");
  }

  // Rewind to the start of the file
  optFile.clear();
  optFile.seekg(0,optFile.beg);

  // Search for the given option string
  while (getline(optFile,str)) {
    // Remove any leading whitespace & see if first word is the input option
    stringstream ss;
    ss.str(str);
    ss >> optKey;
    if (optKey.compare(optName)==0) {
      int nVals;
      if (!(ss >> nVals)) {
        // This could happen if, for example, trying to assign a string to a double
        cerr << "WARNING: Unable to read number of entries for vector option " << optName << endl;
        cerr << "Option not set: " << optName << endl;
        opt.setup(0);
        return;
      }

      opt.setup(nVals);
      for (int i=0; i<nVals; i++) {
        if (!(ss >> opt(i))) {
          cerr << "WARNING: Unable to assign all values to vector option " << optName << endl;
          cerr << "Option not set: " << optName << endl;
          opt.setup(0);
          return;
        }
      }

      closeFile();
      return;
    }
  }

  // Option was not found; setup array to size 0
  opt.setup(0);
}
