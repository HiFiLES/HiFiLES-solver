/*!
 * \file eles.cpp
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
#include <iomanip>
#include <cmath>

#if defined _ACCELERATE_BLAS
#include <Accelerate/Accelerate.h>
#endif

#if defined _MKL_BLAS
#include "mkl.h"
#include "mkl_spblas.h"
#endif

#if defined _STANDARD_BLAS
extern "C"
{
#include "cblas.h"
}
#endif

#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif

#if defined _GPU
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "../include/cuda_kernels.h"
#endif

#include "../include/global.h"
#include "../include/array.h"
#include "../include/flux.h"
#include "../include/source.h"
#include "../include/eles.h"
#include "../include/funcs.h"

using namespace std;

// #### constructors ####

// default constructor

eles::eles()
{
}

// default destructor

eles::~eles() {}

// #### methods ####

// set number of elements

void eles::setup(int in_n_eles, int in_max_n_spts_per_ele)
{
  if (run_input.adv_type==0) {
    RK_a.setup(1);
    RK_b.setup(1);
    RK_c.setup(1);
  }else if (run_input.adv_type==3) {
    RK_a.setup(5);
    RK_b.setup(5);
    RK_c.setup(5);

    RK_a(0) = 0.0;
    RK_a(1) = -0.417890474499852;
    RK_a(2) = -1.192151694642677;
    RK_a(3) = -1.697784692471528;
    RK_a(4) = -1.514183444257156;

    RK_b(0) = 0.149659021999229;
    RK_b(1) = 0.379210312999627;
    RK_b(2) = 0.822955029386982;
    RK_b(3) = 0.699450455949122;
    RK_b(4) = 0.153057247968152;

    RK_c(0) = 0.0;
    RK_c(1) = 1432997174477/9575080441755;
    RK_c(2) = 2526269341429/6820363962896;
    RK_c(3) = 2006345519317/3224310063776;
    RK_c(4) = 2802321613138/2924317926251;
  }

  first_time = true;
  n_eles=in_n_eles;
  max_n_spts_per_ele = in_max_n_spts_per_ele;
  
  if (n_eles!=0)
  {

    order=run_input.order;
    p_res=run_input.p_res;
    viscous =run_input.viscous;
    motion = run_input.motion;
    LES = run_input.LES;
    sgs_model = run_input.SGS_model;
    wall_model = run_input.wall_model;
    
    // Set filter flag before calling setup_ele_type_specific
    filter = 0;
    if(LES)
      if(sgs_model==3 || sgs_model==2 || sgs_model==4)
        filter = 1;
    
    inters_cub_order = run_input.inters_cub_order;
    volume_cub_order = run_input.volume_cub_order;
    n_bdy_eles=0;

    // Initialize the element specific static members
    (*this).setup_ele_type_specific();

    if(run_input.adv_type==0)
    {
      n_adv_levels=1;
    }
    else if(run_input.adv_type==1)
    {
      n_adv_levels=3;
    }
    else if(run_input.adv_type==2)
    {
      n_adv_levels=4;
    }
    else if(run_input.adv_type==3)
    {
      n_adv_levels=2;
    }
    else
    {
      cout << "ERROR: Type of time integration scheme not recongized ... " << endl;
    }

    // Allocate storage for solution
    disu_upts.setup(n_adv_levels);
    for(int i=0;i<n_adv_levels;i++)
    {
      disu_upts(i).setup(n_upts_per_ele,n_eles,n_fields);
    }

    // Allocate storage for timestep
    // If using global minimum, only one timestep
    if (run_input.dt_type == 1)
      dt_local.setup(1);
    // If using local, one timestep per element
    else
      dt_local.setup(n_eles);
    
    // If in parallel and using global minumum timestep, allocate storage
    // for minimum timesteps in each partition
#ifdef _MPI
    if (run_input.dt_type == 1)
    {
      MPI_Comm_size(MPI_COMM_WORLD,&nproc);
      dt_local_mpi.setup(nproc);
      dt_local_mpi.initialize_to_zero();
    }
#endif
    
    // Initialize to zero
    for (int m=0;m<n_adv_levels;m++)
      disu_upts(m).initialize_to_zero();
    
    // Set no. of diagnostic fields
    n_diagnostic_fields = run_input.n_diagnostic_fields;

    // Set no. of diagnostic fields
    n_average_fields = run_input.n_average_fields;

    // Allocate storage for time-averaged velocity components    
    if(n_average_fields > 0) {
      disu_average_upts.setup(n_upts_per_ele,n_eles,n_average_fields);
      disu_average_upts.initialize_to_zero();
    }
    
    // Allocate extra arrays for LES models
    if(LES) {
      
      sgsf_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
      sgsf_fpts.setup(n_fpts_per_ele,n_eles,n_fields,n_dims);
      
      // SVV model requires filtered solution
      if(sgs_model==3 || sgs_model==2 || sgs_model==4) {
        disuf_upts.setup(n_upts_per_ele,n_eles,n_fields);
      }
      // allocate dummy array for passing to GPU routine
      else {
        disuf_upts.setup(1);
      }
      
      // Similarity model requires product terms and Leonard tensors
      if(sgs_model==2 || sgs_model==4) {
        
        // Leonard tensor and velocity-velocity product for momentum SGS term
        if(n_dims==2) {
          Lu.setup(n_upts_per_ele,n_eles,3);
          uu.setup(n_upts_per_ele,n_eles,3);
        }
        else if(n_dims==3) {
          Lu.setup(n_upts_per_ele,n_eles,6);
          uu.setup(n_upts_per_ele,n_eles,6);
        }
        
        // Leonard tensor and velocity-energy product for energy SGS term
        Le.setup(n_upts_per_ele,n_eles,n_dims);
        ue.setup(n_upts_per_ele,n_eles,n_dims);
        
      }
      // allocate dummy arrays
      else {
        Lu.setup(1);
        uu.setup(1);
        Le.setup(1);
        ue.setup(1);
      }
    }
    // Dummy arrays to pass to GPU kernel wrapper
    else {
      disuf_upts.setup(1);
      Lu.setup(1);
      uu.setup(1);
      Le.setup(1);
      ue.setup(1);
    }

    // Allocate array for wall distance if using a RANS-based turbulence model or LES wall model
    if (run_input.turb_model > 0) {
      wall_distance.setup(n_upts_per_ele,n_eles,n_dims);
      wall_distance_mag.setup(n_upts_per_ele,n_eles);
      zero_array(wall_distance);
      zero_array(wall_distance_mag);
      twall.setup(1);
    }
    else if (wall_model > 0) {
      wall_distance.setup(n_upts_per_ele,n_eles,n_dims);
      twall.setup(n_upts_per_ele,n_eles,n_fields);
      zero_array(wall_distance);
      zero_array(twall);
      wall_distance_mag.setup(1);
    }
    else {
      wall_distance.setup(1);
      wall_distance_mag.setup(1);
      twall.setup(1);
    }
    
    // Allocate SGS flux array if using LES or wall model
    if(LES != 0 || wall_model != 0) {
      temp_sgsf.setup(n_fields,n_dims);
      if(motion)
        temp_sgsf_ref.setup(n_fields,n_dims);
    }

    // Initialize source term
    src_upts.setup(n_upts_per_ele, n_eles, n_fields);
    zero_array(src_upts);

    // Allocate array for grid velocity
    temp_v.setup(n_dims);
    temp_v.initialize_to_zero();
    temp_v_ref.setup(n_dims);
    temp_v_ref.initialize_to_zero();

    int n_comp;
    if(n_dims == 2)
        n_comp = 3;
    else if(n_dims == 3)
        n_comp = 6;
    
    set_shape(in_max_n_spts_per_ele);
    ele2global_ele.setup(n_eles);
    bctype.setup(n_eles,n_inters_per_ele);

    // TODO: reduce unused allocation space (i.e. more spts alloc'd than needed)
    nodal_s_basis_fpts.setup(in_max_n_spts_per_ele,n_fpts_per_ele,n_eles);
    nodal_s_basis_upts.setup(in_max_n_spts_per_ele,n_upts_per_ele,n_eles);
    nodal_s_basis_ppts.setup(in_max_n_spts_per_ele,n_ppts_per_ele,n_eles);
    //d_nodal_s_basis_fpts.setup(n_fpts_per_ele,n_eles,n_dims,in_max_n_spts_per_ele);
    //d_nodal_s_basis_upts.setup(n_upts_per_ele,n_eles,n_dims,in_max_n_spts_per_ele);
    d_nodal_s_basis_upts.setup(n_dims,in_max_n_spts_per_ele,n_upts_per_ele,n_eles);
    d_nodal_s_basis_fpts.setup(n_dims,in_max_n_spts_per_ele,n_fpts_per_ele,n_eles);

    nodal_s_basis_vol_cubpts.setup(in_max_n_spts_per_ele,n_cubpts_per_ele,n_eles);
    d_nodal_s_basis_vol_cubpts.setup(n_dims,in_max_n_spts_per_ele,n_cubpts_per_ele,n_eles);
    nodal_s_basis_inters_cubpts.setup(n_inters_per_ele);
    d_nodal_s_basis_inters_cubpts.setup(n_inters_per_ele);
    for (int iface=0; iface<n_inters_per_ele; iface++) {
      nodal_s_basis_inters_cubpts(iface).setup(in_max_n_spts_per_ele,n_cubpts_per_inter(iface),n_eles);
      d_nodal_s_basis_inters_cubpts(iface).setup(n_dims,in_max_n_spts_per_ele,n_cubpts_per_inter(iface),n_eles);
    }
    
    // for mkl sparse blas
    matdescra[0]='G';
    matdescra[3]='F';
    transa='N';
    one=1.0;
    zero=0.0;
    
    n_fields_mul_n_eles=n_fields*n_eles;
    n_dims_mul_n_upts_per_ele=n_dims*n_upts_per_ele;
    
    div_tconf_upts.setup(n_adv_levels);
    for(int i=0;i<n_adv_levels;i++)
    {
      div_tconf_upts(i).setup(n_upts_per_ele,n_eles,n_fields);
    }
    
    // Initialize to zero
    for (int m=0;m<n_adv_levels;m++)
      div_tconf_upts(m).initialize_to_zero();
    
    disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
    tdisf_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
    norm_tdisf_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
    norm_tconf_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
    
    if (motion && run_input.GCL) {
      tdisf_GCL_upts.setup(n_upts_per_ele,n_eles,n_dims);
      tdisf_GCL_fpts.setup(n_upts_per_ele,n_eles,n_dims);
      norm_tdisf_GCL_fpts.setup(n_fpts_per_ele,n_eles);
      norm_tconf_GCL_fpts.setup(n_fpts_per_ele,n_eles);
      tdisf_GCL_upts.initialize_to_zero();
      tdisf_GCL_fpts.initialize_to_zero();
      norm_tdisf_GCL_fpts.initialize_to_zero();
      norm_tconf_GCL_fpts.initialize_to_zero();

      Jbar_upts.setup(n_adv_levels);
      Jbar_fpts.setup(n_adv_levels);
      div_tconf_GCL_upts.setup(n_adv_levels);
      //div_tconf_GCL_fpts.setup(n_adv_levels);
      for(int i=0;i<n_adv_levels;i++)
      {
        Jbar_upts(i).setup(n_upts_per_ele,n_eles);
        Jbar_fpts(i).setup(n_fpts_per_ele,n_eles);
        div_tconf_GCL_upts(i).setup(n_upts_per_ele,n_eles);
        //div_tconf_GCL_fpts(i).setup(n_fpts_per_ele,n_eles);
        div_tconf_GCL_upts(i).initialize_to_zero();
        //div_tconf_GCL_fpts(i).initialize_to_zero();
        Jbar_upts(i).initialize_to_zero();
        Jbar_fpts(i).initialize_to_zero();
      }
    }

    if(viscous)
    {
      delta_disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields);
      grad_disu_upts.setup(n_upts_per_ele,n_eles,n_fields,n_dims);
      grad_disu_fpts.setup(n_fpts_per_ele,n_eles,n_fields,n_dims);
    }

    if(run_input.ArtifOn)
    {
      if(run_input.artif_type == 1)
        sensor.setup(n_eles);

      if(run_input.artif_type == 0)
      {
          epsilon.setup(n_eles);
          epsilon_upts.setup(n_upts_per_ele,n_eles);
          epsilon_fpts.setup(n_fpts_per_ele,n_eles);
          sensor.setup(n_eles);
          //dt_local.setup(n_eles);
          min_dt_local.setup(1);
      }
    }
    
    // Set connectivity array. Needed for Paraview output.
    if (ele_type==3) // prism
      connectivity_plot.setup(8,n_peles_per_ele);
    else
      connectivity_plot.setup(n_verts_per_ele,n_peles_per_ele);
    
    set_connectivity_plot();
  }
  
}

void eles::set_disu_upts_to_zero_other_levels(void)
{
  
  if (n_eles!=0)
  {
    // Initialize to zero
    for (int m=1;m<n_adv_levels;m++)
    {
      disu_upts(m).initialize_to_zero();
      
#ifdef _GPU
      if (n_eles!=0)
      {
        disu_upts(m).cp_cpu_gpu();
      }
#endif
    }
  }
}

array<int> eles::get_connectivity_plot()
{
  return connectivity_plot;
}

// set initial conditions

void eles::set_ics(double& time)
{
  int i,j,k;
  
  double rho,vx,vy,vz,p;
  double gamma=run_input.gamma;
  time = 0.;
  
  array<double> loc(n_dims);
  array<double> pos(n_dims);
  array<double> ics(n_fields);
  
  array<double> grad_rho(n_dims);
  
  for(i=0;i<n_eles;i++)
  {
    for(j=0;j<n_upts_per_ele;j++)
    {
      for(k=0;k<n_dims;k++)
      {
        loc(k)=loc_upts(k,j);
      }
      
      // calculate position of solution point
      
      calc_pos(loc,i,pos);
      
      // evaluate solution at solution point
      if(run_input.ic_form==0)
      {
        eval_isentropic_vortex(pos,time,rho,vx,vy,vz,p,n_dims);
        
        ics(0)=rho;
        ics(1)=rho*vx;
        ics(2)=rho*vy;
        if(n_dims==2)
        {
          ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));
        }
        else if(n_dims==3)
        {
          ics(3)=rho*vz;
          ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));
        }
        else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
      }
      else if(run_input.ic_form==1)
      {
        rho=run_input.rho_c_ic;
        vx=run_input.u_c_ic;
        vy=run_input.v_c_ic;
        vz=run_input.w_c_ic;
        p=run_input.p_c_ic;
        
        ics(0)=rho;
        ics(1)=rho*vx;
        ics(2)=rho*vy;
        if(n_dims==2)
        {
          ics(3)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)));

          if (run_input.turb_model==1) {
            ics(4) = run_input.mu_tilde_c_ic;
          }
        }
        else if(n_dims==3)
        {
          ics(3)=rho*vz;
          ics(4)=(p/(gamma-1.0))+(0.5*rho*((vx*vx)+(vy*vy)+(vz*vz)));

          if(run_input.turb_model==1) {
            ics(5) = run_input.mu_tilde_c_ic;
          }
        }
        else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
        
      }
      else if (run_input.ic_form==2) // Sine wave (single)
      {
        eval_sine_wave_single(pos,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
        ics(0) = rho;
      }
      else if (run_input.ic_form==3) // Sine wave (group)
      {
        eval_sine_wave_group(pos,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
        ics(0) = rho;
      }
      else if (run_input.ic_form==4) // Spherical distribution
      {
        eval_sphere_wave(pos,run_input.wave_speed,time,rho,n_dims);
        ics(0) = rho;
      }
      else if (run_input.ic_form==5) // Constant for adv-diff
      {
        ics(0) = 1.0;
      }
      else if (run_input.ic_form==6) // Up to 4th order polynomials for u, v, w
      {
        rho=run_input.rho_c_ic;
        p=run_input.p_c_ic;
        eval_poly_ic(pos,rho,ics,n_dims);
        ics(0) = rho;
        if(n_dims==2)
          ics(3)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2));
        else if(n_dims==3)
          ics(4)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2)+ics(3)*ics(3));
      }
      else if (run_input.ic_form==7) // Taylor-Green Vortex initial conditions
      {
        rho=run_input.rho_c_ic;
        ics(0) = rho;
        if(n_dims==2)
        {
          // Simple 2D div-free vortex
          p = 100 + rho/16.0*(cos(2.0*pos(0)) + cos(2.0*pos(1)))*(cos(2.0*pos(2)) + 2.0);
          ics(1) = sin(pos(0)/2.)*cos(pos(1)/2.);
          ics(2) = -1.0*cos(pos(0)/2.)*sin(pos(1)/2.);
          ics(3)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2));
        }
        else if(n_dims==3)
        {
          // ONERA benchmark setup
          p = 100 + rho/16.0*(cos(2.0*pos(0)) + cos(2.0*pos(1)))*(cos(2.0*pos(2)) + 2.0);
          ics(1) = sin(pos(0))*cos(pos(1))*cos(pos(2));
          ics(2) = -1.0*cos(pos(0))*sin(pos(1))*cos(pos(2));
          ics(3) = 0.0;
          ics(4)=p/(gamma-1.0)+0.5*rho*(ics(1)*ics(1)+ics(2)*ics(2)+ics(3)*ics(3));
        }
      }
      else
      {
        cout << "ERROR: Invalid form of initial condition ... (File: " << __FILE__ << ", Line: " << __LINE__ << ")" << endl;
        exit (1);
      }
      
      // Add perturbation to channel
      if(run_input.perturb_ic==1 and n_dims==3)
      {
        // Constructs the function
        // u = alpha exp(-((x-L_x/2)/L_x)^2) exp(-(y/L_y)^2) cos(4 pi z/L_z)
        // Where L_x, L_y, L_z are the domain dimensions, u_0=bulk velocity,
        // alpha=scale factor, u=wall normal velocity
        double alpha, L_x, L_y, L_z;
        alpha = 0.1;
        L_x = 2.*pi;
        L_y = pi;
        L_z = 2.;
        ics(3) += alpha*exp(-pow((pos(0)-L_x/2.)/L_x,2))*exp(-pow(pos(1)/L_y,2))*cos(4.*pi*pos(2)/L_z);
      }
      
      // set solution at solution point
      for(k=0;k<n_fields;k++)
      {
        disu_upts(0)(j,i,k)=ics(k);
      }
    }
  }

  // If required, calculate element reference lengths
  if (run_input.dt_type > 0) {
    // Allocate array
    h_ref.setup(n_upts_per_ele,n_eles);
    
    // Call element specific function to obtain length
    double h_ref_dumb;
    for (int i=0;i<n_eles;i++) {
      h_ref_dumb = (*this).calc_h_ref_specific(i);
      for (int j=0;j<n_upts_per_ele;j++) {
        // TODO: Make more memory efficient!!!
        h_ref(j,i) = h_ref_dumb;
      }
    }
  }
  else {
    h_ref.setup(1);
  }
  h_ref.cp_cpu_gpu();
}


// set initial conditions


void eles::read_restart_data(ifstream& restart_file)
{
  
  if (n_eles==0) return;
  
  int num_eles_to_read;
  string ele_name,str;
  if (ele_type==0) ele_name="TRIS";
  else if (ele_type==1) ele_name="QUADS";
  else if (ele_type==2) ele_name="TETS";
  else if (ele_type==3) ele_name="PRIS";
  else if (ele_type==4) ele_name="HEXAS";
  
  // Move cursor to correct element type
  while(1) {
    getline(restart_file,str);
    if (str==ele_name) break;
    if (restart_file.eof()) return; // Restart file doesn't contain my elements
  }
  
  // Move cursor to n_eles
  while(1) {
    getline(restart_file,str);
    if (str=="n_eles") break;
  }
  
  // Read number of elements to read
  restart_file >> num_eles_to_read;
  getline(restart_file,str);
  
  //Skip ele2global_ele lines
  getline(restart_file,str);
  getline(restart_file,str);
  getline(restart_file,str);
  
  int ele,index;
  array<double> disu_upts_rest;
  disu_upts_rest.setup(n_upts_per_ele_rest,n_fields);
  
  for (int i=0;i<num_eles_to_read;i++)
  {
    restart_file >> ele ;
    index = index_locate_int(ele,ele2global_ele.get_ptr_cpu(),n_eles);
    
    if (index!=-1) // Ele belongs to processor
    {
      for (int j=0;j<n_upts_per_ele_rest;j++)
        for (int k=0;k<n_fields;k++)
          restart_file >> disu_upts_rest(j,k);
      
      // Now compute transformed solution at solution points using opp_r
      for (int m=0;m<n_fields;m++)
      {
        for (int j=0;j<n_upts_per_ele;j++)
        {
          double value = 0.;
          for (int k=0;k<n_upts_per_ele_rest;k++)
            value += opp_r(j,k)*disu_upts_rest(k,m);
          
          disu_upts(0)(j,index,m) = value;
        }
      }
      
    }
    else // Skip the data (doesn't belong to current processor)
    {
      // Skip rest of ele line
      getline(restart_file,str);
      for (int j=0;j<n_upts_per_ele_rest;j++)
        getline(restart_file,str);
    }
  }
  
  // If required, calculate element reference lengths
  if (run_input.dt_type > 0) {
    // Allocate array
    h_ref.setup(n_upts_per_ele,n_eles);

    // Call element specific function to obtain length
    double h_ref_dumb;
    for (int i=0;i<n_eles;i++) {
      h_ref_dumb = (*this).calc_h_ref_specific(i);
      for (int j=0;j<n_upts_per_ele;j++) {
        // TODO: Make more memory efficient!!!
        h_ref(j,i) = h_ref_dumb;
      }
    }
  }
  else {
    h_ref.setup(1);
  }
  h_ref.cp_cpu_gpu();
}


void eles::write_restart_data(ofstream& restart_file)
{
  restart_file << "n_eles" << endl;
  restart_file << n_eles << endl;
  restart_file << "ele2global_ele array" << endl;
  for (int i=0;i<n_eles;i++)
    restart_file << ele2global_ele(i) << " ";
  restart_file << endl;
  
  restart_file << "data" << endl;
  
  for (int i=0;i<n_eles;i++)
  {
    restart_file << ele2global_ele(i) << endl;
    for (int j=0;j<n_upts_per_ele;j++)
    {
      for (int k=0;k<n_fields;k++)
      {
        restart_file << disu_upts(0)(j,i,k) << " ";
      }
      restart_file << endl;
    }
  }
  restart_file << endl;
}

void eles::write_restart_mesh(ofstream& restart_file)
{
  restart_file << "n_eles" << endl;
  restart_file << n_eles << endl;
  restart_file << "ele2global_ele array" << endl;
  for (int i=0;i<n_eles;i++)
    restart_file << ele2global_ele(i) << " ";
  restart_file << endl;

  restart_file << "data" << endl;

  for (int i=0;i<n_eles;i++)
  {
    restart_file << ele2global_ele(i) << endl;
    for (int j=0;j<n_upts_per_ele;j++)
    {
      for (int k=0;k<n_dims;k++)
      {
        if (motion!=STATIC_MESH)
          restart_file << dyn_pos_upts(j,i,k) << " ";
        else
          restart_file << pos_upts(j,i,k) << " ";
      }
      restart_file << endl;
    }
  }
  restart_file << endl;
}

// move all to from cpu to gpu

void eles::mv_all_cpu_gpu(void)
{
#ifdef _GPU
  if (n_eles!=0)
  {
    disu_upts(0).cp_cpu_gpu();
    div_tconf_upts(0).cp_cpu_gpu();
    src_upts.cp_cpu_gpu();
    
    for(int i=1;i<n_adv_levels;i++)
    {
      disu_upts(i).cp_cpu_gpu();
      div_tconf_upts(i).mv_cpu_gpu();
    }
    
    disu_fpts.mv_cpu_gpu();
    tdisf_upts.mv_cpu_gpu();
    norm_tdisf_fpts.mv_cpu_gpu();
    norm_tconf_fpts.mv_cpu_gpu();
    
    //TODO: mv instead of cp
    if(viscous)
    {
      delta_disu_fpts.mv_cpu_gpu();
      grad_disu_upts.cp_cpu_gpu();
      grad_disu_fpts.mv_cpu_gpu();
      
      //tdisvisf_upts.mv_cpu_gpu();
      //norm_tdisvisf_fpts.mv_cpu_gpu();
      //norm_tconvisf_fpts.mv_cpu_gpu();
    }
    
    // LES and wall model arrays
    filter_upts.mv_cpu_gpu();
    disuf_upts.mv_cpu_gpu();
    sgsf_upts.mv_cpu_gpu();
    sgsf_fpts.mv_cpu_gpu();
    uu.mv_cpu_gpu();
    ue.mv_cpu_gpu();
    Lu.mv_cpu_gpu();
    Le.mv_cpu_gpu();
    twall.mv_cpu_gpu();

    // Grid Velocity-related arrays for moving meshes
    vel_spts.cp_cpu_gpu();
    grid_vel_upts.mv_cpu_gpu();
    grid_vel_fpts.mv_cpu_gpu();

    if (motion) {
      run_input.bound_vel_simple(0).mv_cpu_gpu();
    }

    if(run_input.ArtifOn){
        // Needed for shock capturing routines
        sensor.cp_cpu_gpu();
        inv_vandermonde.mv_cpu_gpu();
        inv_vandermonde2D.mv_cpu_gpu();
        vandermonde2D.mv_cpu_gpu();

        if(run_input.artif_type == 1)
        {
          concentration_array.mv_cpu_gpu();
          sigma.mv_cpu_gpu();
        }

        if(run_input.artif_type == 0)
        {
          epsilon.mv_cpu_gpu();
          epsilon_upts.cp_cpu_gpu();
          epsilon_fpts.cp_cpu_gpu();
          ele2global_ele_code.mv_cpu_gpu();
          area_coord_upts.mv_cpu_gpu();
          area_coord_fpts.mv_cpu_gpu();
          n_spts_per_ele.mv_cpu_gpu();
          dt_local.cp_cpu_gpu();
          min_dt_local.cp_cpu_gpu();
        }
    }
  }
#endif
}

// move wall distance array to gpu
void eles::mv_wall_distance_cpu_gpu(void)
{
#ifdef _GPU
  
  wall_distance.mv_cpu_gpu();
  
#endif
}

// move wall distance magnitude array to gpu
void eles::mv_wall_distance_mag_cpu_gpu(void)
{
#ifdef _GPU

  wall_distance_mag.mv_cpu_gpu();

#endif
}

// copy discontinuous solution at solution points to cpu
void eles::cp_disu_upts_gpu_cpu(void)
{
#ifdef _GPU
  if (n_eles!=0)
  {
    disu_upts(0).cp_gpu_cpu();
  }
#endif
}


// copy discontinuous solution at solution points to gpu
void eles::cp_disu_upts_cpu_gpu(void)
{
#ifdef _GPU
  if (n_eles!=0)
  {
    disu_upts(0).cp_cpu_gpu();
  }
#endif
}

// copy gradient of discontinuous solution at solution points to cpu
void eles::cp_grad_disu_upts_gpu_cpu(void)
{
  if (n_eles!=0)
  {
#ifdef _GPU
    grad_disu_upts.cp_gpu_cpu();
#endif
  }
}

// copy determinant of jacobian at solution points to cpu
void eles::cp_detjac_upts_gpu_cpu(void)
{
#ifdef _GPU
  
  detjac_upts.cp_gpu_cpu();
  
#endif
}

// copy divergence at solution points to cpu
void eles::cp_div_tconf_upts_gpu_cpu(void)
{
  if (n_eles!=0)
  {
#ifdef _GPU
    
    div_tconf_upts(0).cp_gpu_cpu();
    
#endif
  }
}

// copy local time stepping reference length at solution points to cpu
void eles::cp_h_ref_gpu_cpu(void)
{
#ifdef _GPU

  h_ref.cp_gpu_cpu();

#endif
}

// copy source term at solution points to cpu
void eles::cp_src_upts_gpu_cpu(void)
{
  if (n_eles!=0)
  {
#ifdef _GPU

    src_upts.cp_gpu_cpu();

#endif
  }
}

// copy sensor in each element to cpu
void eles::cp_sensor_gpu_cpu(void)
{
#ifdef _GPU
  if (n_eles!=0)
    {
      sensor.cp_gpu_cpu();
    }
#endif
}

// copy sensor in each element to cpu
void eles::cp_epsilon_upts_gpu_cpu(void)
{
#ifdef _GPU
  if (n_eles!=0)
    {
      epsilon_upts.cp_gpu_cpu();
    }
#endif
}

// remove transformed discontinuous solution at solution points from cpu

void eles::rm_disu_upts_cpu(void)
{
#ifdef _GPU
  
  disu_upts(0).rm_cpu();
  
#endif
}

// remove determinant of jacobian at solution points from cpu

void eles::rm_detjac_upts_cpu(void)
{
#ifdef _GPU
  
  detjac_upts.rm_cpu();
  
#endif
}

// advance solution

void eles::AdvanceSolution(int in_step, int adv_type) {
  
  if (n_eles!=0)
  {
    
    /*! Time integration using a forwards Euler integration. */
    
    if (adv_type == 0) {
      
      /*!
       Performs B = B + (alpha*A) where: \n
       alpha = -run_input.dt \n
       A = div_tconf_upts(0)\n
       B = disu_upts(0)
       */
      
#ifdef _CPU
      // If using global minimum timestep based on CFL, determine
      // global minimum
      if (run_input.dt_type == 1)
      {
        // Find minimum timestep
        dt_local(0) = 1e12; // Set to large value
        
        for (int ic=0; ic<n_eles; ic++)
        {
          dt_local_new = calc_dt_local(ic);
          
          if (dt_local_new < dt_local(0))
            dt_local(0) = dt_local_new;
        }
        
        // If running in parallel, gather minimum timestep values from
        // each partition and find global minumum across partitions
#ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(&dt_local(0),1,MPI_DOUBLE,dt_local_mpi.get_ptr_cpu(),
                      1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        
        dt_local(0) = dt_local_mpi.get_min();
#endif
      }
      
      // If using local timestepping, just compute and store all local
      // timesteps
      if (run_input.dt_type == 2)
      {
        for (int ic=0; ic<n_eles; ic++)
          dt_local(ic) = calc_dt_local(ic);
      }
      
      for (int i=0;i<n_fields;i++)
      {
        for (int ic=0;ic<n_eles;ic++)
        {
          for (int inp=0;inp<n_upts_per_ele;inp++)
          {
            // User supplied timestep
            if (run_input.dt_type != 0)
            {
              // Global minimum timestep
              if (run_input.dt_type == 1)
                run_input.dt = dt_local(0);
            
              // Element local timestep
              else if (run_input.dt_type == 2)
                run_input.dt = dt_local(ic);

              else
                FatalError("ERROR: dt_type not recognized!")
            }
              
            disu_upts(0)(inp,ic,i) -= run_input.dt*(div_tconf_upts(0)(inp,ic,i)/detjac_upts(inp,ic) - run_input.const_src - src_upts(inp,ic,i));
          }
        }
      }

#endif
      
#ifdef _GPU
      RK11_update_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,disu_upts(0).get_ptr_gpu(),div_tconf_upts(0).get_ptr_gpu(),detjac_upts.get_ptr_gpu(),src_upts.get_ptr_gpu(),h_ref.get_ptr_gpu(),run_input.dt,run_input.const_src,run_input.CFL,run_input.gamma,run_input.mu_inf,run_input.order,viscous,run_input.dt_type);
#endif
      
    }
    
    /*! Time integration using a RK45 method. */
    
    else if (adv_type == 3) {
      
      double rk4a, rk4b;
      if (in_step==0) {
        rk4a=    0.0;
        rk4b=   0.149659021999229;
      }
      else if (in_step==1) {
        rk4a=   -0.417890474499852;
        rk4b=   0.379210312999627;
      }
      else if (in_step==2) {
        rk4a=   -1.192151694642677;
        rk4b=   0.822955029386982;
      }
      else if (in_step==3) {
        rk4a=   -1.697784692471528;
        rk4b=   0.699450455949122;
      }
      else if (in_step==4) {
        rk4a=   -1.514183444257156;
        rk4b=   0.153057247968152;
      }
      
#ifdef _CPU
      // for first stage only, compute timestep
      if (in_step == 0)
      {
        // For global timestepping, find minimum timestep
        if (run_input.dt_type == 1)
        {
          dt_local(0) = 1e12;
          
          for (int ic=0; ic<n_eles; ic++)
          {
            dt_local_new = calc_dt_local(ic);
            
            if (dt_local_new < dt_local(0))
            {
              dt_local(0) = dt_local_new;
            }
          }
          
          
          // If using MPI, find minimum across partitions
#ifdef _MPI
          MPI_Barrier(MPI_COMM_WORLD);
          MPI_Allgather(&dt_local(0),1,MPI_DOUBLE,dt_local_mpi.get_ptr_cpu(),1,MPI_DOUBLE,MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
          
          dt_local(0) = dt_local_mpi.get_min();
#endif
        }
        
        // For local timestepping, find element local timesteps
        if (run_input.dt_type == 2)
        {
          for (int ic=0; ic<n_eles; ic++)
          {
            dt_local(ic) = calc_dt_local(ic);
          }
        }
      }
      
      double res, rhs;
      for (int ic=0;ic<n_eles;ic++)
      {
        for (int i=0;i<n_fields;i++)
        {
          for (int inp=0;inp<n_upts_per_ele;inp++)
          {
            rhs = -div_tconf_upts(0)(inp,ic,i)/detjac_upts(inp,ic) + run_input.const_src + src_upts(inp,ic,i);
            res = disu_upts(1)(inp,ic,i);
            
            if (run_input.dt_type != 0)
            {
              if (run_input.dt_type == 1)
                run_input.dt = dt_local(0);
              else if (run_input.dt_type == 2)
                run_input.dt = dt_local(ic);
            }
            
            res = rk4a*res + run_input.dt*rhs;
            disu_upts(1)(inp,ic,i) = res;
            disu_upts(0)(inp,ic,i) += rk4b*res;
          }
        }
      }
      
#endif
      
#ifdef _GPU
      
      RK45_update_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,disu_upts(0).get_ptr_gpu(),disu_upts(1).get_ptr_gpu(),div_tconf_upts(0).get_ptr_gpu(),detjac_upts.get_ptr_gpu(),src_upts.get_ptr_gpu(),h_ref.get_ptr_gpu(),rk4a,rk4b,run_input.dt,run_input.const_src,run_input.CFL,run_input.gamma,run_input.mu_inf,run_input.order,viscous,run_input.dt_type,in_step);
      
#endif
      
    }
    
    /*! Time integration not implemented. */
    
    else {
      cout << "ERROR: Time integration type not recognised ... " << endl;
    }
    
  }
  
}

double eles::calc_dt_local(int in_ele)
{
  double lam_inv, lam_inv_new;
  double lam_visc, lam_visc_new;
  double out_dt_local;
  double dt_inv, dt_visc;
  
  // 2-D Elements
  if (n_dims == 2)
  {
    double u,v,p,c;
    
    lam_inv = 0;
    lam_visc = 0;
    
    // Calculate maximum internal wavespeed per element
    for (int i=0; i<n_upts_per_ele; i++)
    {
      u = disu_upts(0)(i,in_ele,1)/disu_upts(0)(i,in_ele,0);
      v = disu_upts(0)(i,in_ele,2)/disu_upts(0)(i,in_ele,0);
      p = (run_input.gamma - 1.0) * (disu_upts(0)(i,in_ele,3) - 0.5*disu_upts(0)(i,in_ele,0)*(u*u+v*v));
      c = sqrt(run_input.gamma * p/disu_upts(0)(i,in_ele,0));
      
      lam_inv_new = sqrt(u*u + v*v) + c;
      lam_visc_new = 4.0/3.0*run_input.mu_inf/disu_upts(0)(i,in_ele,0);
      
      if (lam_inv < lam_inv_new)
        lam_inv = lam_inv_new;

      if (lam_visc < lam_visc_new)
        lam_visc = lam_visc_new;
    }

    if (viscous)
    {
      dt_visc = (run_input.CFL * 0.25 * h_ref(0,in_ele) * h_ref(0,in_ele))/(lam_visc) * 1.0/(2.0*run_input.order+1.0);
      dt_inv = run_input.CFL*h_ref(0,in_ele)/lam_inv*1.0/(2.0*run_input.order + 1.0);
    }
    else
    {
      dt_visc = 1e16;
      dt_inv = run_input.CFL*h_ref(0,in_ele)/lam_inv * 1.0/(2.0*run_input.order + 1.0);
    }
      out_dt_local = min(dt_visc,dt_inv);
  }
  
  else if (n_dims == 3)
  {
    FatalError("Timestep type is not implemented in 3D yet.");
  }
  
  return out_dt_local;
}

// calculate the discontinuous solution at the flux points

void eles::extrapolate_solution(int in_disu_upts_from)
{
  if (n_eles!=0) {
    
    /*!
     Performs C = (alpha*A*B) + (beta*C) where: \n
     alpha = 1.0 \n
     beta = 0.0 \n
     A = opp_0 \n
     B = disu_upts(in_disu_upts_from) \n
     C = disu_fpts
     
     opp_0 is the polynomial extrapolation matrix;
     has dimensions n_f_pts_per_ele by n_upts_per_ele
     
     Recall: opp_0(j,i) = value of the ith nodal basis at the
     jth flux point location in the reference domain
     
     (vector of solution values at flux points) = opp_0 * (vector of solution values at nodes)
     */
    
    Arows =  n_fpts_per_ele;
    Acols = n_upts_per_ele;
    
    Brows = Acols;
    Bcols = n_fields*n_eles;
    
    Astride = Arows;
    Bstride = Brows;
    Cstride = Arows;
    
#ifdef _CPU
    
    if(opp_0_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_0.get_ptr_cpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_cpu(),Bstride,0.0,disu_fpts.get_ptr_cpu(),Cstride);
      
#elif defined _NO_BLAS
      dgemm(Arows,Bcols,Acols,1.0,0.0,opp_0.get_ptr_cpu(),disu_upts(in_disu_upts_from).get_ptr_cpu(),disu_fpts.get_ptr_cpu());
      
#endif
    }
    else if(opp_0_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      mkl_dcsrmm(&transa,&n_fpts_per_ele,&n_fields_mul_n_eles,&n_upts_per_ele,&one,matdescra,opp_0_data.get_ptr_cpu(),opp_0_cols.get_ptr_cpu(),opp_0_b.get_ptr_cpu(),opp_0_e.get_ptr_cpu(),disu_upts(in_disu_upts_from).get_ptr_cpu(),&n_upts_per_ele,&zero,disu_fpts.get_ptr_cpu(),&n_fpts_per_ele);
      
#endif
    }
    else { cout << "ERROR: Unknown storage for opp_0 ... " << endl; }
    
#endif
    
#ifdef _GPU
    if(opp_0_sparse==0)
    {
      cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_0.get_ptr_gpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_gpu(),Bstride,0.0,disu_fpts.get_ptr_gpu(),Cstride);
    }
    else if (opp_0_sparse==1)
    {
      bespoke_SPMV(n_fpts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_0_ell_data.get_ptr_gpu(),opp_0_ell_indices.get_ptr_gpu(),opp_0_nnz_per_row,disu_upts(in_disu_upts_from).get_ptr_gpu(),disu_fpts.get_ptr_gpu(),ele_type,order,0);
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_0 ... " << endl;
    }
#endif
    
  }
  
}

// calculate the transformed discontinuous inviscid flux at the solution points

void eles::evaluate_invFlux(int in_disu_upts_from)
{
  if (n_eles!=0)
  {
    
#ifdef _CPU
    
    int i,j,k,l,m;
    
    for(i=0;i<n_eles;i++)
    {
      for(j=0;j<n_upts_per_ele;j++)
      {
        for(k=0;k<n_fields;k++)
        {
          temp_u(k)=disu_upts(in_disu_upts_from)(j,i,k);
        }

        if (motion) {
          // Transform solution from static frame to dynamic frame
          for (k=0; k<n_fields; k++) {
            temp_u(k) /= J_dyn_upts(j,i);
          }
          // Get mesh velocity in dynamic frame
          for (k=0; k<n_dims; k++) {
            temp_v(k) = grid_vel_upts(j,i,k);
          }
          // Temporary flux vector for dynamic->static transformation
          temp_f_ref.setup(n_fields,n_dims);
        }else{
          temp_v.initialize_to_zero();
        }
        
        if(n_dims==2)
        {
          calc_invf_2d(temp_u,temp_f);
          if (motion)
            calc_alef_2d(temp_u, temp_v, temp_f);
        }
        else if(n_dims==3)
        {
          calc_invf_3d(temp_u,temp_f);
          if (motion)
            calc_alef_3d(temp_u, temp_v, temp_f);
        }
        else
        {
          FatalError("Invalid number of dimensions!");
        }

        // Transform from dynamic-physical space to static-physical space
        if (motion) {
          for(k=0; k<n_fields; k++) {
            for(l=0; l<n_dims; l++) {
              temp_f_ref(k,l)=0.;
              for(m=0; m<n_dims; m++) {
                temp_f_ref(k,l) += JGinv_dyn_upts(l,m,j,i)*temp_f(k,m);
              }
            }
          }

          // Copy Static-Physical Domain flux back to temp_f
          for (k=0; k<n_fields; k++) {
            for (l=0; l<n_dims; l++) {
              temp_f(k,l) = temp_f_ref(k,l);
            }
          }
        }
        
        // Transform from static physical space to computational space
        for(k=0;k<n_fields;k++) {
          for(l=0;l<n_dims;l++) {
            tdisf_upts(j,i,k,l)=0.;
            for(m=0;m<n_dims;m++) {
              tdisf_upts(j,i,k,l) += JGinv_upts(l,m,j,i)*temp_f(k,m);//JGinv_upts(j,i,l,m)*temp_f(k,m);
            }
          }
        }
      }
    }
    
#endif
    
#ifdef _GPU
    evaluate_invFlux_gpu_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,disu_upts(in_disu_upts_from).get_ptr_gpu(),tdisf_upts.get_ptr_gpu(),detjac_upts.get_ptr_gpu(),J_dyn_upts.get_ptr_gpu(),JGinv_upts.get_ptr_gpu(),JGinv_dyn_upts.get_ptr_gpu(),grid_vel_upts.get_ptr_gpu(),run_input.gamma,motion,run_input.equation,run_input.wave_speed(0),run_input.wave_speed(1),run_input.wave_speed(2),run_input.turb_model);
#endif
  }
}


// calculate the normal transformed discontinuous flux at the flux points

void eles::extrapolate_totalFlux()
{
  if (n_eles!=0)
  {
#ifdef _CPU
    
    if(opp_1_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(0).get_ptr_cpu(),n_fpts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,0),n_upts_per_ele,0.0,norm_tdisf_fpts.get_ptr_cpu(),n_fpts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(i).get_ptr_cpu(),n_fpts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele,1.0,norm_tdisf_fpts.get_ptr_cpu(),n_fpts_per_ele);
      }
      
#elif defined _NO_BLAS
      dgemm(n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,0.0,opp_1(0).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,0),norm_tdisf_fpts.get_ptr_cpu());
      for (int i=1;i<n_dims;i++)
      {
        dgemm(n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,1.0,opp_1(i).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,i),norm_tdisf_fpts.get_ptr_cpu());
      }
#endif
    }
    else if(opp_1_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      
      mkl_dcsrmm(&transa,&n_fpts_per_ele,&n_fields_mul_n_eles,&n_upts_per_ele,&one,matdescra,opp_1_data(0).get_ptr_cpu(),opp_1_cols(0).get_ptr_cpu(),opp_1_b(0).get_ptr_cpu(),opp_1_e(0).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,0),&n_upts_per_ele,&zero,norm_tdisf_fpts.get_ptr_cpu,&n_fpts_per_ele);
      
      for (int i=1;i<n_dims;i++) {
        mkl_dcsrmm(&transa,&n_fpts_per_ele,&n_fields_mul_n_eles,&n_upts_per_ele,&one,matdescra,opp_1_data(i).get_ptr_cpu(),opp_1_cols(i).get_ptr_cpu(),opp_1_b(i).get_ptr_cpu(),opp_1_e(i).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,i),&n_upts_per_ele,&one,norm_tdisf_fpts.get_ptr_cpu(),&n_fpts_per_ele);
      }
      
#endif
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_1 ... " << endl;
    }
    
#endif
    
#ifdef _GPU
    
    if (opp_1_sparse==0)
    {
      cublasDgemm('N','N',n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(0).get_ptr_gpu(),n_fpts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,0),n_upts_per_ele,0.0,norm_tdisf_fpts.get_ptr_gpu(),n_fpts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
        cublasDgemm('N','N',n_fpts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_1(i).get_ptr_gpu(),n_fpts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele,1.0,norm_tdisf_fpts.get_ptr_gpu(),n_fpts_per_ele);
      }
    }
    else if (opp_1_sparse==1)
    {
      bespoke_SPMV(n_fpts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_1_ell_data(0).get_ptr_gpu(),opp_1_ell_indices(0).get_ptr_gpu(),opp_1_nnz_per_row(0),tdisf_upts.get_ptr_gpu(0,0,0,0),norm_tdisf_fpts.get_ptr_gpu(),ele_type,order,0);
      for (int i=1;i<n_dims;i++)
      {
        bespoke_SPMV(n_fpts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_1_ell_data(i).get_ptr_gpu(),opp_1_ell_indices(i).get_ptr_gpu(),opp_1_nnz_per_row(i),tdisf_upts.get_ptr_gpu(0,0,0,i),norm_tdisf_fpts.get_ptr_gpu(),ele_type,order,1);
      }
    }
#endif
    
  }
  
  /*
   #ifdef _GPU
   tdisinvf_upts.cp_gpu_cpu();
   #endif
   
   cout << "Before" << endl;
   for (int i=0;i<n_fpts_per_ele;i++)
   for (int j=0;j<n_eles;j++)
   for (int k=0;k<n_fields;k++)
   for (int m=0;m<n_dims;m++)
   cout << setprecision(10)  << i << " " << j<< " " << k << " " << tdisinvf_upts(i,j,k,m) << endl;
   */
  
  /*
   cout << "After,ele_type =" << ele_type << endl;
   #ifdef _GPU
   norm_tdisinvf_fpts.cp_gpu_cpu();
   #endif
   
   for (int i=0;i<n_fpts_per_ele;i++)
   for (int j=0;j<n_eles;j++)
   for (int k=0;k<n_fields;k++)
   cout << setprecision(10)  << i << " " << j<< " " << k << " " << norm_tdisinvf_fpts(i,j,k) << endl;
   */
}


// calculate the divergence of the transformed discontinuous flux at the solution points

void eles::calculate_divergence(int in_div_tconf_upts_to)
{
  if (n_eles!=0)
  {
#ifdef _CPU
    
    if(opp_2_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(0).get_ptr_cpu(),n_upts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,0),n_upts_per_ele,0.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),n_upts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(i).get_ptr_cpu(),n_upts_per_ele,tdisf_upts.get_ptr_cpu(0,0,0,i),n_upts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),n_upts_per_ele);
      }
      
#elif defined _NO_BLAS
      dgemm(n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,0.0,opp_2(0).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,0),div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu());
      for (int i=1;i<n_dims;i++)
      {
        dgemm(n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,1.0,opp_2(i).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,i),div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu());
      }
      
#endif
    }
    else if(opp_2_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      
      mkl_dcsrmm(&transa,&n_upts_per_ele,&n_fields_mul_n_eles,&n_upts_per_ele,&one,matdescra,opp_2_data(0).get_ptr_cpu(),opp_2_cols(0).get_ptr_cpu(),opp_2_b(0).get_ptr_cpu(),opp_2_e(0).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,0),&n_upts_per_ele,&zero,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),&n_upts_per_ele);
      for (int i=1;i<n_dims;i++)
      {
        mkl_dcsrmm(&transa,&n_upts_per_ele,&n_fields_mul_n_eles,&n_upts_per_ele,&one,matdescra,opp_2_data(i).get_ptr_cpu(),opp_2_cols(i).get_ptr_cpu(),opp_2_b(i).get_ptr_cpu(),opp_2_e(i).get_ptr_cpu(),tdisf_upts.get_ptr_cpu(0,0,0,i),&n_upts_per_ele,&one,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),&n_upts_per_ele);
      }
      
#endif
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_2 ... " << endl;
    }
    
#endif
    
    
#ifdef _GPU
    
    if (opp_2_sparse==0)
    {
      cublasDgemm('N','N',n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(0).get_ptr_gpu(),n_upts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,0),n_upts_per_ele,0.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),n_upts_per_ele);
      for (int i=1;i<n_dims;i++) {
        cublasDgemm('N','N',n_upts_per_ele,n_fields*n_eles,n_upts_per_ele,1.0,opp_2(i).get_ptr_gpu(),n_upts_per_ele,tdisf_upts.get_ptr_gpu(0,0,0,i),n_upts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),n_upts_per_ele);
      }
    }
    else if (opp_2_sparse==1)
    {
      bespoke_SPMV(n_upts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_2_ell_data(0).get_ptr_gpu(),opp_2_ell_indices(0).get_ptr_gpu(),opp_2_nnz_per_row(0),tdisf_upts.get_ptr_gpu(0,0,0,0),div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),ele_type,order,0);
      for (int i=1;i<n_dims;i++) {
        bespoke_SPMV(n_upts_per_ele,n_upts_per_ele,n_fields,n_eles,opp_2_ell_data(i).get_ptr_gpu(),opp_2_ell_indices(i).get_ptr_gpu(),opp_2_nnz_per_row(i),tdisf_upts.get_ptr_gpu(0,0,0,i),div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),ele_type,order,1);
      }
      
    }
#endif
    
  }
  
  /*
   for (int j=0;j<n_eles;j++)
   for (int i=0;i<n_upts_per_ele;i++)
   //for (int k=0;k<n_fields;k++)
   cout << scientific << setw(16) << setprecision(12) << div_tconf_upts(0)(i,j,0) << endl;
   */
}


// calculate divergence of the transformed continuous flux at the solution points

void eles::calculate_corrected_divergence(int in_div_tconf_upts_to)
{
  if (n_eles!=0)
  {
#ifdef _CPU
    
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    
    cblas_daxpy(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_cpu(),1,norm_tconf_fpts.get_ptr_cpu(),1);
    
#elif defined _NO_BLAS
    
    daxpy(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_cpu(),norm_tconf_fpts.get_ptr_cpu());
    
#endif
    
    if(opp_3_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_upts_per_ele,n_fields*n_eles,n_fpts_per_ele,1.0,opp_3.get_ptr_cpu(),n_upts_per_ele,norm_tconf_fpts.get_ptr_cpu(),n_fpts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),n_upts_per_ele);
      
#elif defined _NO_BLAS
      dgemm(n_upts_per_ele,n_fields*n_eles,n_fpts_per_ele,1.0,1.0,opp_3.get_ptr_cpu(),norm_tconf_fpts.get_ptr_cpu(),div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu());
      
#endif
    }
    else if(opp_3_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      
      mkl_dcsrmm(&transa,&n_upts_per_ele,&n_fields_mul_n_eles,&n_fpts_per_ele,&one,matdescra,opp_3_data.get_ptr_cpu(),opp_3_cols.get_ptr_cpu(),opp_3_b.get_ptr_cpu(),opp_3_e.get_ptr_cpu(),norm_tconf_fpts.get_ptr_cpu(),&n_fpts_per_ele,&one,div_tconf_upts(in_div_tconf_upts_to).get_ptr_cpu(),&n_upts_per_ele);
      
#endif
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_3 ... " << endl;
    }
    
    for (int i=0;i<n_upts_per_ele;i++)
      for (int j=0;j<n_eles;j++)
        for (int k=0;k<n_fields;k++)
          if (isnan(div_tconf_upts(in_div_tconf_upts_to)(j,i,k)))
            FatalError("NaN in residual, exiting.");

#endif
    
#ifdef _GPU
    
    cublasDaxpy(n_eles*n_fields*n_fpts_per_ele,-1.0,norm_tdisf_fpts.get_ptr_gpu(),1,norm_tconf_fpts.get_ptr_gpu(),1);
    
    if (opp_3_sparse==0)
    {
      cublasDgemm('N','N',n_upts_per_ele,n_fields*n_eles,n_fpts_per_ele,1.0,opp_3.get_ptr_gpu(),n_upts_per_ele,norm_tconf_fpts.get_ptr_gpu(),n_fpts_per_ele,1.0,div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),n_upts_per_ele);
    }
    else if (opp_3_sparse==1)
    {
      bespoke_SPMV(n_upts_per_ele,n_fpts_per_ele,n_fields,n_eles,opp_3_ell_data.get_ptr_gpu(),opp_3_ell_indices.get_ptr_gpu(),opp_3_nnz_per_row,norm_tconf_fpts.get_ptr_gpu(),div_tconf_upts(in_div_tconf_upts_to).get_ptr_gpu(),ele_type,order,1);
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_3 ... " << endl;
    }
#endif

  }
}


// calculate uncorrected transformed gradient of the discontinuous solution at the solution points
// (mixed derivative)

void eles::calculate_gradient(int in_disu_upts_from)
{
  if (n_eles!=0)
  {
    
    /*!
     Performs C = (alpha*A*B) + (beta*C) where: \n
     alpha = 1.0 \n
     beta = 0.0 \n
     A = opp_4 \n
     B = disu_upts \n
     C = grad_disu_upts
     
     opp_4 is the polynomial gradient matrix;
     has dimensions n_upts_per_ele by n_upts_per_ele
     Recall: opp_4(i)(k,j) = eval_d_nodal_basis(j,i,loc);
     = derivative of the jth nodal basis at the
     kth nodal (solution) point location in the reference domain
     for the ith dimension
     
     (vector of gradient values at solution points) = opp_4 *
     (vector of solution values at solution points in all elements of the same type)
     */
    
    Arows =  n_upts_per_ele;
    Acols = n_upts_per_ele;
    
    Brows = Acols;
    Bcols = n_fields*n_eles;
    
    Astride = Arows;
    Bstride = Brows;
    Cstride = Arows;
    
#ifdef _CPU
    
    if(opp_4_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      for (int i=0;i<n_dims;i++) {
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_4(i).get_ptr_cpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_cpu(),Bstride,0.0,grad_disu_upts.get_ptr_cpu(0,0,0,i),Cstride);
      }
      
#elif defined _NO_BLAS
      for (int i=0;i<n_dims;i++) {
        dgemm(Arows,Bcols,Acols,1.0,0.0,opp_4(i).get_ptr_cpu(),disu_upts(in_disu_upts_from).get_ptr_cpu(),grad_disu_upts.get_ptr_cpu(0,0,0,i));
      }
      
#endif
    }
    else if(opp_4_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      
      // implement
      
#endif
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_4 ... " << endl;
    }
    
#endif
    
#ifdef _GPU
    
    if (opp_4_sparse==0)
    {
      for (int i=0;i<n_dims;i++)
      {
        cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_4(i).get_ptr_gpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_gpu(),Bstride,0.0,grad_disu_upts.get_ptr_gpu(0,0,0,i),Cstride);
      }
    }
    else if (opp_4_sparse==1)
    {
      for (int i=0;i<n_dims;i++)
      {
        bespoke_SPMV(Arows,Acols,n_fields,n_eles,opp_4_ell_data(i).get_ptr_gpu(),opp_4_ell_indices(i).get_ptr_gpu(),opp_4_nnz_per_row(i),disu_upts(in_disu_upts_from).get_ptr_gpu(),grad_disu_upts.get_ptr_gpu(0,0,0,i),ele_type,order,0);
      }
    }
#endif
  }

  /*
   cout << "OUTPUT" << endl;
   #ifdef _GPU
   grad_disu_upts.cp_gpu_cpu();
   #endif
   
   for (int i=0;i<n_upts_per_ele;i++)
   for (int j=0;j<n_eles;j++)
   for (int k=0;k<n_fields;k++)
   for (int m=0;m<n_dims;m++)
   {
   if (ele2global_ele(j)==53)
   cout << setprecision(10)  << i << " " << ele2global_ele(j) << " " << k << " " << m << " " << grad_disu_upts(i,j,k,m) << endl;
   }
   */
}

// calculate corrected gradient of the discontinuous solution at solution points

void eles::correct_gradient(void)
{
  if (n_eles!=0)
  {
    Arows =  n_upts_per_ele;
    Acols = n_fpts_per_ele;
    
    Brows = Acols;
    Bcols = n_fields*n_eles;
    
    Astride = Arows;
    Bstride = Brows;
    Cstride = Arows;
    
#ifdef _CPU
    
    if(opp_5_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      
      for (int i=0;i<n_dims;i++)
      {
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_5(i).get_ptr_cpu(),Astride,delta_disu_fpts.get_ptr_cpu(),Bstride,1.0,grad_disu_upts.get_ptr_cpu(0,0,0,i),Cstride);
      }
      
#elif defined _NO_BLAS
      for (int i=0;i<n_dims;i++) {
        dgemm(Arows,Bcols,Acols,1.0,1.0,opp_5(i).get_ptr_cpu(),delta_disu_fpts.get_ptr_cpu(),grad_disu_upts.get_ptr_cpu(0,0,0,i));
      }
      
#endif
    }
    else if(opp_5_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      
      // impelement
      
#endif
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_5 ... " << endl;
    }
    
    // Transform to physical space
    double detjac;
    double inv_detjac;
    double rx,ry,rz,sx,sy,sz,tx,ty,tz;
    double Xx,Xy,Xz,Yx,Yy,Yz,Zx,Zy,Zz;
    double ur,us,ut,uX,uY,uZ;
    
    for (int i=0;i<n_eles;i++)
    {
      for (int j=0;j<n_upts_per_ele;j++)
      {
        // Transform to static-physical domain
        detjac = detjac_upts(j,i);
        inv_detjac = 1.0/detjac;
        
        rx = JGinv_upts(0,0,j,i)*inv_detjac;
        ry = JGinv_upts(0,1,j,i)*inv_detjac;
        sx = JGinv_upts(1,0,j,i)*inv_detjac;
        sy = JGinv_upts(1,1,j,i)*inv_detjac;
        
        //physical gradient
        if(n_dims==2)
        {
          for(int k=0;k<n_fields;k++)
          {
            ur = grad_disu_upts(j,i,k,0);
            us = grad_disu_upts(j,i,k,1);
            
            grad_disu_upts(j,i,k,0) = ur*rx + us*sx;
            grad_disu_upts(j,i,k,1) = ur*ry + us*sy;
          }
        }
        if (n_dims==3)
        {
          rz = JGinv_upts(0,2,j,i)*inv_detjac;
          sz = JGinv_upts(1,2,j,i)*inv_detjac;
          
          tx = JGinv_upts(2,0,j,i)*inv_detjac;
          ty = JGinv_upts(2,1,j,i)*inv_detjac;
          tz = JGinv_upts(2,2,j,i)*inv_detjac;
          
          for (int k=0;k<n_fields;k++)
          {
            ur = grad_disu_upts(j,i,k,0);
            us = grad_disu_upts(j,i,k,1);
            ut = grad_disu_upts(j,i,k,2);
            
            grad_disu_upts(j,i,k,0) = ur*rx + us*sx + ut*tx;
            grad_disu_upts(j,i,k,1) = ur*ry + us*sy + ut*ty;
            grad_disu_upts(j,i,k,2) = ur*rz + us*sz + ut*tz;
          }
        }

        if (motion) {
          // Transform to dynamic-physical domain
          detjac = J_dyn_upts(j,i);
          inv_detjac = 1.0/detjac;

          Xx = JGinv_dyn_upts(0,0,j,i)*inv_detjac;
          Xy = JGinv_dyn_upts(0,1,j,i)*inv_detjac;
          Yx = JGinv_dyn_upts(1,1,j,i)*inv_detjac;
          Yy = JGinv_dyn_upts(1,1,j,i)*inv_detjac;

          //physical gradient
          if(n_dims==2)
          {
            for(int k=0;k<n_fields;k++)
            {
              uX = grad_disu_upts(j,i,k,0);
              uY = grad_disu_upts(j,i,k,1);

              grad_disu_upts(j,i,k,0) = uX*Xx + uY*Yx;
              grad_disu_upts(j,i,k,1) = uX*Xy + uY*Yy;
            }
          }
          if (n_dims==3)
          {
            Xz = JGinv_dyn_upts(j,i,0,2)*inv_detjac;
            Yz = JGinv_dyn_upts(j,i,1,2)*inv_detjac;

            Zx = JGinv_dyn_upts(j,i,2,0)*inv_detjac;
            Zy = JGinv_dyn_upts(j,i,2,1)*inv_detjac;
            Zz = JGinv_dyn_upts(j,i,2,2)*inv_detjac;

            for (int k=0;k<n_fields;k++)
            {
              uX = grad_disu_upts(j,i,k,0);
              uY = grad_disu_upts(j,i,k,1);
              uZ = grad_disu_upts(j,i,k,2);

              grad_disu_upts(j,i,k,0) = uX*Xx + uY*Yx + uZ*Zx;
              grad_disu_upts(j,i,k,1) = uX*Xy + uY*Yy + uZ*Zy;
              grad_disu_upts(j,i,k,2) = uX*Xz + uY*Yz + uZ*Zz;
            }
          }
        }
      }
    }
    
#endif
    
#ifdef _GPU
    
    if (opp_5_sparse==0)
    {
      for (int i=0;i<n_dims;i++)
      {
        cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_5(i).get_ptr_gpu(),Astride,delta_disu_fpts.get_ptr_gpu(),Bstride,1.0,grad_disu_upts.get_ptr_gpu(0,0,0,i),Cstride);
      }
    }
    else if (opp_5_sparse==1)
    {
      for (int i=0;i<n_dims;i++)
      {
        bespoke_SPMV(Arows,Acols,n_fields,n_eles,opp_5_ell_data(i).get_ptr_gpu(),opp_5_ell_indices(i).get_ptr_gpu(),opp_5_nnz_per_row(i),delta_disu_fpts.get_ptr_gpu(),grad_disu_upts.get_ptr_gpu(0,0,0,i),ele_type,order,1);
      }
    }
    
    transform_grad_disu_upts_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,grad_disu_upts.get_ptr_gpu(),detjac_upts.get_ptr_gpu(),J_dyn_upts.get_ptr_gpu(),JGinv_upts.get_ptr_gpu(),JGinv_dyn_upts.get_ptr_gpu(),run_input.equation,motion);
    
#endif
    
  }
  
  /*
   for (int i=0;i<n_fpts_per_ele;i++)
   for (int j=0;j<n_eles;j++)
   for (int k=0;k<n_fields;k++)
   {
   if (ele2global_ele(j)==53)
   {
   cout << setprecision(10)  << i << " " << ele2global_ele(j) << " " << k << " " << " " << delta_disu_fpts(i,j,k) << endl;
   }
   }
   */
  
  /*
   cout << "OUTPUT" << endl;
   #ifdef _GPU
   grad_disu_upts.cp_gpu_cpu();
   #endif
   
   for (int i=0;i<n_upts_per_ele;i++)
   for (int j=0;j<n_eles;j++)
   for (int k=0;k<n_fields;k++)
   for (int m=0;m<n_dims;m++)
   {
   if (ele2global_ele(j)==53)
   {
   cout << setprecision(10)  << i << " " << ele2global_ele(j) << " " << k << " " << m << " " << grad_disu_upts(i,j,k,m) << endl;
   }
   }
   */
}


// calculate corrected gradient of the discontinuous solution at flux points

void eles::extrapolate_corrected_gradient(void)
{
  if (n_eles!=0)
  {
    Arows =  n_fpts_per_ele;
    Acols = n_upts_per_ele;
    
    Brows = Acols;
    Bcols = n_fields*n_eles;
    
    Astride = Arows;
    Bstride = Brows;
    Cstride = Arows;
    
#ifdef _CPU
    
    if(opp_6_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      
      for (int i=0;i<n_dims;i++)
      {
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_6.get_ptr_cpu(),Astride,grad_disu_upts.get_ptr_cpu(0,0,0,i),Bstride,0.0,grad_disu_fpts.get_ptr_cpu(0,0,0,i),Cstride);
      }
      
#elif defined _NO_BLAS
      for (int i=0;i<n_dims;i++) {
        dgemm(Arows,Bcols,Acols,1.0,0.0,opp_6.get_ptr_cpu(),grad_disu_upts.get_ptr_cpu(0,0,0,i),grad_disu_fpts.get_ptr_cpu(0,0,0,i));
      }
#endif
    }
    else if(opp_6_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      
      // implement
      
#endif
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_6 ... " << endl;
    }
    
#endif
    
#ifdef _GPU
    
    if (opp_6_sparse==0)
    {
      for (int i=0;i<n_dims;i++)
      {
        cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_6.get_ptr_gpu(),Astride,grad_disu_upts.get_ptr_gpu(0,0,0,i),Bstride,0.0,grad_disu_fpts.get_ptr_gpu(0,0,0,i),Cstride);
      }
    }
    else if (opp_6_sparse==1)
    {
      for (int i=0;i<n_dims;i++)
      {
        bespoke_SPMV(Arows,Acols,n_fields,n_eles,opp_6_ell_data.get_ptr_gpu(),opp_6_ell_indices.get_ptr_gpu(),opp_6_nnz_per_row,grad_disu_upts.get_ptr_gpu(0,0,0,i),grad_disu_fpts.get_ptr_gpu(0,0,0,i),ele_type,order,0);
      }
    }
    
#endif
    
  }
  
  /*
   cout << "OUTPUT" << endl;
   #ifdef _GPU
   grad_disu_fpts.cp_gpu_cpu();
   #endif
   
   for (int i=0;i<n_fpts_per_ele;i++)
   for (int j=0;j<n_eles;j++)
   for (int k=0;k<n_fields;k++)
   for (int m=0;m<n_dims;m++)
   cout << setprecision(10)  << i << " " << j<< " " << k << " " << m << " " << grad_disu_fpts(i,j,k,m) << endl;
   */
}

/*! If at first RK step and using certain LES models, compute some model-related quantities.
 If using similarity or WALE-similarity (WSM) models, compute filtered solution and Leonard tensors.
 If using spectral vanishing viscosity (SVV) model, compute filtered solution. */

void eles::calc_sgs_terms(int in_disu_upts_from)
{
  if (n_eles!=0) {
    
    int i,j,k,l;
    int dim3;
    double diag, rsq;
    array <double> utemp(n_fields);
    
    /*! Filter solution */
    
    Arows =  n_upts_per_ele;
    Acols = n_upts_per_ele;
    Brows = Acols;
    Bcols = n_fields*n_eles;
    
    Astride = Arows;
    Bstride = Brows;
    Cstride = Arows;
    
#ifdef _CPU
    
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,filter_upts.get_ptr_cpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_cpu(),Bstride,0.0,disuf_upts.get_ptr_cpu(),Cstride);
    
#elif defined _NO_BLAS
    dgemm(Arows,Bcols,Acols,1.0,0.0,filter_upts.get_ptr_cpu(),disu_upts(in_disu_upts_from).get_ptr_cpu(),disuf_upts.get_ptr_cpu());
    
#else
    
    /*! slow matrix multiplication */
    for(i=0;i<n_upts_per_ele;i++) {
      for(j=0;j<n_eles;j++) {
        for(k=0;k<n_fields;k++) {
          disuf_upts(i,j,k) = 0.0;
          for(l=0;l<n_upts_per_ele;l++) {
            disuf_upts(i,j,k) += filter_upts(i,l)*disu_upts(in_disu_upts_from)(l,j,k);
          }
        }
      }
    }
    
#endif
    
    /*! Check for NaNs */
    for(i=0;i<n_upts_per_ele;i++)
      for(j=0;j<n_eles;j++)
        for(k=0;k<n_fields;k++)
          if(isnan(disuf_upts(i,j,k)))
            FatalError("nan in filtered solution");
    
    /*! If SVV model, copy filtered solution back to solution */
    if(sgs_model==3)
      for(i=0;i<n_upts_per_ele;i++)
        for(j=0;j<n_eles;j++)
          for(k=0;k<n_fields;k++)
            disu_upts(in_disu_upts_from)(i,j,k) = disuf_upts(i,j,k);
    
    /*! If Similarity model, compute product terms and Leonard tensors */
    else if(sgs_model==2 || sgs_model==4) {
      
      /*! third dimension of Lu, uu arrays */
      if(n_dims==2)      dim3 = 3;
      else if(n_dims==3) dim3 = 6;
      
      /*! Calculate velocity and energy product arrays uu, ue */
      for(i=0;i<n_upts_per_ele;i++) {
        for(j=0;j<n_eles;j++) {
          for(k=0;k<n_fields;k++) {
            utemp(k) = disu_upts(in_disu_upts_from)(i,j,k);
          }
          
          rsq = utemp(0)*utemp(0);
          
          /*! note that product arrays are symmetric */
          if(n_dims==2) {
            /*! velocity-velocity product */
            uu(i,j,0) = utemp(1)*utemp(1)/rsq;
            uu(i,j,1) = utemp(2)*utemp(2)/rsq;
            uu(i,j,2) = utemp(1)*utemp(2)/rsq;
            
            /*! velocity-energy product */
            utemp(3) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2))/utemp(0); // internal energy*rho
            
            ue(i,j,0) = utemp(1)*utemp(3)/rsq;
            ue(i,j,1) = utemp(2)*utemp(3)/rsq;
          }
          else if(n_dims==3) {
            /*! velocity-velocity product */
            uu(i,j,0) = utemp(1)*utemp(1)/rsq;
            uu(i,j,1) = utemp(2)*utemp(2)/rsq;
            uu(i,j,2) = utemp(3)*utemp(3)/rsq;
            uu(i,j,3) = utemp(1)*utemp(2)/rsq;
            uu(i,j,4) = utemp(1)*utemp(3)/rsq;
            uu(i,j,5) = utemp(2)*utemp(3)/rsq;
            
            /*! velocity-energy product */
            utemp(4) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2)+utemp(3)*utemp(3))/utemp(0); // internal energy*rho
            
            ue(i,j,0) = utemp(1)*utemp(4)/rsq;
            ue(i,j,1) = utemp(2)*utemp(4)/rsq;
            ue(i,j,2) = utemp(3)*utemp(4)/rsq;
          }
        }
      }
      
      /*! Filter products uu and ue */
      
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      
      Bcols = dim3*n_eles;
      
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,filter_upts.get_ptr_cpu(),Astride,uu.get_ptr_cpu(),Bstride,0.0,Lu.get_ptr_cpu(),Cstride);
      
      Bcols = n_dims*n_eles;
      
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,filter_upts.get_ptr_cpu(),Astride,ue.get_ptr_cpu(),Bstride,0.0,Le.get_ptr_cpu(),Cstride);
      
#elif defined _NO_BLAS
      
      Bcols = dim3*n_eles;
      
      dgemm(Arows,Bcols,Acols,1.0,0.0,filter_upts.get_ptr_cpu(),uu.get_ptr_cpu(),Lu.get_ptr_cpu());
      
      Bcols = n_dims*n_eles;
      
      dgemm(Arows,Bcols,Acols,1.0,0.0,filter_upts.get_ptr_cpu(),ue.get_ptr_cpu(),Le.get_ptr_cpu());
      
#else
      
      /*! slow matrix multiplication */
      for(i=0;i<n_upts_per_ele;i++) {
        for(j=0;j<n_eles;j++) {
          
          for(k=0;k<dim3;k++)
            for(l=0;l<n_upts_per_ele;l++)
              Lu(i,j,k) += filter_upts(i,l)*uu(l,j,k);
          
          for(k=0;k<n_dims;k++)
            for(l=0;l<n_upts_per_ele;l++)
              Le(i,j,k) += filter_upts(i,l)*ue(l,j,k);
          
        }
      }
      
#endif
      
      /*! Subtract product of unfiltered quantities from Leonard tensors */
      for(i=0;i<n_upts_per_ele;i++) {
        for(j=0;j<n_eles;j++) {
          
          // filtered solution
          for(k=0;k<n_fields;k++)
            utemp(k) = disuf_upts(i,j,k);
          
          rsq = utemp(0)*utemp(0);
          
          if(n_dims==2) {
            
            Lu(i,j,0) -= (utemp(1)*utemp(1))/rsq;
            Lu(i,j,1) -= (utemp(2)*utemp(2))/rsq;
            Lu(i,j,2) -= (utemp(1)*utemp(2))/rsq;
            
            diag = (Lu(i,j,0)+Lu(i,j,1))/3.0;
            
            // internal energy*rho
            utemp(3) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2))/utemp(0);
            
            Le(i,j,0) = (Le(i,j,0) - utemp(1)*utemp(3))/rsq;
            Le(i,j,1) = (Le(i,j,1) - utemp(2)*utemp(3))/rsq;
            
          }
          else if(n_dims==3) {
            
            Lu(i,j,0) -= (utemp(1)*utemp(1))/rsq;
            Lu(i,j,1) -= (utemp(2)*utemp(2))/rsq;
            Lu(i,j,2) -= (utemp(3)*utemp(3))/rsq;
            Lu(i,j,3) -= (utemp(1)*utemp(2))/rsq;
            Lu(i,j,4) -= (utemp(1)*utemp(3))/rsq;
            Lu(i,j,5) -= (utemp(2)*utemp(3))/rsq;
            
            diag = (Lu(i,j,0)+Lu(i,j,1)+Lu(i,j,2))/3.0;
            
            // internal energy*rho
            utemp(4) -= 0.5*(utemp(1)*utemp(1)+utemp(2)*utemp(2)+utemp(3)*utemp(3))/utemp(0);
            
            Le(i,j,0) = (Le(i,j,0) - utemp(1)*utemp(4))/rsq;
            Le(i,j,1) = (Le(i,j,1) - utemp(2)*utemp(4))/rsq;
            Le(i,j,2) = (Le(i,j,2) - utemp(3)*utemp(4))/rsq;
            
          }
          
          /*! subtract diagonal from Lu */
          for (k=0;k<n_dims;++k) Lu(i,j,k) -= diag;
          
        }
      }
    }
    
#endif
    
    /*! GPU version of the above */
#ifdef _GPU
    
    /*! Filter solution (CUDA BLAS library) */
    cublasDgemm('N','N',Arows,Bcols,Acols,1.0,filter_upts.get_ptr_gpu(),Astride,disu_upts(in_disu_upts_from).get_ptr_gpu(),Bstride,0.0,disuf_upts.get_ptr_gpu(),Cstride);
    
    /*! Check for NaNs */
    disuf_upts.cp_gpu_cpu();
    
    for(i=0;i<n_upts_per_ele;i++)
      for(j=0;j<n_eles;j++)
        for(k=0;k<n_fields;k++)
          if(isnan(disuf_upts(i,j,k)))
            FatalError("nan in filtered solution");
    
    /*! If Similarity model */
    if(sgs_model==2 || sgs_model==4) {
      
      /*! compute product terms uu, ue (pass flag=0 to wrapper function) */
      calc_similarity_model_kernel_wrapper(0, n_fields, n_upts_per_ele, n_eles, n_dims, disu_upts(in_disu_upts_from).get_ptr_gpu(), disuf_upts.get_ptr_gpu(), uu.get_ptr_gpu(), ue.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu());
      
      /*! third dimension of Lu, uu arrays */
      if(n_dims==2)
        dim3 = 3;
      else if(n_dims==3)
        dim3 = 6;
      
      Bcols = dim3*n_eles;
      
      /*! Filter product terms uu and ue */
      cublasDgemm('N','N',Arows,Bcols,Acols,1.0,filter_upts.get_ptr_gpu(),Astride,uu.get_ptr_gpu(),Bstride,0.0,Lu.get_ptr_gpu(),Cstride);
      
      Bcols = n_dims*n_eles;
      
      cublasDgemm('N','N',Arows,Bcols,Acols,1.0,filter_upts.get_ptr_gpu(),Astride,ue.get_ptr_gpu(),Bstride,0.0,Le.get_ptr_gpu(),Cstride);
      
      /*! compute Leonard tensors Lu, Le (pass flag=1 to wrapper function) */
      calc_similarity_model_kernel_wrapper(1, n_fields, n_upts_per_ele, n_eles, n_dims, disu_upts(in_disu_upts_from).get_ptr_gpu(), disuf_upts.get_ptr_gpu(), uu.get_ptr_gpu(), ue.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu());
      
    }
    
    /*! If SVV model, copy filtered solution back to original solution */
    else if(sgs_model==3) {
      for(i=0;i<n_upts_per_ele;i++) {
        for(j=0;j<n_eles;j++) {
          for(k=0;k<n_fields;k++) {
            disu_upts(in_disu_upts_from)(i,j,k) = disuf_upts(i,j,k);
          }
        }
      }
      /*! copy back to GPU */
      disu_upts(in_disu_upts_from).cp_cpu_gpu();
    }
    
#endif
    
  }
}

// calculate transformed discontinuous viscous flux at solution points

void eles::evaluate_viscFlux(int in_disu_upts_from)
{
  if (n_eles!=0)
  {
#ifdef _CPU
    
    int i,j,k,l,m;
    double detjac;

    for(i=0;i<n_eles;i++) {
      
      // Calculate viscous flux
      for(j=0;j<n_upts_per_ele;j++)
      {
        detjac = detjac_upts(j,i);
        
        // solution in static-physical domain
        for(k=0;k<n_fields;k++)
        {
          temp_u(k)=disu_upts(in_disu_upts_from)(j,i,k);
          
          // gradient in dynamic-physical domain
          for (m=0;m<n_dims;m++)
          {
            temp_grad_u(k,m) = grad_disu_upts(j,i,k,m);
          }
        }

        // Transform to dynamic-physical domain
        if (motion) {
          for (k=0; k<n_fields; k++) {
            temp_u(k) /= J_dyn_upts(j,i);
          }
        }

        if(n_dims==2)
        {
          calc_visf_2d(temp_u,temp_grad_u,temp_f);
        }
        else if(n_dims==3)
        {
          calc_visf_3d(temp_u,temp_grad_u,temp_f);
        }
        else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
        
        // If LES or wall model, calculate SGS viscous flux
        if(LES != 0 || wall_model != 0) {
          
          calc_sgsf_upts(temp_u,temp_grad_u,detjac,i,j,temp_sgsf);
          
          // Add SGS or wall flux to viscous flux
          for(k=0;k<n_fields;k++)
            for(l=0;l<n_dims;l++)
              temp_f(k,l) += temp_sgsf(k,l);
          
        }
        
        // If LES, add SGS flux to global array (needed for interface flux calc)
        if(LES > 0) {

          // Transfer back to static-phsycial domain
          if (motion) {
            temp_sgsf_ref.initialize_to_zero();
            for(k=0;k<n_fields;k++) {
              for(l=0;l<n_dims;l++) {
                for(m=0;m<n_dims;m++) {
                  temp_sgsf_ref(k,l)+=JGinv_dyn_upts(l,m,j,i)*temp_sgsf(k,m);
                }
              }
            }
            // Copy back to original flux array
            for (k=0; k<n_fields; k++) {
              for(l=0; l<n_dims; l++) {
                temp_sgsf(k,l) = temp_sgsf_ref(k,l);
              }
            }
          }

          // Transfer back to computational domain
          for(k=0;k<n_fields;k++) {
            for(l=0;l<n_dims;l++) {
              sgsf_upts(j,i,k,l) = 0.0;
              for(m=0;m<n_dims;m++) {
                sgsf_upts(j,i,k,l)+=JGinv_upts(l,m,j,i)*temp_sgsf(k,m);
              }
            }
          }
        }

        // Transfer back to static-phsycial domain
        if (motion) {
          temp_f_ref.initialize_to_zero();
          for(k=0;k<n_fields;k++) {
            for(l=0;l<n_dims;l++) {
              for(m=0;m<n_dims;m++) {
                temp_f_ref(k,l)+=JGinv_dyn_upts(l,m,j,i)*temp_f(k,m);
              }
            }
          }
          // Copy back to original flux array
          for(l=0; l<n_dims; l++) {
            for (k=0; k<n_fields; k++) {
              temp_f(k,l) = temp_f_ref(k,l);
            }
          }
        }
        
        // Transform viscous flux
        for(k=0;k<n_fields;k++)
        {
          for(l=0;l<n_dims;l++)
          {
            for(m=0;m<n_dims;m++)
            {
              tdisf_upts(j,i,k,l)+=JGinv_upts(l,m,j,i)*temp_f(k,m);
            }
          }
        }
      }
    }
#endif
    
#ifdef _GPU
    
    evaluate_viscFlux_gpu_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, ele_type, order, run_input.filter_ratio, LES, motion, sgs_model, wall_model, run_input.wall_layer_t, wall_distance.get_ptr_gpu(), twall.get_ptr_gpu(), Lu.get_ptr_gpu(), Le.get_ptr_gpu(), disu_upts(in_disu_upts_from).get_ptr_gpu(), tdisf_upts.get_ptr_gpu(), sgsf_upts.get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(), detjac_upts.get_ptr_gpu(), J_dyn_upts.get_ptr_gpu(), JGinv_upts.get_ptr_gpu(), JGinv_dyn_upts.get_ptr_gpu(), run_input.gamma, run_input.prandtl, run_input.rt_inf, run_input.mu_inf, run_input.c_sth, run_input.fix_vis, run_input.equation, run_input.diff_coeff, run_input.turb_model, run_input.c_v1, run_input.omega, run_input.prandtl_t);
    
#endif
    
  }
}

// Calculate SGS flux at solution points
void eles::calc_sgsf_upts(array<double>& temp_u, array<double>& temp_grad_u, double& detjac, int ele, int upt, array<double>& temp_sgsf)
{
  int i,j,k;
  int eddy, sim, wall;
  double Cs;
  double diag=0.0;
  double Smod=0.0;
  double ke=0.0;
  double Pr=0.5; // turbulent Prandtl number
  double delta, mu, mu_t, vol;
  double rho, inte, rt_ratio;
  array<double> u(n_dims);
  array<double> drho(n_dims), dene(n_dims), dke(n_dims), de(n_dims);
  array<double> dmom(n_dims,n_dims), du(n_dims,n_dims), S(n_dims,n_dims);
  
  // quantities for wall model
  array<double> norm(n_dims);
  array<double> tau(n_dims,n_dims);
  array<double> Mrot(n_dims,n_dims);
  array<double> temp(n_dims,n_dims);
  array<double> urot(n_dims);
  array<double> tw(n_dims);
  double y, qw, utau, yplus;
  
  // primitive variables
  rho = temp_u(0);
  for (i=0;i<n_dims;i++) {
    u(i) = temp_u(i)/rho;
    ke += 0.5*pow(u(i),2);
  }
  inte = temp_u(n_fields-1)/rho - ke;
  
  // fluid properties
  rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
  mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
  mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);
  
  // Initialize SGS flux array to zero
  zero_array(temp_sgsf);
  
  // Compute SGS flux using wall model if sufficiently close to solid boundary
  wall = 0;
  
  if(wall_model != 0) {
    
    // Magnitude of wall distance vector
    y = 0.0;
    for (i=0;i<n_dims;i++)
      y += wall_distance(upt,ele,i)*wall_distance(upt,ele,i);
    
    y = sqrt(y);
    
    // get subgrid momentum flux at previous timestep
    //utau = 0.0;
    for (i=0;i<n_dims;i++) {
      tw(i) = twall(upt,ele,i+1);
      //utau += tw(i)*tw(i);
    }
    // shear velocity
    //utau = pow((utau/rho/rho),0.25);
    
    // Wall distance in wall units
    //yplus = y*rho*utau/mu;
    
    if(y < run_input.wall_layer_t) wall = 1;
    //if(yplus < 100.0) wall = 1;
    //cout << "tw, y, y+ " << tw(0) << ", " << y << ", " << yplus << endl;
  }
  
  // calculate SGS flux from a wall model
  if(wall) {
    
    for (i=0;i<n_dims;i++) {
      // Get approximate normal from wall distance vector
      norm(i) = wall_distance(upt,ele,i)/y;
    }
    
    // subgrid energy flux from previous timestep
    qw = twall(upt,ele,n_fields-1);
    
    // Calculate local rotation matrix
    Mrot = calc_rotation_matrix(norm);
    
    // Rotate velocity to surface
    if(n_dims==2) {
      urot(0) = u(0)*Mrot(0,1)+u(1)*Mrot(1,1);
      urot(1) = 0.0;
    }
    else {
      urot(0) = u(0)*Mrot(0,1)+u(1)*Mrot(1,1)+u(2)*Mrot(2,1);
      urot(1) = u(0)*Mrot(0,2)+u(1)*Mrot(1,2)+u(2)*Mrot(2,2);
      urot(2) = 0.0;
    }
    
    // Calculate wall shear stress
    calc_wall_stress(rho,urot,inte,mu,run_input.prandtl,run_input.gamma,y,tw,qw);
    
    // correct the sign of wall shear stress and wall heat flux? - see SD3D
    
    // Set arrays for next timestep
    for(i=0;i<n_dims;++i) twall(upt,ele,i+1) = tw(i); // momentum flux
    
    twall(upt,ele,0)          = 0.0; // density flux
    twall(upt,ele,n_fields-1) = qw;  // energy flux
    
    // populate ndims*ndims rotated stress array
    zero_array(tau);
    
    for(i=0;i<n_dims-1;i++) tau(i+1,0) = tau(0,i+1) = tw(i);
    
    // rotate stress array back to Cartesian coordinates
    zero_array(temp);
    for(i=0;i<n_dims;++i)
      for(j=0;j<n_dims;++j)
        for(k=0;k<n_dims;++k)
          temp(i,j) += tau(i,k)*Mrot(k,j);
    
    zero_array(tau);
    for(i=0;i<n_dims;++i)
      for(j=0;j<n_dims;++j)
        for(k=0;k<n_dims;++k)
          tau(i,j) += Mrot(k,i)*temp(k,j);
    
    // set SGS fluxes
    for(i=0;i<n_dims;i++) {
      
      // density
      temp_sgsf(0,i) = 0.0;
      
      // velocity
      for(j=0;j<n_dims;j++) {
        temp_sgsf(j+1,i) = 0.5*(tau(i,j)+tau(j,i));
      }
      
      // energy
      temp_sgsf(n_fields-1,i) = qw*norm(i);
    }
  }
  
  // Free-stream SGS flux
  else {
    
    // Set wall shear stress to 0 to prevent NaNs
    if(wall_model != 0) for(i=0;i<n_dims;++i) twall(upt,ele,i) = 0.0;
    
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
    else {
      FatalError("SGS model not implemented");
    }
    
    if(eddy==1) {
      
      // Delta is the cutoff length-scale representing local grid resolution.
      
      // OPTION 1. Approx resolution in 1D element. Interval is [-1:1]
      // Appropriate for quads, hexes and tris. Not sure about tets.
      //dlt = 2.0/order;
      
      // OPTION 2. Deardorff definition (Deardorff, JFM 1970)
      vol = (*this).calc_ele_vol(detjac);
      delta = run_input.filter_ratio*pow(vol,1./n_dims)/(order+1.);
      
      // OPTION 3. Suggested by Bardina, AIAA 1980:
      // delta = sqrt((dx^2+dy^2+dz^2)/3)
      
      // Filtered solution gradient
      for (i=0;i<n_dims;i++) {
        drho(i) = temp_grad_u(0,i); // density gradient
        dene(i) = temp_grad_u(n_fields-1,i); // energy gradient
        
        for (j=1;j<n_fields-1;j++) {
          dmom(i,j-1) = temp_grad_u(j,i); // momentum gradients
        }
      }
      
      // Velocity and energy gradients
      for (i=0;i<n_dims;i++) {
        dke(i) = ke*drho(i);
        
        for (j=0;j<n_dims;j++) {
          du(i,j) = (dmom(i,j)-u(j)*drho(j))/rho;
          dke(i) += rho*u(j)*du(i,j);
        }
        
        de(i) = (dene(i)-dke(i)-drho(i)*inte)/rho;
      }
      
      // Strain rate tensor
      for (i=0;i<n_dims;i++) {
        for (j=0;j<n_dims;j++) {
          S(i,j) = (du(i,j)+du(j,i))/2.0;
        }
        diag += S(i,i)/3.0;
      }
      
      // Subtract diag
      for (i=0;i<n_dims;i++) S(i,i) -= diag;
      
      // Strain modulus
      for (i=0;i<n_dims;i++)
        for (j=0;j<n_dims;j++)
          Smod += 2.0*S(i,j)*S(i,j);
      
      Smod = sqrt(Smod);
      
      // Eddy viscosity
      
      // Smagorinsky model
      if(sgs_model==0) {
        
        Cs=0.1;
        mu_t = rho*Cs*Cs*delta*delta*Smod;
        
      }
      
      //  Wall-Adapting Local Eddy-viscosity (WALE) SGS Model
      //
      //  NICOUD F., DUCROS F.: "Subgrid-Scale Stress Modelling Based on the Square
      //                         of the Velocity Gradient Tensor"
      //  Flow, Turbulence and Combustion 62: 183-200, 1999.
      //
      //                                            (sqij*sqij)^3/2
      //  Output: mu_t = rho*Cs^2*delta^2 * -----------------------------
      //                                     (Sij*Sij)^5/2+(sqij*sqij)^5/4
      //
      //  Typically Cw = 0.5.
      
      else if(sgs_model==1 || sgs_model==2) {
        
        Cs=0.5;
        double num=0.0;
        double denom=0.0;
        double eps=1.e-12;
        array<double> Sq(n_dims,n_dims);
        diag = 0.0;
        
        // Square of gradient tensor
        // This needs optimising!
        for (i=0;i<n_dims;i++) {
          for (j=0;j<n_dims;j++) {
            Sq(i,j) = 0.0;
            for (k=0;k<n_dims;++k) {
              Sq(i,j) += (du(i,k)*du(k,j)+du(j,k)*du(k,i))/2.0;
            }
            diag += du(i,j)*du(j,i)/3.0;
          }
        }
        
        // Subtract diag
        for (i=0;i<n_dims;i++) Sq(i,i) -= diag;
        
        // Numerator and denominator
        for (i=0;i<n_dims;i++) {
          for (j=0;j<n_dims;j++) {
            num += Sq(i,j)*Sq(i,j);
            denom += S(i,j)*S(i,j);
          }
        }
        
        denom = pow(denom,2.5) + pow(num,1.25);
        num = pow(num,1.5);
        mu_t = rho*Cs*Cs*delta*delta*num/(denom+eps);
      }
      
      // Add eddy-viscosity term to SGS fluxes
      for (j=0;j<n_dims;j++) {
        temp_sgsf(0,j) = 0.0; // Density flux
        temp_sgsf(n_fields-1,j) = -1.0*run_input.gamma*mu_t/Pr*de(j); // Energy flux
        
        for (i=1;i<n_fields-1;i++) {
          temp_sgsf(i,j) = -2.0*mu_t*S(i-1,j); // Velocity flux
        }
      }
    }
    
    // Add similarity term to SGS fluxes if WSM or Similarity model
    if(sim==1) {
      for (j=0;j<n_dims;j++) {
        temp_sgsf(0,j) += 0.0; // Density flux
        temp_sgsf(n_fields-1,j) += run_input.gamma*rho*Le(upt,ele,j); // Energy flux
      }
      
      // Momentum fluxes
      if(n_dims==2) {
        temp_sgsf(1,0) += rho*Lu(upt,ele,0);
        temp_sgsf(1,1) += rho*Lu(upt,ele,2);
        temp_sgsf(2,0) += temp_sgsf(1,1);
        temp_sgsf(2,1) += rho*Lu(upt,ele,1);
      }
      else if(n_dims==3) {
        temp_sgsf(1,0) += rho*Lu(upt,ele,0);
        temp_sgsf(1,1) += rho*Lu(upt,ele,3);
        temp_sgsf(1,2) += rho*Lu(upt,ele,4);
        temp_sgsf(2,0) += temp_sgsf(1,1);
        temp_sgsf(2,1) += rho*Lu(upt,ele,1);
        temp_sgsf(2,2) += rho*Lu(upt,ele,5);
        temp_sgsf(3,0) += temp_sgsf(1,2);
        temp_sgsf(3,1) += temp_sgsf(2,2);
        temp_sgsf(3,2) += rho*Lu(upt,ele,2);
      }
    }
  }
}


// calculate source term for SA turbulence model at solution points
void eles::calc_src_upts_SA(int in_disu_upts_from)
{
  if (n_eles!=0)
  {
#ifdef _CPU

    int i,j,k,l,m;

    for(i=0; i<n_eles; i++) {
      for(j=0; j<n_upts_per_ele; j++) {

        // physical solution
        for(k=0; k<n_fields; k++) {
          temp_u(k)=disu_upts(in_disu_upts_from)(j,i,k);
        }

        // physical gradient
        for(k=0; k<n_fields; k++) {
          for (m=0; m<n_dims; m++) {
            temp_grad_u(k,m) = grad_disu_upts(j,i,k,m);
          }
        }

        // source term
        if(n_dims==2)
          calc_source_SA_2d(temp_u, temp_grad_u, wall_distance_mag(j,i), src_upts(j,i,n_fields-1));
        else if(n_dims==3)
          calc_source_SA_3d(temp_u, temp_grad_u, wall_distance_mag(j,i), src_upts(j,i,n_fields-1));
        else
          cout << "ERROR: Invalid number of dimensions ... " << endl;
      }
    }

#endif

#ifdef _GPU
    calc_src_upts_SA_gpu_kernel_wrapper(n_upts_per_ele, n_dims, n_fields, n_eles, disu_upts(in_disu_upts_from).get_ptr_gpu(), grad_disu_upts.get_ptr_gpu(), wall_distance_mag.get_ptr_gpu(), src_upts.get_ptr_gpu(), run_input.gamma, run_input.prandtl, run_input.rt_inf, run_input.mu_inf, run_input.c_sth, run_input.fix_vis, run_input.c_v1, run_input.c_v2, run_input.c_v3, run_input.c_b1, run_input.c_b2, run_input.c_w2, run_input.c_w3, run_input.omega, run_input.Kappa);
#endif

  }
}


/*! If using a RANS or LES near-wall model, calculate distance
 of each solution point to nearest no-slip wall by a brute-force method */

void eles::calc_wall_distance(int n_seg_noslip_inters, int n_tri_noslip_inters, int n_quad_noslip_inters, array< array<double> > loc_noslip_bdy)
{
  if(n_eles!=0)
  {
    int i,j,k,m,n,p;
    int n_fpts_per_inter_seg = order+1;
    int n_fpts_per_inter_tri = (order+2)*(order+1)/2;
    int n_fpts_per_inter_quad = (order+1)*(order+1);
    double dist;
    double distmin;
    array<double> pos(n_dims);
    array<double> pos_bdy(n_dims);
    array<double> vec(n_dims);
    array<double> vecmin(n_dims);
    
    // hold our breath and go round the brute-force loop...
    for (i=0;i<n_eles;++i) {
      for (j=0;j<n_upts_per_ele;++j) {
        
        // get coords of current solution point
        calc_pos_upt(j,i,pos);
        
        // initialize wall distance
        distmin = 1e20;
        
        // line segment boundaries
        for (k=0;k<n_seg_noslip_inters;++k) {
          
          for (m=0;m<n_fpts_per_inter_seg;++m) {
            
            dist = 0.0;
            // get coords of boundary flux point
            for (n=0;n<n_dims;++n) {
              pos_bdy(n) = loc_noslip_bdy(0)(m,k,n);
              vec(n) = pos(n) - pos_bdy(n);
              dist += vec(n)*vec(n);
            }
            dist = sqrt(dist);
            
            // update shortest vector
            if (dist < distmin) {
              for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
              distmin = dist;
            }
          }
        }
        
        // tri boundaries
        for (k=0;k<n_tri_noslip_inters;++k) {
          
          for (m=0;m<n_fpts_per_inter_tri;++m) {
            
            dist = 0.0;
            // get coords of boundary flux point
            for (n=0;n<n_dims;++n) {
              pos_bdy(n) = loc_noslip_bdy(1)(m,k,n);
              vec(n) = pos(n) - pos_bdy(n);
              dist += vec(n)*vec(n);
            }
            dist = sqrt(dist);
            
            // update shortest vector
            if (dist < distmin) {
              for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
              distmin = dist;
            }
          }
        }
        
        // quad boundaries
        for (k=0;k<n_quad_noslip_inters;++k) {
          
          for (m=0;m<n_fpts_per_inter_quad;++m) {
            
            dist = 0.0;
            // get coords of boundary flux point
            for (n=0;n<n_dims;++n) {
              pos_bdy(n) = loc_noslip_bdy(2)(m,k,n);
              vec(n) = pos(n) - pos_bdy(n);
              dist += vec(n)*vec(n);
            }
            dist = sqrt(dist);
            
            // update shortest vector
            if (dist < distmin) {
              for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
              distmin = dist;
            }
          }
        }
        for (n=0;n<n_dims;++n) wall_distance(j,i,n) = vecmin(n);

        if (run_input.turb_model > 0) {
          wall_distance_mag(j,i) = distmin;
        }
      }
    }
  }
}

#ifdef _MPI

void eles::calc_wall_distance_parallel(array<int> n_seg_inters_array, array<int> n_tri_inters_array, array<int> n_quad_inters_array, array< array<double> > loc_noslip_bdy_global, int nproc)
{
  if(n_eles!=0)
  {
    int i,j,k,m,n,p;
    int n_fpts_per_inter_seg = order+1;
    int n_fpts_per_inter_tri = (order+2)*(order+1)/2;
    int n_fpts_per_inter_quad = (order+1)*(order+1);
    double dist;
    double distmin;
    array<double> pos(n_dims);
    array<double> pos_bdy(n_dims);
    array<double> vec(n_dims);
    array<double> vecmin(n_dims);
    
    // hold our breath and go round the brute-force loop...
    for (i=0;i<n_eles;++i) {
      for (j=0;j<n_upts_per_ele;++j) {
        
        // get coords of current solution point
        calc_pos_upt(j,i,pos);
        
        // initialize wall distance
        distmin = 1e20;
        
        // loop over all partitions
        for (p=0;p<nproc;++p) {
          
          // line segment boundaries
          for (k=0;k<n_seg_inters_array(p);++k) {
            
            for (m=0;m<n_fpts_per_inter_seg;++m) {
              
              dist = 0.0;
              // get coords of boundary flux point
              for (n=0;n<n_dims;++n) {
                pos_bdy(n) = loc_noslip_bdy_global(0)(m,k,p*n_dims+n);
                vec(n) = pos(n) - pos_bdy(n);
                dist += vec(n)*vec(n);
              }
              dist = sqrt(dist);
              
              // update shortest vector
              if (dist < distmin) {
                for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
                distmin = dist;
              }
            }
          }
          
          // tri boundaries
          for (k=0;k<n_tri_inters_array(p);++k) {
            
            for (m=0;m<n_fpts_per_inter_tri;++m) {
              
              dist = 0.0;
              // get coords of boundary flux point
              for (n=0;n<n_dims;++n) {
                pos_bdy(n) = loc_noslip_bdy_global(1)(m,k,p*n_dims+n);
                vec(n) = pos(n) - pos_bdy(n);
                dist += vec(n)*vec(n);
              }
              dist = sqrt(dist);
              
              // update shortest vector
              if (dist < distmin) {
                for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
                distmin = dist;
              }
            }
          }
          
          // quad boundaries
          for (k=0;k<n_quad_inters_array(p);++k) {
            
            for (m=0;m<n_fpts_per_inter_quad;++m) {
              
              dist = 0.0;
              // get coords of boundary flux point
              for (n=0;n<n_dims;++n) {
                pos_bdy(n) = loc_noslip_bdy_global(2)(m,k,p*n_dims+n);
                vec(n) = pos(n) - pos_bdy(n);
                dist += vec(n)*vec(n);
              }
              dist = sqrt(dist);
              
              // update shortest vector
              if (dist < distmin) {
                for (n=0;n<n_dims;++n) vecmin(n) = vec(n);
                distmin = dist;
              }
            }
          }
        }
        
        for (n=0;n<n_dims;++n) wall_distance(j,i,n) = vecmin(n);

        if (run_input.turb_model > 0) {
          wall_distance_mag(j,i) = distmin;
        }
      }
    }
  }
}

#endif

array<double> eles::calc_rotation_matrix(array<double>& norm)
{
  array <double> mrot(n_dims,n_dims);
  double nn;
  
  // Create rotation matrix
  if(n_dims==2) {
    if(abs(norm(1)) > 0.7) {
      mrot(0,0) = norm(0);
      mrot(1,0) = norm(1);
      mrot(0,1) = norm(1);
      mrot(1,1) = -norm(0);
    }
    else {
      mrot(0,0) = -norm(0);
      mrot(1,0) = -norm(1);
      mrot(0,1) = norm(1);
      mrot(1,1) = -norm(0);
    }
  }
  else if(n_dims==3) {
    if(abs(norm(2)) > 0.7) {
      nn = sqrt(norm(1)*norm(1)+norm(2)*norm(2));
      
      mrot(0,0) = norm(0)/nn;
      mrot(1,0) = norm(1)/nn;
      mrot(2,0) = norm(2)/nn;
      mrot(0,1) = 0.0;
      mrot(1,1) = -norm(2)/nn;
      mrot(2,1) = norm(1)/nn;
      mrot(0,2) = nn;
      mrot(1,2) = -norm(0)*norm(1)/nn;
      mrot(2,2) = -norm(0)*norm(2)/nn;
    }
    else {
      nn = sqrt(norm(0)*norm(0)+norm(1)*norm(1));
      
      mrot(0,0) = norm(0)/nn;
      mrot(1,0) = norm(1)/nn;
      mrot(2,0) = norm(2)/nn;
      mrot(0,1) = norm(1)/nn;
      mrot(1,1) = -norm(0)/nn;
      mrot(2,1) = 0.0;
      mrot(0,2) = norm(0)*norm(2)/nn;
      mrot(1,2) = norm(1)*norm(2)/nn;
      mrot(2,2) = -nn;
    }
  }
  
  return mrot;
}

void eles::calc_wall_stress(double rho, array<double>& urot, double ene, double mu, double Pr, double gamma, double y, array<double>& tau_wall, double q_wall)
{
  double eps = 1.e-10;
  double Rey, Rey_c, u, uplus, utau, tw, qw;
  double Pr_t = 0.9;
  double c0;
  double ymatch = 11.8;
  int i,j;
  
  // Magnitude of surface velocity
  u = 0.0;
  for(i=0;i<n_dims;++i) u += urot(i)*urot(i);
  
  u = sqrt(u);
  
  if(u > eps) {
    
    /*! Simple power-law wall model Werner and Wengle (1991)
     
     u+ = y+               for y+ < 11.8
     u+ = 8.3*(y+)^(1/7)   for y+ > 11.8
     */
    
    if(run_input.wall_model == 1) {
      
      Rey_c = ymatch*ymatch;
      Rey = rho*u*y/mu;
      
      if(Rey < Rey_c) uplus = sqrt(Rey);
      else            uplus = pow(8.3,0.875)*pow(Rey,0.125);
      
      utau = u/uplus;
      tw = rho*utau*utau;
      
      for (i=0;i<n_dims;i++) tau_wall(i) = tw*urot(i)/u;
      
      // Wall heat flux
      if(Rey < Rey_c) q_wall = ene*gamma*tw / (Pr * u);
      else            q_wall = ene*gamma*tw / (Pr * (u + utau * sqrt(Rey_c) * (Pr/Pr_t-1.0)));
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
    
    else if(run_input.wall_model == 2) {
      
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
      phi = rho*y/mu;
      Rey0 = u*phi;
      utau = 0.0;
      for (i=0;i<n_dims;i++)
        utau += tau_wall(i)*tau_wall(i);
      
      utau = pow((utau/rho/rho),0.25);
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
          yplusL = yplus;
          ReyL = Rey-Rey0;
          
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
        Rey = wallfn_br(yplus,A,B,E,kappa);
        
        if(Rey > eps) utau = u*yplus/Rey;
        else          utau = 0.0;
        yplus = utau*phi;
      }
      
      tw = rho*utau*utau;
      
      // why different to WW model?
      for (i=0;i<n_dims;i++) tau_wall(i) = abs(tw*urot(i)/u);
      
      // Wall heat flux
      if(yplus <= ymatch) q_wall = ene*gamma*tw / (Pr * u);
      else                q_wall = ene*gamma*tw / (Pr * (u + utau * ymatch * (Pr/Pr_t-1.0)));
    }
  }
  
  // if velocity is 0
  else {
    for (i=0;i<n_dims;i++) tau_wall(i) = 0.0;
    q_wall = 0.0;
  }
}

double eles::wallfn_br(double yplus, double A, double B, double E, double kappa) {
  double Rey;
  
  if     (yplus < 0.5)  Rey = yplus*yplus;
  else if(yplus > 30.0) Rey = yplus*log(E*yplus)/kappa;
  else                  Rey = yplus*(A*log(yplus)+B);
  
  return Rey;
}

/*! Calculate SGS flux at solution points */
void eles::evaluate_sgsFlux(void)
{
  if (n_eles!=0) {
    
    /*!
     Performs C = (alpha*A*B) + (beta*C) where: \n
     alpha = 1.0 \n
     beta = 0.0 \n
     A = opp_0 \n
     B = sgsf_upts \n
     C = sgsf_fpts
     */
    
    Arows =  n_fpts_per_ele;
    Acols = n_upts_per_ele;
    
    Brows = Acols;
    Bcols = n_fields*n_eles;
    
    Astride = Arows;
    Bstride = Brows;
    Cstride = Arows;
    
#ifdef _CPU
    
    if(opp_0_sparse==0) // dense
    {
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
      
      for (int i=0;i<n_dims;i++) {
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Arows,Bcols,Acols,1.0,opp_0.get_ptr_cpu(),Astride,sgsf_upts.get_ptr_cpu(0,0,0,i),Bstride,0.0,sgsf_fpts.get_ptr_cpu(0,0,0,i),Cstride);
      }
      
#elif defined _NO_BLAS
      for (int i=0;i<n_dims;i++) {
        dgemm(Arows,Bcols,Acols,1.0,0.0,opp_0.get_ptr_cpu(),sgsf_upts.get_ptr_cpu(0,0,0,i),sgsf_fpts.get_ptr_cpu(0,0,0,i));
      }
      
#endif
    }
    else if(opp_0_sparse==1) // mkl blas four-array csr format
    {
#if defined _MKL_BLAS
      
      for (int i=0;i<n_dims;i++) {
        mkl_dcsrmm(&transa, &n_fpts_per_ele, &n_fields_mul_n_eles, &n_upts_per_ele, &one, matdescra, opp_0_data.get_ptr_cpu(), opp_0_cols.get_ptr_cpu(), opp_0_b.get_ptr_cpu(), opp_0_e.get_ptr_cpu(), sgsf_upts.get_ptr_cpu(0,0,0,i), &n_upts_per_ele, &zero, sgsf_fpts.get_ptr_cpu(0,0,0,i), &n_fpts_per_ele);
      }
      
#endif
    }
    else { cout << "ERROR: Unknown storage for opp_0 ... " << endl; }
    
#endif
    
#ifdef _GPU
    
    if(opp_0_sparse==0)
    {
      for (int i=0;i<n_dims;i++) {
        cublasDgemm('N','N',Arows,Bcols,Acols,1.0,opp_0.get_ptr_gpu(),Astride,sgsf_upts.get_ptr_gpu(0,0,0,i),Bstride,0.0,sgsf_fpts.get_ptr_gpu(0,0,0,i),Cstride);
      }
    }
    else if (opp_0_sparse==1)
    {
      for (int i=0;i<n_dims;i++) {
        bespoke_SPMV(n_fpts_per_ele, n_upts_per_ele, n_fields, n_eles, opp_0_ell_data.get_ptr_gpu(), opp_0_ell_indices.get_ptr_gpu(), opp_0_nnz_per_row, sgsf_upts.get_ptr_gpu(0,0,0,i), sgsf_fpts.get_ptr_gpu(0,0,0,i), ele_type, order, 0);
      }
    }
    else
    {
      cout << "ERROR: Unknown storage for opp_0 ... " << endl;
    }
#endif
  }
}

// sense shock and filter (for concentration method) - only on GPUs

void eles::shock_capture_concentration(int in_disu_upts_from)
{
  if (n_eles!=0){
    #ifdef _GPU
      shock_capture_concentration_gpu_kernel_wrapper(n_eles, n_upts_per_ele, n_fields, order, ele_type, run_input.artif_type, run_input.s0, run_input.kappa, disu_upts(in_disu_upts_from).get_ptr_gpu(), inv_vandermonde.get_ptr_gpu(), inv_vandermonde2D.get_ptr_gpu(), vandermonde2D.get_ptr_gpu(), concentration_array.get_ptr_gpu(), sensor.get_ptr_gpu(), sigma.get_ptr_gpu());
    #endif

    #ifdef _CPU
        shock_capture_concentration_cpu(n_eles, n_upts_per_ele, n_fields, order, ele_type, run_input.artif_type, run_input.s0, run_input.kappa, disu_upts(in_disu_upts_from).get_ptr_cpu(), inv_vandermonde.get_ptr_cpu(), inv_vandermonde2D.get_ptr_cpu(), vandermonde2D.get_ptr_cpu(), concentration_array.get_ptr_cpu(), sensor.get_ptr_cpu(), sigma.get_ptr_cpu());
    #endif
  }
}

void eles::shock_capture_concentration_cpu(int in_n_eles, int in_n_upts_per_ele, int in_n_fields, int in_order, int in_ele_type, int in_artif_type, double s0, double kappa, double* in_disu_upts_ptr, double* in_inv_vandermonde_ptr, double* in_inv_vandermonde2D_ptr, double* in_vandermonde2D_ptr, double* concentration_array_ptr, double* out_sensor, double* sigma)
{
    int stride = in_n_upts_per_ele*in_n_eles;
    double tmp_sensor = 0;

    double nodal_rho[8];  // Array allocated so that it can handle upto p=7
    double modal_rho[8];
    double uE[8];
    double temp;
    double p = 3;	// exponent in concentration method
    double J = 0.15;
    int shock_found = 0;

    if(in_n_eles!=0){
        // X-slices
        for(int m=0; m<in_n_eles; m++)
        {
          tmp_sensor = 0;
            for(int i=0; i<in_order+1; i++)
            {
                for(int j=0; j<in_order+1; j++){
                    nodal_rho[j] = in_disu_upts_ptr[m*in_n_upts_per_ele + i*(in_order+1) + j];
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
                    temp = pow(uE[j],p)*pow(in_order+1,p/2);

                    if(temp >= J)
                        shock_found++;

                    if(temp > tmp_sensor)
                        tmp_sensor = temp;
                }

            }

            // Y-slices
            for(int i=0; i<in_order+1; i++)
            {
                for(int j=0; j<in_order+1; j++){
                    nodal_rho[j] = in_disu_upts_ptr[m*in_n_upts_per_ele + j*(in_order+1) + i];
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
                    temp = pow(abs(uE[j]),p)*pow(in_order+1,p/2);

                    if(temp >= J)
                        shock_found++;

                    if(temp > tmp_sensor)
                        tmp_sensor = temp;
                }
            }

            out_sensor[m] = tmp_sensor;

            /* -------------------------------------------------------------------------------------- */
            /* Exponential modal filter */

            if(tmp_sensor > s0 + kappa && in_artif_type == 1) {
                double nodal_sol[36];
                double modal_sol[36];

                for(int k=0; k<in_n_fields; k++) {

                    for(int i=0; i<in_n_upts_per_ele; i++){
                        nodal_sol[i] = in_disu_upts_ptr[m*in_n_upts_per_ele + k*stride + i];
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

                        in_disu_upts_ptr[m*in_n_upts_per_ele + k*stride + i] = nodal_sol[i];
                    }
                }
            }
        }
    }
}

// get the type of element

int eles::get_ele_type(void)
{
  return ele_type;
}

// get number of elements

int eles::get_n_eles(void)
{
  return n_eles;
}

// get number of ppts_per_ele
int eles::get_n_ppts_per_ele(void)
{
  return n_ppts_per_ele;
}

// get number of peles_per_ele
int eles::get_n_peles_per_ele(void)
{
  return n_peles_per_ele;
}

// get number of verts_per_ele
int eles::get_n_verts_per_ele(void)
{
  return n_verts_per_ele;
}

// get number of elements

int eles::get_n_dims(void)
{
  return n_dims;
}

// get number of solution fields

int eles::get_n_fields(void)
{
  return n_fields;
}

// get number of solutions points per element

int eles::get_n_upts_per_ele(void)
{
  return n_upts_per_ele;
}

// set the shape array
void eles::set_shape(int in_max_n_spts_per_ele)
{
  shape.setup(n_dims,in_max_n_spts_per_ele,n_eles);
  shape_dyn.setup(n_dims,in_max_n_spts_per_ele,n_eles,5);
  //shape_dyn_old.setup(n_dims,in_max_n_spts_per_ele,n_eles,4);
  n_spts_per_ele.setup(n_eles);
}

// set a shape node

void eles::set_shape_node(int in_spt, int in_ele, array<double>& in_pos)
{
  for(int i=0;i<n_dims;i++)
  {
    shape(i,in_spt,in_ele)=in_pos(i);
    for (int j=0; j<5; j++)
      shape_dyn(i,in_spt,in_ele,j)=in_pos(i);
  }
}

void eles::set_dynamic_shape_node(int in_spt, int in_ele, array<double> &in_pos)
{
  for(int i=0;i<n_dims;i++) {
    shape_dyn(i,in_spt,in_ele)=in_pos(i);
  }
}

void eles::set_rank(int in_rank)
{
  rank = in_rank;
}

// set bc type
void eles::set_bctype(int in_ele,int in_inter, int in_bctype)
{
  bctype(in_ele, in_inter) = in_bctype;
}

// set number of shape points

void eles::set_n_spts(int in_ele, int in_n_spts)
{
  n_spts_per_ele(in_ele) = in_n_spts;
  
  // Allocate storage for the s_nodal_basis
  
  d_nodal_s_basis.setup(in_n_spts,n_dims);
  
  int n_comp;
  if(n_dims == 2)
    n_comp = 3;
  else if(n_dims == 3)
    n_comp = 6;  
}

// set global element number

void eles::set_ele2global_ele(int in_ele, int in_global_ele)
{
  ele2global_ele(in_ele) = in_global_ele;
}


// set opp_0 (transformed discontinuous solution at solution points to transformed discontinuous solution at flux points)

void eles::set_opp_0(int in_sparse)
{
  int i,j,k;
  
  array<double> loc(n_dims);
  
  opp_0.setup(n_fpts_per_ele,n_upts_per_ele);
  
  for(i=0;i<n_upts_per_ele;i++)
  {
    for(j=0;j<n_fpts_per_ele;j++)
    {
      for(k=0;k<n_dims;k++)
      {
        loc(k)=tloc_fpts(k,j);
      }
      
      opp_0(j,i)=eval_nodal_basis(i,loc);
    }
  }
  
#ifdef _GPU
  opp_0.cp_cpu_gpu();
#endif
  
  //cout << "opp_0" << endl;
  //cout << "ele_type=" << ele_type << endl;
  //opp_0.print();
  //cout << endl;
  
  if(in_sparse==0)
  {
    opp_0_sparse=0;
  }
  else if(in_sparse==1)
  {
    opp_0_sparse=1;
    
#ifdef _CPU
    array_to_mklcsr(opp_0,opp_0_data,opp_0_cols,opp_0_b,opp_0_e);
#endif
    
#ifdef _GPU
    array_to_ellpack(opp_0, opp_0_ell_data, opp_0_ell_indices, opp_0_nnz_per_row);
    opp_0_ell_data.cp_cpu_gpu();
    opp_0_ell_indices.cp_cpu_gpu();
#endif
    
  }
  else
  {
    cout << "ERROR: Invalid sparse matrix form ... " << endl;
  }
  
  
  
}

// set opp_1 (transformed discontinuous flux at solution points to normal transformed discontinuous flux at flux points)

void eles::set_opp_1(int in_sparse)
{
  int i,j,k,l;
  array<double> loc(n_dims);
  
  opp_1.setup(n_dims);
  for (int i=0;i<n_dims;i++)
    opp_1(i).setup(n_fpts_per_ele,n_upts_per_ele);
  
  for(i=0;i<n_dims;i++)
  {
    for(j=0;j<n_upts_per_ele;j++)
    {
      for(k=0;k<n_fpts_per_ele;k++)
      {
        for(l=0;l<n_dims;l++)
        {
          loc(l)=tloc_fpts(l,k);
        }
        
        opp_1(i)(k,j)=eval_nodal_basis(j,loc)*tnorm_fpts(i,k);
      }
    }
    //cout << "opp_1,i =" << i << endl;
    //cout << "ele_type=" << ele_type << endl;
    //opp_1(i).print();
    //cout << endl;
  }
  
#ifdef _GPU
  for (int i=0;i<n_dims;i++)
    opp_1(i).cp_cpu_gpu();
#endif
  
  
  if(in_sparse==0)
  {
    opp_1_sparse=0;
  }
  else if(in_sparse==1)
  {
    opp_1_sparse=1;
    
#ifdef _CPU
    for (int i=0;i<n_dims;i++) {
      array_to_mklcsr(opp_1(i),opp_1_data(i),opp_1_cols(i),opp_1_b(i),opp_1_e(i));
    }
#endif
    
#ifdef _GPU
    opp_1_ell_data.setup(n_dims);
    opp_1_ell_indices.setup(n_dims);
    opp_1_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_1(i), opp_1_ell_data(i), opp_1_ell_indices(i), opp_1_nnz_per_row(i));
      opp_1_ell_data(i).cp_cpu_gpu();
      opp_1_ell_indices(i).cp_cpu_gpu();
    }
#endif
    
  }
  else
  {
    cout << "ERROR: Invalid sparse matrix form ... " << endl;
  }
}

// set opp_2 (transformed discontinuous flux at solution points to divergence of transformed discontinuous flux at solution points)

void eles::set_opp_2(int in_sparse)
{
  
  int i,j,k,l;
  
  array<double> loc(n_dims);
  
  opp_2.setup(n_dims);
  for (int i=0;i<n_dims;i++)
    opp_2(i).setup(n_upts_per_ele,n_upts_per_ele);
  
  for(i=0;i<n_dims;i++)
  {
    for(j=0;j<n_upts_per_ele;j++)
    {
      for(k=0;k<n_upts_per_ele;k++)
      {
        for(l=0;l<n_dims;l++)
        {
          loc(l)=loc_upts(l,k);
        }
        
        opp_2(i)(k,j)=eval_d_nodal_basis(j,i,loc);
      }
    }
    
    //cout << "opp_2,i =" << i << endl;
    //cout << "ele_type=" << ele_type << endl;
    //opp_2(i).print();
    //cout << endl;
    
    //cout << "opp_2,i=" << i << endl;
    //opp_2(i).print();
    
  }
  
#ifdef _GPU
  for (int i=0;i<n_dims;i++)
    opp_2(i).cp_cpu_gpu();
#endif
  
  //cout << "opp 2" << endl;
  //opp_2.print();
  
  if(in_sparse==0)
  {
    opp_2_sparse=0;
  }
  else if(in_sparse==1)
  {
    opp_2_sparse=1;
    
#ifdef _CPU
    for (int i=0;i<n_dims;i++) {
      array_to_mklcsr(opp_2(i),opp_2_data(i),opp_2_cols(i),opp_2_b(i),opp_2_e(i));
    }
#endif
    
#ifdef _GPU
    opp_2_ell_data.setup(n_dims);
    opp_2_ell_indices.setup(n_dims);
    opp_2_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_2(i), opp_2_ell_data(i), opp_2_ell_indices(i), opp_2_nnz_per_row(i));
      opp_2_ell_data(i).cp_cpu_gpu();
      opp_2_ell_indices(i).cp_cpu_gpu();
    }
#endif
  }
  else
  {
    cout << "ERROR: Invalid sparse matrix form ... " << endl;
  }
}

// set opp_3 (normal transformed correction flux at edge flux points to divergence of transformed correction flux at solution points)

void eles::set_opp_3(int in_sparse)
{
  
  opp_3.setup(n_upts_per_ele,n_fpts_per_ele);
  (*this).fill_opp_3(opp_3);
  
  //cout << "OPP_3" << endl;
  //cout << "ele_type=" << ele_type << endl;
  //opp_3.print();
  //cout << endl;
  
#ifdef _GPU
  opp_3.cp_cpu_gpu();
#endif
  
  if(in_sparse==0)
  {
    opp_3_sparse=0;
  }
  else if(in_sparse==1)
  {
    opp_3_sparse=1;
    
#ifdef _CPU
    array_to_mklcsr(opp_3,opp_3_data,opp_3_cols,opp_3_b,opp_3_e);
#endif
    
#ifdef _GPU
    array_to_ellpack(opp_3, opp_3_ell_data, opp_3_ell_indices, opp_3_nnz_per_row);
    opp_3_ell_data.cp_cpu_gpu();
    opp_3_ell_indices.cp_cpu_gpu();
#endif
  }
  else
  {
    cout << "ERROR: Invalid sparse matrix form ... " << endl;
  }
}

// set opp_4 (transformed solution at solution points to transformed gradient of transformed solution at solution points)

void eles::set_opp_4(int in_sparse)
{
  int i,j,k,l;
  
  array<double> loc(n_dims);
  
  opp_4.setup(n_dims);
  for (int i=0;i<n_dims;i++)
    opp_4(i).setup(n_upts_per_ele, n_upts_per_ele);
  
  for(i=0; i<n_dims; i++)
  {
    for(j=0; j<n_upts_per_ele; j++)
    {
      for(k=0; k<n_upts_per_ele; k++)
      {
        for(l=0; l<n_dims; l++)
        {
          loc(l)=loc_upts(l,k);
        }
        
        opp_4(i)(k,j) = eval_d_nodal_basis(j,i,loc);
      }
    }
  }
  
#ifdef _GPU
  for (int i=0;i<n_dims;i++)
    opp_4(i).cp_cpu_gpu();
#endif
  
  if(in_sparse==0)
  {
    opp_4_sparse=0;
  }
  else if(in_sparse==1)
  {
    opp_4_sparse=1;
    
#ifdef _CPU
    for (int i=0;i<n_dims;i++)
    {
      array_to_mklcsr(opp_4(i),opp_4_data(i),opp_4_cols(i),opp_4_b(i),opp_4_e(i));
    }
#endif
    
#ifdef _GPU
    opp_4_ell_data.setup(n_dims);
    opp_4_ell_indices.setup(n_dims);
    opp_4_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_4(i), opp_4_ell_data(i), opp_4_ell_indices(i), opp_4_nnz_per_row(i));
      opp_4_ell_data(i).cp_cpu_gpu();
      opp_4_ell_indices(i).cp_cpu_gpu();
    }
#endif
  }
  else
  {
    cout << "ERROR: Invalid sparse matrix form ... " << endl;
  }
}

// transformed solution correction at flux points to transformed gradient correction at solution points

void eles::set_opp_5(int in_sparse)
{
  int i,j,k,l;
  
  array<double> loc(n_dims);
  
  opp_5.setup(n_dims);
  for (int i=0;i<n_dims;i++)
    opp_5(i).setup(n_upts_per_ele, n_fpts_per_ele);

  for(i=0;i<n_dims;i++)
  {
    for(j=0;j<n_fpts_per_ele;j++)
    {
      for(k=0;k<n_upts_per_ele;k++)
      {
        /*
         for(l=0;l<n_dims;l++)
         {
         loc(l)=loc_upts(l,k);
         }
         */
        
        //opp_5(i)(k,j) = eval_div_vcjh_basis(j,loc)*tnorm_fpts(i,j);
        opp_5(i)(k,j) = opp_3(k,j)*tnorm_fpts(i,j);
      }
    }
  }
  
#ifdef _GPU
  for (int i=0;i<n_dims;i++)
    opp_5(i).cp_cpu_gpu();
#endif
  
  //cout << "opp_5" << endl;
  //opp_5.print();
  
  if(in_sparse==0)
  {
    opp_5_sparse=0;
  }
  else if(in_sparse==1)
  {
    opp_5_sparse=1;
    
#ifdef _CPU
    for (int i=0;i<n_dims;i++) {
      array_to_mklcsr(opp_5(i),opp_5_data(i),opp_5_cols(i),opp_5_b(i),opp_5_e(i));
    }
#endif
    
#ifdef _GPU
    opp_5_ell_data.setup(n_dims);
    opp_5_ell_indices.setup(n_dims);
    opp_5_nnz_per_row.setup(n_dims);
    for (int i=0;i<n_dims;i++) {
      array_to_ellpack(opp_5(i), opp_5_ell_data(i), opp_5_ell_indices(i), opp_5_nnz_per_row(i));
      opp_5_ell_data(i).cp_cpu_gpu();
      opp_5_ell_indices(i).cp_cpu_gpu();
    }
#endif
  }
  else
  {
    cout << "ERROR: Invalid sparse matrix form ... " << endl;
  }
}

// transformed gradient at solution points to transformed gradient at flux points

void eles::set_opp_6(int in_sparse)
{
  int i,j,k,l,m;
  
  array<double> loc(n_dims);
  
  opp_6.setup(n_fpts_per_ele, n_upts_per_ele);
  
  for(j=0; j<n_upts_per_ele; j++)
  {
    for(l=0; l<n_fpts_per_ele; l++)
    {
      for(m=0; m<n_dims; m++)
      {
        loc(m) = tloc_fpts(m,l);
      }
      opp_6(l,j) = eval_nodal_basis(j,loc);
    }
  }
  
  //cout << "opp_6" << endl;
  //opp_6.print();
  
#ifdef _GPU
  opp_6.cp_cpu_gpu();
#endif
  
  if(in_sparse==0)
  {
    opp_6_sparse=0;
  }
  else if(in_sparse==1)
  {
    opp_6_sparse=1;
    
#ifdef _CPU
    array_to_mklcsr(opp_6,opp_6_data,opp_6_cols,opp_6_b,opp_6_e);
#endif
    
#ifdef _GPU
    array_to_ellpack(opp_6, opp_6_ell_data, opp_6_ell_indices, opp_6_nnz_per_row);
    opp_6_ell_data.cp_cpu_gpu();
    opp_6_ell_indices.cp_cpu_gpu();
#endif
    
  }
  else
  {
    cout << "ERROR: Invalid sparse matrix form ... " << endl;
  }
}

// set opp_p (solution at solution points to solution at plot points)

void eles::set_opp_p(void)
{
  int i,j,k;
  
  array<double> loc(n_dims);
  
  opp_p.setup(n_ppts_per_ele,n_upts_per_ele);
  
  for(i=0;i<n_upts_per_ele;i++)
  {
    for(j=0;j<n_ppts_per_ele;j++)
    {
      for(k=0;k<n_dims;k++)
      {
        loc(k)=loc_ppts(k,j);
      }
      
      opp_p(j,i)=eval_nodal_basis(i,loc);
    }
  }
  
}

void eles::set_opp_inters_cubpts(void)
{
  
  int i,j,k,l;
  
  array<double> loc(n_dims);
  
  opp_inters_cubpts.setup(n_inters_per_ele);
  
  for (int i=0;i<n_inters_per_ele;i++)
  {
    opp_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_upts_per_ele);
  }
  
  for(l=0;l<n_inters_per_ele;l++)
  {
    for(i=0;i<n_upts_per_ele;i++)
    {
      for(j=0;j<n_cubpts_per_inter(l);j++)
      {
        for(k=0;k<n_dims;k++)
        {
          loc(k)=loc_inters_cubpts(l)(k,j);
        }
        
        opp_inters_cubpts(l)(j,i)=eval_nodal_basis(i,loc);
      }
    }
  }
  
}

void eles::set_opp_volume_cubpts(void)
{
  
  int i,j,k,l;
  array<double> loc(n_dims);
  opp_volume_cubpts.setup(n_cubpts_per_ele,n_upts_per_ele);
  
  for(i=0;i<n_upts_per_ele;i++)
  {
    for(j=0;j<n_cubpts_per_ele;j++)
    {
      for(k=0;k<n_dims;k++)
      {
        loc(k)=loc_volume_cubpts(k,j);
      }
      
      opp_volume_cubpts(j,i)=eval_nodal_basis(i,loc);
    }
  }
}


// set opp_r (solution at restart points to solution at solution points)

void eles::set_opp_r(void)
{
  int i,j,k;
  
  array<double> loc(n_dims);
  
  opp_r.setup(n_upts_per_ele,n_upts_per_ele_rest);
  
  for(i=0;i<n_upts_per_ele_rest;i++)
  {
    for(j=0;j<n_upts_per_ele;j++)
    {
      for(k=0;k<n_dims;k++)
        loc(k)=loc_upts(k,j);
      
      opp_r(j,i)=eval_nodal_basis_restart(i,loc);
    }
  }
}

// calculate position of the plot points

void eles::calc_pos_ppts(int in_ele, array<double>& out_pos_ppts)
{
  int i,j;
  
  array<double> loc(n_dims);
  array<double> pos(n_dims);
  
  for(i=0;i<n_ppts_per_ele;i++)
  {   
    if (motion) {
      calc_pos_dyn_ppt(i,in_ele,pos);
    }else{
      for(j=0;j<n_dims;j++)
      {
        loc(j)=loc_ppts(j,i);
      }
      calc_pos(loc,in_ele,pos);
    }
    
    for(j=0;j<n_dims;j++)  // TODO: can this be made more efficient/simpler?
    {
      out_pos_ppts(i,j)=pos(j);
    }
  }
}

// calculate solution at the plot points
void eles::calc_disu_ppts(int in_ele, array<double>& out_disu_ppts)
{
  if (n_eles!=0)
  {
    
    int i,j,k;
    
    array<double> disu_upts_plot(n_upts_per_ele,n_fields);
    
    for(i=0;i<n_fields;i++)
    {
      for(j=0;j<n_upts_per_ele;j++)
      {
        if (motion) {
          disu_upts_plot(j,i)=disu_upts(0)(j,in_ele,i)/J_dyn_upts(j,in_ele);
          //disu_upts_plot(j,i)=1/J_dyn_upts(j,in_ele);
          //cout << in_ele << "," << j << "," << i << ": " << disu_upts(0)(j,in_ele,i) << ", " << J_dyn_upts(j,in_ele) << endl;
        }else{
          disu_upts_plot(j,i)=disu_upts(0)(j,in_ele,i);
        }
      }
    }
    
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,disu_upts_plot.get_ptr_cpu(),n_upts_per_ele,0.0,out_disu_ppts.get_ptr_cpu(),n_ppts_per_ele);
    
#elif defined _NO_BLAS
    dgemm(n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,0.0,opp_p.get_ptr_cpu(),disu_upts_plot.get_ptr_cpu(),out_disu_ppts.get_ptr_cpu());
    
#else
    
    //HACK (inefficient, but useful if cblas is unavailible)
    
    for(i=0;i<n_ppts_per_ele;i++)
    {
      for(k=0;k<n_fields;k++)
      {
        out_disu_ppts(i,k) = 0.;
        
        for(j=0;j<n_upts_per_ele;j++)
        {
          out_disu_ppts(i,k) += opp_p(i,j)*disu_upts_plot(j,k);
        }
      }
    }
    
#endif
    
  }
}

// calculate gradient of solution at the plot points
void eles::calc_grad_disu_ppts(int in_ele, array<double>& out_grad_disu_ppts)
{
  if (n_eles!=0)
  {
    
    int i,j,k,l;
    
    array<double> grad_disu_upts_temp(n_upts_per_ele,n_fields,n_dims);
    
    for(i=0;i<n_fields;i++)
    {
      for(j=0;j<n_upts_per_ele;j++)
      {
        for(k=0;k<n_dims;k++)
        {
          grad_disu_upts_temp(j,i,k)=grad_disu_upts(j,in_ele,i,k);
        }
      }
    }
    
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    
    for (i=0;i<n_dims;i++) {
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,grad_disu_upts_temp.get_ptr_cpu(0,0,i),n_upts_per_ele,0.0,out_grad_disu_ppts.get_ptr_cpu(0,0,i),n_ppts_per_ele);
    }
    
#elif defined _NO_BLAS
    
    for (i=0;i<n_dims;i++) {
      dgemm(n_ppts_per_ele,n_fields,n_upts_per_ele,1.0,0.0,opp_p.get_ptr_cpu(),grad_disu_upts_temp.get_ptr_cpu(0,0,i),out_grad_disu_ppts.get_ptr_cpu(0,0,i));
    }
    
#else
    
    //HACK (inefficient, but useful if cblas is unavailible)
    
    for(i=0;i<n_ppts_per_ele;i++)
    {
      for(k=0;k<n_fields;k++)
      {
        for(l=0;l<n_dims;l++)
        {
          out_grad_disu_ppts(i,k,l) = 0.;
          for(j=0;j<n_upts_per_ele;j++)
          {
            out_grad_disu_ppts(i,k,l) += opp_p(i,j)*grad_disu_upts_temp(j,k,l);
          }
        }
      }
    }
    
#endif
    
  }
}

// calculate the time averaged field values at plot points
void eles::calc_time_average_ppts(int in_ele, array<double>& out_disu_average_ppts)
{
  if (n_eles!=0)
  {
    
    int i,j,k;
    
    array<double> disu_average_upts_plot(n_upts_per_ele,n_average_fields);
    
    for(i=0;i<n_average_fields;i++)
    {
      for(j=0;j<n_upts_per_ele;j++)
      {
        disu_average_upts_plot(j,i)=disu_average_upts(j,in_ele,i);
      }
    }
    
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,n_average_fields,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,disu_average_upts_plot.get_ptr_cpu(),n_upts_per_ele,0.0,out_disu_average_ppts.get_ptr_cpu(),n_ppts_per_ele);
    
#elif defined _NO_BLAS
    dgemm(n_ppts_per_ele,n_average_fields,n_upts_per_ele,1.0,0.0,opp_p.get_ptr_cpu(),disu_average_upts_plot.get_ptr_cpu(),out_disu_average_ppts.get_ptr_cpu());
    
#else
    
    //HACK (inefficient, but useful if cblas is unavailible)
    
    for(i=0;i<n_ppts_per_ele;i++)
    {
      for(k=0;k<n_average_fields;k++)
      {
        out_disu_average_ppts(i,k) = 0.;
        
        for(j=0;j<n_upts_per_ele;j++)
        {
          out_disu_average_ppts(i,k) += opp_p(i,j)*disu_average_upts_plot(j,k);
        }
      }
    }
    
#endif
    
  }
}

// calculate the sensor values at plot points
void eles::calc_sensor_ppts(int in_ele, array<double>& out_sensor_ppts)
{
    if (n_eles!=0)
    {
      for(int i=0;i<n_ppts_per_ele;i++)
        out_sensor_ppts(i) = sensor(in_ele);
    }
}

// calculate solution at the plot points
void eles::calc_epsilon_ppts(int in_ele, array<double>& out_epsilon_ppts)
{
  if (n_eles!=0)
  {

      int i,j,k;

      array<double> epsilon_upts_plot(n_upts_per_ele);

      for(j=0;j<n_upts_per_ele;j++)
      {
        epsilon_upts_plot(j)=epsilon_upts(j,in_ele);
      }


#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS

      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n_ppts_per_ele,1,n_upts_per_ele,1.0,opp_p.get_ptr_cpu(),n_ppts_per_ele,epsilon_upts_plot.get_ptr_cpu(),n_upts_per_ele,0.0,out_epsilon_ppts.get_ptr_cpu(),n_ppts_per_ele);

#else

      //HACK (inefficient, but useful if cblas is unavailible)

      for(i=0;i<n_ppts_per_ele;i++)
      {
        out_epsilon_ppts(i) = 0.;

        for(j=0;j<n_upts_per_ele;j++)
        {
          out_epsilon_ppts(i) += opp_p(i,j)*epsilon_upts_plot(j);
        }

      }

#endif

  }
}

// calculate diagnostic fields at the plot points
void eles::calc_diagnostic_fields_ppts(int in_ele, array<double>& in_disu_ppts, array<double>& in_grad_disu_ppts, array<double>& in_sensor_ppts, array<double>& in_epsilon_ppts, array<double>& out_diag_field_ppts, double& time)
{
  int i,j,k,m;
  double diagfield_upt;
  double u,v,w;
  double irho,pressure,v_sq;
  double wx,wy,wz;
  double dudx, dudy, dudz;
  double dvdx, dvdy, dvdz;
  double dwdx, dwdy, dwdz;
  
  for(j=0;j<n_ppts_per_ele;j++)
  {
    // Compute velocity square
    v_sq = 0.;
    for (m=0;m<n_dims;m++)
      v_sq += (in_disu_ppts(j,m+1)*in_disu_ppts(j,m+1));
    v_sq /= in_disu_ppts(j,0)*in_disu_ppts(j,0);
    
    // Compute pressure
    pressure = (run_input.gamma-1.0)*( in_disu_ppts(j,n_dims+1) - 0.5*in_disu_ppts(j,0)*v_sq);
    
    // compute diagnostic fields
    for (k=0;k<n_diagnostic_fields;k++)
    {
      irho = 1./in_disu_ppts(j,0);
      
      if (run_input.diagnostic_fields(k)=="u")
        diagfield_upt = in_disu_ppts(j,1)*irho;
      else if (run_input.diagnostic_fields(k)=="v")
        diagfield_upt = in_disu_ppts(j,2)*irho;
      else if (run_input.diagnostic_fields(k)=="w")
      {
        if (n_dims==2)
          diagfield_upt = 0.;
        else if (n_dims==3)
          diagfield_upt = in_disu_ppts(j,3)*irho;
      }
      else if (run_input.diagnostic_fields(k)=="energy")
      {
        if (n_dims==2)
          diagfield_upt = in_disu_ppts(j,3);
        else if (n_dims==3)
          diagfield_upt = in_disu_ppts(j,4);
      }
      // flow properties
      else if (run_input.diagnostic_fields(k)=="mach")
      {
        diagfield_upt = sqrt( v_sq / (run_input.gamma*pressure/in_disu_ppts(j,0)) );
      }
      else if (run_input.diagnostic_fields(k)=="pressure")
      {
        diagfield_upt = pressure;
      }
      // turbulence metrics
      else if (run_input.diagnostic_fields(k)=="vorticity" || run_input.diagnostic_fields(k)=="q_criterion")
      {
        u = in_disu_ppts(j,1)*irho;
        v = in_disu_ppts(j,2)*irho;
        
        dudx = irho*(in_grad_disu_ppts(j,1,0) - u*in_grad_disu_ppts(j,0,0));
        dudy = irho*(in_grad_disu_ppts(j,1,1) - u*in_grad_disu_ppts(j,0,1));
        dvdx = irho*(in_grad_disu_ppts(j,2,0) - v*in_grad_disu_ppts(j,0,0));
        dvdy = irho*(in_grad_disu_ppts(j,2,1) - v*in_grad_disu_ppts(j,0,1));
        
        if (n_dims==2)
        {
          if (run_input.diagnostic_fields(k) == "vorticity")
          {
            diagfield_upt = abs(dvdx-dudy);
          }
          else if (run_input.diagnostic_fields(k) == "q_criterion")
          {
            FatalError("Not implemented in 2D");
          }
        }
        else if (n_dims==3)
        {
          w = in_disu_ppts(j,3)*irho;
          
          dudz = irho*(in_grad_disu_ppts(j,1,2) - u*in_grad_disu_ppts(j,0,2));
          dvdz = irho*(in_grad_disu_ppts(j,2,2) - v*in_grad_disu_ppts(j,0,2));
          
          dwdx = irho*(in_grad_disu_ppts(j,3,0) - w*in_grad_disu_ppts(j,0,0));
          dwdy = irho*(in_grad_disu_ppts(j,3,1) - w*in_grad_disu_ppts(j,0,1));
          dwdz = irho*(in_grad_disu_ppts(j,3,2) - w*in_grad_disu_ppts(j,0,2));
          
          wx = dwdy - dvdz;
          wy = dudz - dwdx;
          wz = dvdx - dudy;
          
          if (run_input.diagnostic_fields(k) == "vorticity")
          {
            diagfield_upt = sqrt(wx*wx+wy*wy+wz*wz);
          }
          else if (run_input.diagnostic_fields(k) == "q_criterion")
          {
            
            wx *= 0.5;
            wy *= 0.5;
            wz *= 0.5;
            
            double Sxx,Syy,Szz,Sxy,Sxz,Syz,SS,OO;
            Sxx = dudx;
            Syy = dvdy;
            Szz = dwdz;
            Sxy = 0.5*(dudy+dvdx);
            Sxz = 0.5*(dudz+dwdx);
            Syz = 0.5*(dvdz+dwdy);
            
            SS = Sxx*Sxx + Syy*Syy + Szz*Szz + 2*Sxy*Sxy + 2*Sxz*Sxz + 2*Syz*Syz;
            OO = 2*wx*wx + 2*wy*wy + 2*wz*wz;
            
            diagfield_upt = 0.5*(OO-SS);
            
          }
        }
      }
      // Artificial Viscosity diagnostics
      else if (run_input.diagnostic_fields(k)=="sensor")
      {
        diagfield_upt = in_sensor_ppts(j);
      }
      else if (run_input.diagnostic_fields(k)=="epsilon")
      {
        diagfield_upt = in_epsilon_ppts(j);
      }

      else {
        cout << "plot_quantity = " << run_input.diagnostic_fields(k) << ": " << flush;
        FatalError("plot_quantity not recognized");
      }
      if (isnan(diagfield_upt)) {
        cout << "In calculation of plot_quantitiy " << run_input.diagnostic_fields(k) << ": " << flush;
        FatalError("NaN");
      }
      
      // set array with solution point value
      out_diag_field_ppts(j,k) = diagfield_upt;
    }
  }
}

// calculate position of solution point

void eles::calc_pos_upt(int in_upt, int in_ele, array<double>& out_pos)
{
  int i;
  
  array<double> loc(n_dims);
  
  for(i=0;i<n_dims;i++)
  {
    loc(i)=loc_upts(i,in_upt);
  }
  
  calc_pos(loc,in_ele,out_pos);
}

double eles::get_loc_upt(int in_upt, int in_dim)
{
  return loc_upts(in_dim,in_upt);
}

// set transforms

void eles::set_transforms(void)
{
  if (n_eles!=0)
  {
    
    int i,j,k;
    
    int n_comp;
    
    if(n_dims == 2)
    {
      n_comp = 3;
    }
    else if(n_dims == 3)
    {
      n_comp = 6;
    }
    
    array<double> loc(n_dims);
    array<double> pos(n_dims);
    array<double> d_pos(n_dims,n_dims);
    array<double> tnorm_dot_inv_detjac_mul_jac(n_dims);
    
    double xr, xs, xt;
    double yr, ys, yt;
    double zr, zs, zt;
    
    double xrr, xss, xtt, xrs, xrt, xst;
    double yrr, yss, ytt, yrs, yrt, yst;
    double zrr, zss, ztt, zrs, zrt, zst;
    
    // Determinant of Jacobian (transformation matrix) (J = |G|)
    detjac_upts.setup(n_upts_per_ele,n_eles);
    // Determinant of Jacobian times inverse of Jacobian (Full vector transform from physcial->reference frame)
    JGinv_upts.setup(n_dims,n_dims,n_upts_per_ele,n_eles);
    // Static-Physical position of solution points
    pos_upts.setup(n_upts_per_ele,n_eles,n_dims);
    
    if (rank==0) {
      cout << " at solution points" << endl;
    }
    
    for(i=0;i<n_eles;i++)
    {
      if ((i%(max(n_eles,10)/10))==0 && rank==0)
        cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;
      
      for(j=0;j<n_upts_per_ele;j++)
      {
        // get coordinates of the solution point
        
        for(k=0;k<n_dims;k++)
        {
          loc(k)=loc_upts(k,j);
        }
        
        calc_pos(loc,i,pos);

        for(k=0;k<n_dims;k++)
        {
          pos_upts(j,i,k)=pos(k);
        }

        // calculate first derivatives of shape functions at the solution point
        calc_d_pos(loc,i,d_pos);
        
        // store quantities at the solution point
        
        if(n_dims==2)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);
          
          yr = d_pos(1,0);
          ys = d_pos(1,1);
          
          // store determinant of jacobian at solution point
          detjac_upts(j,i)= xr*ys - xs*yr;
          
          if (detjac_upts(j,i) < 0)
          {
            FatalError("Negative Jacobian at solution points");
          }
          
          // store inverse of determinant of jacobian multiplied by jacobian at the solution point
          JGinv_upts(0,0,j,i)= ys;
          JGinv_upts(0,1,j,i)= -xs;
          JGinv_upts(1,0,j,i)= -yr;
          JGinv_upts(1,1,j,i)= xr;          
        }
        else if(n_dims==3)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);
          xt = d_pos(0,2);
          
          yr = d_pos(1,0);
          ys = d_pos(1,1);
          yt = d_pos(1,2);
          
          zr = d_pos(2,0);
          zs = d_pos(2,1);
          zt = d_pos(2,2);
          
          // store determinant of jacobian at solution point
          
          detjac_upts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);
          
          JGinv_upts(0,0,j,i) = ys*zt - yt*zs;
          JGinv_upts(0,1,j,i) = xt*zs - xs*zt;
          JGinv_upts(0,2,j,i) = xs*yt - xt*ys;
          JGinv_upts(1,0,j,i) = yt*zr - yr*zt;
          JGinv_upts(1,1,j,i) = xr*zt - xt*zr;
          JGinv_upts(1,2,j,i) = xt*yr - xr*yt;
          JGinv_upts(2,0,j,i) = yr*zs - ys*zr;
          JGinv_upts(2,1,j,i) = xs*zr - xr*zs;
          JGinv_upts(2,2,j,i) = xr*ys - xs*yr;
        }
        else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
      }
    }
    
#ifdef _GPU
    detjac_upts.cp_cpu_gpu(); // Copy since need in write_tec
    JGinv_upts.cp_cpu_gpu(); // Copy since needed for calc_d_pos_dyn
    /*
     if (viscous) {
     tgrad_detjac_upts.mv_cpu_gpu();
     }
     */
#endif
    
    // Compute metrics term at flux points
    /// Determinant of Jacobian (transformation matrix)
    detjac_fpts.setup(n_fpts_per_ele,n_eles);
    /// Determinant of Jacobian times inverse of Jacobian (Full vector transform from physcial->reference frame)
    JGinv_fpts.setup(n_dims,n_dims,n_fpts_per_ele,n_eles);
    tdA_fpts.setup(n_fpts_per_ele,n_eles);
    norm_fpts.setup(n_fpts_per_ele,n_eles,n_dims);
    // Static-Physical position of solution points
    pos_fpts.setup(n_fpts_per_ele,n_eles,n_dims);
    
    if (rank==0)
      cout << endl << " at flux points"  << endl;
    
    for(i=0;i<n_eles;i++)
    {
      if ((i%(max(n_eles,10)/10))==0 && rank==0)
        cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;
      
      for(j=0;j<n_fpts_per_ele;j++)
      {
        // get coordinates of the flux point
        
        for(k=0;k<n_dims;k++)
        {
          loc(k)=tloc_fpts(k,j);
        }
        
        calc_pos(loc,i,pos);
        
        for(k=0;k<n_dims;k++)
        {
          pos_fpts(j,i,k)=pos(k);
        }
        
        // calculate first derivatives of shape functions at the flux points
        
        calc_d_pos(loc,i,d_pos);
        
        // store quantities at the flux point
        
        if(n_dims==2)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);
          
          yr = d_pos(1,0);
          ys = d_pos(1,1);
          
          // store determinant of jacobian at flux point
          
          detjac_fpts(j,i)= xr*ys - xs*yr;
          
          if (detjac_fpts(j,i) < 0)
          {
            FatalError("Negative Jacobian at flux points");
          }
          
          // store inverse of determinant of jacobian multiplied by jacobian at the flux point
          
          JGinv_fpts(0,0,j,i)= ys;
          JGinv_fpts(0,1,j,i)= -xs;
          JGinv_fpts(1,0,j,i)= -yr;
          JGinv_fpts(1,1,j,i)= xr;
          
          // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
          
          tnorm_dot_inv_detjac_mul_jac(0)=(tnorm_fpts(0,j)*d_pos(1,1))-(tnorm_fpts(1,j)*d_pos(1,0));
          tnorm_dot_inv_detjac_mul_jac(1)=-(tnorm_fpts(0,j)*d_pos(0,1))+(tnorm_fpts(1,j)*d_pos(0,0));
          
          // store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
          
          tdA_fpts(j,i)=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                                          tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1));
          
          
          // store normal at flux point
          
          norm_fpts(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/tdA_fpts(j,i);
          norm_fpts(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/tdA_fpts(j,i);
        }
        else if(n_dims==3)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);
          xt = d_pos(0,2);
          
          yr = d_pos(1,0);
          ys = d_pos(1,1);
          yt = d_pos(1,2);
          
          zr = d_pos(2,0);
          zs = d_pos(2,1);
          zt = d_pos(2,2);
          
          // store determinant of jacobian at flux point
          
          detjac_fpts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);
          
          // store inverse of determinant of jacobian multiplied by jacobian at the flux point
          
          JGinv_fpts(0,0,j,i) = ys*zt - yt*zs;
          JGinv_fpts(0,1,j,i) = xt*zs - xs*zt;
          JGinv_fpts(0,2,j,i) = xs*yt - xt*ys;
          JGinv_fpts(1,0,j,i) = yt*zr - yr*zt;
          JGinv_fpts(1,1,j,i) = xr*zt - xt*zr;
          JGinv_fpts(1,2,j,i) = xt*yr - xr*yt;
          JGinv_fpts(2,0,j,i) = yr*zs - ys*zr;
          JGinv_fpts(2,1,j,i) = xs*zr - xr*zs;
          JGinv_fpts(2,2,j,i) = xr*ys - xs*yr;
          
          // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
          
          tnorm_dot_inv_detjac_mul_jac(0)=((tnorm_fpts(0,j)*(d_pos(1,1)*d_pos(2,2)-d_pos(1,2)*d_pos(2,1)))+(tnorm_fpts(1,j)*(d_pos(1,2)*d_pos(2,0)-d_pos(1,0)*d_pos(2,2)))+(tnorm_fpts(2,j)*(d_pos(1,0)*d_pos(2,1)-d_pos(1,1)*d_pos(2,0))));
          tnorm_dot_inv_detjac_mul_jac(1)=((tnorm_fpts(0,j)*(d_pos(0,2)*d_pos(2,1)-d_pos(0,1)*d_pos(2,2)))+(tnorm_fpts(1,j)*(d_pos(0,0)*d_pos(2,2)-d_pos(0,2)*d_pos(2,0)))+(tnorm_fpts(2,j)*(d_pos(0,1)*d_pos(2,0)-d_pos(0,0)*d_pos(2,1))));
          tnorm_dot_inv_detjac_mul_jac(2)=((tnorm_fpts(0,j)*(d_pos(0,1)*d_pos(1,2)-d_pos(0,2)*d_pos(1,1)))+(tnorm_fpts(1,j)*(d_pos(0,2)*d_pos(1,0)-d_pos(0,0)*d_pos(1,2)))+(tnorm_fpts(2,j)*(d_pos(0,0)*d_pos(1,1)-d_pos(0,1)*d_pos(1,0))));
          
          // store magnitude of transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the flux point
          
          tdA_fpts(j,i)=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                                          tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1)+
                                                          tnorm_dot_inv_detjac_mul_jac(2)*tnorm_dot_inv_detjac_mul_jac(2));
          
          // store normal at flux point
          
          norm_fpts(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/tdA_fpts(j,i);
          norm_fpts(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/tdA_fpts(j,i);
          norm_fpts(j,i,2)=tnorm_dot_inv_detjac_mul_jac(2)/tdA_fpts(j,i);
        }
        else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
      }
    }
    
#ifdef _GPU
    tdA_fpts.mv_cpu_gpu();
    pos_fpts.cp_cpu_gpu();

    JGinv_fpts.cp_cpu_gpu();
    detjac_fpts.cp_cpu_gpu();

    if (motion) {
      norm_fpts.cp_cpu_gpu(); // cp b/c needed for set_transforms_dynamic()
    }
    else
    {
      norm_fpts.mv_cpu_gpu(); 
      // move the dummy dynamic-transform pointers to GPUs
      cp_transforms_cpu_gpu();
    }
#endif
    
    if (rank==0) cout << endl;
  } // if n_eles!=0
}

void eles::set_transforms_dynamic(void)
{
  if (n_eles!=0 && motion && first_time) {
    // Determinant of the dynamic -> static reference transformation matrix ( |G| )
    J_dyn_upts.setup(n_upts_per_ele,n_eles);
    // Total dynamic -> static reference transformation matrix ( |G|*G^{-1} )
    JGinv_dyn_upts.setup(n_dims,n_dims,n_upts_per_ele,n_eles);
    dyn_pos_upts.setup(n_upts_per_ele,n_eles,n_dims);

    J_dyn_fpts.setup(n_fpts_per_ele,n_eles);
    JGinv_dyn_fpts.setup(n_dims,n_dims,n_fpts_per_ele,n_eles);
    norm_dyn_fpts.setup(n_fpts_per_ele,n_eles,n_dims);
    dyn_pos_fpts.setup(n_fpts_per_ele,n_eles,n_dims);

    ndA_dyn_fpts.setup(n_fpts_per_ele,n_eles);
  }

#ifdef _CPU
  if (n_eles!=0 && motion)
  {
    int i,j,k;

    int n_comp;

    if(n_dims == 2)
      n_comp = 3;
    else if(n_dims == 3)
      n_comp = 6;

    array<double> pos(n_dims);
    array<double> d_pos(n_dims,n_dims);
    array<double> norm_dot_JGinv(n_dims);  // un-normalized normal vector in moving-physical domain

    double xr, xs, xt;
    double yr, ys, yt;
    double zr, zs, zt;

    if (rank==0 && first_time) {
      cout << " Setting up dynamic->static transforms at solution points" << endl;
    }

    for(i=0;i<n_eles;i++) {
      if ((i%(max(n_eles,10)/10))==0 && rank==0 && first_time)
        cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;

      for(j=0;j<n_upts_per_ele;j++)
      {
        // get coordinates of the solution point

        // calculate first derivatives of shape functions at the solution point
        calc_d_pos_dyn_upt(j,i,d_pos);

        // store quantities at the solution point

        if(n_dims==2)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);

          yr = d_pos(1,0);
          ys = d_pos(1,1);

          // store determinant of jacobian at solution point
          J_dyn_upts(j,i)= xr*ys - xs*yr;

          if (first_time && run_input.GCL) {
          //if (in_rkstep==0) {
            Jbar_upts(0)(j,i) = J_dyn_upts(j,i);
          }

          if (J_dyn_upts(j,i) < 0)
          {
            FatalError("Negative Jacobian at solution points");
          }

          // store determinant of jacobian multiplied by inverse of jacobian at the solution point
          JGinv_dyn_upts(0,0,j,i)=  ys;
          JGinv_dyn_upts(0,1,j,i)= -xs;
          JGinv_dyn_upts(1,1,j,i)= -yr;
          JGinv_dyn_upts(1,1,j,i)=  xr;
        }
        else if(n_dims==3)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);
          xt = d_pos(0,2);

          yr = d_pos(1,0);
          ys = d_pos(1,1);
          yt = d_pos(1,2);

          zr = d_pos(2,0);
          zs = d_pos(2,1);
          zt = d_pos(2,2);

          // store determinant of jacobian at solution point
          J_dyn_upts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

          // store determinant of jacobian multiplied by inverse of jacobian at the solution point
          JGinv_dyn_upts(0,0,j,i) = (ys*zt - yt*zs);
          JGinv_dyn_upts(0,1,j,i) = (xt*zs - xs*zt);
          JGinv_dyn_upts(0,2,j,i) = (xs*yt - xt*ys);
          JGinv_dyn_upts(1,1,j,i) = (yt*zr - yr*zt);
          JGinv_dyn_upts(1,1,j,i) = (xr*zt - xt*zr);
          JGinv_dyn_upts(1,2,j,i) = (xt*yr - xr*yt);
          JGinv_dyn_upts(2,0,j,i) = (yr*zs - ys*zr);
          JGinv_dyn_upts(2,1,j,i) = (xs*zr - xr*zs);
          JGinv_dyn_upts(2,2,j,i) = (xr*ys - xs*yr);
        }
        else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }

        // Have to use the GCL-corrected Jacobain everywhere
        //J_dyn_upts(j,i) = Jbar_upts(0)(j,i);
      }
    }

    // Compute metrics term at flux points
    if (rank==0 && first_time)
      cout << endl << " at flux points"  << endl;

    for(i=0;i<n_eles;i++) {
      if ((i%(max(n_eles,10)/10))==0 && rank==0 && first_time)
        cout << fixed << setprecision(2) <<  (i*1.0/n_eles)*100 << "% " << flush;

      for(j=0;j<n_fpts_per_ele;j++)
      {
        // get coordinates of the flux point

        calc_pos_dyn_fpt(j,i,pos);

        for(k=0;k<n_dims;k++) {
          dyn_pos_fpts(j,i,k)=pos(k);
        }

        // calculate first derivatives of shape functions at the flux points
        calc_d_pos_dyn_fpt(j,i,d_pos);

        // store quantities at the flux point

        if(n_dims==2)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);

          yr = d_pos(1,0);
          ys = d_pos(1,1);

          // store determinant of dynamic transformation Jacobian at flux point
          J_dyn_fpts(j,i)= xr*ys - xs*yr;

          if (first_time && run_input.GCL) {
          //if (in_rkstep==0) {
            Jbar_fpts(0)(j,i) = J_dyn_fpts(j,i);
          }

          if (J_dyn_fpts(j,i) < 0)
          {
            FatalError("Negative Jacobian at flux points");
          }

          // store determinant of jacobian multiplied by inverse of jacobian at the flux point
          // (dynamic -> static transformation matrix)
          JGinv_dyn_fpts(0,0,j,i)=  ys;
          JGinv_dyn_fpts(0,1,j,i)= -xs;
          JGinv_dyn_fpts(1,0,j,i)= -yr;
          JGinv_dyn_fpts(1,1,j,i)=  xr;

//          if (viscous)
//          {
//            // store static->dynamic transformation matrix
//            JinvG_dyn_fpts(j,i,0,0)= xr/J_dyn_fpts(j,i);
//            JinvG_dyn_fpts(j,i,0,1)= xs/J_dyn_fpts(j,i);
//            JinvG_dyn_fpts(j,i,1,0)= yr/J_dyn_fpts(j,i);
//            JinvG_dyn_fpts(j,i,1,1)= ys/J_dyn_fpts(j,i);
//          }

          // temporarily store transformed normal dot determinant of jacobian multiplied by inverse of jacobian at the flux point
          norm_dot_JGinv(0)= ( norm_fpts(j,i,0)*ys -norm_fpts(j,i,1)*yr);
          norm_dot_JGinv(1)= (-norm_fpts(j,i,0)*xs +norm_fpts(j,i,1)*xr);

          // store magnitude of transformed normal dot determinant of jacobian multiplied by inverse of jacobian at the flux point
          ndA_dyn_fpts(j,i)=sqrt(norm_dot_JGinv(0)*norm_dot_JGinv(0)+
                                 norm_dot_JGinv(1)*norm_dot_JGinv(1));

          // store normal at flux point
          norm_dyn_fpts(j,i,0)=norm_dot_JGinv(0)/ndA_dyn_fpts(j,i);
          norm_dyn_fpts(j,i,1)=norm_dot_JGinv(1)/ndA_dyn_fpts(j,i);
        }
        else if(n_dims==3)
        {
          xr = d_pos(0,0);
          xs = d_pos(0,1);
          xt = d_pos(0,2);

          yr = d_pos(1,0);
          ys = d_pos(1,1);
          yt = d_pos(1,2);

          zr = d_pos(2,0);
          zs = d_pos(2,1);
          zt = d_pos(2,2);

          // store determinant of jacobian at flux point

          J_dyn_fpts(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

          // store determinant of jacobian multiplied by inverse of jacobian at the flux point

          JGinv_dyn_fpts(0,0,j,i) = (ys*zt - yt*zs);
          JGinv_dyn_fpts(0,1,j,i) = (xt*zs - xs*zt);
          JGinv_dyn_fpts(0,2,j,i) = (xs*yt - xt*ys);
          JGinv_dyn_fpts(1,0,j,i) = (yt*zr - yr*zt);
          JGinv_dyn_fpts(1,1,j,i) = (xr*zt - xt*zr);
          JGinv_dyn_fpts(1,2,j,i) = (xt*yr - xr*yt);
          JGinv_dyn_fpts(2,0,j,i) = (yr*zs - ys*zr);
          JGinv_dyn_fpts(2,1,j,i) = (xs*zr - xr*zs);
          JGinv_dyn_fpts(2,2,j,i) = (xr*ys - xs*yr);

          // temporarily store moving-physical domain interface normal at the flux point
          // [transformed normal dot determinant of jacobian multiplied by inverse of jacobian]
          norm_dot_JGinv(0)=((norm_fpts(j,i,0)*(ys*zt-yt*zs))+(norm_fpts(j,i,1)*(yt*zr-yr*zt))+(norm_fpts(j,i,2)*(yr*zs-ys*zr)));
          norm_dot_JGinv(1)=((norm_fpts(j,i,0)*(xt*zs-xs*zt))+(norm_fpts(j,i,1)*(xr*zt-xt*zr))+(norm_fpts(j,i,2)*(xs*zr-xr*zs)));
          norm_dot_JGinv(2)=((norm_fpts(j,i,0)*(xs*yt-xt*ys))+(norm_fpts(j,i,1)*(xt*yr-xr*yt))+(norm_fpts(j,i,2)*(xr*ys-xs*yr)));

          // store magnitude of transformed normal dot determinant of jacobian multiplied by inverse of jacobian at the flux point
          ndA_dyn_fpts(j,i)=sqrt(norm_dot_JGinv(0)*norm_dot_JGinv(0)+
                                 norm_dot_JGinv(1)*norm_dot_JGinv(1)+
                                 norm_dot_JGinv(2)*norm_dot_JGinv(2));

          // store normal at flux point
          norm_dyn_fpts(j,i,0)=norm_dot_JGinv(0)/ndA_dyn_fpts(j,i);
          norm_dyn_fpts(j,i,1)=norm_dot_JGinv(1)/ndA_dyn_fpts(j,i);
          norm_dyn_fpts(j,i,2)=norm_dot_JGinv(2)/ndA_dyn_fpts(j,i);
        }
        else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
      }
    }
    if (rank==0 && first_time) cout << endl;

    // To avoid re-setting up ALL transform arrays in the future (for dynamic grids)
    first_time = false;
  }
#endif

#ifdef _GPU
  if (n_eles!=0 && motion)
  {
    if (first_time) cp_transforms_cpu_gpu();

    if (rank == 0 && first_time) cout << "Setting dynamic transformations ... " << flush;
    set_transforms_dynamic_fpts_kernel_wrapper(n_fpts_per_ele,n_eles,n_dims,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),detjac_fpts.get_ptr_gpu(),J_dyn_fpts.get_ptr_gpu(),JGinv_fpts.get_ptr_gpu(),JGinv_dyn_fpts.get_ptr_gpu(),ndA_dyn_fpts.get_ptr_gpu(),norm_fpts.get_ptr_gpu(),norm_dyn_fpts.get_ptr_gpu(),d_nodal_s_basis_fpts.get_ptr_gpu(),shape_dyn.get_ptr_gpu());
    set_transforms_dynamic_upts_kernel_wrapper(n_upts_per_ele,n_eles,n_dims,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),detjac_upts.get_ptr_gpu(),J_dyn_upts.get_ptr_gpu(),JGinv_upts.get_ptr_gpu(),JGinv_dyn_upts.get_ptr_gpu(),d_nodal_s_basis_upts.get_ptr_gpu(),shape_dyn.get_ptr_gpu());
    if (rank == 0 && first_time) cout << "done." << endl;
  }

  first_time = false;
#endif
}

#ifdef _GPU
void eles::cp_transforms_gpu_cpu(void)
{
  J_dyn_upts.cp_gpu_cpu();
  JGinv_dyn_upts.cp_gpu_cpu();

  J_dyn_fpts.cp_gpu_cpu();
  JGinv_dyn_fpts.cp_gpu_cpu();
  ndA_dyn_fpts.cp_gpu_cpu();
  norm_dyn_fpts.cp_gpu_cpu();
  dyn_pos_fpts.cp_gpu_cpu();

  shape_dyn.cp_gpu_cpu();
  vel_spts.cp_gpu_cpu();
}

void eles::cp_transforms_cpu_gpu(void)
{
  J_dyn_upts.cp_cpu_gpu();
  J_dyn_fpts.cp_cpu_gpu();

  JGinv_dyn_upts.cp_cpu_gpu();
  JGinv_dyn_fpts.cp_cpu_gpu();

  ndA_dyn_fpts.cp_cpu_gpu();
  norm_dyn_fpts.cp_cpu_gpu();
  dyn_pos_fpts.cp_cpu_gpu();

  n_spts_per_ele.cp_cpu_gpu();
  shape.cp_cpu_gpu();
  shape_dyn.cp_cpu_gpu();

  //grid_vel_upts.cp_cpu_gpu();
}
#endif

void eles::set_bdy_ele2ele(void)
{
  
  n_bdy_eles=0;
  // Count the number of bdy_eles
  for (int i=0;i<n_eles;i++) {
    for (int j=0;j<n_inters_per_ele;j++) {
      if (bctype(i,j) != 0) {
        n_bdy_eles++;
        break;
      }
    }
  }
  
  if (n_bdy_eles!=0) {
    
    bdy_ele2ele.setup(n_bdy_eles);
    
    n_bdy_eles=0;
    for (int i=0;i<n_eles;i++) {
      for (int j=0;j<n_inters_per_ele;j++) {
        if (bctype(i,j) != 0) {
          bdy_ele2ele(n_bdy_eles++) = i;
          break;
        }
      }
    }
    
  }
  
}


// set transforms

void eles::set_transforms_inters_cubpts(void)
{
  if (n_eles!=0)
    {
      int i,j,k;
      int n_comp;

      double xr, xs, xt;
      double yr, ys, yt;
      double zr, zs, zt;

      // Initialize bdy_ele2ele array
      (*this).set_bdy_ele2ele();

      if(n_dims == 2)
        {
          n_comp = 3;
        }
      if(n_dims == 3)
        {
          n_comp = 6;
        }
      double mag_tnorm;

      array<double> loc(n_dims);
      array<double> d_pos(n_dims,n_dims);
      array<double> tnorm_dot_inv_detjac_mul_jac(n_dims);

      inter_detjac_inters_cubpts.setup(n_inters_per_ele);
      norm_inters_cubpts.setup(n_inters_per_ele);
      vol_detjac_inters_cubpts.setup(n_inters_per_ele);

      for (int i=0;i<n_inters_per_ele;i++)
        {
          inter_detjac_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles);
          norm_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles,n_dims);
          vol_detjac_inters_cubpts(i).setup(n_cubpts_per_inter(i),n_bdy_eles);
        }

      for(i=0;i<n_bdy_eles;i++)
        {
          for (int l=0;l<n_inters_per_ele;l++)
            {
              for(j=0;j<n_cubpts_per_inter(l);j++)
                {
                  // get coordinates of the cubature point

                  for(k=0;k<n_dims;k++)
                    {
                      loc(k)=loc_inters_cubpts(l)(k,j);
                    }

                  // calculate first derivatives of shape functions at the cubature points

                  // TODO: Need mapping between bdy_interfaces and ele
                  calc_d_pos(loc,bdy_ele2ele(i),d_pos);

                  // store quantities at the cubature point

                  if(n_dims==2)
                    {

                      xr = d_pos(0,0);
                      xs = d_pos(0,1);

                      yr = d_pos(1,0);
                      ys = d_pos(1,1);

                      // store determinant of jacobian at cubature point. TODO: what is this for?
                      vol_detjac_inters_cubpts(l)(j,i)= xr*ys - xs*yr;

                      // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the cubature point
                      tnorm_dot_inv_detjac_mul_jac(0)=(tnorm_inters_cubpts(l)(0,j)*d_pos(1,1))-(tnorm_inters_cubpts(l)(1,j)*d_pos(1,0));
                      tnorm_dot_inv_detjac_mul_jac(1)=-(tnorm_inters_cubpts(l)(0,j)*d_pos(0,1))+(tnorm_inters_cubpts(l)(1,j)*d_pos(0,0));

                      // calculate interface area
                      mag_tnorm = sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                       tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1));

                      // store normal at cubature point
                      norm_inters_cubpts(l)(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm;
                      norm_inters_cubpts(l)(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm;

                      inter_detjac_inters_cubpts(l)(j,i) = compute_inter_detjac_inters_cubpts(l,d_pos);
                    }
                  else if(n_dims==3)
                    {

                      xr = d_pos(0,0);
                      xs = d_pos(0,1);
                      xt = d_pos(0,2);

                      yr = d_pos(1,0);
                      ys = d_pos(1,1);
                      yt = d_pos(1,2);

                      zr = d_pos(2,0);
                      zs = d_pos(2,1);
                      zt = d_pos(2,2);

                      // store determinant of jacobian at cubature point
                      vol_detjac_inters_cubpts(l)(j,i) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

                      // temporarily store transformed normal dot inverse of determinant of jacobian multiplied by jacobian at the cubature point
                      tnorm_dot_inv_detjac_mul_jac(0)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(1,1)*d_pos(2,2)-d_pos(1,2)*d_pos(2,1)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(1,2)*d_pos(2,0)-d_pos(1,0)*d_pos(2,2)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(1,0)*d_pos(2,1)-d_pos(1,1)*d_pos(2,0))));
                      tnorm_dot_inv_detjac_mul_jac(1)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(0,2)*d_pos(2,1)-d_pos(0,1)*d_pos(2,2)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(0,0)*d_pos(2,2)-d_pos(0,2)*d_pos(2,0)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(0,1)*d_pos(2,0)-d_pos(0,0)*d_pos(2,1))));
                      tnorm_dot_inv_detjac_mul_jac(2)=((tnorm_inters_cubpts(l)(0,j)*(d_pos(0,1)*d_pos(1,2)-d_pos(0,2)*d_pos(1,1)))+(tnorm_inters_cubpts(l)(1,j)*(d_pos(0,2)*d_pos(1,0)-d_pos(0,0)*d_pos(1,2)))+(tnorm_inters_cubpts(l)(2,j)*(d_pos(0,0)*d_pos(1,1)-d_pos(0,1)*d_pos(1,0))));

                      // calculate interface area
                      mag_tnorm=sqrt(tnorm_dot_inv_detjac_mul_jac(0)*tnorm_dot_inv_detjac_mul_jac(0)+
                                     tnorm_dot_inv_detjac_mul_jac(1)*tnorm_dot_inv_detjac_mul_jac(1)+
                                     tnorm_dot_inv_detjac_mul_jac(2)*tnorm_dot_inv_detjac_mul_jac(2));

                      // store normal at cubature point
                      norm_inters_cubpts(l)(j,i,0)=tnorm_dot_inv_detjac_mul_jac(0)/mag_tnorm;
                      norm_inters_cubpts(l)(j,i,1)=tnorm_dot_inv_detjac_mul_jac(1)/mag_tnorm;
                      norm_inters_cubpts(l)(j,i,2)=tnorm_dot_inv_detjac_mul_jac(2)/mag_tnorm;

                      inter_detjac_inters_cubpts(l)(j,i) = compute_inter_detjac_inters_cubpts(l,d_pos);
                    }
                  else
                    {
                      FatalError("ERROR: Invalid number of dimensions ... ");
                    }
                }
            }
        }

    } // if n_eles!=0
}

// Set transforms at volume cubature points
void eles::set_transforms_vol_cubpts(void)
{
  if(n_eles!=0)
  {
    int i,j,m;
    array<double> d_pos(n_dims,n_dims);
    array<double> loc(n_dims);
    array<double> pos(n_dims);
    
    vol_detjac_vol_cubpts.setup(n_cubpts_per_ele);
    
    for (i=0;i<n_cubpts_per_ele;i++)
      vol_detjac_vol_cubpts(i).setup(n_eles);
    
    for (i=0;i<n_eles;i++)
    {
      for (j=0;j<n_cubpts_per_ele;j++)
      {
        // Get jacobian determinant at cubpts
        for (m=0;m<n_dims;m++)
          loc(m) = loc_volume_cubpts(m,j);
        
        calc_pos(loc,i,pos);
        calc_d_pos(loc,i,d_pos);
        
        if (n_dims==2)
        {
          vol_detjac_vol_cubpts(j)(i) = d_pos(0,0)*d_pos(1,1) - d_pos(0,1)*d_pos(1,0);
        }
        else if (n_dims==3)
        {
          vol_detjac_vol_cubpts(j)(i) = d_pos(0,0)*(d_pos(1,1)*d_pos(2,2) - d_pos(1,2)*d_pos(2,1))
          - d_pos(0,1)*(d_pos(1,0)*d_pos(2,2) - d_pos(1,2)*d_pos(2,0))
          + d_pos(0,2)*(d_pos(1,0)*d_pos(2,1) - d_pos(1,1)*d_pos(2,0));
        }
      }
    }
  }
}

// get a pointer to the transformed discontinuous solution at a flux point

double* eles::get_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return disu_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
  return disu_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
}

// get a pointer to the normal transformed continuous inviscid flux at a flux point

double* eles::get_norm_tconf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return norm_tconf_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
  return norm_tconf_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
  
}

// get a pointer to the determinant of the jacobian at a flux point

double* eles::get_detjac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return detjac_fpts.get_ptr_gpu(fpt,in_ele);
#else
  return detjac_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get a pointer to the determinant of the jacobian at a flux point (dynamic->static)

double* eles::get_detjac_dyn_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
  int i;

  int fpt;

  fpt=in_inter_local_fpt;

  for(i=0;i<in_ele_local_inter;i++)
    {
      fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
  return J_dyn_fpts.get_ptr_gpu(fpt,in_ele);
#else
  return J_dyn_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get a pointer to the magnitude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at flux points

double* eles::get_tdA_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return tdA_fpts.get_ptr_gpu(fpt,in_ele);
#else
  return tdA_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get pointer to the equivalent of 'dA' (face area) at a flux point in dynamic physical space */

double* eles::get_ndA_dyn_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele)
{
  int i;

  int fpt;

  fpt=in_inter_local_fpt;

  for(i=0;i<in_ele_local_inter;i++)
    {
      fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
  return ndA_dyn_fpts.get_ptr_gpu(fpt,in_ele);
#else
  return ndA_dyn_fpts.get_ptr_cpu(fpt,in_ele);
#endif
}

// get a pointer to the normal at a flux point

double* eles::get_norm_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return norm_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
#else
  return norm_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
#endif
}

// get a pointer to the normal at a flux point in dynamic space

double* eles::get_norm_dyn_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
  int i;

  int fpt;

  fpt=in_inter_local_fpt;

  for(i=0;i<in_ele_local_inter;i++)
    {
      fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
  return norm_dyn_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
#else
  return norm_dyn_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
#endif
}

// get a CPU pointer to the coordinates at a flux point

double* eles::get_loc_fpts_ptr_cpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
  return pos_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
}

// get a GPU pointer to the coordinates at a flux point

double* eles::get_loc_fpts_ptr_gpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
  return pos_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
}

// get a CPU pointer to the coordinates at a flux point

double* eles::get_pos_dyn_fpts_ptr_cpu(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele)
{
  int i;

  int fpt;

  fpt=in_inter_local_fpt;

  for(i=0;i<in_ele_local_inter;i++)
    {
      fpt+=n_fpts_per_inter(i);
    }

  return dyn_pos_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
}

// get a pointer to delta of the transformed discontinuous solution at a flux point

double* eles::get_delta_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  //if (ele2global_ele(in_ele)==53)
  //{
  //  cout << "HERE" << endl;
  //  cout << "local_face=" << in_ele_local_inter << endl;
  //}
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return delta_disu_fpts.get_ptr_gpu(fpt,in_ele,in_field);
#else
  return delta_disu_fpts.get_ptr_cpu(fpt,in_ele,in_field);
#endif
}

// get a pointer to gradient of discontinuous solution at a flux point

double* eles::get_grad_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_field, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return grad_disu_fpts.get_ptr_gpu(fpt,in_ele,in_field,in_dim);
#else
  return grad_disu_fpts.get_ptr_cpu(fpt,in_ele,in_field,in_dim);
#endif
}

double* eles::get_normal_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele, array<double> temp_loc, double temp_pos[3])
{
  
  array<double> pos(n_dims);
  double dist = 0.0, min_dist = 1E6;
  int min_index = 0;
  
  // find closest solution point
  
  for (int i=0; i<n_upts_per_ele; i++) {
    
    calc_pos_upt(i, in_ele, pos);
    
    dist = 0.0;
    for(int j=0;j<n_dims;j++) {
      dist += (pos(j)-temp_loc(j))*(pos(j)-temp_loc(j));
    }
    dist = sqrt(dist);
    
    if (dist < min_dist) {
      min_dist = dist;
      min_index = i;
      for(int j=0;j<n_dims;j++) {
        temp_pos[j] = pos(j);
      }
    }
    
  }
  
#ifdef _GPU
  return disu_upts(0).get_ptr_gpu(min_index,in_ele,in_field);
#else
  return disu_upts(0).get_ptr_cpu(min_index,in_ele,in_field);
#endif
  
}

// get a pointer to the normal transformed continuous viscous flux at a flux point
/*
 double* eles::get_norm_tconvisf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele)
 {
 int i;
 
 int fpt;
 
 fpt=in_inter_local_fpt;
 
 for(i=0;i<in_ele_local_inter;i++)
 {
 fpt+=n_fpts_per_inter(i);
 }
 
 #ifdef _GPU
 return norm_tconvisf_fpts.get_ptr_gpu(fpt,in_ele,in_field);
 #else
 return norm_tconvisf_fpts.get_ptr_cpu(fpt,in_ele,in_field);
 #endif
 }
 */

// Get a pointer to the grid velocity at a flux point */
double* eles::get_grid_vel_fpts_ptr(int in_ele, int in_ele_local_inter, int in_inter_local_fpt, int in_dim)
{
    int i;

    int fpt;

    fpt=in_inter_local_fpt;

    for(i=0;i<in_ele_local_inter;i++)
    {
        fpt+=n_fpts_per_inter(i);
    }

#ifdef _GPU
    return grid_vel_fpts.get_ptr_gpu(fpt,in_ele,in_dim);
#else
    return grid_vel_fpts.get_ptr_cpu(fpt,in_ele,in_dim);
#endif
}

// get a pointer to the subgrid-scale flux at a flux point
double* eles::get_sgsf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_dim, int in_ele)
{
  int i;
  
  int fpt;
  
  fpt=in_inter_local_fpt;
  
  for(i=0;i<in_ele_local_inter;i++)
  {
    fpt+=n_fpts_per_inter(i);
  }
  
#ifdef _GPU
  return sgsf_fpts.get_ptr_gpu(fpt,in_ele,in_field,in_dim);
#else
  return sgsf_fpts.get_ptr_cpu(fpt,in_ele,in_field,in_dim);
#endif
}

//#### helper methods ####

// calculate position

void eles::calc_pos(array<double> in_loc, int in_ele, array<double>& out_pos)
{
  int i,j;
  
  for(i=0;i<n_dims;i++)
  {
    out_pos(i)=0.0;
    
    for(j=0;j<n_spts_per_ele(in_ele);j++)
    {
      out_pos(i)+=eval_nodal_s_basis(j,in_loc,n_spts_per_ele(in_ele))*shape(i,j,in_ele);
    }
  }
  
}

/** find the position of a point within the element (r,s,t -> xd,yd,zd) (using positions in dynamic grid) */
void eles::calc_pos_dyn(array<double> in_loc, int in_ele, array<double>& out_pos)
{
    int i,j;

    for(i=0;i<n_dims;i++) {
        out_pos(i)=0.0;

        for(j=0;j<n_spts_per_ele(in_ele);j++) {
            out_pos(i)+=eval_nodal_s_basis(j,in_loc,n_spts_per_ele(in_ele))*shape_dyn(i,j,in_ele);
        }
    }
}

/** find the physical position of a flux point within the element (using positions in dynamic grid) */
void eles::calc_pos_dyn_fpt(int in_fpt, int in_ele, array<double>& out_pos)
{
    int i,j;

    out_pos.initialize_to_zero();
    for(i=0;i<n_dims;i++) {
        for(j=0;j<n_spts_per_ele(in_ele);j++) {
            out_pos(i)+=nodal_s_basis_fpts(j,in_fpt,in_ele)*shape_dyn(i,j,in_ele);
        }
    }
}

/** find the physical position of a solution point within the element (using positions in dynamic grid) */
void eles::calc_pos_dyn_upt(int in_upt, int in_ele, array<double>& out_pos)
{
    int i,j;

    out_pos.initialize_to_zero();
    for(i=0;i<n_dims;i++) {
        for(j=0;j<n_spts_per_ele(in_ele);j++) {
            out_pos(i)+=nodal_s_basis_fpts(j,in_upt,in_ele)*shape_dyn(i,j,in_ele);
        }
    }
}

/** find the physical position of a plot point within the element (using positions in dynamic grid) */
void eles::calc_pos_dyn_ppt(int in_ppt, int in_ele, array<double>& out_pos)
{
    int i,j;

    out_pos.initialize_to_zero();
    for(i=0;i<n_dims;i++) {
        for(j=0;j<n_spts_per_ele(in_ele);j++) {
            out_pos(i)+=nodal_s_basis_ppts(j,in_ppt,in_ele)*shape_dyn(i,j,in_ele);
        }
    }
}

/** find the physical position of a volume cubature point within the element (using positions in dynamic grid) */
void eles::calc_pos_dyn_vol_cubpt(int in_ppt, int in_ele, array<double>& out_pos)
{
    int i,j;

    out_pos.initialize_to_zero();
    for(i=0;i<n_dims;i++) {
        for(j=0;j<n_spts_per_ele(in_ele);j++) {
            out_pos(i)+=nodal_s_basis_vol_cubpts(j,in_ppt,in_ele)*shape_dyn(i,j,in_ele);
        }
    }
}

/** find the physical position of a interface cubature point within the element (using positions in dynamic grid) */
void eles::calc_pos_dyn_inters_cubpt(int in_cubpt, int in_face, int in_ele, array<double>& out_pos)
{
    int i,j;

    out_pos.initialize_to_zero();
    for(i=0;i<n_dims;i++) {
        for(j=0;j<n_spts_per_ele(in_ele);j++) {
            out_pos(i)+=nodal_s_basis_inters_cubpts(in_face)(j,in_cubpt,in_ele)*shape_dyn(i,j,in_ele);
        }
    }
}

// calculate derivative of position - NEEDS TO BE OPTIMIZED
/** Calculate derivative of position wrt computational space (dx/dr, dx/ds, etc.) */
void eles::calc_d_pos(array<double> in_loc, int in_ele, array<double>& out_d_pos)
{
  int i,j,k;

  eval_d_nodal_s_basis(d_nodal_s_basis,in_loc,n_spts_per_ele(in_ele));

  for(j=0;j<n_dims;j++) {
    for(k=0;k<n_dims;k++) {
      out_d_pos(j,k)=0.0;
      for(i=0;i<n_spts_per_ele(in_ele);i++) {
        out_d_pos(j,k)+=d_nodal_s_basis(i,k)*shape(j,i,in_ele);
      }
    }
  }
}

/**
 * Calculate derivative of static position wrt computational-space position at upt
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_upt - ID of solution point within element to evaluate at
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dxi_j
 */
void eles::calc_d_pos_upt(int in_upt, int in_ele, array<double>& out_d_pos)
{
  int i,j,k;

  // Calculate dx/d<c>
  out_d_pos.initialize_to_zero();
  for(j=0;j<n_dims;j++) {
    for(k=0;k<n_dims;k++) {
      for(i=0;i<n_spts_per_ele(in_ele);i++) {
        out_d_pos(j,k)+=d_nodal_s_basis_upts(k,i,in_upt,in_ele)*shape(j,i,in_ele);
        //out_d_pos(j,k)+=d_nodal_s_basis_upts(in_upt,in_ele,k,i)*shape(j,i,in_ele);
      }
    }
  }
}

/**
 * Calculate derivative of static position wrt computational-space position at fpt
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_fpt - ID of flux point within element to evaluate at
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dxi_j
 */
void eles::calc_d_pos_fpt(int in_fpt, int in_ele, array<double>& out_d_pos)
{
  int i,j,k;

  // Calculate dx/d<c>
  out_d_pos.initialize_to_zero();
  for(j=0;j<n_dims;j++) {
    for(k=0;k<n_dims;k++) {
      for(i=0;i<n_spts_per_ele(in_ele);i++) {
        out_d_pos(j,k)+=d_nodal_s_basis_fpts(k,i,in_fpt,in_ele)*shape(j,i,in_ele);
        //out_d_pos(j,k)+=d_nodal_s_basis_fpts(in_fpt,in_ele,k,i)*shape(j,i,in_ele);
      }
    }
  }
}

/**
 * Calculate derivative of dynamic position wrt reference (initial,static) position
 * \param[in] in_loc - position of point in computational space
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
 */
void eles::calc_d_pos_dyn(array<double> in_loc, int in_ele, array<double>& out_d_pos)
{
  int i,j,k;

  eval_d_nodal_s_basis(d_nodal_s_basis,in_loc,n_spts_per_ele(in_ele));

  // Calculate dx/d<c>
  out_d_pos.initialize_to_zero();
  for(j=0;j<n_dims;j++) {
    for(k=0;k<n_dims;k++) {
      for(i=0;i<n_spts_per_ele(in_ele);i++) {
        out_d_pos(j,k)+=d_nodal_s_basis(i,k)*shape_dyn(j,i,in_ele);
      }
    }
  }
}

/**
 * Calculate derivative of dynamic physical position wrt static/reference physical position at fpt
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_fpt - ID of flux point within element to evaluate at
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
 */
void eles::calc_d_pos_dyn_fpt(int in_fpt, int in_ele, array<double>& out_d_pos)
{
  if (run_input.motion==4) {
    // Analytical formula for perturbed motion
    out_d_pos(0,0) = 1 + 0.2*pi*cos(pi*pos_fpts(in_fpt,in_ele,0)/10)*sin(pi*pos_fpts(in_fpt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
    out_d_pos(0,1) = 0.2*pi*sin(pi*pos_fpts(in_fpt,in_ele,0)/10)*cos(pi*pos_fpts(in_fpt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
    out_d_pos(1,0) = 0.2*pi*cos(pi*pos_fpts(in_fpt,in_ele,0)/10)*sin(pi*pos_fpts(in_fpt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
    out_d_pos(1,1) = 1 + 0.2*pi*sin(pi*pos_fpts(in_fpt,in_ele,0)/10)*cos(pi*pos_fpts(in_fpt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
  }
  else
  {
    // For all cases which do not have an analytical solution (i.e. for linear-elasticity deformation)
    int i,j,k;

    // Calculate dx/dr
    array<double> dxdr(n_dims,n_dims);
    dxdr.initialize_to_zero();
    for(i=0; i<n_dims; i++) {
      for(j=0; j<n_dims; j++) {
        for(k=0; k<n_spts_per_ele(in_ele); k++) {
          dxdr(i,j) += shape_dyn(i,k,in_ele)*d_nodal_s_basis_fpts(j,k,in_fpt,in_ele);
          //dxdr(i,j) += shape_dyn(i,k,in_ele)*d_nodal_s_basis_fpts(in_fpt,in_ele,j,k);
        }
      }
    }

    // Calculate dx/dX using transformation matrix
    out_d_pos.initialize_to_zero();
    for(i=0; i<n_dims; i++) {
      for(j=0; j<n_dims; j++) {
        for(k=0; k<n_dims; k++) {
          out_d_pos(i,j) += dxdr(i,k)*JGinv_fpts(k,j,in_fpt,in_ele)/detjac_fpts(in_fpt,in_ele);
        }
      }
    }
  }
}

/**
 * Calculate derivative of dynamic physical position wrt static/reference physical position at upt
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_upt - ID of solution point within element to evaluate at
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
 */
void eles::calc_d_pos_dyn_upt(int in_upt, int in_ele, array<double>& out_d_pos)
{
  if (run_input.motion==4) {
    // Analytical formula for perturbed motion test case
    out_d_pos(0,0) = 1 + 0.2*pi*cos(pi*pos_upts(in_upt,in_ele,0)/10)*sin(pi*pos_upts(in_upt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
    out_d_pos(0,1) = 0.2*pi*sin(pi*pos_upts(in_upt,in_ele,0)/10)*cos(pi*pos_upts(in_upt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
    out_d_pos(1,0) = 0.2*pi*cos(pi*pos_upts(in_upt,in_ele,0)/10)*sin(pi*pos_upts(in_upt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
    out_d_pos(1,1) = 1 + 0.2*pi*sin(pi*pos_upts(in_upt,in_ele,0)/10)*cos(pi*pos_upts(in_upt,in_ele,1)/10)*sin(2*pi*run_input.rk_time/10);
  }
  else
  {
    // For all cases which do not have an analytical solution (i.e. for linear-elasticity deformation)
    int i,j,k;

    // Calculate dx/dr
    array<double> dxdr(n_dims,n_dims);
    dxdr.initialize_to_zero();
    for(i=0; i<n_dims; i++) {
      for(j=0; j<n_dims; j++) {
        for(k=0; k<n_spts_per_ele(in_ele); k++) {
          dxdr(i,j) += shape_dyn(i,k,in_ele)*d_nodal_s_basis_upts(j,k,in_upt,in_ele);
          //dxdr(i,j) += shape_dyn(i,k,in_ele)*d_nodal_s_basis_upts(in_upt,in_ele,j,k);
        }
      }
    }

    // Calculate dx/dX using transformation matrix
    out_d_pos.initialize_to_zero();
    for(i=0; i<n_dims; i++) {
      for(j=0; j<n_dims; j++) {
        for(k=0; k<n_dims; k++) {
          out_d_pos(i,j) += dxdr(i,k)*JGinv_upts(k,j,in_upt,in_ele)/detjac_upts(in_upt,in_ele);
        }
      }
    }
  }
}

/**
 * Calculate derivative of dynamic physical position wrt static/reference physical position at volume cubature point
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_cubpt - ID of volume cubature point within element to evaluate at
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
 */
void eles::calc_d_pos_dyn_vol_cubpt(int in_cubpt, int in_ele, array<double>& out_d_pos)
{
  int i,j,k;

  // Calculate dx/dr
  array<double> dxdr(n_dims,n_dims);
  dxdr.initialize_to_zero();
  for(i=0; i<n_dims; i++) {
    for(j=0; j<n_dims; j++) {
      for(k=0; k<n_spts_per_ele(in_ele); k++) {
        dxdr(i,j) += d_nodal_s_basis_vol_cubpts(j,k,in_cubpt,in_ele)*shape_dyn(i,k,in_ele);
      }
    }
  }

  // Apply chain rule: dx/dX = (dx/dr) / (dX/dr)
  // *** FIX ME *** (see proper method above - fpts, upts method using JGinv)
  out_d_pos.initialize_to_zero();
  for (i=0; i<n_dims; i++) {
    for (j=0; j<n_dims; j++) {
      for (k=0; k<n_dims; k++) {
        //out_d_pos(i,j) += dxdr(i,k)/jac_vol_cubpts(in_cubpt),in_ele,j,k);
      }
    }
  }
}

/**
 * Calculate derivative of dynamic physical position wrt static/reference physical position at interface cubature point
 * Uses pre-computed nodal shape basis derivatives for efficiency
 * \param[in] in_cubpt - ID of interface cubature point within element to evaluate at
 * \param[in] in_face - Local ID of face within element
 * \param[in] in_ele - local element ID
 * \param[out] out_d_pos - array of size (n_dims,n_dims); (i,j) = dx_i / dX_j
 */
void eles::calc_d_pos_dyn_inters_cubpt(int in_cubpt, int in_face, int in_ele, array<double>& out_d_pos)
{
  int i,j,k;

  // Calculate dx/dr
  array<double> dxdr(n_dims,n_dims);
  dxdr.initialize_to_zero();
  for(i=0; i<n_dims; i++) {
    for(j=0; j<n_dims; j++) {
      for(k=0; k<n_spts_per_ele(in_ele); k++) {
        dxdr(i,j) += d_nodal_s_basis_inters_cubpts(in_face)(j,k,in_cubpt,in_ele)*shape_dyn(i,k,in_ele);
      }
    }
  }

  // Apply chain rule: dx/dX = (dx/dr) / (dX/dr)
  // *** FIX ME *** (see proper method above - fpts, upts method using JGinv)
  out_d_pos.initialize_to_zero();
  for (i=0; i<n_dims; i++) {
    for (j=0; j<n_dims; j++) {
      for (k=0; k<n_dims; k++) {
        //out_d_pos(i,j) += dxdr(i,k)/jac_inters_cubpts(in_face)(in_cubpt,in_ele,j,k);
      }
    }
  }
}

/*! Calculate residual sum for monitoring purposes */
double eles::compute_res_upts(int in_norm_type, int in_field) {
  
  int i, j;
  double sum = 0.;
  double cell_sum = 0.;
  
  // NOTE: div_tconf_upts must be on CPU
  
  for (i=0; i<n_eles; i++) {
    cell_sum=0;
    for (j=0; j<n_upts_per_ele; j++) {
      if (in_norm_type == 0) {
        cell_sum = max(cell_sum, abs(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-run_input.const_src-src_upts(j,i,in_field)));
      }
      if (in_norm_type == 1) {
        cell_sum += abs(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-run_input.const_src-src_upts(j,i,in_field));
      }
      else if (in_norm_type == 2) {
        cell_sum += (div_tconf_upts(0)(j, i, in_field)/detjac_upts(j,i)-run_input.const_src-src_upts(j,i,in_field))*(div_tconf_upts(0)(j, i, in_field)/detjac_upts(j, i)-run_input.const_src-src_upts(j,i,in_field));
      }
    }
    if (in_norm_type==0)
      sum = max(cell_sum,sum);
    else
      sum += cell_sum;
  }
  
  return sum;
  
}


array<double> eles::compute_error(int in_norm_type, double& time)
{
  array<double> disu_cubpt(n_fields);
  array<double> grad_disu_cubpt(n_fields,n_dims);
  double detjac;
  array<double> pos(n_dims);
  
  array<double> error(2,n_fields);  //storage
  array<double> error_sum(2,n_fields);  //output
  
  for (int i=0; i<n_fields; i++)
  {
    error_sum(0,i) = 0.;
    error_sum(1,i) = 0.;
  }
  
  for (int i=0;i<n_eles;i++)
  {
    for (int j=0;j<n_cubpts_per_ele;j++)
    {
      // Get jacobian determinant at cubpts
      detjac = vol_detjac_vol_cubpts(j)(i);
      
      // Get the solution at cubature point
      for (int m=0;m<n_fields;m++)
      {
        disu_cubpt(m) = 0.;
        for (int k=0;k<n_upts_per_ele;k++)
        {
          disu_cubpt(m) += opp_volume_cubpts(j,k)*disu_upts(0)(k,i,m);
        }
      }
      
      // Get the gradient at cubature point
      if (viscous==1)
      {
        for (int m=0;m<n_fields;m++) {
          for (int n=0;n<n_dims;n++) {
            double value=0.;
            for (int k=0;k<n_upts_per_ele;k++) {
              value += opp_volume_cubpts(j,k)*grad_disu_upts(k,i,m,n);
            }
            grad_disu_cubpt(m,n) = value;
            //cout << value << endl;
          }
        }
      }
      
      error = get_pointwise_error(disu_cubpt,grad_disu_cubpt,pos,time,in_norm_type);
      
      for (int m=0;m<n_fields;m++) {
        error_sum(0,m) += error(0,m)*weight_volume_cubpts(j)*detjac;
        error_sum(1,m) += error(1,m)*weight_volume_cubpts(j)*detjac;
      }
    }
  }
  
  cout << "time   " << time << endl;
  
  return error_sum;
}


array<double> eles::get_pointwise_error(array<double>& sol, array<double>& grad_sol, array<double>& loc, double& time, int in_norm_type)
{
  array<double> error(2,n_fields);  //output
  
  array<double> error_sol(n_fields);
  array<double> error_grad_sol(n_fields,n_dims);
  
  for (int i=0; i<n_fields; i++) {
    error_sol(i) = 0.;
    
    error(0,i) = 0.;
    error(1,i) = 0.;
    
    for (int j=0; j<n_dims; j++) {
      error_grad_sol(i,j) = 0.;
    }
  }
  
  if (run_input.test_case==1) // Isentropic vortex
  {
    // Computing error in all quantities
    double rho,vx,vy,vz,p;
    eval_isentropic_vortex(loc,time,rho,vx,vy,vz,p,n_dims);
    
    error_sol(0) = sol(0) - rho;
    error_sol(1) = sol(1) - rho*vx;
    error_sol(2) = sol(2) - rho*vy;
    error_sol(3) = sol(3) - (p/(run_input.gamma-1) + 0.5*rho*(vx*vx+vy*vy));
  }
  else if (run_input.test_case==2) // Sine Wave (single)
  {
    double rho;
    array<double> grad_rho(n_dims);
    
    if(viscous) {
      eval_sine_wave_single(loc,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
    }
    else {
      eval_sine_wave_single(loc,run_input.wave_speed,0.,time,rho,grad_rho,n_dims);
    }
    
    error_sol(0) = sol(0) - rho;
    
    for (int j=0; j<n_dims; j++) {
      error_grad_sol(0,j) = grad_sol(0,j) - grad_rho(j);
    }
    
  }
  else if (run_input.test_case==3) // Sine Wave (group)
  {
    double rho;
    array<double> grad_rho(n_dims);
    
    if(viscous) {
      eval_sine_wave_group(loc,run_input.wave_speed,run_input.diff_coeff,time,rho,grad_rho,n_dims);
    }
    else {
      eval_sine_wave_group(loc,run_input.wave_speed,0.,time,rho,grad_rho,n_dims);
    }
    
    error_sol(0) = sol(0) - rho;
    
    for (int j=0; j<n_dims; j++) {
      error_grad_sol(0,j) = grad_sol(0,j) - grad_rho(j);
    }
  }
  else if (run_input.test_case==4) // Sphere Wave
  {
    double rho;
    eval_sphere_wave(loc,run_input.wave_speed,time,rho,n_dims);
    error_sol(0) = sol(0) - rho;
  }
  else if (run_input.test_case==5) // Couette flow
  {
    int ind;
    double ene, u_wall;
    array<double> grad_ene(n_dims);
    
    u_wall = run_input.v_wall(0);
    
    eval_couette_flow(loc,run_input.gamma, run_input.R_ref, u_wall, run_input.T_wall, run_input.p_bound, run_input.prandtl, time, ene, grad_ene, n_dims);
    
    ind = n_dims+1;
    
    error_sol(ind) = sol(ind) - ene;
    
    for (int j=0; j<n_dims; j++) {
      error_grad_sol(ind,j) = grad_sol(ind,j) - grad_ene(j);
    }
  }
  else {
    FatalError("Test case not recognized in compute error, exiting");
  }
  
  if (in_norm_type==1)
  {
    for (int m=0;m<n_fields;m++) {
      error(0,m) += abs(error_sol(m));
      
      for(int n=0;n<n_dims;n++) {
        error(1,m) += abs(error_grad_sol(m,n)); //might be incorrect
      }
    }
  }
  
  if (in_norm_type==2)
  {
    for (int m=0;m<n_fields;m++) {
      error(0,m) += error_sol(m)*error_sol(m);
      
      for(int n=0;n<n_dims;n++) {
        error(1,m) += error_grad_sol(m,n)*error_grad_sol(m,n);
      }
    }
  }
  
  return error;
}

// Calculate body forcing term for periodic channel flow. HARDCODED FOR THE CHANNEL AND PERIODIC HILL!

void eles::evaluate_body_force(int in_file_num)
{
//#ifdef _CPU
  
  if (n_eles!=0) {
    int i,j,k,l,m,ele;
    double area, vol, detjac, ubulk, wgt;
    double mdot0, mdot_old, alpha, dt;
    array <int> inflowinters(n_bdy_eles,n_inters_per_ele);
    array <double> body_force(n_fields);
    array <double> disu_cubpt(4);
    array <double> integral(4);
    array <double> norm(n_dims), flow(n_dims), loc(n_dims), pos(n_dims);
    ofstream write_mdot;
    bool open_mdot;

    for (i=0;i<4;i++)
    {
      integral(i)=0.0;
    }

    // zero the interface flags
    for (i=0;i<n_bdy_eles;i++)
    {
      for (l=0;l<n_inters_per_ele;l++)
      {
        inflowinters(i,l)=0;
      }
    }

    // Mass flux on inflow boundary
    // Integrate density and x-velocity over inflow area
    for (i=0;i<n_bdy_eles;i++)
    {
      ele = bdy_ele2ele(i);
      for (l=0;l<n_inters_per_ele;l++)
      {
        if(inflowinters(i,l)!=1) // only unflagged inters
        {
          // HACK: Inlet is always a Cyclic (9) BC
          if(bctype(ele,l) == 9)
          {
            // Get the normal
            for (m=0;m<n_dims;m++)
            {
              norm(m) = norm_inters_cubpts(l)(0,i,m);
            }

            // HACK: inflow plane normal direction is -x
            if(norm(0)==-1)
            {
              inflowinters(i,l)=1; // Flag this interface
            }
          }
        }
      }
    }
    
    // Now loop over flagged inters
    for (i=0;i<n_bdy_eles;i++)
    {
      ele = bdy_ele2ele(i);
      for (l=0;l<n_inters_per_ele;l++)
      {
        if(inflowinters(i,l)==1)
        {
          for (j=0;j<n_cubpts_per_inter(l);j++)
          {
            wgt = weight_inters_cubpts(l)(j);
            detjac = inter_detjac_inters_cubpts(l)(j,i);

            for (m=0;m<4;m++)
            {
              disu_cubpt(m) = 0.;
            }

            // Get the solution at cubature point
            for (k=0;k<n_upts_per_ele;k++)
            {
              for (m=0;m<4;m++)
              {
                disu_cubpt(m) += opp_inters_cubpts(l)(j,k)*disu_upts(0)(k,ele,m);
              }
            }
            for (m=0;m<4;m++)
            {
              integral(m) += wgt*disu_cubpt(m)*detjac;
            }
          }
        }
      }
    }

#ifdef _MPI

    array<double> integral_global(4);
    for (m=0;m<4;m++)
    {
      integral_global(m) = 0.;
      MPI_Allreduce(&integral(m), &integral_global(m), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      integral(m) = integral_global(m);
    }

#endif

    // case-specific parameters
    // periodic channel:
    //area = 2*pi;
    //vol = 4*pi**2;

    // periodic hill (HIOCFD3 Case 3.4):
    area = 9.162;
    vol = 114.34;
    mdot0 = 9.162; // initial mass flux

    // get old mass flux
    if(run_input.restart_flag==0 and in_file_num == 0)
      mdot_old = mdot0;
    else if(run_input.restart_flag==1 and in_file_num == run_input.restart_iter)
      mdot_old = mdot0;
    else
      mdot_old = mass_flux;

    // get timestep
    if (run_input.dt_type == 0)
      dt = run_input.dt;
    else if (run_input.dt_type == 1)
      dt = dt_local(0);
    else if (run_input.dt_type == 2)
      FatalError("Not sure what value of timestep to use in body force term when using local timestepping.");

    // bulk velocity
    if(integral(0)==0)
      ubulk = 0.0;
    else
      ubulk = integral(1)/integral(0);

    // compute new mass flux
    mass_flux = ubulk*integral(0);

    //alpha = 1; // relaxation parameter

    // set body force for streamwise momentum and energy
    body_force(0) = 0.;
    //body_force(1) = alpha/area/dt*(mdot0 - mass_flux); // modified SD3D version
    body_force(1) = 1.0/area/dt*(mdot0 - 2.0*mass_flux + mdot_old); // HIOCFD C3.4 version
    body_force(2) = 0.;
    body_force(3) = 0.;
    body_force(4) = body_force(1)*ubulk; // energy forcing

    if (rank == 0) cout << "iter, mdot0, mdot_old, mass_flux, body_force(1): " << in_file_num << ", " << setprecision(8) << mdot0 << ", " << mdot_old << ", " << mass_flux << ", " << body_force(1) << endl;

    // write out mass flux to file
    if (rank == 0) {
      if (run_input.restart_flag==0 and in_file_num == 1) {
        // write file header
        write_mdot.open("massflux.dat", ios::out);
        write_mdot << "Iteration, massflux, Ubulk, bodyforce(x)" << endl;
        write_mdot.close();
      }
      else {
        // append subsequent dqata
        write_mdot.open("massflux.dat", ios::app);
        write_mdot.precision(15);
        write_mdot << in_file_num;
        write_mdot << ", " << mass_flux;
        write_mdot << ", " << ubulk;
        write_mdot << ", " << body_force(1) << endl;
        write_mdot.close();
      }
    }
    // error checking
    if(isnan(body_force(1))) {
      FatalError("ERROR: NaN body force, exiting");
    }

//#endif

//TODO: GPU version of above?
//#ifdef _GPU

//#endif

#ifdef _CPU

    // Add to source term at solution points
    for (i=0;i<n_eles;i++)
      for (j=0;j<n_upts_per_ele;j++)
        for(k=0;k<n_fields;k++)
          src_upts(j,i,k) += body_force(k);

#endif

#ifdef _GPU
    body_force.cp_cpu_gpu();
    evaluate_body_force_gpu_kernel_wrapper(n_upts_per_ele,n_dims,n_fields,n_eles,src_upts.get_ptr_gpu(),body_force.get_ptr_gpu());
#endif
  }
}

// Compute integral quantities
void eles::CalcIntegralQuantities(int n_integral_quantities, array <double>& integral_quantities)
{
  array<double> disu_cubpt(n_fields);
  array<double> grad_disu_cubpt(n_fields,n_dims);
  array<double> S(n_dims,n_dims);
  double wx, wy, wz;
  double dudx, dudy, dudz;
  double dvdx, dvdy, dvdz;
  double dwdx, dwdy, dwdz;
  double diagnostic, tke, pressure, diag, irho, detjac;
  
  // Sum over elements
  for (int i=0;i<n_eles;i++)
  {
    for (int j=0;j<n_cubpts_per_ele;j++)
    {
      // Get jacobian determinant at cubpts
      detjac = vol_detjac_vol_cubpts(j)(i);
      
      // Get the solution at cubature point
      for (int m=0;m<n_fields;m++)
      {
        disu_cubpt(m) = 0.;
        for (int k=0;k<n_upts_per_ele;k++)
        {
          disu_cubpt(m) += opp_volume_cubpts(j,k)*disu_upts(0)(k,i,m);
        }
      }
      // Get the solution gradient at cubature point
      for (int m=0;m<n_fields;m++)
      {
        for (int n=0;n<n_dims;n++)
        {
          grad_disu_cubpt(m,n)=0.;
          for (int k=0;k<n_upts_per_ele;k++)
          {
            grad_disu_cubpt(m,n) += opp_volume_cubpts(j,k)*grad_disu_upts(k,i,m,n);
          }
        }
      }
      irho = 1./disu_cubpt(0);
      dudx = irho*(grad_disu_cubpt(1,0) - disu_cubpt(1)*irho*grad_disu_cubpt(0,0));
      dudy = irho*(grad_disu_cubpt(1,1) - disu_cubpt(1)*irho*grad_disu_cubpt(0,1));
      dvdx = irho*(grad_disu_cubpt(2,0) - disu_cubpt(2)*irho*grad_disu_cubpt(0,0));
      dvdy = irho*(grad_disu_cubpt(2,1) - disu_cubpt(2)*irho*grad_disu_cubpt(0,1));
      
      if (n_dims==3)
      {
        dudz = irho*(grad_disu_cubpt(1,2) - disu_cubpt(1)*irho*grad_disu_cubpt(0,2));
        dvdz = irho*(grad_disu_cubpt(2,2) - disu_cubpt(2)*irho*grad_disu_cubpt(0,2));
        dwdx = irho*(grad_disu_cubpt(3,0) - disu_cubpt(3)*irho*grad_disu_cubpt(0,0));
        dwdy = irho*(grad_disu_cubpt(3,1) - disu_cubpt(3)*irho*grad_disu_cubpt(0,1));
        dwdz = irho*(grad_disu_cubpt(3,2) - disu_cubpt(3)*irho*grad_disu_cubpt(0,2));
      }
      
      // Now calculate integral quantities
      for (int m=0;m<n_integral_quantities;++m)
      {
        diagnostic = 0.0;
        if (run_input.integral_quantities(m)=="kineticenergy")
        {
          // Compute kinetic energy
          tke = 0.0;
          for (int n=1;n<n_fields-1;n++)
            tke += 0.5*disu_cubpt(n)*disu_cubpt(n);
          
          diagnostic = irho*tke;
        }
        else if (run_input.integral_quantities(m)=="vorticity")
        {
          // Compute vorticity squared
          wz = dvdx - dudy;
          diagnostic = wz*wz;
          if (n_dims==3)
          {
            wx = dwdy - dvdz;
            wy = dudz - dwdx;
            diagnostic += wx*wx+wy*wy;
          }
          diagnostic *= 0.5/irho;
        }
        else if (run_input.integral_quantities(m)=="pressuredilatation")
        {
          // Kinetic energy
          tke = 0.0;
          for (int n=1;n<n_fields-1;n++)
            tke += 0.5*disu_cubpt(n)*disu_cubpt(n);
          
          // Compute pressure
          pressure = (run_input.gamma-1.0)*(disu_cubpt(n_fields-1) - irho*tke);
          
          // Multiply pressure by divergence of velocity
          if (n_dims==2) {
            diagnostic = pressure*(dudx+dvdy);
          }
          else if (n_dims==3) {
            diagnostic = pressure*(dudx+dvdy+dwdz);
          }
        }
        else if (run_input.integral_quantities(m)=="straincolonproduct" || run_input.integral_quantities(m)=="devstraincolonproduct")
        {
          // Rate of strain tensor
          S(0,0) = dudx;
          S(0,1) = (dudy+dvdx)/2.0;
          S(1,0) = S(0,1);
          S(1,1) = dvdy;
          diag = (S(0,0)+S(1,1))/3.0;
          
          if (n_dims==3)
          {
            S(0,2) = (dudz+dwdx)/2.0;
            S(1,2) = (dvdz+dwdy)/2.0;
            S(2,0) = S(0,2);
            S(2,1) = S(1,2);
            S(2,2) = dwdz;
            diag += S(2,2)/3.0;
          }
          
          // Subtract diag if deviatoric strain
          if (run_input.integral_quantities(m)=="devstraincolonproduct") {
            for (int i=0;i<n_dims;i++)
              S(i,i) -= diag;
          }
          
          for (int i=0;i<n_dims;i++)
            for (int j=0;j<n_dims;j++)
              diagnostic += S(i,j)*S(i,j);
          
        }
        else
        {
          FatalError("integral diagnostic quantity not recognized");
        }
        // Add contribution to global integral
        integral_quantities(m) += diagnostic*weight_volume_cubpts(j)*detjac;
      }
    }
  }
}

// Compute time-averaged quantities
void eles::CalcTimeAverageQuantities(double& time)
{
  double current_value, average_value;
  double a, b, dt;
  double spinup_time = run_input.spinup_time;
  double rho;
  int i, j, k;

  for(j=0;j<n_upts_per_ele;j++) {
    for(k=0;k<n_eles;k++) {
      for(i=0;i<n_average_fields;++i) {

        rho = disu_upts(0)(j,k,0);

        if(run_input.average_fields(i)=="rho_average") {
          current_value = rho;
          average_value = disu_average_upts(j,k,0);
        }
        else if(run_input.average_fields(i)=="u_average") {
          current_value = disu_upts(0)(j,k,1)/rho;
          average_value = disu_average_upts(j,k,1);
        }
        else if(run_input.average_fields(i)=="v_average") {
          current_value = disu_upts(0)(j,k,2)/rho;
          average_value = disu_average_upts(j,k,2);
        }
        else if(run_input.average_fields(i)=="w_average") {
          current_value = disu_upts(0)(j,k,3)/rho;
          average_value = disu_average_upts(j,k,3);
        }
        else if(run_input.average_fields(i)=="e_average") {
          if(n_dims==2) {
            current_value = disu_upts(0)(j,k,3)/rho;
            average_value = disu_average_upts(j,k,3);
          }
          else {
            current_value = disu_upts(0)(j,k,4)/rho;
            average_value = disu_average_upts(j,k,4);
          }
        }

        // get timestep
        if (run_input.dt_type == 0)
          dt = run_input.dt;
        else if (run_input.dt_type == 1)
          dt = dt_local(0);
        else if (run_input.dt_type == 2)
          FatalError("Not sure what value of timestep to use in time average calculation when using local timestepping.");

        // set average value to current value if before spinup time
        // and prevent division by a very small number if time = spinup time
        if(time-spinup_time < 1.0e-12) {
          a = 0.0;
          b = 1.0;
        }
        // calculate running average
        else {
          a = (time-spinup_time-dt)/(time-spinup_time);
          b = dt/(time-spinup_time);
        }

        // Set new average value for next timestep
        if(run_input.average_fields(i)=="rho_average") {
          disu_average_upts(j,k,0) = a*average_value + b*current_value;
        }
        else if(run_input.average_fields(i)=="u_average") {
          disu_average_upts(j,k,1) = a*average_value + b*current_value;
        }
        else if(run_input.average_fields(i)=="v_average") {
          disu_average_upts(j,k,2) = a*average_value + b*current_value;
        }
        else if(run_input.average_fields(i)=="w_average") {
          disu_average_upts(j,k,3) = a*average_value + b*current_value;
        }
        else if(run_input.average_fields(i)=="e_average") {
          if(n_dims==2) {
            disu_average_upts(j,k,3) = a*average_value + b*current_value;
          }
          else {
            disu_average_upts(j,k,4) = a*average_value + b*current_value;
          }
        }
      }
    }
  }
}

void eles::compute_wall_forces( array<double>& inv_force, array<double>& vis_force,  double& temp_cl, double& temp_cd, ofstream& coeff_file, bool write_forces)
{
  
  array<double> u_l(n_fields),norm(n_dims);
  double p_l,v_sq,vn_l;
  array<double> grad_u_l(n_fields,n_dims);
  array<double> dv(n_dims,n_dims);
  array<double> de(n_dims);
  array<double> drho(n_dims);
  array<double> taun(n_dims);
  array<double> tautan(n_dims);
  array<double> Finv(n_dims);
  array<double> Fvis(n_dims);
  array<double> loc(n_dims);
  array<double> pos(n_dims);
  double inte, mu, rt_ratio, gamma=run_input.gamma;
  double diag, tauw, taundotn, wgt, detjac;
  double factor, aoa, aos, cp, cf, cl, cd;
  
  // Need to add a reference area to the input file... Not needed for Cp/Cf,
  // but will be needed for forces if not equal to 1.0 for different geometries
  
  double area_ref = 1.0;
  
  for (int m=0;m<n_dims;m++)
  {
    Finv(m) = 0.;
    Fvis(m) = 0.;
    inv_force(m) = 0.;
    vis_force(m) = 0.;
  }
  
  temp_cd = 0.0;
  temp_cl = 0.0;
  
  // angle of attack
  aoa = atan2(run_input.v_c_ic, run_input.u_c_ic);
  
  // angle of side slip
  if (n_dims == 3)
  {
    aos = atan2(run_input.w_c_ic, run_input.u_c_ic);
  }
  
  // one over the dynamic pressure - factor for computing friction coeff, pressure coeff, forces
  factor = 1.0 / (0.5*run_input.rho_c_ic*(run_input.u_c_ic*run_input.u_c_ic+run_input.v_c_ic*run_input.v_c_ic+run_input.w_c_ic*run_input.w_c_ic));

  // Add a header to the force file
  if (write_forces) { coeff_file << setw(18) << "x" << setw(18) << "Cp" << setw(18) << "Cf" << endl; }
  
  // loop over the boundary elements
  for (int i=0;i<n_bdy_eles;i++) {

      int ele = bdy_ele2ele(i);
    
      // loop over the interfaces of the element
      for (int l=0;l<n_inters_per_ele;l++) {

          if (bctype(ele,l) == 7 || bctype(ele,l) == 11 || bctype(ele,l) == 12 || bctype(ele,l)==16) {

              // Compute force on this interface
              for (int j=n_cubpts_per_inter(l)-1;j>=0;j--)
                {
                  // Get determinant of Jacobian (=area of interface)
                  detjac = inter_detjac_inters_cubpts(l)(j,i);

                  // Get cubature weight
                  wgt = weight_inters_cubpts(l)(j);

                  // Get position of cubature point
                  for (int m=0;m<n_dims;m++)
                    loc(m) = loc_inters_cubpts(l)(m,j);

                  calc_pos(loc,ele,pos);

                  // Compute solution at current cubature point
                  for (int m=0;m<n_fields;m++) {
                      double value = 0.;
                      for (int k=0;k<n_upts_per_ele;k++) {
                          value += opp_inters_cubpts(l)(j,k)*disu_upts(0)(k,ele,m);
                        }
                      u_l(m) = value;
                    }

                  // If viscous, extrapolate the gradient at the cubature points
                  if (viscous==1)
                    {
                      for (int m=0;m<n_fields;m++) {
                          for (int n=0;n<n_dims;n++) {
                              double value=0.;
                              for (int k=0;k<n_upts_per_ele;k++) {
                                  value += opp_inters_cubpts(l)(j,k)*grad_disu_upts(k,ele,m,n);
                                }
                              grad_u_l(m,n) = value;
                            }
                        }
                    }

                  // Get the normal
                  for (int m=0;m<n_dims;m++)
                    {
                      norm(m) = norm_inters_cubpts(l)(j,i,m);
                    }

                  // Get pressure

                  // Not dual consistent
                  if (bctype(ele,l)!=16) {
                      v_sq = 0.;
                      for (int m=0;m<n_dims;m++)
                        v_sq += (u_l(m+1)*u_l(m+1));
                      p_l   = (gamma-1.0)*( u_l(n_dims+1) - 0.5*v_sq/u_l(0));
                    }
                  else
                    {
                      //Dual consistent approach
                      vn_l = 0.;
                      for (int m=0;m<n_dims;m++)
                        vn_l += u_l(m+1)*norm(m);
                      vn_l /= u_l(0);

                      for (int m=0;m<n_dims;m++)
                        u_l(m+1) = u_l(m+1)-(vn_l)*norm(m);

                      v_sq = 0.;
                      for (int m=0;m<n_dims;m++)
                        v_sq += (u_l(m+1)*u_l(m+1));
                      p_l   = (gamma-1.0)*( u_l(n_dims+1) - 0.5*v_sq/u_l(0));
                    }
                  
                  // calculate pressure coefficient at current point on the surface
                  cp = (p_l-run_input.p_c_ic)*factor;
                  
                  // Inviscid force
                  for (int m=0;m<n_dims;m++)
                    {
                      Finv(m) = wgt*(p_l-run_input.p_c_ic)*norm(m)*detjac*factor;
                    }
                  
                  // inviscid component of the lift and drag coefficients
                  
                  if (n_dims==2)
                  {
                    cl = -Finv(0)*sin(aoa) + Finv(1)*cos(aoa);
                    cd = Finv(0)*cos(aoa) + Finv(1)*sin(aoa);
                  }
                  else if (n_dims==3)
                  {
                    cl = -Finv(0)*sin(aoa) + Finv(1)*cos(aoa);
                    cd = Finv(0)*cos(aoa)*cos(aos) + Finv(1)*sin(aoa) + Finv(2)*sin(aoa)*cos(aos);
                  }
                  
                  // write to file
                  if (write_forces) { coeff_file << scientific << setw(18) << setprecision(12) << pos(0) << " " << setw(18) << setprecision(12) << cp;}

                  if (viscous==1)
                    {
                      // TODO: Have a function that returns tau given u and grad_u
                      // Computing the n_dims derivatives of rho,u,v,w and ene
                      for (int m=0;m<n_dims;m++)
                        {
                          drho(m) = grad_u_l(0,m);
                          for (int n=0;n<n_dims;n++)
                            {
                              dv(n,m) = 1.0/u_l(0)*(grad_u_l(n+1,m)-drho(m)*u_l(n+1));;
                            }
                          de(m) = 1.0/u_l(0)*(grad_u_l(n_dims+1,m)-drho(m)*u_l(n_dims+1));;
                        }

                      // trace of stress tensor
                      diag = 0.;
                      for (int m=0;m<n_dims;m++)
                        {
                          diag += dv(m,m);
                        }
                      diag /= 3.0;

                      // internal energy
                      inte = u_l(n_dims+1)/u_l(0);
                      for (int m=0;m<n_dims;m++)
                        {
                          inte -= 0.5*u_l(m+1)*u_l(m+1);
                        }

                      // get viscosity
                      rt_ratio = (run_input.gamma-1.0)*inte/(run_input.rt_inf);
                      mu = (run_input.mu_inf)*pow(rt_ratio,1.5)*(1+(run_input.c_sth))/(rt_ratio+(run_input.c_sth));
                      mu = mu + run_input.fix_vis*(run_input.mu_inf - mu);

                      // Compute the coefficient of friction and wall shear stress
                      if (n_dims==2)
                        {
                          // stresses w.r.t. normal
                          taun(0) = mu*(2.*(dv(0,0)-diag)*norm(0) + (dv(0,1)+dv(1,0))*norm(1));
                          taun(1) = mu*(2.*(dv(1,1)-diag)*norm(1) + (dv(0,1)+dv(1,0))*norm(0));

                          // take dot product with normal
                          taundotn = taun(0)*norm(0)+taun(1)*norm(1);
                          
                          // stresses tangent to wall
                          tautan(0) = taun(0) - taundotn*norm(0);
                          tautan(1) = taun(1) - taundotn*norm(1);

                          // wall shear stress
                          tauw = sqrt(pow(tautan(0),2)+pow(tautan(1),2));

                          // coefficient of friction
                          cf = tauw*factor;
                          
                          if (write_forces) { coeff_file << " " << setw(18) <<setprecision(12) << cf; }

                          // viscous force
                          for (int m=0;m<n_dims;m++)
                          {
                            Fvis(m) = -wgt*taun(m)*detjac*factor;
                          }
                          
                          // viscous component of the lift and drag coefficients
                          
                          cl += -Fvis(0)*sin(aoa) + Fvis(1)*cos(aoa);
                          cd += Fvis(0)*cos(aoa) + Fvis(1)*sin(aoa);
                          
                        }

                      if (n_dims==3)
                        {
                          // stresses w.r.t. normal
                          taun(0) = mu*(2.*(dv(0,0)-diag)*norm(0) + (dv(0,1)+dv(1,0))*norm(1) + (dv(0,2)+dv(2,0))*norm(2));
                          taun(1) = mu*(2.*(dv(1,1)-diag)*norm(1) + (dv(0,1)+dv(1,0))*norm(0) + (dv(1,2)+dv(2,1))*norm(2));
                          taun(2) = mu*(2.*(dv(2,2)-diag)*norm(2) + (dv(0,2)+dv(2,0))*norm(0) + (dv(1,2)+dv(2,1))*norm(1));
                          
                          // take dot product with normal
                          taundotn = taun(0)*norm(0)+taun(1)*norm(1)+taun(2)*norm(2);
                          
                          // stresses tangent to wall
                          tautan(0) = taun(0) - taundotn*norm(0);
                          tautan(1) = taun(1) - taundotn*norm(1);
                          tautan(2) = taun(2) - taundotn*norm(2);
                          
                          // wall shear stress
                          tauw = sqrt(pow(tautan(0),2)+pow(tautan(1),2)+pow(tautan(2),2));
                          
                          // coefficient of friction
                          cf = tauw*factor;
                          
                          if (write_forces) { coeff_file << " " << setw(18) <<setprecision(12) << cf; }
                          
                          // viscous force
                          for (int m=0;m<n_dims;m++)
                            {
                              Fvis(m) = -wgt*taun(m)*detjac*factor;
                            }

                          // viscous component of the lift and drag coefficients
                          // TODO: make this work for any orientation of axes
                          
                          cl += -Fvis(0)*sin(aoa) + Fvis(1)*cos(aoa);
                          cd += Fvis(0)*cos(aoa)*cos(aos) + Fvis(1)*sin(aoa) + Fvis(2)*sin(aoa)*cos(aos);
                          
                        }
                    } // End of if viscous

                  if (write_forces) { coeff_file << endl; }
                  
                  // Add force and coefficient contributions from current face
                  for (int m=0;m<n_dims;m++)
                  {
                    inv_force(m) += Finv(m);
                    vis_force(m) += Fvis(m);
                  }
                  temp_cl += cl;
                  temp_cd += cd;
                }
            }
        }
    }
}

/*! Set the grid velocity at one shape point
 *  TODO: CUDA */
void eles::set_grid_vel_spt(int in_ele, int in_spt, array<double> in_vel)
{
  for (int i=0; i<n_dims; i++)
    vel_spts(i,in_spt,in_ele) = in_vel(i);
}

/*! Store nodal basis at flux points to avoid re-calculating every time
 *  TODO: CUDA (mv to GPU) */
void eles::store_nodal_s_basis_fpts(void)
{
  int ic,fpt,j,k;
  array<double> loc(n_dims);
  for (ic=0; ic<n_eles; ic++) {
    for (fpt=0; fpt<n_fpts_per_ele; fpt++) {
      for(k=0;k<n_dims;k++) {
        loc(k) = tloc_fpts(k,fpt);
      }
      for(j=0;j<n_spts_per_ele(ic);j++) {
        nodal_s_basis_fpts(j,fpt,ic) = eval_nodal_s_basis(j,loc,n_spts_per_ele(ic));
      }
    }
  }
#ifdef _GPU
  nodal_s_basis_fpts.cp_cpu_gpu();
#endif
}

void eles::store_nodal_s_basis_upts(void)
{
  int ic,upt,j,k;
  array<double> loc(n_dims);
  for (ic=0; ic<n_eles; ic++) {
    for (upt=0; upt<n_upts_per_ele; upt++) {
      for(k=0;k<n_dims;k++) {
        loc(k) = loc_upts(k,upt);
      }
      for(j=0;j<n_spts_per_ele(ic);j++) {
        nodal_s_basis_upts(j,upt,ic) = eval_nodal_s_basis(j,loc,n_spts_per_ele(ic));
      }
    }
  }
#ifdef _GPU
  nodal_s_basis_upts.cp_cpu_gpu();
#endif
}

void eles::store_nodal_s_basis_ppts(void)
{
  int ic,ppt,j,k;

  array<double> loc(n_dims);
  for(ic=0; ic<n_eles; ic++) {
    for(ppt=0; ppt<n_ppts_per_ele; ppt++) {
      for(k=0; k<n_dims; k++) {
        loc(k)=loc_ppts(k,ppt);
      }
      for (j=0; j<n_spts_per_ele(ic); j++) {
        nodal_s_basis_ppts(j,ppt,ic) = eval_nodal_s_basis(j,loc,n_spts_per_ele(ic));
      }
    }
  }
}

void eles::store_nodal_s_basis_vol_cubpts(void)
{
  int ic,cubpt,j,k;

  array<double> loc(n_dims);
  for(ic=0; ic<n_eles; ic++) {
    for(cubpt=0; cubpt<n_cubpts_per_ele; cubpt++) {
      for(k=0; k<n_dims; k++) {
        loc(k)=loc_volume_cubpts(k,cubpt);
      }
      for (j=0; j<n_spts_per_ele(ic); j++) {
        nodal_s_basis_vol_cubpts(j,cubpt,ic) = eval_nodal_s_basis(j,loc,n_spts_per_ele(ic));
      }
    }
  }
}

void eles::store_nodal_s_basis_inters_cubpts()
{
  int ic,iface,cubpt,j,k;

  array<double> loc(n_dims);
  for(ic=0; ic<n_eles; ic++) {
    for(iface=0; iface<n_inters_per_ele; iface++) {
      for(cubpt=0; cubpt<n_cubpts_per_inter(iface); cubpt++) {
        for(k=0; k<n_dims; k++) {
          loc(k)=loc_inters_cubpts(iface)(k,cubpt);
        }
        for (j=0; j<n_spts_per_ele(ic); j++) {
          nodal_s_basis_inters_cubpts(iface)(j,cubpt,ic) = eval_nodal_s_basis(j,loc,n_spts_per_ele(ic));
        }
      }
    }
  }
}


void eles::store_d_nodal_s_basis_fpts(void)
{
  int ic,fpt,j,k;
  array<double> loc(n_dims);
  array<double> d_nodal_basis;

  for (ic=0; ic<n_eles; ic++) {
    for (fpt=0; fpt<n_fpts_per_ele; fpt++) {
      for(k=0;k<n_dims;k++) {
        loc(k) = tloc_fpts(k,fpt);
      }
      d_nodal_basis.setup(n_spts_per_ele(ic),n_dims);
      eval_d_nodal_s_basis(d_nodal_basis,loc,n_spts_per_ele(ic));
      for (j=0; j<n_spts_per_ele(ic); j++) {
        for (k=0; k<n_dims; k++) {
          d_nodal_s_basis_fpts(k,j,fpt,ic) = d_nodal_basis(j,k);
          //d_nodal_s_basis_fpts(fpt,ic,k,j) = d_nodal_basis(j,k);
        }
      }
    }
  }
#ifdef _GPU
  d_nodal_s_basis_fpts.cp_cpu_gpu();
#endif
}


void eles::store_d_nodal_s_basis_upts(void)
{
  int ic,upt,j,k;
  array<double> loc(n_dims);
  array<double> d_nodal_basis;

  for (ic=0; ic<n_eles; ic++) {
    for (upt=0; upt<n_upts_per_ele; upt++) {
      for(k=0;k<n_dims;k++) {
        loc(k) = loc_upts(k,upt);
      }
      d_nodal_basis.setup(n_spts_per_ele(ic),n_dims);
      eval_d_nodal_s_basis(d_nodal_basis,loc,n_spts_per_ele(ic));
      for (j=0; j<n_spts_per_ele(ic); j++) {
        for (k=0; k<n_dims; k++) {
          //d_nodal_s_basis_upts(upt,ic,k,j) = d_nodal_basis(j,k);
          d_nodal_s_basis_upts(k,j,upt,ic) = d_nodal_basis(j,k);
        }
      }
    }
  }
#ifdef _GPU
  d_nodal_s_basis_upts.cp_cpu_gpu();
#endif
}

void eles::store_d_nodal_s_basis_vol_cubpts(void)
{
  int ic,cubpt,j,k;
  array<double> loc(n_dims);
  array<double> d_nodal_basis;

  for (ic=0; ic<n_eles; ic++) {
    for (cubpt=0; cubpt<n_cubpts_per_ele; cubpt++) {
      for(k=0;k<n_dims;k++) {
        loc(k) = loc_volume_cubpts(k,cubpt);
      }
      d_nodal_basis.setup(n_spts_per_ele(ic),n_dims);
      eval_d_nodal_s_basis(d_nodal_basis,loc,n_spts_per_ele(ic));
      for (j=0; j<n_spts_per_ele(ic); j++) {
        for (k=0; k<n_dims; k++) {
          d_nodal_s_basis_vol_cubpts(k,j,cubpt,ic) = d_nodal_basis(j,k);
        }
      }
    }
  }
}

void eles::store_d_nodal_s_basis_inters_cubpts(void)
{
  int ic,iface,cubpt,j,k;
  array<double> loc(n_dims);
  array<double> d_nodal_basis;

  for (ic=0; ic<n_eles; ic++) {
    for (iface=0; iface<n_inters_per_ele; iface++) {
      for (cubpt=0; cubpt<n_cubpts_per_inter(iface); cubpt++) {
        for(k=0;k<n_dims;k++) {
          loc(k) = loc_inters_cubpts(iface)(k,cubpt);
        }
        d_nodal_basis.setup(n_spts_per_ele(ic),n_dims);
        eval_d_nodal_s_basis(d_nodal_basis,loc,n_spts_per_ele(ic));
        for (j=0; j<n_spts_per_ele(ic); j++) {
          for (k=0; k<n_dims; k++) {
            d_nodal_s_basis_inters_cubpts(iface)(k,j,cubpt,ic) = d_nodal_basis(j,k);
          }
        }
      }
    }
  }
}

/*! Setup arrays for storing grid velocity */
void eles::initialize_grid_vel(int in_max_n_spts_per_ele)
{
  if (motion)
  {
    // At solution & flux points
    grid_vel_fpts.setup(n_fpts_per_ele,n_eles,n_dims);
    grid_vel_fpts.initialize_to_zero();

    grid_vel_upts.setup(n_upts_per_ele,n_eles,n_dims);
    grid_vel_upts.initialize_to_zero();

    // At output / plotting points
    vel_ppts.setup(n_dims,n_ppts_per_ele,n_eles);
    vel_ppts.initialize_to_zero();

    // at shape points
    vel_spts.setup(n_dims,in_max_n_spts_per_ele,n_eles);
    vel_spts.initialize_to_zero();

    /// TODO: after other mesh stuff implemented in CUDA, *.mv_cpu_gpu()
    /// ALSO: *_nodal_s_basis data is redundant: same for every element (of same type)
  }
}

/*! Interpolate the grid velocity from shape points to flux points
 *  TODO: Find a way to speed up with BLAS or something
 *  TODO: Implement these routines in CUDA (just 'for' loops - easy!)
 *  (would have to use sparse BLAS - think block-diag matrix) */
void eles::set_grid_vel_fpts(int in_rk_step)
{
  int ic,fpt,j,k;
//  if (run_input.motion==3) {
//    double rk_time;
//    rk_time = run_input.time+RK_c(in_rk_step)*run_input.dt;
//    for (ic=0; ic<n_eles; ic++) {
//      for (fpt=0; fpt<n_fpts_per_ele; fpt++) {
//        grid_vel_fpts(fpt,ic,0) = 4*pi/10*sin(pi*pos_fpts(fpt,ic,0)/10)*sin(pi*pos_fpts(fpt,ic,1)/10)*cos(2*pi*rk_time/10);
//        grid_vel_fpts(fpt,ic,1) = 4*pi/10*sin(pi*pos_fpts(fpt,ic,0)/10)*sin(pi*pos_fpts(fpt,ic,1)/10)*cos(2*pi*rk_time/10);
//      }
//    }
//  }
//  else
//  {
    for (ic=0; ic<n_eles; ic++) {
      for (fpt=0; fpt<n_fpts_per_ele; fpt++) {
        for(k=0;k<n_dims;k++) {
          grid_vel_fpts(fpt,ic,k) = 0.0;
          for(j=0;j<n_spts_per_ele(ic);j++) {
            grid_vel_fpts(fpt,ic,k)+=nodal_s_basis_fpts(j,fpt,ic)*vel_spts(k,j,ic);
          }
        }
      }
    }
//  }
#ifdef _GPU
  //grid_vel_fpts.cp_cpu_gpu();
  eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_fpts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_fpts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_fpts.get_ptr_gpu());
#endif
}

/*! Interpolate the grid velocity from shape points to solution points
 *  TODO: Find a way to speed up with BLAS or something
 *  TODO: Implement these routines in CUDA (just 'for' loops - easy!) */
void eles::set_grid_vel_upts(int in_rk_step)
{
  int ic,upt,j,k;
//  if (run_input.motion==3) {
//    double rk_time;
//    rk_time = run_input.time+RK_c(in_rk_step)*run_input.dt;
//    for (ic=0; ic<n_eles; ic++) {
//      for (upt=0; upt<n_upts_per_ele; upt++) {
//        grid_vel_upts(upt,ic,0) = 4*pi/10*sin(pi*pos_upts(upt,ic,0)/10)*sin(pi*pos_upts(upt,ic,1)/10)*cos(2*pi*rk_time/10);
//        grid_vel_upts(upt,ic,1) = 4*pi/10*sin(pi*pos_upts(upt,ic,0)/10)*sin(pi*pos_upts(upt,ic,1)/10)*cos(2*pi*rk_time/10);
//      }
//    }
//  }
//  else
//  {
    for (ic=0; ic<n_eles; ic++) {
      for (upt=0; upt<n_upts_per_ele; upt++) {
        for(k=0;k<n_dims;k++) {
          grid_vel_upts(upt,ic,k) = 0.0;
          for(j=0;j<n_spts_per_ele(ic);j++) {
            grid_vel_upts(upt,ic,k)+=nodal_s_basis_upts(j,upt,ic)*vel_spts(k,j,ic);
          }
        }
      }
    }
//  }
#ifdef _GPU
  //grid_vel_upts.cp_cpu_gpu();
  eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_upts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_upts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_upts.get_ptr_gpu());
#endif
}


/*! Interpolate the grid velocity from shape points to solution points
 *  TODO: Find a way to speed up with BLAS or something */
void eles::set_grid_vel_ppts(void)
{
  int ic,ppt,j,k;
  for (ic=0; ic<n_eles; ic++) {
    for (ppt=0; ppt<n_ppts_per_ele; ppt++) {
      for(k=0;k<n_dims;k++) {
        vel_ppts(k,ppt,ic) = 0.0;
        for(j=0;j<n_spts_per_ele(ic);j++) {
          vel_ppts(k,ppt,ic)+=nodal_s_basis_ppts(j,ppt,ic)*vel_spts(k,j,ic);
        }
      }
    }
  }
}

array<double> eles::get_grid_vel_ppts(void)
{
  return vel_ppts;
}

#ifdef _GPU
void eles::rigid_move(double rk_time)
{
  if (n_eles!=0) {
    rigid_motion_kernel_wrapper(n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),shape.get_ptr_gpu(),shape_dyn.get_ptr_gpu(),run_input.bound_vel_simple(0).get_ptr_gpu(),rk_time);
  }
}

void eles::perturb_shape(double rk_time)
{
  if (n_eles!=0) {
    push_back_shape_dyn_kernel_wrapper(n_dims,n_eles,max_n_spts_per_ele,5,n_spts_per_ele.get_ptr_gpu(),shape_dyn.get_ptr_gpu());
    perturb_shape_kernel_wrapper(n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),shape.get_ptr_gpu(),shape_dyn.get_ptr_gpu(),rk_time);
  }
}

void eles::rigid_grid_velocity(double rk_time)
{
  if (n_eles!=0) {
    calc_rigid_grid_vel_spts_kernel_wrapper(n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),run_input.bound_vel_simple(0).get_ptr_gpu(),vel_spts.get_ptr_gpu(),rk_time);
    eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_upts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_upts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_upts.get_ptr_gpu());
    eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_fpts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_fpts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_fpts.get_ptr_gpu());
  }
}

void eles::perturb_grid_velocity(double rk_time)
{
  if (n_eles!=0) {
    calc_perturb_grid_vel_spts_kernel_wrapper(n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),shape.get_ptr_gpu(),vel_spts.get_ptr_gpu(),rk_time);
    eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_upts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_upts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_upts.get_ptr_gpu());
    eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_fpts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_fpts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_fpts.get_ptr_gpu());
  }
}

void eles::calc_grid_velocity(void)
{
  if (n_eles!=0) {
    calc_grid_vel_spts_kernel_wrapper(n_dims,n_eles,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),shape_dyn.get_ptr_gpu(),vel_spts.get_ptr_gpu(),run_input.dt);
    eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_upts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_upts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_upts.get_ptr_gpu());
    eval_grid_vel_pts_kernel_wrapper(n_dims,n_eles,n_fpts_per_ele,max_n_spts_per_ele,n_spts_per_ele.get_ptr_gpu(),nodal_s_basis_fpts.get_ptr_gpu(),vel_spts.get_ptr_gpu(),grid_vel_fpts.get_ptr_gpu());
  }
}
#endif
