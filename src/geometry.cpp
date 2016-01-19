/*!
 * \file geometry.cpp
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
#include <sstream>
#include <cmath>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/input.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/funcs.h"
#include "../include/error.h"
#include "../include/solution.h"

#ifdef _TECIO
#include "TECIO.h"
#endif

#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif

#ifdef _GPU
#include "../include/util.h"
#endif

using namespace std;

#define MAX_V_PER_F 4
#define MAX_F_PER_C 6
#define MAX_E_PER_C 12
#define MAX_V_PER_C 27

void SetInput(struct solution* FlowSol) {

  /*! Basic allocation using the input file. */
  FlowSol->rank               = 0;
  FlowSol->n_steps            = run_input.n_steps;
  FlowSol->adv_type           = run_input.adv_type;
  FlowSol->viscous            = run_input.viscous;
  FlowSol->plot_freq          = run_input.plot_freq;
  FlowSol->restart_dump_freq  = run_input.restart_dump_freq;
  FlowSol->write_type         = run_input.write_type;
  FlowSol->ini_iter           = 0;

  /*! Number of edges/faces for different type of cells. */
  FlowSol->num_f_per_c.setup(5);
  FlowSol->num_f_per_c(0) = 3;
  FlowSol->num_f_per_c(1) = 4;
  FlowSol->num_f_per_c(2) = 4;
  FlowSol->num_f_per_c(3) = 5;
  FlowSol->num_f_per_c(4) = 6;

#ifdef _MPI

  /*! Get MPI rank and nproc. */
  MPI_Comm_rank(MPI_COMM_WORLD,&FlowSol->rank);
  MPI_Comm_size(MPI_COMM_WORLD,&FlowSol->nproc);
  cout << "my_rank=" << FlowSol->rank << endl;

#ifdef _GPU

  /*! Associate a GPU to each rank. */

  // Cluster:
#ifdef _YOSEMITESAM
  if (FlowSol->rank==0) { cout << "setting CUDA devices on yosemitesam ..." << endl; }
  if ((FlowSol->rank%2)==0) { cudaSetDevice(0); }
  if ((FlowSol->rank%2)==1) { cudaSetDevice(1); }
#endif

  // Enrico:
#ifdef _ENRICO
  if (FlowSol->rank==0) { cout << "setting CUDA devices on enrico ..." << endl; }
  if (FlowSol->rank==0) { cudaSetDevice(2); }
  else if (FlowSol->rank==1) { cudaSetDevice(0); }
  else if (FlowSol->rank==2) { cudaSetDevice(3); }
#endif

#ifndef _ENRICO
#ifndef _YOSEMITESAM
  // NOTE: depening on system arcihtecture, this may not be the GPU device you want
  // i.e. one of the devices may be a (non-scientific-computing) graphics card
  cudaSetDevice(FlowSol->rank);
#endif
#endif

#endif

#endif

}

int get_bc_number(string& bcname) {

  int bcflag;

  std::transform(bcname.begin(), bcname.end(), bcname.begin(), ::tolower);

  if (!bcname.compare("sub_in_simp")) bcflag = 1;         // Subsonic inflow simple (free pressure) //
  else if (!bcname.compare("sub_out_simp")) bcflag = 2;   // Subsonic outflow simple (fixed pressure) //
  else if (!bcname.compare("sub_in_char")) bcflag = 3;    // Subsonic inflow characteristic //
  else if (!bcname.compare("sub_out_char")) bcflag = 4;   // Subsonic outflow characteristic //
  else if (!bcname.compare("sup_in")) bcflag = 5;         // Supersonic inflow //
  else if (!bcname.compare("sup_out")) bcflag = 6;        // Supersonic outflow //
  else if (!bcname.compare("slip_wall")) bcflag = 7;      // Slip wall //
  else if (!bcname.compare("cyclic")) bcflag = 9;
  else if (!bcname.compare("isotherm_fix")) bcflag = 11;  // Isothermal, no-slip wall //
  else if (!bcname.compare("adiabat_fix")) bcflag = 12;   // Adiabatic, no-slip wall //
  else if (!bcname.compare("isotherm_move")) bcflag = 13; // Isothermal, no-slip moving wall //
  else if (!bcname.compare("adiabat_move")) bcflag = 14;  // Adiabatic, no-slip moving wall //
  else if (!bcname.compare("char")) bcflag = 15;          // Characteristic //
  else if (!bcname.compare("slip_wall_dual")) bcflag = 16;// Dual consistent BC //
  else if (!bcname.compare("ad_wall")) bcflag = 50;       // Advection, Advection-Diffusion Boundary Conditions //
  else
  {
    cout << "Boundary = " << bcname << endl;
    FatalError("Boundary condition not recognized");
  }

  return bcflag;
}

void GeoPreprocess(struct solution* FlowSol, mesh &Mesh) {
  array<double> xv;
  array<int> c2v,c2n_v,ctype,bctype_c,ic2icg,iv2ivg;

  /*! Reading vertices and cells. */
  ReadMesh(run_input.mesh_file, xv, c2v, c2n_v, ctype, ic2icg, iv2ivg, FlowSol->num_eles, FlowSol->num_verts, Mesh.n_verts_global, FlowSol);

  // ** TODO: clean up duplicate/redundant data between Mesh and FlowSol **
  Mesh.setup(FlowSol,xv,c2v,c2n_v,iv2ivg,ctype);

  /////////////////////////////////////////////////
  /// Set connectivity
  /////////////////////////////////////////////////

  array<int> f2c,f2loc_f,c2f,c2e,f2v,f2nv;
  array<int> rot_tag,unmatched_inters;
  int n_unmatched_inters;

  // cannot have more than num_eles*6 faces
  int max_inters = FlowSol->num_eles*MAX_F_PER_C;

  f2c.setup(max_inters,2);
  f2v.setup(max_inters,MAX_V_PER_F); // each edge/face cannot have more than 4 vertices
  f2nv.setup(max_inters);
  f2loc_f.setup(max_inters,2);
  c2f.setup(FlowSol->num_eles,MAX_F_PER_C); // one cell cannot have more than 8 faces
  c2e.setup(FlowSol->num_eles,MAX_E_PER_C); // one cell cannot have more than 8 faces
  rot_tag.setup(max_inters);
  unmatched_inters.setup(max_inters);

  Mesh.v2e.setup(Mesh.n_verts);
  Mesh.v2n_e.setup(Mesh.n_verts);
  Mesh.v2n_e.initialize_to_zero();

  // Initialize arrays to -1
  f2c.initialize_to_value(-1);
  f2loc_f.initialize_to_value(-1);
  c2f.initialize_to_value(-1);

  array<int> icvsta, icvert;

  // Compute connectivity
  if (FlowSol->rank==0) cout << "Setting up mesh connectivity" << endl;

  //CompConnectivity(c2v, c2n_v, ctype, c2f, c2e, f2c, f2loc_f, f2v, f2nv, rot_tag, unmatched_inters, n_unmatched_inters, icvsta, icvert, FlowSol->num_inters, FlowSol->num_edges, FlowSol);
  CompConnectivity(c2v, c2n_v, ctype, c2f, c2e, f2c, f2loc_f, f2v, f2nv, Mesh.e2v, Mesh.v2n_e, Mesh.v2e, rot_tag,
                   unmatched_inters, n_unmatched_inters, icvsta, icvert, FlowSol->num_inters, FlowSol->num_edges, FlowSol);

  if (FlowSol->rank==0) cout << "Done setting up mesh connectivity" << endl;

  // Reading boundaries
  //ReadBound(run_input.mesh_file,c2v,c2n_v,ctype,bctype_c,ic2icg,icvsta,icvert,iv2ivg,FlowSol->num_eles,FlowSol->num_verts, FlowSol);
  ReadBound(run_input.mesh_file,c2v,c2n_v,c2f,f2v,f2nv,ctype,bctype_c,Mesh.boundPts,Mesh.bc_list,Mesh.bound_flags,ic2icg,
            icvsta,icvert,iv2ivg,FlowSol->num_eles,FlowSol->num_verts,FlowSol);

  // ** TODO: clean up duplicate/redundant data **
  Mesh.c2f = c2f;
  Mesh.c2e = c2e;
  Mesh.f2c = f2c;
  Mesh.f2n_v = f2nv;
  Mesh.n_faces = FlowSol->num_inters;
  Mesh.n_bnds = Mesh.bc_list.get_dim(0);
  Mesh.nBndPts.setup(Mesh.n_bnds);
  for (int i=0; i<Mesh.n_bnds; i++) {
      Mesh.nBndPts(i) = Mesh.boundPts(i).get_dim(0);
  }

  /////////////////////////////////////////////////
  /// Initializing Elements
  /////////////////////////////////////////////////

  // Count the number of elements of each type
  int num_tris = 0;
  int num_quads= 0;
  int num_tets= 0;
  int num_pris= 0;
  int num_hexas= 0;

  for (int i=0;i<FlowSol->num_eles;i++) {
      if (ctype(i)==0) num_tris++;
      else if (ctype(i)==1) num_quads++;
      else if (ctype(i)==2) num_tets++;
      else if (ctype(i)==3) num_pris++;
      else if (ctype(i)==4) num_hexas++;
    }

  // Error checking
  if (FlowSol->n_dims == 2 && (num_tets != 0 || num_pris != 0 || num_hexas != 0)) {
      FatalError("Error in mesh reader, n_dims=2 and 3d elements exists");
    }
  if (FlowSol->n_dims == 3 && (num_tris!= 0 || num_quads != 0)) {
      FatalError("Error in mesh reader, n_dims=3 and 2d elements exists");
    }

  // For each element type, count the maximum number of shape points per element
  int max_n_spts_per_tri = 0;
  int max_n_spts_per_quad= 0;
  int max_n_spts_per_tet = 0;
  int max_n_spts_per_pri = 0;
  int max_n_spts_per_hexa = 0;

  for (int i=0;i<FlowSol->num_eles;i++) {
      if (ctype(i)==0 && c2n_v(i) > max_n_spts_per_tri ) // triangle
        max_n_spts_per_tri = c2n_v(i);
      else if (ctype(i)==1 && c2n_v(i) > max_n_spts_per_quad ) // quad
        max_n_spts_per_quad = c2n_v(i);
      else if (ctype(i)==2 && c2n_v(i) > max_n_spts_per_tet) // tet
        max_n_spts_per_tet = c2n_v(i);
      else if (ctype(i)==3 && c2n_v(i) > max_n_spts_per_pri) // pri
        max_n_spts_per_pri = c2n_v(i);
      else if (ctype(i)==4 && c2n_v(i) > max_n_spts_per_hexa) // hexa
        max_n_spts_per_hexa = c2n_v(i);
    }

  // Initialize the mesh_eles
  FlowSol->n_ele_types=5;
  FlowSol->mesh_eles.setup(FlowSol->n_ele_types);

  FlowSol->mesh_eles(0) = &FlowSol->mesh_eles_tris;
  FlowSol->mesh_eles(1) = &FlowSol->mesh_eles_quads;
  FlowSol->mesh_eles(2) = &FlowSol->mesh_eles_tets;
  FlowSol->mesh_eles(3) = &FlowSol->mesh_eles_pris;
  FlowSol->mesh_eles(4) = &FlowSol->mesh_eles_hexas;

  array<int> max_n_spts(FlowSol->n_ele_types);
  max_n_spts(0) = max_n_spts_per_tri;
  max_n_spts(1) = max_n_spts_per_quad;
  max_n_spts(2) = max_n_spts_per_tet;
  max_n_spts(3) = max_n_spts_per_pri;
  max_n_spts(4) = max_n_spts_per_hexa;

  for (int i=0;i<FlowSol->n_ele_types;i++)
    {
      FlowSol->mesh_eles(i)->set_rank(FlowSol->rank);
    }

  if (FlowSol->rank==0)
    cout << endl << "---------------- Flux Reconstruction Preprocessing ----------------" << endl;

  if (FlowSol->rank==0) cout << "initializing elements" << endl;
  if (FlowSol->rank==0) cout << "tris" << endl;
  FlowSol->mesh_eles_tris.setup(num_tris,max_n_spts_per_tri);
  if (FlowSol->rank==0) cout << "quads" << endl;
  FlowSol->mesh_eles_quads.setup(num_quads,max_n_spts_per_quad);
  if (FlowSol->rank==0) cout << "tets" << endl;
  FlowSol->mesh_eles_tets.setup(num_tets,max_n_spts_per_tet);
  if (FlowSol->rank==0) cout << "pris" << endl;
  FlowSol->mesh_eles_pris.setup(num_pris,max_n_spts_per_pri);
  if (FlowSol->rank==0) cout << "hexas" << endl;
  FlowSol->mesh_eles_hexas.setup(num_hexas,max_n_spts_per_hexa);
  if (FlowSol->rank==0) cout << "done initializing elements" << endl;

  // Set shape for each cell
  array<int> local_c(FlowSol->num_eles);

  int tris_count = 0;
  int quads_count = 0;
  int tets_count = 0;
  int pris_count = 0;
  int hexas_count = 0;

  array<double> pos(FlowSol->n_dims);

  if (FlowSol->rank==0) cout << "setting elements shape ... ";
  for (int i=0;i<FlowSol->num_eles;i++) {
      if (ctype(i) == 0) //tri
        {
          local_c(i) = tris_count;
          FlowSol->mesh_eles_tris.set_n_spts(tris_count,c2n_v(i));
          FlowSol->mesh_eles_tris.set_ele2global_ele(tris_count,ic2icg(i));

          for (int j=0;j<c2n_v(i);j++)
            {
              pos(0) = xv(c2v(i,j),0);
              pos(1) = xv(c2v(i,j),1);
              FlowSol->mesh_eles_tris.set_shape_node(j,tris_count,pos);
            }

          for (int j=0;j<3;j++) {
              FlowSol->mesh_eles_tris.set_bctype(tris_count,j,bctype_c(i,j));
            }

          tris_count++;
        }
      else if (ctype(i) == 1) // quad
        {
          local_c(i) = quads_count;
          FlowSol->mesh_eles_quads.set_n_spts(quads_count,c2n_v(i));
          FlowSol->mesh_eles_quads.set_ele2global_ele(quads_count,ic2icg(i));
          for (int j=0;j<c2n_v(i);j++)
            {
              pos(0) = xv(c2v(i,j),0);
              pos(1) = xv(c2v(i,j),1);
              FlowSol->mesh_eles_quads.set_shape_node(j,quads_count,pos);
            }

          for (int j=0;j<4;j++) {
              FlowSol->mesh_eles_quads.set_bctype(quads_count,j,bctype_c(i,j));
            }

          quads_count++;
        }
      else if (ctype(i) == 2) //tet
        {
          local_c(i) = tets_count;
          FlowSol->mesh_eles_tets.set_n_spts(tets_count,c2n_v(i));
          FlowSol->mesh_eles_tets.set_ele2global_ele(tets_count,ic2icg(i));
          for (int j=0;j<c2n_v(i);j++)
            {
              pos(0) = xv(c2v(i,j),0);
              pos(1) = xv(c2v(i,j),1);
              pos(2) = xv(c2v(i,j),2);
              FlowSol->mesh_eles_tets.set_shape_node(j,tets_count,pos);
            }

          for (int j=0;j<4;j++) {
              FlowSol->mesh_eles_tets.set_bctype(tets_count,j,bctype_c(i,j));
            }

          tets_count++;
        }
      else if (ctype(i) == 3) //pri
        {
          local_c(i) = pris_count;
          FlowSol->mesh_eles_pris.set_n_spts(pris_count,c2n_v(i));
          FlowSol->mesh_eles_pris.set_ele2global_ele(pris_count,ic2icg(i));
          for (int j=0;j<c2n_v(i);j++)
            {
              pos(0) = xv(c2v(i,j),0);
              pos(1) = xv(c2v(i,j),1);
              pos(2) = xv(c2v(i,j),2);
              FlowSol->mesh_eles_pris.set_shape_node(j,pris_count,pos);
            }

          for (int j=0;j<5;j++) {
              FlowSol->mesh_eles_pris.set_bctype(pris_count,j,bctype_c(i,j));
            }

          pris_count++;
        }
      else if (ctype(i) == 4) //hex
        {
          local_c(i) = hexas_count;
          FlowSol->mesh_eles_hexas.set_n_spts(hexas_count,c2n_v(i));
          FlowSol->mesh_eles_hexas.set_ele2global_ele(hexas_count,ic2icg(i));
          for (int j=0;j<c2n_v(i);j++)
            {
              pos(0) = xv(c2v(i,j),0);
              pos(1) = xv(c2v(i,j),1);
              pos(2) = xv(c2v(i,j),2);
              FlowSol->mesh_eles_hexas.set_shape_node(j,hexas_count,pos);
            }

          for (int j=0;j<6;j++) {
              FlowSol->mesh_eles_hexas.set_bctype(hexas_count,j,bctype_c(i,j));
            }

          hexas_count++;
        }
    }
  if (FlowSol->rank==0) cout << "done." << endl;

  // Pre-compute shape basis - CRITICAL for deforming-mesh performance
  if (FlowSol->rank==0) cout << "pre-computing nodal shape-basis functions ... " << flush;
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      FlowSol->mesh_eles(i)->store_nodal_s_basis_fpts();
      FlowSol->mesh_eles(i)->store_nodal_s_basis_upts();
      FlowSol->mesh_eles(i)->store_nodal_s_basis_ppts();
      FlowSol->mesh_eles(i)->store_d_nodal_s_basis_fpts();
      FlowSol->mesh_eles(i)->store_d_nodal_s_basis_upts();
      FlowSol->mesh_eles(i)->store_nodal_s_basis_inters_cubpts();
      FlowSol->mesh_eles(i)->store_d_nodal_s_basis_inters_cubpts();
    }
  }
  if (FlowSol->rank==0) cout << "done." << endl;

  // set transforms
  if (FlowSol->rank==0) cout << "setting element transforms ... " << endl;
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      FlowSol->mesh_eles(i)->set_transforms();
      if (run_input.motion)
        FlowSol->mesh_eles(i)->set_transforms_dynamic();
    }
  }
  if (FlowSol->rank==0) cout << "done." << endl;

  // Initialize grid velocity variables & set to 0
  if (FlowSol->rank==0) cout << "initializing grid velocity to 0 ... " << flush;
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      FlowSol->mesh_eles(i)->initialize_grid_vel(max_n_spts(i));
    }
  }
  if (FlowSol->rank==0) cout << "done." << endl;

  // Set metrics at interface cubpts
  if (FlowSol->rank==0) cout << "setting element transforms at interface cubpts ... ";
  for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
          FlowSol->mesh_eles(i)->set_transforms_inters_cubpts();
        }
    }
  if (FlowSol->rank==0) cout << "done." << endl;

  // Set metrics at volume cubpts. Only needed for computing error and integral diagnostic quantities.
  if (run_input.test_case != 0 || run_input.monitor_integrals_freq!=0) {
    if (FlowSol->rank==0) cout << "setting element transforms at volume cubpts ... " << endl;
    for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
        FlowSol->mesh_eles(i)->store_nodal_s_basis_vol_cubpts();
        FlowSol->mesh_eles(i)->store_d_nodal_s_basis_vol_cubpts();
        FlowSol->mesh_eles(i)->set_transforms_vol_cubpts();
      }
    }
  }

  // set on gpu (important - need to do this before we set connectivity, so that pointers point to GPU memory)
#ifdef _GPU
      for(int i=0;i<FlowSol->n_ele_types;i++) {
          if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

              if (FlowSol->rank==0) cout << "Moving eles to GPU ... " << endl;
              FlowSol->mesh_eles(i)->mv_all_cpu_gpu();
            }
        }
#endif

  // ------------------------------------
  // Initializing Interfaces
  // ------------------------------------

  int n_int_inters= 0;
  int n_bdy_inters= 0;
  int n_cyc_loc = 0;

  // -------------------------------------------------------
  // Split the cyclic faces as being internal or mpi faces
  // -------------------------------------------------------

  array<double> loc_center_inter_0(FlowSol->n_dims),loc_center_inter_1(FlowSol->n_dims);
  array<double> loc_vert_0(MAX_V_PER_F,FlowSol->n_dims),loc_vert_1(MAX_V_PER_F,FlowSol->n_dims);

  array<double> delta_cyclic(FlowSol->n_dims);
  delta_cyclic(0) = run_input.dx_cyclic;
  delta_cyclic(1) = run_input.dy_cyclic;
  if (FlowSol->n_dims==3) {
      delta_cyclic(2) = run_input.dz_cyclic;
    }

  double tol = 1.e-6;
  int bctype_f, found, rtag;
  int ic_l,ic_r;

  for(int i=0;i<FlowSol->num_inters;i++)
    {
      bctype_f = bctype_c( f2c(i,0),f2loc_f(i,0));
      if (bctype_f==9) {

          for (int m=0;m<FlowSol->n_dims;m++)
            loc_center_inter_0(m) = 0.;

          for (int k=0;k<f2nv(i);k++)
            for (int m=0;m<FlowSol->n_dims;m++)
              loc_center_inter_0(m) += xv(f2v(i,k),m)/f2nv(i);

          found = 0;
          for (int j=0;j<n_unmatched_inters;j++) {

              int i2 = unmatched_inters(j);

              for (int m=0;m<FlowSol->n_dims;m++)
                loc_center_inter_1(m) = 0.;

              for (int k=0;k<f2nv(i2);k++)
                for (int m=0;m<FlowSol->n_dims;m++)
                  loc_center_inter_1(m) += xv(f2v(i2,k),m)/f2nv(i2);

              if (check_cyclic(delta_cyclic,loc_center_inter_0,loc_center_inter_1,tol,FlowSol))
                {

                  found = 1;
                  f2c(i,1) = f2c(i2,0);
                  bctype_c(f2c(i,0),f2loc_f(i,0)) = 0;
                  // Change the flag of matching cyclic inter so that it's not counted as interior inter
                  bctype_c(f2c(i2,0),f2loc_f(i2,0)) = 99;

                  f2loc_f(i,1) = f2loc_f(i2,0);
                  n_cyc_loc++;

                  for(int k=0;k<f2nv(i);k++)
                    {
                      for (int m=0;m<FlowSol->n_dims;m++)
                        {
                          loc_vert_0(k,m) = xv(f2v(i,k),m);
                          loc_vert_1(k,m) = xv(f2v(i2,k),m);
                        }
                    }


                  compare_cyclic_faces(loc_vert_0,loc_vert_1,f2nv(i),rtag,delta_cyclic,tol,FlowSol);
                  rot_tag(i) = rtag;
                  break;
                }
            }
          if (found==0) // Corresponding cyclic edges belongs to another processsor
            {
              f2c(i,1) = -1;
              bctype_c(f2c(i,0),f2loc_f(i,0)) = 0;
            }
        }
    }

#ifdef _MPI


  // ---------------------------------
  //  Initialize MPI faces
  //  --------------------------------

  array<int> f_mpi2f(max_inters);
  FlowSol->n_mpi_inters = 0;
  int n_seg_mpi_inters=0;
  int n_tri_mpi_inters=0;
  int n_quad_mpi_inters=0;

  for (int i=0;i<FlowSol->num_inters;i++) {
      bctype_f = bctype_c( f2c(i,0),f2loc_f(i,0));
      ic_r = f2c(i,1);

      if (bctype_f==0 && ic_r==-1) { // mpi_face

          if (FlowSol->nproc==1)
            {
              cout << "ic=" << f2c(i,0) << endl;
              cout << "local_face=" << f2loc_f(i,0) << endl;
              FatalError("Should not be here");
            }

          bctype_c( f2c(i,0),f2loc_f(i,0)) = 10;
          f_mpi2f(FlowSol->n_mpi_inters) = i;
          FlowSol->n_mpi_inters++;

          if (f2nv(i)==2) n_seg_mpi_inters++;
          if (f2nv(i)==3) n_tri_mpi_inters++;
          if (f2nv(i)==4) n_quad_mpi_inters++;
        }
    }

  FlowSol->n_mpi_inter_types=3;
  FlowSol->mesh_mpi_inters.setup(FlowSol->n_mpi_inter_types);

  for (int i=0;i<FlowSol->n_mpi_inter_types;i++)
    FlowSol->mesh_mpi_inters(i).set_nproc(FlowSol->nproc,FlowSol->rank);

  FlowSol->mesh_mpi_inters(0).setup(n_seg_mpi_inters,0);
  FlowSol->mesh_mpi_inters(1).setup(n_tri_mpi_inters,1);
  FlowSol->mesh_mpi_inters(2).setup(n_quad_mpi_inters,2);

  array<int> mpifaces_part(FlowSol->nproc);

  // Call function that takes in f_mpi2f,f2v and returns a new array f_mpi2f, and an array mpiface_part
  // that contains the number of faces to send to each processor
  // the new array f_mpi2f is in good order i.e. proc1,proc2,....

  match_mpifaces(f2v,f2nv,xv,f_mpi2f,mpifaces_part,delta_cyclic,FlowSol->n_mpi_inters,tol,FlowSol);

  array<int> rot_tag_mpi(FlowSol->n_mpi_inters);
  find_rot_mpifaces(f2v,f2nv,xv,f_mpi2f,rot_tag_mpi,mpifaces_part,delta_cyclic,FlowSol->n_mpi_inters,tol,FlowSol);

  //Initialize the mpi faces

  int i_seg_mpi = 0;
  int i_tri_mpi = 0;
  int i_quad_mpi = 0;

  for(int i_mpi=0;i_mpi<FlowSol->n_mpi_inters;i_mpi++)
    {
      int i = f_mpi2f(i_mpi);
      bctype_f = bctype_c( f2c(i,0),f2loc_f(i,0) );
      ic_l = f2c(i,0);

      if (f2nv(i)==2) {
          FlowSol->mesh_mpi_inters(0).set_mpi(i_seg_mpi,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),rot_tag_mpi(i_mpi),FlowSol);
          i_seg_mpi++;
        }
      else if (f2nv(i)==3) {
          FlowSol->mesh_mpi_inters(1).set_mpi(i_tri_mpi,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),rot_tag_mpi(i_mpi),FlowSol);
          i_tri_mpi++;
        }
      else if (f2nv(i)==4) {
          FlowSol->mesh_mpi_inters(2).set_mpi(i_quad_mpi,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),rot_tag_mpi(i_mpi),FlowSol);
          i_quad_mpi++;
        }
    }

  // Initialize Nout_proc
  int icount = 0;

  int request_seg=0;
  int request_tri=0;
  int request_quad=0;

  for (int p=0;p<FlowSol->nproc;p++)
    {
      // For all faces to send to processor p, split between face types
      int Nout_seg = 0;
      int Nout_tri = 0;
      int Nout_quad = 0;

      for (int j=0;j<mpifaces_part(p);j++)
        {
          int i_mpi = icount + j;
          int i = f_mpi2f(i_mpi);
          if (f2nv(i)==2)  Nout_seg++;
          else if (f2nv(i)==3)  Nout_tri++;
          else if (f2nv(i)==4)  Nout_quad++;
        }
      icount += mpifaces_part(p);

      if (Nout_seg!=0) {
          FlowSol->mesh_mpi_inters(0).set_nout_proc(Nout_seg,p);
          request_seg++;
        }
      if (Nout_tri!=0) {
          FlowSol->mesh_mpi_inters(1).set_nout_proc(Nout_tri,p);
          request_tri++;
        }
      if (Nout_quad!=0) {
          FlowSol->mesh_mpi_inters(2).set_nout_proc(Nout_quad,p);
          request_quad++;
        }
    }

  FlowSol->mesh_mpi_inters(0).set_mpi_requests(request_seg);
  FlowSol->mesh_mpi_inters(1).set_mpi_requests(request_tri);
  FlowSol->mesh_mpi_inters(2).set_mpi_requests(request_quad);

#ifdef _GPU
      for(int i=0;i<FlowSol->n_mpi_inter_types;i++)
        FlowSol->mesh_mpi_inters(i).mv_all_cpu_gpu();
#endif

#endif

  // ---------------------------------------
  // Initializing internal and bdy faces
  // ---------------------------------------

  // TODO: Need to count quad and triangle faces

  // Count the number of int_inters and bdy_inters
  int n_seg_int_inters = 0;
  int n_tri_int_inters = 0;
  int n_quad_int_inters = 0;

  int n_seg_bdy_inters = 0;
  int n_tri_bdy_inters = 0;
  int n_quad_bdy_inters = 0;

  for (int i=0; i<FlowSol->num_inters; i++)
    {
      bctype_f = bctype_c( f2c(i,0),f2loc_f(i,0));
      ic_r = f2c(i,1);

      if (bctype_f!=10)
        {
          if (bctype_f==0)  // internal interface
            {
              if (ic_r==-1)
                {
                  FatalError("Error: Interior interface has i_cell_right=-1. Should not be here, exiting");
                }
              n_int_inters++;
              if (f2nv(i)==2) n_seg_int_inters++;
              if (f2nv(i)==3) n_tri_int_inters++;
              if (f2nv(i)==4) n_quad_int_inters++;
            }
          else // boundary interface
            {
              if (bctype_f!=99) //  Not a deleted cyclic interface
                {
                  n_bdy_inters++;
                  if (f2nv(i)==2) n_seg_bdy_inters++;
                  if (f2nv(i)==3) n_tri_bdy_inters++;
                  if (f2nv(i)==4) n_quad_bdy_inters++;
                }
            }
        }
    }

  FlowSol->n_int_inter_types=3;
  FlowSol->mesh_int_inters.setup(FlowSol->n_int_inter_types);
  FlowSol->mesh_int_inters(0).setup(n_seg_int_inters,0);
  FlowSol->mesh_int_inters(1).setup(n_tri_int_inters,1);
  FlowSol->mesh_int_inters(2).setup(n_quad_int_inters,2);

  FlowSol->n_bdy_inter_types=3;
  FlowSol->mesh_bdy_inters.setup(FlowSol->n_bdy_inter_types);
  FlowSol->mesh_bdy_inters(0).setup(n_seg_bdy_inters,0);
  FlowSol->mesh_bdy_inters(1).setup(n_tri_bdy_inters,1);
  FlowSol->mesh_bdy_inters(2).setup(n_quad_bdy_inters,2);

  int i_seg_int=0;
  int i_tri_int=0;
  int i_quad_int=0;

  int i_seg_bdy=0;
  int i_tri_bdy=0;
  int i_quad_bdy=0;

  for(int i=0;i<FlowSol->num_inters;i++)
    {
      bctype_f = bctype_c( f2c(i,0),f2loc_f(i,0) );
      ic_l = f2c(i,0);
      ic_r = f2c(i,1);

      if(bctype_f!=10) // internal or boundary edge
        {
          if(bctype_f==0)
            {
              if (f2nv(i)==2) {
                  FlowSol->mesh_int_inters(0).set_interior(i_seg_int,ctype(ic_l),ctype(ic_r),local_c(ic_l),local_c(ic_r),f2loc_f(i,0),f2loc_f(i,1),rot_tag(i),FlowSol);
                  i_seg_int++;
                }
              if (f2nv(i)==3) {
                  FlowSol->mesh_int_inters(1).set_interior(i_tri_int,ctype(ic_l),ctype(ic_r),local_c(ic_l),local_c(ic_r),f2loc_f(i,0),f2loc_f(i,1),rot_tag(i),FlowSol);
                  i_tri_int++;
                }
              if (f2nv(i)==4) {
                  FlowSol->mesh_int_inters(2).set_interior(i_quad_int,ctype(ic_l),ctype(ic_r),local_c(ic_l),local_c(ic_r),f2loc_f(i,0),f2loc_f(i,1),rot_tag(i),FlowSol);
                  i_quad_int++;
                }
            }
          else // boundary face other than cyclic face
            {
              if (bctype_f!=99) //  Not a deleted cyclic face
                {
                  if (f2nv(i)==2){
                      FlowSol->mesh_bdy_inters(0).set_boundary(i_seg_bdy,bctype_f,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),FlowSol);
                      i_seg_bdy++;
                    }
                  else if (f2nv(i)==3){
                      FlowSol->mesh_bdy_inters(1).set_boundary(i_tri_bdy,bctype_f,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),FlowSol);
                      i_tri_bdy++;
                    }
                  else if (f2nv(i)==4){
                      FlowSol->mesh_bdy_inters(2).set_boundary(i_quad_bdy,bctype_f,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),FlowSol);
                      i_quad_bdy++;
                    }
                }
            }
        }
    }

  if (run_input.motion)
    Mesh.ic2loc_c = local_c;

  // Flag interfaces for calculating LES wall model
  if(run_input.wall_model>0 or run_input.turb_model>0) {

    if (FlowSol->rank==0) cout << "calculating wall distance... " << endl;

    int n_seg_noslip_inters = 0;
    int n_tri_noslip_inters = 0;
    int n_quad_noslip_inters = 0;
    int order = run_input.order;
    int n_fpts_per_inter_seg = order+1;
    int n_fpts_per_inter_tri = (order+2)*(order+1)/2;
    int n_fpts_per_inter_quad = (order+1)*(order+1);

    for(int i=0;i<FlowSol->num_inters;i++) {

      bctype_f = bctype_c( f2c(i,0),f2loc_f(i,0) );

      // All types of no-slip wall
      if(bctype_f == 11 || bctype_f == 12 || bctype_f == 13 || bctype_f == 14) {

        // segs
        if (f2nv(i)==2) n_seg_noslip_inters++;

        // tris
        if (f2nv(i)==3) n_tri_noslip_inters++;

        // quads
        if (f2nv(i)==4) n_quad_noslip_inters++;
      }
    }

#ifdef _MPI

    /*! Code paraphrased from SU2 */

    array<int> n_seg_inters_array(FlowSol->nproc);
    array<int> n_tri_inters_array(FlowSol->nproc);
    array<int> n_quad_inters_array(FlowSol->nproc);
    int n_global_seg_noslip_inters = 0;
    int n_global_tri_noslip_inters = 0;
    int n_global_quad_noslip_inters = 0;
    int max_seg_noslip_inters = 0;
    int max_tri_noslip_inters = 0;
    int max_quad_noslip_inters = 0;
    int buf;

    /*! Communicate to all processors the total number of no-slip boundary
    inters, the maximum number of no-slip boundary inters on any single
    partition, and the number of no-slip inters on each partition. */

    MPI_Allreduce(&n_seg_noslip_inters, &n_global_seg_noslip_inters, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&n_tri_noslip_inters, &n_global_tri_noslip_inters, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&n_quad_noslip_inters, &n_global_quad_noslip_inters, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&n_seg_noslip_inters, &max_seg_noslip_inters, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&n_tri_noslip_inters, &max_tri_noslip_inters, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&n_quad_noslip_inters, &max_quad_noslip_inters, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allgather(&n_seg_noslip_inters, 1, MPI_INT, n_seg_inters_array.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&n_tri_noslip_inters, 1, MPI_INT, n_tri_inters_array.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&n_quad_noslip_inters, 1, MPI_INT, n_quad_inters_array.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);

    // Set loop counters to max. inters on any partition
    n_seg_noslip_inters = max_seg_noslip_inters;
    n_tri_noslip_inters = max_tri_noslip_inters;
    n_quad_noslip_inters = max_quad_noslip_inters;

#endif

    // Allocate arrays for coordinates of points on no-slip boundaries
    FlowSol->loc_noslip_bdy.setup(FlowSol->n_bdy_inter_types);
    FlowSol->loc_noslip_bdy(0).setup(n_fpts_per_inter_seg,n_seg_noslip_inters,FlowSol->n_dims);
    FlowSol->loc_noslip_bdy(1).setup(n_fpts_per_inter_tri,n_tri_noslip_inters,FlowSol->n_dims);
    FlowSol->loc_noslip_bdy(2).setup(n_fpts_per_inter_quad,n_quad_noslip_inters,FlowSol->n_dims);

    n_seg_noslip_inters = 0;
    n_tri_noslip_inters = 0;
    n_quad_noslip_inters = 0;

    // Get coordinates
    for(int i=0;i<FlowSol->num_inters;i++) {

      ic_l = f2c(i,0);
      bctype_f = bctype_c( f2c(i,0),f2loc_f(i,0) );

      // All types of no-slip wall
      if(bctype_f == 11 || bctype_f == 12 || bctype_f == 13 || bctype_f == 14) {

        // segs
        if (f2nv(i)==2) {
          for(int j=0;j<n_fpts_per_inter_seg;j++) {
            for(int k=0;k<FlowSol->n_dims;k++) {

              // find coordinates
              FlowSol->loc_noslip_bdy(0)(j,n_seg_noslip_inters,k) = *get_loc_fpts_ptr_cpu(ctype(ic_l), local_c(ic_l), f2loc_f(i,0), j, k, FlowSol);
            }
          }
          n_seg_noslip_inters++;
        }
        // tris
        if (f2nv(i)==3) {
          for(int j=0;j<n_fpts_per_inter_tri;j++) {
            for(int k=0;k<FlowSol->n_dims;k++) {

              FlowSol->loc_noslip_bdy(1)(j,n_tri_noslip_inters,k) = *get_loc_fpts_ptr_cpu(ctype(ic_l), local_c(ic_l), f2loc_f(i,0), j, k, FlowSol);

            }
          }
          n_tri_noslip_inters++;
        }
        // quads
        if (f2nv(i)==4) {
          for(int j=0;j<n_fpts_per_inter_quad;j++) {
            for(int k=0;k<FlowSol->n_dims;k++) {

              FlowSol->loc_noslip_bdy(2)(j,n_quad_noslip_inters,k) = *get_loc_fpts_ptr_cpu(ctype(ic_l), local_c(ic_l), f2loc_f(i,0), j, k, FlowSol);

            }
          }

          n_quad_noslip_inters++;
        }
      }
    }

#ifdef _MPI

    // Allocate global arrays for coordinates of points on no-slip boundaries
    FlowSol->loc_noslip_bdy_global.setup(FlowSol->n_bdy_inter_types);

    FlowSol->loc_noslip_bdy_global(0).setup(n_fpts_per_inter_seg,max_seg_noslip_inters,FlowSol->nproc*FlowSol->n_dims);

    FlowSol->loc_noslip_bdy_global(1).setup(n_fpts_per_inter_tri,max_tri_noslip_inters,FlowSol->nproc*FlowSol->n_dims);

    FlowSol->loc_noslip_bdy_global(2).setup(n_fpts_per_inter_quad,max_quad_noslip_inters,FlowSol->nproc*FlowSol->n_dims);

    // Broadcast coordinates of interface points to all partitions

    buf = max_seg_noslip_inters*n_fpts_per_inter_seg*FlowSol->n_dims;

    MPI_Allgather(FlowSol->loc_noslip_bdy(0).get_ptr_cpu(), buf, MPI_DOUBLE, FlowSol->loc_noslip_bdy_global(0).get_ptr_cpu(), buf, MPI_DOUBLE, MPI_COMM_WORLD);

    buf = max_tri_noslip_inters*n_fpts_per_inter_tri*FlowSol->n_dims;

    MPI_Allgather(FlowSol->loc_noslip_bdy(1).get_ptr_cpu(), buf, MPI_DOUBLE, FlowSol->loc_noslip_bdy_global(1).get_ptr_cpu(), buf, MPI_DOUBLE, MPI_COMM_WORLD);

    buf = max_quad_noslip_inters*n_fpts_per_inter_quad*FlowSol->n_dims;

    MPI_Allgather(FlowSol->loc_noslip_bdy(2).get_ptr_cpu(), buf, MPI_DOUBLE, FlowSol->loc_noslip_bdy_global(2).get_ptr_cpu(), buf, MPI_DOUBLE, MPI_COMM_WORLD);

    // Calculate distance of every solution point to nearest point on no-slip boundary for every partition
    for(int i=0;i<FlowSol->n_ele_types;i++)
      FlowSol->mesh_eles(i)->calc_wall_distance_parallel(n_seg_inters_array,n_tri_inters_array,n_quad_inters_array,FlowSol->loc_noslip_bdy_global,FlowSol->nproc);

#else // serial

    // Calculate distance of every solution point to nearest point on no-slip boundary
    for(int i=0;i<FlowSol->n_ele_types;i++)
      FlowSol->mesh_eles(i)->calc_wall_distance(n_seg_noslip_inters,n_tri_noslip_inters,n_quad_noslip_inters,FlowSol->loc_noslip_bdy);

#endif
  }

  // set on GPU
#ifdef _GPU
      if (FlowSol->rank==0) cout << "Moving interfaces to GPU ... " << endl;
      for(int i=0;i<FlowSol->n_int_inter_types;i++)
        FlowSol->mesh_int_inters(i).mv_all_cpu_gpu();

      for(int i=0;i<FlowSol->n_bdy_inter_types;i++)
        FlowSol->mesh_bdy_inters(i).mv_all_cpu_gpu();

      if (FlowSol->rank==0) cout << "Moving wall_distance to GPU ... " << endl;
      for(int i=0;i<FlowSol->n_ele_types;i++)
        FlowSol->mesh_eles(i)->mv_wall_distance_cpu_gpu();

      for(int i=0;i<FlowSol->n_ele_types;i++)
        FlowSol->mesh_eles(i)->mv_wall_distance_mag_cpu_gpu();
#endif

}

void ReadMesh(string& in_file_name, array<double>& out_xv, array<int>& out_c2v, array<int>& out_c2n_v, array<int>& out_ctype, array<int>& out_ic2icg,
              array<int>& out_iv2ivg, int& out_n_cells, int& out_n_verts, int& out_n_verts_global, struct solution* FlowSol)
{
  if (FlowSol->rank==0)
    cout << endl << "----------------------- Mesh Preprocessing ------------------------" << endl;

  if (FlowSol->rank==0) cout << "reading connectivity ... " << endl;
  if (run_input.mesh_format==0) { // Gambit
    read_connectivity_gambit(in_file_name, out_n_cells, out_c2v, out_c2n_v, out_ctype, out_ic2icg, FlowSol);
  }
  else if (run_input.mesh_format==1) { // Gmsh
    read_connectivity_gmsh(in_file_name, out_n_cells, out_c2v, out_c2n_v, out_ctype, out_ic2icg, FlowSol);
  }
  else {
    FatalError("Mesh format not recognized");
  }
  if (FlowSol->rank==0) cout << "done reading connectivity" << endl;

#ifdef _MPI
  // Call method to repartition the mesh
  if (FlowSol->nproc != 1)
    repartition_mesh(out_n_cells, out_c2v, out_c2n_v, out_ctype, out_ic2icg,FlowSol);
#endif

  if (FlowSol->rank==0) cout << "reading vertices" << endl;

  // Call method to create array iv2ivg and modify c2v using local vertex indices
  array<int> iv2ivg;
  int n_verts;
  create_iv2ivg(iv2ivg,out_c2v,n_verts,out_n_cells);
  out_iv2ivg=iv2ivg;

  // Now read position of vertices in mesh file
  out_xv.setup(n_verts,FlowSol->n_dims);

  if (run_input.mesh_format==0) { read_vertices_gambit(in_file_name, n_verts, out_n_verts_global, out_iv2ivg, out_xv, FlowSol); }
  else if (run_input.mesh_format==1) { read_vertices_gmsh(in_file_name, n_verts, out_n_verts_global, out_iv2ivg, out_xv, FlowSol); }
  else { FatalError("Mesh format not recognized"); }

  out_n_verts = n_verts;
  if (FlowSol->rank==0) cout << "done reading vertices" << endl;

}

void ReadBound(string& in_file_name, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_c2f, array<int>& in_f2v, array<int>& in_f2nv,
               array<int>& in_ctype, array<int>& out_bctype, array<array<int> >& out_boundpts, array<int> &out_bc_list, array<int>& out_bound_flag,
               array<int>& in_ic2icg, array<int>& in_icvsta, array<int>&in_icvert, array<int>& in_iv2ivg, int& in_n_cells, int& in_n_verts,
               struct solution* FlowSol)
{

  if (FlowSol->rank==0) cout << "reading boundary conditions" << endl;
  // Set the boundary conditions
  // HACK
  out_bctype.setup(in_n_cells,MAX_F_PER_C);

  // initialize to 0 (as interior edges)
  out_bctype.initialize_to_zero();

  if (run_input.mesh_format==0) {
    array<array<int> > bccells;
    array<array<int> > bcfaces;
    read_boundary_gambit(in_file_name, in_n_cells, in_ic2icg, out_bctype, out_bc_list, bccells, bcfaces);
    if (run_input.motion != 0)
      create_boundpts(out_boundpts, out_bc_list, out_bound_flag, bccells, bcfaces, in_c2f, in_f2v, in_f2nv);
  }
  else if (run_input.mesh_format==1) {
    read_boundary_gmsh(in_file_name, in_n_cells, in_ic2icg, in_c2v, in_c2n_v, out_bctype, out_bc_list, out_bound_flag, out_boundpts, in_iv2ivg, in_n_verts, in_ctype, in_icvsta, in_icvert, FlowSol);
  }
  else {
    FatalError("Mesh format not recognized");
  }

  if (FlowSol->rank==0) cout << "done reading boundary conditions" << endl;
}

void create_boundpts(array<array<int> >& out_boundpts, array<int>& in_bclist, array<int>& out_bound_flag, array<array<int> >& in_bccells,
                     array<array<int> >& in_bcfaces, array<int>& in_c2f, array<int>& in_f2v, array<int>& in_f2nv)
{
  int iv, ivg, ic, k, loc_k, nv;
  int n_faces, bcflag;

  int n_bcs = in_bclist.get_dim(0);

  out_bound_flag.setup(n_bcs);
  out_bound_flag.initialize_to_zero();

  /** Find boundaries which are moving */
  for (int i=0; i<run_input.n_moving_bnds; i++) {
    if (run_input.boundary_flags(i).compare("FLUID"))  // if NOT 'FLUID'
    {
      bcflag = get_bc_number(run_input.boundary_flags(i));
      for (int j=0; j<n_bcs; j++) {
        if (in_bclist(j)==bcflag) {
          out_bound_flag(j) = 1;
          break;
        }
      }
    }
  }

  /** --- CREATE BOUNDARY->POINTS STRUCTURE ---
    want: iv = boundpts(bcflag,ivert); */
  out_boundpts.setup(n_bcs);
  array<set<int> > Bounds(n_bcs);
  for (int i=0; i<n_bcs; i++) {
    nv = 0;
    bcflag = in_bclist(i);
    n_faces = in_bcfaces(i).get_dim(0);
    for (int j=0; j<n_faces; j++) {
      // find (semi-)global face index
      ic = in_bccells(i)(j);
      loc_k = in_bcfaces(i)(j);
      k = in_c2f(ic,loc_k);
      for (int m=0; m<in_f2nv(k); m++) {
        // find vertex index
        iv = in_f2v(k,m);
        // add vertex to bc
        Bounds(i).insert(iv);
      }
    }
    out_boundpts(i).setup(Bounds(i).size());
    int j = 0;
    for (set<int>::iterator it=Bounds(i).begin(); it!=Bounds(i).end(); it++) {
      out_boundpts(i)(j) = (*it);
      j++;
    }
  }
}

// Method to read boundary edges in mesh file
void read_boundary_gambit(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& out_bctype, array<int> &out_bclist,
                          array<array<int> >& out_bccells, array<array<int> >& out_bcfaces)
{

  // input: ic2icg
  // output: bctype

  char buf[BUFSIZ]={""};
  ifstream mesh_file;

  array<int> cell_list(in_n_cells);

  for (int i=0;i<in_n_cells;i++)
    cell_list(i) = in_ic2icg(i);

  // Sort the cells
  qsort(cell_list.get_ptr_cpu(),in_n_cells,sizeof(int),compare_ints);

  // Read Gambit Neutral file format

  mesh_file.open(&in_file_name[0]);
  if (!mesh_file)
    FatalError("Unable to open mesh file");

  // Skip 6-line header
  for (int i=0;i<6;i++) mesh_file.getline(buf,BUFSIZ);

  int n_verts_global,n_cells_global;
  int n_mats,n_bcs,dummy;
  // Find number of vertices and number of cells
  mesh_file       >> n_verts_global // num vertices in mesh
                  >> n_cells_global // num elements
                  >> n_mats         // num material groups
                  >> n_bcs          // num boundary groups
                  >> dummy;         // num space dimensions
  //cout << "Gambit mesh specs from header: " << ", " << n_verts_global << ", " << n_cells_global << ", " << n_mats << ", " << n_bcs << ", " << dummy << endl;
  mesh_file.getline(buf,BUFSIZ);  // clear rest of line
  mesh_file.getline(buf,BUFSIZ);  // Skip 2 lines
  mesh_file.getline(buf,BUFSIZ);

  // ------------------------------
  // Skip a bunch of lines
  // ------------------------------

  // Skip the x,y,z location of vertices
  for (int i=0;i<n_verts_global;i++)
    mesh_file.getline(buf,BUFSIZ);

  mesh_file.getline(buf,BUFSIZ); // Skip "ENDOFSECTION"
  mesh_file.getline(buf,BUFSIZ); // Skip "ELEMENTS/CELLS"

  // Skip the Elements connectivity
  int c2n_v;
  for (int i=0;i<n_cells_global;i++)
  {
    mesh_file >> dummy >> dummy >> c2n_v;
    mesh_file.getline(buf,BUFSIZ); // skip end of line
    if (c2n_v>7) mesh_file.getline(buf,BUFSIZ); // skip another line
    if (c2n_v>14) mesh_file.getline(buf,BUFSIZ); // skip another line
    if (c2n_v>21) mesh_file.getline(buf,BUFSIZ); // skip another line
  }
  mesh_file.getline(buf,BUFSIZ); // Skip "ENDOFSECTION"
  mesh_file.getline(buf,BUFSIZ); // Skip "ELEMENTS GROUP"

  // Skip materials section
  int gnel,dummy2;
  for (int i=0;i<n_mats;i++)
    {
      mesh_file.getline(buf,BUFSIZ); // Read GROUP: 1 ELEMENTS
      int nread = sscanf(buf,"%*s%d%*s%d%*s%d",&dummy,&gnel,&dummy2);
      if (3!=nread) {
          cout << "ERROR while reading Gambit file" << endl;
          cout << "nread =" << nread << endl; exit(1);
        }
      mesh_file.getline(buf,BUFSIZ); // Read group name
      mesh_file.getline(buf,BUFSIZ); // Skip solver dependant flag
      for (int k=0;k<gnel;k++) mesh_file >> dummy;
      mesh_file.getline(buf,BUFSIZ); // Clear end of line
      mesh_file.getline(buf,BUFSIZ); // skip "ENDOFSECTION"
      mesh_file.getline(buf,BUFSIZ); // skip "Element Group"
    }

  // ---------------------------------
  // Read the boundary regions
  // --------------------------------

  int bcNF, bcID, bcflag,icg,k, real_face;
  string bcname;
  char bcTXT[100];

  out_bcfaces.setup(n_bcs);
  out_bccells.setup(n_bcs);
  out_bclist.setup(n_bcs);

  for (int i=0;i<n_bcs;i++)
  {
    mesh_file.getline(buf,BUFSIZ);  // Load ith boundary group
    if (strstr(buf,"ENDOFSECTION")){
      continue; // may not have a boundary section
    }

    sscanf(buf,"%s %d %d", bcTXT, &bcID, &bcNF);

    bcname.assign(bcTXT,0,14);
    bcflag = get_bc_number(bcname);

    out_bclist(i) = bcflag;
    out_bcfaces(i).setup(bcNF);
    out_bccells(i).setup(bcNF);

    int bdy_count = 0;
    int eleType;
    for (int bf=0;bf<bcNF;bf++)
    {
      mesh_file >> icg >> eleType >> k;
      icg--;  // 1-indexed -> 0-indexed
      // Matching Gambit faces with face convention in code
      if (eleType==2 || eleType==3)
        real_face = k-1;
      // Hex
      else if (eleType==4)
      {
        if (k==1)
          real_face = 0;
        else if (k==2)
          real_face = 3;
        else if (k==3)
          real_face = 5;
        else if (k==4)
          real_face = 1;
        else if (k==5)
          real_face = 4;
        else if (k==6)
          real_face = 2;
      }
      // Tet
      else if (eleType==6)
      {
        if (k==1)
          real_face = 3;
        else if (k==2)
          real_face = 2;
        else if (k==3)
          real_face = 0;
        else if (k==4)
          real_face = 1;
      }
      else if (eleType==5)
      {
        if (k==1)
          real_face = 2;
        else if (k==2)
          real_face = 3;
        else if (k==3)
          real_face = 4;
        else if (k==4)
          real_face = 0;
        else if (k==5)
          real_face = 1;
      }
      else
      {
        cout << "Element Type = " << eleType << endl;
        FatalError("Cannot handle other element type in readbnd");
      }

      // Check if cell icg belongs to processor
      int cellID = index_locate_int(icg,cell_list.get_ptr_cpu(),in_n_cells);

      // If it does, find local cell ic corresponding to icg
      if (cellID!=-1)
      {
        bdy_count++;
        out_bctype(cellID,real_face) = bcflag;
        out_bccells(i)(bf) = cellID;
        out_bcfaces(i)(bf) = real_face;
      }
    }

    mesh_file.getline(buf,BUFSIZ); // Clear "end of line"
    mesh_file.getline(buf,BUFSIZ); // Skip "ENDOFSECTION"
    mesh_file.getline(buf,BUFSIZ); // Skip "Element group"
  }

  mesh_file.close();
}

void read_boundary_gmsh(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& out_bctype,
                        array<int> &out_bclist, array<int> &out_bound_flag, array<array<int> >& out_boundpts, array<int>& in_iv2ivg,
                        int in_n_verts, array<int>& in_ctype, array<int>& in_icvsta, array<int>& in_icvert, struct solution* FlowSol)
{
  string str;

  ifstream mesh_file;

  mesh_file.open(&in_file_name[0]);
  if (!mesh_file)
    FatalError("Unable to open mesh file");

  // Move cursor to $PhysicalNames
  while(1) {
      getline(mesh_file,str);
      if (str.find("$PhysicalNames")!=string::npos) break;
      if(mesh_file.eof()) FatalError("$PhysicalNames tag not found!");
    }

  // Read number of boundaries and fields defined
  int dummy;
  int n_bcs;;
  int id,elmtype,ntags,bcid,bcdim,bcflag;

  char buf[BUFSIZ]={""};
  char bcTXT[100][100];// can read up to 100 different boundary conditions
  char bc_txt_temp[100];

  mesh_file >> n_bcs;
  mesh_file.getline(buf,BUFSIZ);  // clear rest of line
  out_bclist.setup(n_bcs);
  for(int i=0;i<n_bcs;i++)
  {
    mesh_file.getline(buf,BUFSIZ);
    sscanf(buf,"%d %d \"%s", &bcdim, &bcid, bc_txt_temp);
    strcpy(bcTXT[bcid],bc_txt_temp);
    out_bclist(i) = bcid;
  }

  // Move cursor to $Elements
  while(1) {
    getline(mesh_file,str);
    if (str.find("$Elements")!=string::npos) break;
    if(mesh_file.eof()) FatalError("$Elements tag not found!");
  }

  // Each processor reads number of entities
  int n_entities;
  // Read number of elements and bdys
  mesh_file   >> n_entities;   // num cells in mesh
  mesh_file.getline(buf,BUFSIZ);  // clear rest of line

  array<int> vlist_bound(9), vlist_cell(9);
  array<int> vlist_local(9);

  int found, num_v_per_f;
  int num_face_vert;

  string bcname;
  int bdy_count=0;

  int sta_ind,end_ind;

  // --- setup vertex->bcflag array ---
  out_boundpts.setup(n_bcs);
  array<set<int> > Bounds;
  Bounds.setup(n_bcs);

  //--- overwrite bc_list with bcflag (previously held gmsh bcid) ---
  for (int i=0; i<n_bcs; i++) {
    bcname.assign(bcTXT[out_bclist(i)],0,14);
    bcname.erase(bcname.find_last_not_of(" \n\r\t")+1);
    bcname.erase(bcname.find_last_not_of("\"")+1);
    // ignore FLUID region as a "boundary"
    if (bcname.compare("FLUID")==0) {
      out_bclist(i) = -1;
    }else{
      bcflag = get_bc_number(bcname);
      out_bclist(i) = bcflag;
    }
  }

  //--- Find boundaries which are moving ---//
  out_bound_flag.setup(n_bcs);
  out_bound_flag.initialize_to_zero();
  for (int i=0; i<run_input.n_moving_bnds; i++) {
    if (run_input.boundary_flags(i).compare("FLUID"))  // if NOT 'FLUID'
    {
      bcflag = get_bc_number(run_input.boundary_flags(i));
      for (int j=0; j<n_bcs; j++) {
        if (out_bclist(j)==bcflag) {
          out_bound_flag(j) = 1;
          break;
        }
      }
    }
  }

  for (int i=0;i<n_entities;i++)
    {
      mesh_file >> id >> elmtype >> ntags;
      mesh_file >> bcid;
      for (int tag=0; tag<ntags-1; tag++)
        mesh_file >> dummy;

      if (strstr(bcTXT[bcid],"FLUID"))
        {
          mesh_file.getline(buf,BUFSIZ);  // skip line
          continue;
        }
      else
        {
          bcname.assign(bcTXT[bcid],0,14);
          bcname.erase(bcname.find_last_not_of(" \n\r\t")+1);
          bcname.erase(bcname.find_last_not_of("\"")+1);
          bcflag = get_bc_number(bcname);
        }

      bdy_count++;

      if (elmtype==1 || elmtype==8) // Edge
      {
        num_face_vert = 2;
        // Read the two vertices
        mesh_file >> vlist_bound(0) >> vlist_bound(1);
      }
      else if (elmtype==3) // Quad face
      {
        num_face_vert = 4;
        mesh_file >> vlist_bound(0) >> vlist_bound(1) >> vlist_bound(3) >> vlist_bound(2);
      }
      else if (elmtype==2) // Linear tri face
      {
        num_face_vert = 3;
        mesh_file >> vlist_bound(0) >> vlist_bound(1) >> vlist_bound(2);
      }
      else if (elmtype == 9) // Quadratic Tri Face
      {
        num_face_vert = 3;
        mesh_file >> vlist_bound(0) >> vlist_bound(1) >> vlist_bound(2);
        mesh_file >> vlist_bound(3) >> vlist_bound(4) >> vlist_bound(5);
      }
      else if (elmtype==10) // Quadratic quad face
      {
        num_face_vert = 9;
        mesh_file >> vlist_bound(0) >> vlist_bound(2) >> vlist_bound(8) >> vlist_bound(6);
        mesh_file >> vlist_bound(1) >> vlist_bound(5) >> vlist_bound(7) >> vlist_bound(3) >> vlist_bound(4);
      }
      else 
      {
        cout << "Gmsh boundary element type: " << elmtype << endl;
        FatalError("Boundary elmtype not recognized");
      }

      // Shift by -1 (1-indexed -> 0-indexed)
      for (int j=0;j<num_face_vert;j++)
      {
        vlist_bound(j)--;
      }

      mesh_file.getline(buf,BUFSIZ);  // Get rest of line

      // Check if all vertices belong to processor
      bool belong_to_proc = true;
      for (int j=0;j<num_face_vert;j++)
        {
          vlist_local(j) = index_locate_int(vlist_bound(j),in_iv2ivg.get_ptr_cpu(),in_n_verts);
          if (vlist_local(j) == -1)
            belong_to_proc = false;

          Bounds(bcid-1).insert(vlist_local(j));
        }

      if (belong_to_proc)
        {
          // All vertices on face belong to processor
          // Try to find the cell that they belong to
          found=0;

          // Loop over cells touching that vertex
          sta_ind = in_icvsta(vlist_local(0));
          end_ind = in_icvsta(vlist_local(0)+1)-1;

          for (int ind=sta_ind;ind<=end_ind;ind++)
            {
              int ic=in_icvert(ind);
              for (int k=0;k<FlowSol->num_f_per_c(in_ctype(ic));k++)
                {
                  // Get local vertices of local face k of cell ic
                  get_vlist_loc_face(in_ctype(ic),in_c2n_v(ic),k,vlist_cell,num_v_per_f);

                  if (num_v_per_f != num_face_vert)
                    continue;

                  for (int j=0;j<num_v_per_f;j++)
                  {
                    vlist_cell(j) = in_c2v(ic,vlist_cell(j));
                  }

                  compare_faces_boundary(vlist_local,vlist_cell,num_v_per_f,found);

                  if (found==1)
                  {
                    out_bctype(ic,k)=bcflag;
                    break;
                  }
                }
              if (found==1)
                break;
            }
          if (found==0)
          {
            cout << "vlist_bound(2)=" << vlist_bound(2) << " vlist_bound(3)=" << vlist_bound(3) << endl;
            FatalError("All nodes of boundary face belong to processor but could not find the coresponding faces");
          }

        } // If all vertices belong to processor

    } // Loop over entities

  set<int>::iterator it;
  for (int i=0; i<n_bcs; i++) {
    out_boundpts(i).setup(Bounds(i).size());
    int j=0;
    for (it=Bounds(i).begin(); it!=Bounds(i).end(); it++) {
      out_boundpts(i)(j) = (*it);
      j++;
    }
  }

  mesh_file.close();

  //cout << "  Number of Boundary Faces: " << bdy_count << endl;
}

void read_vertices_gambit(string& in_file_name, int in_n_verts, int &out_n_verts_global, array<int> &in_iv2ivg, array<double> &out_xv, solution *FlowSol)
{

  // Now open gambit file and read the vertices
  ifstream mesh_file;
  char buf[BUFSIZ]={""};

  mesh_file.open(&in_file_name[0]);

  if (!mesh_file)
    FatalError("Could not open mesh file");

  // Skip 6-line header
  for (int i=0;i<6;i++) mesh_file.getline(buf,BUFSIZ);

  // Find number of vertices and number of cells
  int dummy;
  mesh_file       >> out_n_verts_global // num vertices in mesh
                  >> dummy   // num elements
                  >> dummy   // num material groups
                  >> dummy   // num boundary groups
                  >> FlowSol->n_dims;        // num space dimensions

  mesh_file.getline(buf,BUFSIZ);  // clear rest of line
  mesh_file.getline(buf,BUFSIZ);  // Skip 2 lines
  mesh_file.getline(buf,BUFSIZ);

  // Read the location of vertices
  int icount = 0;
  int id,index;
  double pos;
  for (int i=0;i<out_n_verts_global;i++)
    {
      mesh_file >> id;
      index = index_locate_int(id-1,in_iv2ivg.get_ptr_cpu(),in_n_verts);

      if (index!=-1) // Vertex belongs to this processor
        {
          for (int m=0;m<FlowSol->n_dims;m++) {
              mesh_file >> pos;
              out_xv(index,m) = pos;
            }
        }
      mesh_file.getline(buf,BUFSIZ);
    }

  mesh_file.close();

}

void read_vertices_gmsh(string& in_file_name, int in_n_verts, int& out_n_verts_global, array<int> &in_iv2ivg, array<double> &out_xv, struct solution* FlowSol)
{

  string str;

  // Now open gambit file and read the vertices
  ifstream mesh_file;
  char buf[BUFSIZ]={""};

  mesh_file.open(&in_file_name[0]);

  if (!mesh_file)
    FatalError("Could not open mesh file");

  // Move cursor to $Nodes
  while(1) {
      getline(mesh_file,str);
      if (str.find("$Nodes")!=string::npos) break;
      if(mesh_file.eof()) FatalError("$Nodes tag not found!");
    }

  double pos;

  mesh_file >> out_n_verts_global ;// num vertices in mesh
  mesh_file.getline(buf,BUFSIZ);

  int id;
  int index;

  for (int i=0;i<out_n_verts_global;i++)
    {
      mesh_file >> id;
      index = index_locate_int(id-1,in_iv2ivg.get_ptr_cpu(),in_n_verts);

      if (index!=-1) // Vertex belongs to this processor
        {
          for (int m=0;m<FlowSol->n_dims;m++) {
              mesh_file >> pos;
              out_xv(index,m) = pos;
            }
        }
      mesh_file.getline(buf,BUFSIZ);
    }

  mesh_file.close();

}

void create_iv2ivg(array<int> &inout_iv2ivg, array<int> &inout_c2v, int &out_n_verts, int in_n_cells)
{

  array<int> vrtlist(in_n_cells*MAX_V_PER_C);
  array<int> temp(in_n_cells*MAX_V_PER_C);

  int icount = 0;
  for (int i=0;i<in_n_cells;i++)
    for (int j=0;j<MAX_V_PER_C;j++)
      vrtlist(icount++) = inout_c2v(i,j);

  // Sort the vertices
  qsort(vrtlist.get_ptr_cpu(),in_n_cells*MAX_V_PER_C,sizeof(int),compare_ints);

  int staind;
  // Get rid of -1 at beginning
  for (int i=0;i<MAX_V_PER_C*in_n_cells;i++)
    {
      if (vrtlist(i)!=-1)
        {
          staind = i;
          break;
        }
    }

  // Get rid of repeated digits
  temp(0) = vrtlist(staind);
  out_n_verts=1;
  for(int i=staind+1;i<MAX_V_PER_C*in_n_cells;i++) {
      if (vrtlist(i)!=vrtlist(i-1)) {
          temp(out_n_verts) = vrtlist(i);
          out_n_verts++;
        }
    }

  inout_iv2ivg.setup(out_n_verts);

  for (int i=0;i<out_n_verts;i++)
    inout_iv2ivg(i) = temp(i);

  //vrtlist.~array();
  //temp.~array();

#ifdef _MPI

  // Now modify array ic2icg
  for (int i=0;i<in_n_cells;i++) {
      for (int j=0;j<MAX_V_PER_C;j++) {
          if (inout_c2v(i,j) != -1) {
              int index = index_locate_int(inout_c2v(i,j),inout_iv2ivg.get_ptr_cpu(), out_n_verts);
              if (index==-1) {
                  FatalError("Could not find value in index_locate");
                }
              else {
                  inout_c2v(i,j) = index;
                }
            }
        }
    }

#endif

}

void read_connectivity_gambit(string& in_file_name, int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol)
{

  int n_verts_global,n_cells_global;
  int dummy,dummy2;

  char buf[BUFSIZ]={""};
  ifstream mesh_file;

  mesh_file.open(&in_file_name[0]);
  if (!mesh_file)
    FatalError("Unable to open mesh file");

  // Skip 6-line header
  for (int i=0;i<6;i++)
    mesh_file.getline(buf,BUFSIZ);

  // Find number of vertices and number of cells
  mesh_file >> n_verts_global   // num vertices in mesh
            >> n_cells_global     // num elements
            >> dummy              // num material groups
            >> dummy              // num boundary groups
            >> FlowSol->n_dims;  // num space dimensions

  if (FlowSol->n_dims != 2 && FlowSol->n_dims != 3) {
      FatalError("Invalid mesh dimensionality. Expected 2D or 3D.");
    }

  mesh_file.getline(buf,BUFSIZ);  // clear rest of line
  mesh_file.getline(buf,BUFSIZ);  // Skip 2 lines
  mesh_file.getline(buf,BUFSIZ);

  // Skip the x,y,z location of vertices
  for (int i=0;i<n_verts_global;i++)
    mesh_file.getline(buf,BUFSIZ);

  mesh_file.getline(buf,BUFSIZ); // Skip "ENDOFSECTION"
  mesh_file.getline(buf,BUFSIZ); // Skip "ELEMENTS/CELLS"

  int kstart;
#ifdef _MPI
  // Assign a number of cells for each processor
  out_n_cells = (int) ( (double)(n_cells_global)/(double)FlowSol->nproc);
  kstart = FlowSol->rank*out_n_cells;

  // Last processor has more cells
  if (FlowSol->rank==(FlowSol->nproc-1))
    out_n_cells += (n_cells_global-FlowSol->nproc*out_n_cells);
#else

  kstart = 0;
  out_n_cells = n_cells_global;

#endif

  //    c2n_v(i) is the number of nodes that define the element i
  out_c2v.setup(out_n_cells,MAX_V_PER_C); // stores the vertices making that cell
  out_c2n_v.setup(out_n_cells); // stores the number of nodes making that cell
  out_ctype.setup(out_n_cells); // stores the type of cell
  out_ic2icg.setup(out_n_cells);

  // Initialize arrays to -1
  for (int i=0;i<out_n_cells;i++) {
      out_c2n_v(i)=-1;
      out_ctype(i) = -1;
      out_ic2icg(i) = -1;
      for (int k=0;k<MAX_V_PER_C;k++)
        out_c2v(i,k)=-1;
    }

  // Skip elements being read by other processors

  for (int i=0;i<kstart;i++) {
      mesh_file >> dummy >> dummy >> dummy2;
      mesh_file.getline(buf,BUFSIZ); // skip end of line
      if (dummy2>7) mesh_file.getline(buf,BUFSIZ); // skip another line
      if (dummy2>14) mesh_file.getline(buf,BUFSIZ); // skip another line
      if (dummy2>21) mesh_file.getline(buf,BUFSIZ); // skip another line
    }

  // Each proc reads a block of elements

  // Start reading elements
  int eleType;
  for (int i=0;i<out_n_cells;i++)
    {
      //  ctype is the element type:  1=edge, 2=quad, 3=tri, 4=brick, 5=wedge, 6=tet, 7=pyramid
      mesh_file >> out_ic2icg(i) >> eleType >> out_c2n_v(i);

      if (eleType==3) out_ctype(i)=TRI;
      else if (eleType==2) out_ctype(i)=QUAD;
      else if (eleType==6) out_ctype(i)=TET;
      else if (eleType==5) out_ctype(i)=PRISM;
      else if (eleType==4) out_ctype(i)=HEX;

      // triangle
      if (out_ctype(i)==TRI)
        {
          if (out_c2n_v(i)==3) // linear triangle
            mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2);
          else if (out_c2n_v(i)==6) // quadratic triangle
            mesh_file >> out_c2v(i,0) >> out_c2v(i,3) >>  out_c2v(i,1) >> out_c2v(i,4) >> out_c2v(i,2) >> out_c2v(i,5);
          else
            FatalError("triangle element type not implemented");
        }
      // quad
      else if (out_ctype(i)==QUAD)
        {
          if (out_c2n_v(i)==4) // linear quadrangle
            mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,3) >> out_c2v(i,2);
          else if (out_c2n_v(i)==8)  // quadratic quad
            mesh_file >> out_c2v(i,0) >> out_c2v(i,4) >> out_c2v(i,1) >> out_c2v(i,5) >> out_c2v(i,2) >> out_c2v(i,6) >> out_c2v(i,3) >> out_c2v(i,7);
          else
            FatalError("quad element type not implemented");
        }
      // tet
      else if (out_ctype(i)==TET)
        {
          if (out_c2n_v(i)==4) // linear tets
            {
              mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3);
            }
          else if (out_c2n_v(i)==10) // quadratic tet
            {
              mesh_file >> out_c2v(i,0) >> out_c2v(i,4) >> out_c2v(i,1) >> out_c2v(i,5) >> out_c2v(i,7);
              mesh_file >> out_c2v(i,2) >> out_c2v(i,6) >> out_c2v(i,9) >> out_c2v(i,8) >> out_c2v(i,3);
            }
          else
            FatalError("tet element type not implemented");
        }
      // prisms
      else if (out_ctype(i)==PRISM)
        {
          if (out_c2n_v(i)==6) // linear prism
            mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3) >> out_c2v(i,4) >> out_c2v(i,5);
          else if (out_c2n_v(i)==15) // quadratic prism
            mesh_file >> out_c2v(i,0) >> out_c2v(i,6) >> out_c2v(i,1) >> out_c2v(i,8) >> out_c2v(i,7) >> out_c2v(i,2) >> out_c2v(i,9) >> out_c2v(i,10) >> out_c2v(i,11) >> out_c2v(i,3) >> out_c2v(i,12) >> out_c2v(i,4) >> out_c2v(i,14) >> out_c2v(i,13) >> out_c2v(i,5) ;
          else
            FatalError("Prism element type not implemented");
        }
      // hexa
      else if (out_ctype(i)==HEX)
        {
          if (out_c2n_v(i)==8) // linear hexas
            mesh_file >> out_c2v(i,0) >> out_c2v(i,2) >> out_c2v(i,4) >> out_c2v(i,6) >> out_c2v(i,1) >> out_c2v(i,3) >> out_c2v(i,5) >> out_c2v(i,7);
          else if (out_c2n_v(i)==20) // quadratic hexas
            mesh_file >> out_c2v(i,0) >> out_c2v(i,11) >> out_c2v(i,3) >> out_c2v(i,12) >> out_c2v(i,15) >> out_c2v(i,4) >> out_c2v(i,19) >> out_c2v(i,7) >> out_c2v(i,8) >> out_c2v(i,10) >> out_c2v(i,16) >> out_c2v(i,18) >> out_c2v(i,1) >> out_c2v(i,9) >> out_c2v(i,2) >> out_c2v(i,13) >> out_c2v(i,14) >> out_c2v(i,5) >> out_c2v(i,17) >> out_c2v(i,6);
          else
            FatalError("Hexa element type not implemented");
        }
      else
      {
        cout << "Element Type = " << out_ctype(i) << endl;
        FatalError("Haven't implemented this element type in gambit_meshreader3, exiting ");
      }
      mesh_file.getline(buf,BUFSIZ); // skip end of line

      // Shift every values of c2v by -1
      for(int k=0;k<out_c2n_v(i);k++)
        if(out_c2v(i,k)!=0)
          out_c2v(i,k)--;

      // Also shift every value of ic2icg
      out_ic2icg(i)--;

    }

#ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  mesh_file.close();

}

void read_connectivity_gmsh(string& in_file_name, int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol)
{
  int n_verts_global,n_cells_global,n_bnds;
  int dummy,dummy2;

  char buf[BUFSIZ]={""};
  char bcTXT[100][100];// can read up to 100 different boundary conditions
  char bc_txt_temp[100];
  ifstream mesh_file;

  string str;
  
  mesh_file.open(&in_file_name[0]);
  if (!mesh_file)
    FatalError("Unable to open mesh file");

  int id,elmtype,ntags,bcid,bcdim;

  // Move cursor to $PhysicalNames
  int argh = 0;
  while(1) {
    getline(mesh_file,str);
    if (str.find("$PhysicalNames")!=string::npos) break;
    if(mesh_file.eof()) FatalError("$PhysicalNames tag not found!");
  }

  // Read number of boundaries and fields defined
  mesh_file >> n_bnds;
  mesh_file.getline(buf,BUFSIZ);  // clear rest of line
  for(int i=0;i<n_bnds;i++)
  {
    mesh_file.getline(buf,BUFSIZ);
    sscanf(buf,"%d %d %s", &bcdim, &bcid, bc_txt_temp);
    strcpy(bcTXT[bcid],bc_txt_temp);
    if (strstr(bc_txt_temp,"FLUID")) {
      FlowSol->n_dims=bcdim;
    }
  }
  if (FlowSol->n_dims != 2 && FlowSol->n_dims != 3) {
    FatalError("Invalid mesh dimensionality. Expected 2D or 3D.");
  }
  if (run_input.turb_model==1 && FlowSol->n_dims == 3) {
    FatalError("ERROR: 3D geometry not supported with RANS equation yet ... ");
  }

  // Move cursor to $Elements
  while(1) {
    getline(mesh_file,str);
    if (str.find("$Elements")!=string::npos) break;
    if(mesh_file.eof()) FatalError("$Elements tag not found!");
  }

  // -------------------------------
  //  Read element connectivity
  //  ------------------------------

  // Each processor first reads number of global cells
  int n_entities;
  // Read number of elements and bdys
  mesh_file >> n_entities;   // num cells in mesh
  mesh_file.getline(buf,BUFSIZ);  // clear rest of line

  int icount=0;

  for (int i=0;i<n_entities;i++)
  {
    mesh_file >> id >> elmtype >> ntags;
    mesh_file >> bcid;

    for (int tag=0; tag<ntags-1; tag++)
      mesh_file >> dummy;

    if (strstr(bcTXT[bcid],"FLUID"))
      icount++;

    mesh_file.getline(buf,BUFSIZ);  // clear rest of line

  }
  n_cells_global=icount;

  // Now assign kstart to each processor
  int kstart;
#ifdef _MPI
  // Assign a number of cells for each processor
  out_n_cells = (int) ( (double)(n_cells_global)/(double)FlowSol->nproc);
  kstart = FlowSol->rank*out_n_cells;

  // Last processor has more cells
  if (FlowSol->rank==(FlowSol->nproc-1))
    out_n_cells += (n_cells_global-FlowSol->nproc*out_n_cells);
#else
  kstart = 0;
  out_n_cells = n_cells_global;
#endif

  // Allocate memory
  out_c2v.setup(out_n_cells,MAX_V_PER_C);
  out_c2n_v.setup(out_n_cells);
  out_ctype.setup(out_n_cells);
  out_ic2icg.setup(out_n_cells);

  // Initialize arrays to -1
  for (int i=0;i<out_n_cells;i++) {
      out_c2n_v(i)=-1;
      out_ctype(i) = -1;
      out_ic2icg(i) = -1;
      for (int k=0;k<MAX_V_PER_C;k++)
        out_c2v(i,k)=-1;
    }

  // Move cursor to $Elements
  mesh_file.clear();
  mesh_file.seekg(0, ios::beg);
  while(1) {
      getline(mesh_file,str);
      if (str.find("$Elements")!=string::npos) break;
      if(mesh_file.eof()) FatalError("$Elements tag not found!");
    }

  mesh_file >> n_entities;   // num cells in mesh
  mesh_file.getline(buf,BUFSIZ);  // clear rest of line

  // Skip elements being read by other processors
  icount=0;
  int i=0;

  // ctype is the element type:  for HiFiLES: 0=tri, 1=quad, 2=tet, 3=prism, 4=hex
  // For Gmsh node ordering, see: http://geuz.org/gmsh/doc/texinfo/gmsh.html#Node-ordering

  for (int k=0;k<n_entities;k++)
    {
      mesh_file >> id >> elmtype >> ntags;
      mesh_file >> bcid;
      for (int tag=0; tag<ntags-1; tag++)
        mesh_file >> dummy;

      if (strstr(bcTXT[bcid],"FLUID"))
        {
          if (icount>=kstart && i< out_n_cells) // Read this cell
            {
              out_ic2icg(i) = icount;
              if (elmtype ==2 || elmtype==9 || elmtype==21) // Triangle
                {
                  out_ctype(i) = 0;
                  if (elmtype==2) // linear triangle
                    {
                      out_c2n_v(i) =3;
                      mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2);
                    }
                  else if (elmtype==9) // quadratic triangle
                    {
                      out_c2n_v(i) =6;
                      mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3) >> out_c2v(i,4) >> out_c2v(i,5) ;
                    }
                  else if (elmtype==21) // cubic triangle
                    {
                      FatalError("Cubic triangle not implemented");
                    }
                }
              else if (elmtype==3 || elmtype==16 || elmtype==10) // Quad
                {
                  out_ctype(i) = 1;
                  if (elmtype==3) // linear quadrangle
                    {
                      out_c2n_v(i) = 4;
                      mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,3) >> out_c2v(i,2);
                    }
                  else if (elmtype==16) // quadratic quadrangle
                    {
                      out_c2n_v(i) = 8;
                      mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3) >> out_c2v(i,4) >> out_c2v(i,5) >> out_c2v(i,6) >> out_c2v(i,7);
                    }
                  else if (elmtype==10) // quadratic quadrangle
                    {
                      out_c2n_v(i) = 9;
                      // not sure this is correct
                      mesh_file >> out_c2v(i,0) >> out_c2v(i,2) >> out_c2v(i,8) >> out_c2v(i,6) >> out_c2v(i,1) >> out_c2v(i,5) >> out_c2v(i,7) >> out_c2v(i,3) >> out_c2v(i,4);
                    }
                }
              else if (elmtype==4 || elmtype==11) // Tetrahedron
              {
                out_ctype(i) = 2;
                if (elmtype==4) // Linear tet
                {
                  out_c2n_v(i) = 4;
                  mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3);
                }
                else if (elmtype==11) // Quadratic tet
                {
                  out_c2n_v(i) = 10;                  
                  //mesh_file >> out_c2v(i,0) >> out_c2v(i,8) >> out_c2v(i,5) >> out_c2v(i,2) >> out_c2v(i,3);
                  //mesh_file >> out_c2v(i,6) >> out_c2v(i,7) >> out_c2v(i,4) >> out_c2v(i,9) >> out_c2v(i,1);
                  mesh_file >> out_c2v(i,0) >> out_c2v(i,5) >> out_c2v(i,4) >> out_c2v(i,2) >> out_c2v(i,8);
                  mesh_file >> out_c2v(i,1) >> out_c2v(i,7) >> out_c2v(i,3) >> out_c2v(i,9) >> out_c2v(i,6);
                }
              }
              else if (elmtype==5 || elmtype==12) // Hexahedron
                {
                  out_ctype(i) = 4;
                  if (elmtype==5) // linear quadrangle
                    {
                      out_c2n_v(i) = 8;
                      mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,3) >> out_c2v(i,2);
                      mesh_file >> out_c2v(i,4) >> out_c2v(i,5) >> out_c2v(i,7) >> out_c2v(i,6);
                    }
                  else if (elmtype==12) // 27-node quadratic hexahedron
                    {
                      out_c2n_v(i) = 27;
                      // vertices
                      mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3) >> out_c2v(i,4) >> out_c2v(i,5) >> out_c2v(i,6) >> out_c2v(i,7);
                      // edges
                      mesh_file >> out_c2v(i,8) >> out_c2v(i,9) >> out_c2v(i,10) >> out_c2v(i,11) >> out_c2v(i,12) >> out_c2v(i,13);
                      mesh_file >> out_c2v(i,14) >> out_c2v(i,15) >> out_c2v(i,16) >> out_c2v(i,17) >> out_c2v(i,18) >> out_c2v(i,19);
                      // faces
                      mesh_file >> out_c2v(i,20) >> out_c2v(i,21) >> out_c2v(i,22) >> out_c2v(i,23) >> out_c2v(i,24) >> out_c2v(i,25);
                      // volume
                      mesh_file >> out_c2v(i,26);
                    }
                }
              else
                {
                  cout << "elmtype=" << elmtype << endl;
                  FatalError("element type not recognized");
                }

              // Shift every values of c2v by -1
              for(int k=0;k<out_c2n_v(i);k++)
                {
                  if(out_c2v(i,k)!=0)
                    {
                      out_c2v(i,k)--;
                    }
                }

              i++;
              mesh_file.getline(buf,BUFSIZ); // skip end of line
            }
          else //
            {
              mesh_file.getline(buf,BUFSIZ); // skip line, cell doesn't belong to this processor
            }
          icount++; // FLUID cell, increase icount
        }
      else // Not FLUID cell, skip line
        {
          mesh_file.getline(buf,BUFSIZ); // skip line, cell doesn't belong to this processor
        }

    } // End of loop over entities

#ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  mesh_file.close();

}

#ifdef _MPI
void repartition_mesh(int &out_n_cells, array<int> &out_c2v, array<int> &out_c2n_v, array<int> &out_ctype, array<int> &out_ic2icg, struct solution* FlowSol)
{


  array<int> c2v_temp = out_c2v;
  array<int> c2n_v_temp = out_c2n_v;
  array<int> ctype_temp = out_ctype;
  array<int> ic2icg_temp = out_ic2icg;

  // Create array that stores the number of cells per proc
  int klocal = out_n_cells;
  array<int> kprocs(FlowSol->nproc);

  MPI_Allgather(&klocal,1,MPI_INT,kprocs.get_ptr_cpu(),1,MPI_INT,MPI_COMM_WORLD);

  int kstart = 0;
  for (int p=0;p<FlowSol->nproc-1;p++)
    kstart += kprocs(p);

  // element distribution
  int *elmdist= (int*) calloc(FlowSol->nproc+1,sizeof(int));
  elmdist[0] =0;
  for (int p=0;p<FlowSol->nproc;p++)
    elmdist[p+1] = elmdist[p] + kprocs(p);

  // list of element starts
  int *eptr = (int*) calloc(klocal+1,sizeof(int));
  eptr[0] = 0;
  for (int i=0;i<klocal;i++)
    {

      if (ctype_temp(i)==0)
        eptr[i+1] = eptr[i] + 3;
      else if (ctype_temp(i)==1)
        eptr[i+1] = eptr[i] + 4;
      else if (ctype_temp(i)==2)
        eptr[i+1] = eptr[i] + 4;
      else if (ctype_temp(i)==3)
        eptr[i+1] = eptr[i] + 6;
      else if (ctype_temp(i)==4)
        eptr[i+1] = eptr[i] + 8;
      else
        cout << "unknown element type, in repartitioning" << endl;
    }

  // local element to vertex
  int *eind = (int*) calloc(eptr[klocal],sizeof(int));
  int sk=0;
  int n_vertices;
  int j_spt;
  for (int i=0;i<klocal;i++)
    {
      if (ctype_temp(i) == 0) { n_vertices=3; }
      else if(ctype_temp(i) == 1) {n_vertices=4;}
      else if(ctype_temp(i) == 2) {n_vertices=4;}
      else if(ctype_temp(i) == 3) {n_vertices=6;}
      else if(ctype_temp(i) == 4) {n_vertices=8;}

      for (int j=0;j<n_vertices;j++)
        {
          get_vert_loc(ctype_temp(i),c2n_v_temp(i),j,j_spt);
          eind[sk] = c2v_temp(i,j_spt);
          sk++;
        }
    }

  //weight per element
  // TODO: Haven't tested this feature yet
  int *elmwgt = (int*) calloc(klocal,sizeof(int));
  for (int i=0;i<klocal;i++)
    {
      if (ctype_temp(i) == 0)
        elmwgt[i] = 1;
      else if (ctype_temp(i) == 1)
        elmwgt[i] = 1;
      else if (ctype_temp(i) == 2)
        elmwgt[i] = 1;
      else if (ctype_temp(i) == 3)
        elmwgt[i] = 1;
      else if (ctype_temp(i) == 4)
        elmwgt[i] = 1;
    }

  int wgtflag = 0;
  int numflag = 0;
  int ncon=1;

  int ncommonnodes;

  // TODO: assumes non-mixed grids
  // HACK
  if (FlowSol->n_dims==2)
    ncommonnodes=2;
  else if (FlowSol->n_dims==3)
    ncommonnodes=3;

  int nparts = FlowSol->nproc;

  float *tpwgts = (float*) calloc(klocal,sizeof(float));
  for (int i=0;i<klocal;i++)
    tpwgts[i] = 1./ (float)FlowSol->nproc;

  float *ubvec = (float*) calloc(ncon,sizeof(float));
  for (int i=0;i<ncon;i++)
    ubvec[i] = 1.05;

  int options[10];

  options[0] = 1;
  options[1] = 7;
  options[2] = 0;

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD,&comm);

  int edgecut;
  int *part= (int*) calloc(klocal,sizeof(int));

  if (FlowSol->rank==0) cout << "Before parmetis" << endl;

  ParMETIS_V3_PartMeshKway
      (elmdist,
       eptr,
       eind,
       elmwgt,
       &wgtflag,
       &numflag,
       &ncon,
       &ncommonnodes,
       &nparts,
       (real_t*)tpwgts,
       (real_t*)ubvec,
       options,
       &edgecut,
       part,
       &comm);

  if (FlowSol->rank==0) cout << "After parmetis " << endl;

  // Printing results of parmetis
  //array<int> part_array(klocal);
  //for (i=0;i<klocal;i++)
  //{
  //  part_array(i) = part[i];
  //}
  //part_array.print_to_file(FlowSol->rank);
  //cout << "After print" << endl;

  // Now creating new c2v array
  int **outlist = (int**) calloc(FlowSol->nproc,sizeof(int*));
  int **outlist_c2n_v = (int**) calloc(FlowSol->nproc,sizeof(int*));
  int **outlist_ctype = (int**) calloc(FlowSol->nproc,sizeof(int*));
  int **outlist_ic2icg = (int**) calloc(FlowSol->nproc,sizeof(int*));

  int *outK = (int*) calloc(FlowSol->nproc,sizeof(int));
  int *inK =  (int*) calloc(FlowSol->nproc,sizeof(int));

  for (int i=0;i<klocal;i++)
    ++outK[part[i]];

  MPI_Alltoall(outK,1,MPI_INT,
               inK, 1,MPI_INT,
               MPI_COMM_WORLD);

  // count totals on each processor
  int *newkprocs =  (int*) calloc(FlowSol->nproc,sizeof(int));
  MPI_Allreduce(outK,newkprocs,FlowSol->nproc,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  int totalinK = 0;
  for (int p=0;p<FlowSol->nproc;p++)
    totalinK += inK[p];

  // declare new array c2v
  int **new_c2v = (int**) calloc(totalinK,sizeof(int*));
  new_c2v[0] = (int*) calloc(totalinK*MAX_V_PER_C,sizeof(int));
  for (int i=1;i<totalinK;i++)
    new_c2v[i] = new_c2v[i-1]+MAX_V_PER_C;

  // declare new c2n_v,ctype
  int *new_c2n_v  = (int*) calloc(totalinK,sizeof(int*));
  int *new_ctype = (int*) calloc(totalinK,sizeof(int*));
  int *new_ic2icg = (int*) calloc(totalinK,sizeof(int*));

  MPI_Request *inrequests = (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));
  MPI_Request *inrequests_c2n_v = (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));
  MPI_Request *inrequests_ctype = (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));
  MPI_Request *inrequests_ic2icg= (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));

  MPI_Request *outrequests = (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));
  MPI_Request *outrequests_c2n_v = (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));
  MPI_Request *outrequests_ctype = (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));
  MPI_Request *outrequests_ic2icg= (MPI_Request*) calloc(FlowSol->nproc,sizeof(MPI_Request));

  MPI_Status *instatus = (MPI_Status*) calloc(FlowSol->nproc,sizeof(MPI_Status));
  MPI_Status *outstatus= (MPI_Status*) calloc(FlowSol->nproc,sizeof(MPI_Status));

  // Make exchange for arrays c2v,c2n_v,ctype,ic2icg
  int cnt=0;
  for (int p=0;p<FlowSol->nproc;p++)
    {
      MPI_Irecv(new_c2v[cnt], MAX_V_PER_C*inK[p],MPI_INT,p,666+p,MPI_COMM_WORLD,inrequests+p);
      MPI_Irecv(&new_c2n_v[cnt], inK[p],MPI_INT,p,666+p,MPI_COMM_WORLD,inrequests_c2n_v+p);
      MPI_Irecv(&new_ctype[cnt], inK[p],MPI_INT,p,666+p,MPI_COMM_WORLD,inrequests_ctype+p);
      MPI_Irecv(&new_ic2icg[cnt], inK[p],MPI_INT,p,666+p,MPI_COMM_WORLD,inrequests_ic2icg+p);
      cnt = cnt + inK[p];
    }

  for (int p=0;p<FlowSol->nproc;p++)
    {
      int cnt = 0;
      int cnt2 = 0;
      outlist[p] = (int*) calloc(MAX_V_PER_C*outK[p],sizeof(int));
      outlist_c2n_v[p] = (int*) calloc(outK[p],sizeof(int));
      outlist_ctype[p] = (int*) calloc(outK[p],sizeof(int));
      outlist_ic2icg[p] = (int*) calloc(outK[p],sizeof(int));

      for (int i=0;i<klocal;i++)
        {
          if (part[i]==p)
            {
              for (int v=0;v<MAX_V_PER_C;v++)
                {
                  outlist[p][cnt] = c2v_temp(i,v);
                  ++cnt;
                }
              outlist_c2n_v[p][cnt2] = c2n_v_temp(i);
              outlist_ctype[p][cnt2] = ctype_temp(i);
              outlist_ic2icg[p][cnt2] = ic2icg_temp(i);
              cnt2++;
            }
        }

      MPI_Isend(outlist[p],MAX_V_PER_C*outK[p],MPI_INT,p,666+FlowSol->rank,MPI_COMM_WORLD,outrequests+p);
      MPI_Isend(outlist_c2n_v[p],outK[p],MPI_INT,p,666+FlowSol->rank,MPI_COMM_WORLD,outrequests_c2n_v+p);
      MPI_Isend(outlist_ctype[p],outK[p],MPI_INT,p,666+FlowSol->rank,MPI_COMM_WORLD,outrequests_ctype+p);
      MPI_Isend(outlist_ic2icg[p],outK[p],MPI_INT,p,666+FlowSol->rank,MPI_COMM_WORLD,outrequests_ic2icg+p);
    }

  MPI_Waitall(FlowSol->nproc,inrequests,instatus);
  MPI_Waitall(FlowSol->nproc,inrequests_c2n_v,instatus);
  MPI_Waitall(FlowSol->nproc,inrequests_ctype,instatus);
  MPI_Waitall(FlowSol->nproc,inrequests_ic2icg,instatus);

  MPI_Waitall(FlowSol->nproc,outrequests,outstatus);
  MPI_Waitall(FlowSol->nproc,outrequests_c2n_v,outstatus);
  MPI_Waitall(FlowSol->nproc,outrequests_ctype,outstatus);
  MPI_Waitall(FlowSol->nproc,outrequests_ic2icg,outstatus);

  out_c2v.setup(totalinK,MAX_V_PER_C);
  out_c2n_v.setup(totalinK);
  out_ctype.setup(totalinK);
  out_ic2icg.setup(totalinK);

  for (int i=0;i<totalinK;i++)
    {
      for (int j=0;j<MAX_V_PER_C;j++)
        out_c2v(i,j) = new_c2v[i][j];

      out_c2n_v(i) = new_c2n_v[i];
      out_ctype(i) = new_ctype[i];
      out_ic2icg(i) = new_ic2icg[i];

    }
  out_n_cells = totalinK;

  MPI_Barrier(MPI_COMM_WORLD);

}

#endif

/*! method to create list of faces & edges from the mesh */
void CompConnectivity(array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_ctype, array<int>& out_c2f, array<int>& out_c2e,
                      array<int>& out_f2c, array<int>& out_f2loc_f, array<int>& out_f2v, array<int>& out_f2nv,
                      array<int>& out_e2v, array<int>& out_v2n_e, array<array<int> >& out_v2e,
                      array<int>& out_rot_tag, array<int>& out_unmatched_faces, int& out_n_unmatched_faces,
                      array<int>& out_icvsta, array<int>& out_icvert, int& out_n_faces, int& out_n_edges,
                      struct solution* FlowSol)
{

  // inputs:   in_c2v (clls to vertex) , in_ctype (type of cell)
  // outputs:  f2c (face to cell), c2f (cell to face), f2loc_f (face to local face index of right and left cells), rot_tag,  n_faces (number of faces in the mesh)

  int sta_ind,end_ind;
  int n_cells,n_verts;
  int num_v_per_f,num_v_per_f2;
  int iface, iface_old;
  int found,rtag;

  n_cells = in_c2v.get_dim(0);
  n_verts = in_c2v.get_max()+1;

  //array<int> num_v_per_c(5); // for 5 element types
  array<int> vlist_loc(MAX_V_PER_F),vlist_loc2(MAX_V_PER_F),vlist_glob(MAX_V_PER_F),vlist_glob2(MAX_V_PER_F); // faces cannot have more than 4 vertices

  array<int> v2n_c;

  /**
   * "Moving pointer" that points to next unassigned entry in icvert (for each vertex)
   * (see descriptions of icvsta & icvert)
   */
  array<int> icvsta2;

  // Number of vertices for different type of cells
  //num_v_per_c(0) = 3; // tri
  //num_v_per_c(1) = 4; // quads
  //num_v_per_c(2) = 4; // tets
  //num_v_per_c(3) = 6; // pris
  //num_v_per_c(4) = 8; // hexas

  v2n_c.setup(n_verts);
  icvsta2.setup(n_verts);
  // Assumes there won't be more than X cells around 1 vertex
  int max_cells_per_vert = 100; //50

  /**
   * List of cells around each vertex
   * First v2n_c(0) entries are all cells around node 0,
   * next v2n_c(1) etries are all cells around node 1, etc.
   */
  out_icvert.setup(max_cells_per_vert*n_verts+1);

  /**
   * Index of icvert corresponding to start of each vertices' entries
   * (see description of icvert)
   * Similar in function to the "row_ptr" of a compressed-sparse-row matrix
   */
  out_icvsta.setup(n_verts+1);

  // Initialize arrays to zero
  v2n_c.initialize_to_zero();
  icvsta2.initialize_to_zero();
  out_icvsta.initialize_to_zero();
  out_icvert.initialize_to_zero();
  vlist_loc.initialize_to_zero();
  vlist_loc2.initialize_to_zero();
  vlist_glob.initialize_to_zero();
  vlist_glob2.initialize_to_zero();

  // Determine how many cells share each node
  for (int ic=0;ic<n_cells;ic++) {
    for(int k=0;k<in_c2n_v(ic);k++) {
      v2n_c(in_c2v(ic,k))++;
    }
  }

  int k=0;
  int max_nc = 0;
  for(int iv=0;iv<n_verts;iv++)
  {
    if (v2n_c(iv)>max_nc)
      max_nc = v2n_c(iv);

    out_icvsta(iv) = k;
    icvsta2(iv) = k;
    k = k+v2n_c(iv);
  }

  //cout << "Maximum number of cells who share same vertex = " << max << endl;
  if (max_nc>max_cells_per_vert)
    FatalError("ERROR: some vertices are shared by more than max_cells_per_vert");

  out_icvsta(n_verts) = out_icvsta(n_verts-1)+v2n_c(n_verts-1);

  int iv,ic2,k2;
  for(int ic=0;ic<n_cells;ic++)
  {
    for(int k=0;k<in_c2n_v(ic);k++)
    {
      iv = in_c2v(ic,k);
      out_icvert(icvsta2(iv)) = ic;
      icvsta2(iv)++;
    }
  }

  out_n_edges=-1;
  if (FlowSol->n_dims==3 || run_input.motion!=0)
  {
    vector<int> e2v;
    vector<set<int> > v2e(n_verts);

      // Create array ic2e
      array<int> num_e_per_c(5);
      out_c2e.initialize_to_value(-1);

      num_e_per_c(0) = 3;
      num_e_per_c(1) = 4;
      num_e_per_c(2) = 6;
      num_e_per_c(3) = 9;
      num_e_per_c(4) = 12;

      for (int ic=0;ic<n_cells;ic++)
      {
          for(int k=0;k<num_e_per_c(in_ctype(ic));k++)
          {
              found = 0;
              if(out_c2e(ic,k) != -1) continue; // we have counted that face already

              out_n_edges++;
              out_c2e(ic,k) = out_n_edges;

              // Get global indices of points on this edge
              if (FlowSol->n_dims==3) {
                  get_vlist_loc_edge(in_ctype(ic),in_c2n_v(ic),k,vlist_loc);
              }else{
                  get_vlist_loc_face(in_ctype(ic),in_c2n_v(ic),k,vlist_loc,num_v_per_f);
              }

              for (int i=0;i<2;i++) {
                vlist_glob(i) = in_c2v(ic,vlist_loc(i));

                int iv = vlist_glob(i);
                e2v.push_back(iv);
                v2e[iv].insert(out_n_edges);
              }

              // loop over the cells touching vertex vlist_glob(0)
              sta_ind = out_icvsta(vlist_glob(0));
              end_ind = out_icvsta(vlist_glob(0)+1)-1;

              for(int ind=sta_ind;ind<=end_ind;ind++)
              {
                int ic2 = out_icvert(ind);
                if(ic2==ic) continue; // ic2 is the same as ic1 so skip it

                // Loop over edges of cell ic2 touching vertex vlist_glob(0)
                for(int k2=0;k2<num_e_per_c(in_ctype(ic2));k2++)
                {
                  // Get local vertices of local edge k2 of cell ic2
                  if (FlowSol->n_dims==3) {
                    get_vlist_loc_edge(in_ctype(ic2),in_c2n_v(ic2),k2,vlist_loc2);
                  }else{
                    get_vlist_loc_face(in_ctype(ic2),in_c2n_v(ic2),k2,vlist_loc2,num_v_per_f);
                  }

                  // get global vertices corresponding to local vertices
                  for (int i2=0;i2<2;i2++)
                    vlist_glob2(i2) = in_c2v(ic2,vlist_loc2(i2));

                  if ( (vlist_glob(0)==vlist_glob2(0) && vlist_glob(1)==vlist_glob2(1)) ||
                       (vlist_glob(0)==vlist_glob2(1) && vlist_glob(1)==vlist_glob2(0)) )
                  {
                    out_c2e(ic2,k2) = out_n_edges;
                  }
                }
              }
          } // Loop over edges
      } // Loop over cells
      out_n_edges++; // 0-index -> actual value

      set<int>::iterator it;
      // consider reversing for better use of CPU cache
      out_e2v.setup(out_n_edges,2);
      for (int ie=0; ie<out_n_edges; ie++) {
        out_e2v(ie,0) = e2v[2*ie];
        out_e2v(ie,1) = e2v[2*ie+1];
      }
      //out_v2e.setup(n_verts); //already setup
      if (n_verts != out_v2e.get_dim(0)) FatalError("n_verts & out_v2e not same size!!");
      for (int iv=0; iv<n_verts; iv++) {
        out_v2n_e(iv) = v2e[iv].size();
        out_v2e(iv).setup(out_v2n_e(iv));
        out_v2e(iv).initialize_to_zero();

        int ie = 0;
        for (it=v2e[iv].begin(); it!=v2e[iv].end(); ++it) {
          out_v2e(iv)(ie) = (*it);
          ie++;
        }
      }

  } // if n_dims=3 || motion != 0

  iface = 0;
  out_n_unmatched_faces= 0;

  // Loop over all the cells
  for(int ic=0;ic<n_cells;ic++)
    {
      //Loop over all faces of that cell
      for(int k=0;k< FlowSol->num_f_per_c(in_ctype(ic));k++)
        {
          found = 0;
          iface_old = iface;
          if(out_c2f(ic,k) != -1) continue; // we have counted that face already

          // Get local vertices of local face k of cell ic
          get_vlist_loc_face(in_ctype(ic),in_c2n_v(ic),k,vlist_loc,num_v_per_f);

          // get global vertices corresponding to local vertices
          for(int i=0;i<num_v_per_f;i++)
          {
            vlist_glob(i) = in_c2v(ic,vlist_loc(i));
          }

          // loop over the cells touching vertex vlist_glob(0)
          sta_ind = out_icvsta(vlist_glob(0));
          end_ind = out_icvsta(vlist_glob(0)+1)-1;

          for(int ind=sta_ind;ind<=end_ind;ind++)
          {
              ic2 = out_icvert(ind);

              if(ic2==ic) continue; // ic2 is the same as ic1 so skip it

              // Loop over faces of cell ic2 (which touches vertex vlist_glob(0))
              for(k2=0;k2<FlowSol->num_f_per_c(in_ctype(ic2));k2++)
              {
                  // Get local vertices of local face k2 of cell ic2
                  get_vlist_loc_face(in_ctype(ic2),in_c2n_v(ic2),k2,vlist_loc2,num_v_per_f2);

                  if (num_v_per_f2==num_v_per_f)
                  {
                      // get global vertices corresponding to local vertices
                      for (int i2=0;i2<num_v_per_f2;i2++)
                        vlist_glob2(i2) = in_c2v(ic2,vlist_loc2(i2));

                      // Compare the list of vertices
                      // If faces match returns 1
                      // For 3D returns the orientation of face2 wrt face1 (rtag)
                      // (see compare_faces for explanation of rtag)
                      compare_faces(vlist_glob,vlist_glob2,num_v_per_f,found,rtag);

                      if (found==1) break;
                  }
              }

              if (found==1) break;
          }

          if(found==1)
          {
            out_c2f(ic,k) = iface;
            out_c2f(ic2,k2) = iface;

            out_f2c(iface,0) = ic;
            out_f2c(iface,1) = ic2;

            out_f2loc_f(iface,0) = k;
            out_f2loc_f(iface,1) = k2;

            for(int i=0;i<num_v_per_f;i++)
              out_f2v(iface,i) = vlist_glob(i);

            out_f2nv(iface) = num_v_per_f;
            out_rot_tag(iface) = rtag;
            iface++;
          }
          else
          {
              // If loops over ic2 and k2 were unsuccesful, it means that face is not shared by two cells
              // Then continue here and set f2c( ,1) = -1
              if (out_f2c(iface_old+1,0)==-1)
              {
                  out_f2c(iface,0) = ic;
                  out_f2c(iface,1) = -1;

                  out_f2loc_f(iface,0) = k;
                  out_f2loc_f(iface,1) = -1;

                  out_c2f(ic,k) = iface;
                  for(int i=0;i<num_v_per_f;i++)
                  {
                    out_f2v(iface,i) = vlist_glob(i);
                  }

                  out_f2nv(iface) = num_v_per_f;

                  out_n_unmatched_faces++;
                  out_unmatched_faces(out_n_unmatched_faces-1) = iface;

                  iface=iface_old+1;
              }
          }

      }  // end of loop over k
  } // end of loop over ic

  out_n_faces = iface;
}


void get_vert_loc(int& in_ctype, int& in_n_spts, int& in_vert, int& out_v)
{
  if (in_ctype==0) // Tri
    {
      if (in_n_spts == 3 || in_n_spts==6)
        {
          out_v = in_vert;
        }
      else
        FatalError("in_nspt not implemented");
    }
  else if (in_ctype==1) // Quad
    {
      if (is_perfect_square(in_n_spts))
        {
          int n_spts_1d = round(sqrt(in_n_spts));
          if (in_vert==0)
            out_v = 0;
          else if (in_vert==1)
            out_v = n_spts_1d-1;
          else if (in_vert==2)
            out_v = in_n_spts-1;
          else if (in_vert==3)
            out_v = in_n_spts-n_spts_1d;
        }
      else if (in_n_spts==8)
        {
          out_v = in_vert;
        }
      else
        FatalError("in_nspt not implemented");
    }
  else if (in_ctype==2) // Tet
    {
      if (in_n_spts==4 || in_n_spts==10)
        out_v = in_vert;
      else
        FatalError("in_nspt not implemented");
    }
  else if (in_ctype==3) // Prism
    {
      if (in_n_spts==6 || in_n_spts==15)
        out_v = in_vert;
      else
        FatalError("in_nspt not implemented");
    }
  else if (in_ctype==4) // Hex
    {
      if (is_perfect_cube(in_n_spts))
        {
          int n_spts_1d = round(pow(in_n_spts,1./3.));
          int shift = n_spts_1d*n_spts_1d*(n_spts_1d-1);
          if(in_vert==0)
            {
              out_v = 0;
            }
          else if(in_vert==1)
            {
              out_v = n_spts_1d-1;
            }
          else if(in_vert==2)
            {
              out_v = n_spts_1d*n_spts_1d-1;
            }
          else if(in_vert==3)
            {
              out_v = n_spts_1d*(n_spts_1d-1);
            }
          else if(in_vert==4)
            {
              out_v = shift;
            }
          else if(in_vert==5)
            {
              out_v = n_spts_1d-1+shift;
            }
          else if(in_vert==6)
            {
              out_v = in_n_spts-1;
            }
          else if(in_vert==7)
            {
              out_v = in_n_spts-n_spts_1d;
            }
        }
      else if (in_n_spts==20)
        {
          out_v = in_vert;
        }
      else
        FatalError("in_nspt not implemented");
    }
}



void get_vlist_loc_edge(int& in_ctype, int& in_n_spts, int& in_edge, array<int>& out_vlist_loc)
{
  if (in_ctype==0 || in_ctype==1)
    {
      FatalError("2D elements not supported in get_vlist_loc_edge");
    }
  else if (in_ctype==2) // Tet
  {
    if (in_n_spts == 4) // Linear Tet
    {
      if(in_edge==0)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 1;
      }
      else if(in_edge==1)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 2;
      }
      else if(in_edge==2)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 3;
      }
      else if(in_edge==3)
      {
        out_vlist_loc(0) = 1;
        out_vlist_loc(1) = 3;
      }
      else if(in_edge==4)
      {
        out_vlist_loc(0) = 1;
        out_vlist_loc(1) = 2;
      }
      else if(in_edge==5)
      {
        out_vlist_loc(0) = 2;
        out_vlist_loc(1) = 3;
      }
    }
    else if (in_n_spts == 10)
    {
      if(in_edge==0)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 5;
      }
      else if(in_edge==1)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 3;
      }
      else if(in_edge==2)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 7;
      }
      else if(in_edge==3)
      {
        out_vlist_loc(0) = 5;
        out_vlist_loc(1) = 7;
      }
      else if(in_edge==4)
      {
        out_vlist_loc(0) = 5;
        out_vlist_loc(1) = 3;
      }
      else if(in_edge==5)
      {
        out_vlist_loc(0) = 3;
        out_vlist_loc(1) = 7;
      }
    }
  }
  else if (in_ctype==3) // Prism
    {
      if(in_edge==0)
        {
          out_vlist_loc(0) = 0;
          out_vlist_loc(1) = 1;
        }
      else if(in_edge==1)
        {
          out_vlist_loc(0) = 1;
          out_vlist_loc(1) = 2;
        }
      else if(in_edge==2)
        {
          out_vlist_loc(0) = 2;
          out_vlist_loc(1) = 0;
        }
      else if(in_edge==3)
        {
          out_vlist_loc(0) = 3;
          out_vlist_loc(1) = 4;
        }
      else if(in_edge==4)
        {
          out_vlist_loc(0) = 4;
          out_vlist_loc(1) = 5;
        }
      else if(in_edge==5)
        {
          out_vlist_loc(0) = 5;
          out_vlist_loc(1) = 3;
        }
      else if(in_edge==6)
        {
          out_vlist_loc(0) = 0;
          out_vlist_loc(1) = 3;
        }
      else if(in_edge==7)
        {
          out_vlist_loc(0) = 1;
          out_vlist_loc(1) = 4;
        }
      else if(in_edge==8)
        {
          out_vlist_loc(0) = 2;
          out_vlist_loc(1) = 5;
        }
    }
  else if (in_ctype==4) // Hexa
    {
      if (is_perfect_cube(in_n_spts))
        {
          int n_spts_1d = round(pow(in_n_spts,1./3.));
          int shift = n_spts_1d*n_spts_1d*(n_spts_1d-1);
          if(in_edge==0)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = n_spts_1d-1;
            }
          else if(in_edge==1)
            {
              out_vlist_loc(0) = n_spts_1d-1;
              out_vlist_loc(1) = n_spts_1d*n_spts_1d-1;

            }
          else if(in_edge==2)
            {
              out_vlist_loc(0) = n_spts_1d*n_spts_1d-1;
              out_vlist_loc(1) = n_spts_1d*(n_spts_1d-1);
            }
          else if(in_edge==3)
            {
              out_vlist_loc(0) = n_spts_1d*(n_spts_1d-1);
              out_vlist_loc(1) = 0;
            }
          else if(in_edge==4)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = shift;
            }
          else if(in_edge==5)
            {
              out_vlist_loc(0) = n_spts_1d-1;
              out_vlist_loc(1) = n_spts_1d-1+shift;
            }
          else if(in_edge==6)
            {
              out_vlist_loc(0) = n_spts_1d*n_spts_1d-1;
              out_vlist_loc(1) = in_n_spts-1;
            }
          else if(in_edge==7)
            {
              out_vlist_loc(0) = n_spts_1d*(n_spts_1d-1);
              out_vlist_loc(1) = in_n_spts-n_spts_1d;
            }
          else if(in_edge==8)
            {
              out_vlist_loc(0) = shift;
              out_vlist_loc(1) = n_spts_1d-1+shift;
            }
          else if(in_edge==9)
            {
              out_vlist_loc(0) = n_spts_1d-1+shift;
              out_vlist_loc(1) = in_n_spts-1;
            }
          else if(in_edge==10)
            {
              out_vlist_loc(0) = in_n_spts-1;
              out_vlist_loc(1) = in_n_spts-n_spts_1d;
            }
          else if(in_edge==11)
            {
              out_vlist_loc(0) = in_n_spts-n_spts_1d;
              out_vlist_loc(1) = shift;
            }
        }
      else if (in_n_spts==20)
        {
          if(in_edge==0)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = 1;
            }
          else if(in_edge==1)
            {
              out_vlist_loc(0) = 1;
              out_vlist_loc(1) = 2;
            }
          else if(in_edge==2)
            {
              out_vlist_loc(0) = 2;
              out_vlist_loc(1) = 3;
            }
          else if(in_edge==3)
            {
              out_vlist_loc(0) = 3;
              out_vlist_loc(1) = 0;
            }
          else if(in_edge==4)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = 4;
            }
          else if(in_edge==5)
            {
              out_vlist_loc(0) = 1;
              out_vlist_loc(1) = 5;
            }
          else if(in_edge==6)
            {
              out_vlist_loc(0) = 2;
              out_vlist_loc(1) = 6;
            }
          else if(in_edge==7)
            {
              out_vlist_loc(0) = 3;
              out_vlist_loc(1) = 7;
            }
          else if(in_edge==8)
            {
              out_vlist_loc(0) = 4;
              out_vlist_loc(1) = 5;
            }
          else if(in_edge==9)
            {
              out_vlist_loc(0) = 5;
              out_vlist_loc(1) = 6;
            }
          else if(in_edge==10)
            {
              out_vlist_loc(0) = 6;
              out_vlist_loc(1) = 7;
            }
          else if(in_edge==11)
            {
              out_vlist_loc(0) = 7;
              out_vlist_loc(1) = 4;
            }
        }
    }
}

void get_vlist_loc_face(int& in_ctype, int& in_n_spts, int& in_face, array<int>& out_vlist_loc, int& num_v_per_f)
{

  if (in_ctype==0) // Triangle
    {
      num_v_per_f = 2;
      if(in_face==0)
        {
          out_vlist_loc(0) = 0;
          out_vlist_loc(1) = 1;
        }
      else if(in_face==1)
        {
          out_vlist_loc(0) = 1;
          out_vlist_loc(1) = 2;

        }
      else if(in_face==2)
        {
          out_vlist_loc(0) = 2;
          out_vlist_loc(1) = 0;
        }
    }
  else if (in_ctype==1) // Quad
    {
      num_v_per_f = 2;
      if (is_perfect_square(in_n_spts))
        {
          int n_spts_1d = round(sqrt(in_n_spts));
          if (in_face==0)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = n_spts_1d-1;
            }
          else if (in_face==1)
            {
              out_vlist_loc(0) = n_spts_1d-1;
              out_vlist_loc(1) = in_n_spts-1;
            }
          else if (in_face==2)
            {
              out_vlist_loc(0) = in_n_spts-1;
              out_vlist_loc(1) = in_n_spts-n_spts_1d;
            }
          else if (in_face==3)
            {
              out_vlist_loc(0) = in_n_spts-n_spts_1d;
              out_vlist_loc(1) = 0;
            }
        }
      else if (in_n_spts==8)
        {
          if(in_face==0)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = 1;
            }
          else if(in_face==1)
            {
              out_vlist_loc(0) = 1;
              out_vlist_loc(1) = 2;

            }
          else if(in_face==2)
            {
              out_vlist_loc(0) = 2;
              out_vlist_loc(1) = 3;
            }
          else if(in_face==3)
            {
              out_vlist_loc(0) = 3;
              out_vlist_loc(1) = 0;
            }
        }
      else
        {
          cout << "in_nspts=" << in_n_spts << endl;
          cout << "ctype=" << in_ctype<< endl;
          FatalError("in_nspt not implemented");
        }
    }
  else if (in_ctype==2) // Tet
  {
    num_v_per_f = 3;
    if (in_n_spts == 4) // Linear Tet
    {
      if(in_face==0)
      {
        out_vlist_loc(0) = 1;
        out_vlist_loc(1) = 2;
        out_vlist_loc(2) = 3;
      }
      else if(in_face==1)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 3;
        out_vlist_loc(2) = 2;

      }
      else if(in_face==2)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 1;
        out_vlist_loc(2) = 3;
      }
      else if(in_face==3)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 2;
        out_vlist_loc(2) = 1;
      }
    }
    else if (in_n_spts == 10)  // Quadratic Tet
    {
      if(in_face==0)
      {
        out_vlist_loc(0) = 5;
        out_vlist_loc(1) = 3;
        out_vlist_loc(2) = 7;
      }
      else if(in_face==1)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 7;
        out_vlist_loc(2) = 3;
      }
      else if(in_face==2)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 5;
        out_vlist_loc(2) = 7;
      }
      else if(in_face==3)
      {
        out_vlist_loc(0) = 0;
        out_vlist_loc(1) = 3;
        out_vlist_loc(2) = 5;
      }
    }
  }
  else if (in_ctype==3) // Prism
    {
      if(in_face==0)
        {
          num_v_per_f = 3;
          out_vlist_loc(0) = 0;
          out_vlist_loc(1) = 2;
          out_vlist_loc(2) = 1;
        }
      else if(in_face==1)
        {
          num_v_per_f = 3;
          out_vlist_loc(0) = 3;
          out_vlist_loc(1) = 4;
          out_vlist_loc(2) = 5;

        }
      else if(in_face==2)
        {
          num_v_per_f = 4;
          out_vlist_loc(0) = 0;
          out_vlist_loc(1) = 1;
          out_vlist_loc(2) = 4;
          out_vlist_loc(3) = 3;
        }
      else if(in_face==3)
        {
          num_v_per_f = 4;
          out_vlist_loc(0) = 1;
          out_vlist_loc(1) = 2;
          out_vlist_loc(2) = 5;
          out_vlist_loc(3) = 4;
        }
      else if(in_face==4)
        {
          num_v_per_f = 4;
          out_vlist_loc(0) = 2;
          out_vlist_loc(1) = 0;
          out_vlist_loc(2) = 3;
          out_vlist_loc(3) = 5;
        }
    }
  else if (in_ctype==4) // Hexas
    {
      num_v_per_f = 4;

      if (is_perfect_cube(in_n_spts))
        {
          int n_spts_1d = round(pow(in_n_spts,1./3.));
          int shift = n_spts_1d*n_spts_1d*(n_spts_1d-1);
          if(in_face==0)
            {
              out_vlist_loc(0) = n_spts_1d-1;
              out_vlist_loc(1) = 0;
              out_vlist_loc(2) = n_spts_1d*(n_spts_1d-1);
              out_vlist_loc(3) = n_spts_1d*n_spts_1d-1;

            }
          else if(in_face==1)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = n_spts_1d-1;
              out_vlist_loc(2) = n_spts_1d-1+shift;
              out_vlist_loc(3) = shift;
            }
          else if(in_face==2)
            {
              out_vlist_loc(0) = n_spts_1d-1;
              out_vlist_loc(1) = n_spts_1d*n_spts_1d-1;
              out_vlist_loc(2) = in_n_spts-1;
              out_vlist_loc(3) = n_spts_1d-1+shift;
            }
          else if(in_face==3)
            {
              out_vlist_loc(0) = n_spts_1d*n_spts_1d-1;
              out_vlist_loc(1) = n_spts_1d*(n_spts_1d-1);
              out_vlist_loc(2) = in_n_spts-n_spts_1d;
              out_vlist_loc(3) = in_n_spts-1;
            }
          else if(in_face==4)
            {
              out_vlist_loc(0) = n_spts_1d*(n_spts_1d-1);
              out_vlist_loc(1) = 0;
              out_vlist_loc(2) = shift;
              out_vlist_loc(3) = in_n_spts-n_spts_1d;
            }
          else if(in_face==5)
            {
              out_vlist_loc(0) = shift;
              out_vlist_loc(1) = n_spts_1d-1+shift;
              out_vlist_loc(2) = in_n_spts-1;
              out_vlist_loc(3) = in_n_spts-n_spts_1d;
            }
        }
      else if (in_n_spts==20)
        {
          if(in_face==0)
            {
              out_vlist_loc(0) = 1;
              out_vlist_loc(1) = 0;
              out_vlist_loc(2) = 3;
              out_vlist_loc(3) = 2;

            }
          else if(in_face==1)
            {
              out_vlist_loc(0) = 0;
              out_vlist_loc(1) = 1;
              out_vlist_loc(2) = 5;
              out_vlist_loc(3) = 4;
            }
          else if(in_face==2)
            {
              out_vlist_loc(0) = 1;
              out_vlist_loc(1) = 2;
              out_vlist_loc(2) = 6;
              out_vlist_loc(3) = 5;
            }
          else if(in_face==3)
            {
              out_vlist_loc(0) = 2;
              out_vlist_loc(1) = 3;
              out_vlist_loc(2) = 7;
              out_vlist_loc(3) = 6;
            }
          else if(in_face==4)
            {
              out_vlist_loc(0) = 3;
              out_vlist_loc(1) = 0;
              out_vlist_loc(2) = 4;
              out_vlist_loc(3) = 7;
            }
          else if(in_face==5)
            {
              out_vlist_loc(0) = 4;
              out_vlist_loc(1) = 5;
              out_vlist_loc(2) = 6;
              out_vlist_loc(3) = 7;
            }
        }
      else
        FatalError("n_spts not implemented");
    }
  else
  {
    cout << "in_ctype = " << in_ctype << endl;
    FatalError("ERROR: Haven't implemented other 3D Elements yet");
  }

}

void compare_faces(array<int>& vlist1, array<int>& vlist2, int& num_v_per_f, int& found, int& rtag)
{
  /* Looking at a face from *inside* the cell, the nodes *must* be numbered in *CW* order
   * (this is in agreement with Gambit; Gmsh does not care about local face numberings)
   *
   * The variable 'rtag' matches up the local node numbers of two overlapping faces from
   * different cells (specifically, it is which node from face 2 matches node 0 from face 1)
   */
  if(num_v_per_f==2) // edge
    {
      if ( (vlist1(0)==vlist2(0) && vlist1(1)==vlist2(1)) ||
           (vlist1(0)==vlist2(1) && vlist1(1)==vlist2(0)) )
        {
          found = 1;
          rtag = 0;
        }
      else
        found= 0;
    }
  else if(num_v_per_f==3) // tri face
    {
      //rot_tag==0
      if (vlist1(0)==vlist2(0) &&
          vlist1(1)==vlist2(2) &&
          vlist1(2)==vlist2(1) )
        {
          rtag = 0;
          found= 1;
        }
      //rot_tag==1
      else if (vlist1(0)==vlist2(2) &&
               vlist1(1)==vlist2(1) &&
               vlist1(2)==vlist2(0) )
        {
          rtag = 1;
          found= 1;
        }
      //rot_tag==2
      else if (vlist1(0)==vlist2(1) &&
               vlist1(1)==vlist2(0) &&
               vlist1(2)==vlist2(2) )
        {
          rtag = 2;
          found= 1;
        }
      else
        found= 0;
    }
  else if(num_v_per_f==4) // quad face
    {
      //rot_tag==0
      if (vlist1(0)==vlist2(1) &&
          vlist1(1)==vlist2(0) &&
          vlist1(2)==vlist2(3) &&
          vlist1(3)==vlist2(2) )
        {
          rtag = 0;
          found= 1;
        }
      //rot_tag==1
      else if (vlist1(0)==vlist2(3) &&
               vlist1(1)==vlist2(2) &&
               vlist1(2)==vlist2(1) &&
               vlist1(3)==vlist2(0) )
        {
          rtag = 1;
          found= 1;
        }
      //rot_tag==2
      else if (vlist1(0)==vlist2(0) &&
               vlist1(1)==vlist2(3) &&
               vlist1(2)==vlist2(2) &&
               vlist1(3)==vlist2(1) )
        {
          rtag = 2;
          found= 1;
        }
      //rot_tag==3
      else if (vlist1(0)==vlist2(2) &&
               vlist1(1)==vlist2(1) &&
               vlist1(2)==vlist2(0) &&
               vlist1(3)==vlist2(3) )
        {
          rtag = 3;
          found= 1;
        }
      else
        found= 0;

    }
  else
    {
      FatalError("ERROR: Haven't implemented this face type in compare_face yet....");
    }

}

void compare_faces_boundary(array<int>& vlist1, array<int>& vlist2, int& num_v_per_f, int& found)
{

  if ( !(num_v_per_f==2 || num_v_per_f==3 || num_v_per_f==4))
    FatalError("Boundary face type not recognized (expecting a linear edge, tri, or quad)");

  int count = 0;
  for (int j=0;j<num_v_per_f;j++) {
    for (int k=0;k<num_v_per_f;k++) {
      if (vlist1(j)==vlist2(k)) {
        count ++;
        break;
      }
    }
  }

  if (count==num_v_per_f)
    found=1;
  else
    found=0;
}



// method to compare two faces and check if they match

void compare_cyclic_faces(array<double> &xvert1, array<double> &xvert2, int& num_v_per_f, int& rtag, array<double> &delta_cyclic, double tol, struct solution* FlowSol)
{
  int found = 0;
  if (FlowSol->n_dims==2)
    {
      found = 1;
      rtag = 0;
    }
  else if (FlowSol->n_dims==3)
    {
      if(num_v_per_f==4) // quad face
        {
          //printf("cell 1, x1=%2.8f, x2=%2.8f, x3=%2.8f\n cell 2(1), x1=%2.8f, x2=%2.8f, x3=%2.8f\n",xvert1(0,0),xvert1(0,1),xvert1(0,2),xvert2(0,0),xvert2(0,1),xvert2(0,2));
          //printf("cell 2(2), x1=%2.8f, x2=%2.8f, x3=%2.8f\n cell 2(3), x1=%2.8f, x2=%2.8f, x3=%2.8f\n cell 2(4), x1=%2.8f, x2=%2.8f, x3=%2.8f\n",xvert2(1,0),xvert2(1,1),xvert2(1,2),xvert2(2,0),xvert2(2,1),xvert2(2,2),xvert2(3,0),xvert2(3,1),xvert2(3,2));
          // Determine rot_tag based on xvert1(0)
          // vert1(0) matches vert2(1), rot_tag=0
          if(
             (    abs(abs(xvert1(0,0)-xvert2(1,0))-delta_cyclic(0)) < tol
                  && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(1,1))-delta_cyclic(1)) < tol
                  && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(1,1))               ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(1,2))-delta_cyclic(2)) < tol )
             )
            {
              rtag = 0;
              found= 1;
            }
          // vert1(0) matches vert2(3), rot_tag=1
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(3,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(3,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(3,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(3,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(3,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(3,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(3,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(3,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(3,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 1;
              found= 1;
            }
          // vert1(0) matches vert2(0), rot_tag=2
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(0,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(0,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(0,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 2;
              found= 1;
            }
          // vert1(0) matches vert2(2), rot_tag=3
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 3;
              found= 1;
            }
        }
      else if(num_v_per_f==3) // tri face
        {
          //printf("cell 1, x1=%f, x2=%f, x3=%f\n cell 2, x1=%f, x2=%f, x3=%f\n",xvert1(0,0),xvert1(1,0),xvert1(2,0),xvert2(0,0),xvert2(1,0),xvert2(2,0));
          //printf("cell 1, y1=%f, y2=%f, y3=%f\n cell 2, y1=%f, y2=%f, y3=%f\n",xvert1(0,1),xvert1(1,1),xvert1(2,1),xvert2(0,1),xvert2(1,1),xvert2(2,1));
          // Determine rot_tag based on xvert1(0)
          // vert1(0) matches vert2(0), rot_tag=0
          if(
             (    abs(abs(xvert1(0,0)-xvert2(0,0))-delta_cyclic(0)) < tol
                  && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(0,1))-delta_cyclic(1)) < tol
                  && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(0,2))-delta_cyclic(2)) < tol )
             )
            {
              rtag = 0;
              found= 1;
            }
          // vert1(0) matches vert2(2), rot_tag=1
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 1;
              found= 1;
            }
          // vert1(0) matches vert2(1), rot_tag=2
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(1,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(1,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(1,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 2;
              found= 1;
            }
        }
      else
        {
          FatalError("ERROR: Haven't implemented this face type in compare_cyclic_face yet....");
        }
    }

  if (found==0)
    FatalError("Could not match vertices in compare faces");

}

bool check_cyclic(array<double> &delta_cyclic, array<double> &loc_center_inter_0, array<double> &loc_center_inter_1, double tol, struct solution* FlowSol)
{

  bool output;
  if (FlowSol->n_dims==3)
    {
      output =
          (abs(abs(loc_center_inter_0(0)-loc_center_inter_1(0))-delta_cyclic(0))<tol
           && abs(loc_center_inter_0(1)-loc_center_inter_1(1))<tol
           && abs(loc_center_inter_0(2)-loc_center_inter_1(2))<tol ) ||

          ( abs(loc_center_inter_0(0)-loc_center_inter_1(0))<tol
            && abs(abs(loc_center_inter_0(1)-loc_center_inter_1(1))-delta_cyclic(1))<tol
            && abs(loc_center_inter_0(2)-loc_center_inter_1(2))<tol ) ||

          ( abs(loc_center_inter_0(0)-loc_center_inter_1(0))<tol
            && abs(loc_center_inter_0(1)-loc_center_inter_1(1))<tol
            && abs(abs(loc_center_inter_0(2)-loc_center_inter_1(2))-delta_cyclic(2))<tol);
    }
  else if (FlowSol->n_dims==2)
    {
      output =
          (abs(abs(loc_center_inter_0(0)-loc_center_inter_1(0))-delta_cyclic(0))<tol
           && abs(loc_center_inter_0(1)-loc_center_inter_1(1))<tol ) ||

          ( abs(loc_center_inter_0(0)-loc_center_inter_1(0))<tol
            && abs(abs(loc_center_inter_0(1)-loc_center_inter_1(1))-delta_cyclic(1))<tol);
    }

  return output;
}

#ifdef _MPI

void match_mpifaces(array<int> &in_f2v, array<int> &in_f2nv, array<double>& in_xv, array<int>& inout_f_mpi2f, array<int>& out_mpifaces_part, array<double> &delta_cyclic, int n_mpi_faces, double tol, struct solution* FlowSol)
{

  // TODO: THIS IS NOT OPTIMAL: GOES AS N^2 operations, try sorting and searching instead (cost will be 2*N*log(N)
  // TODO: TIDY UP

  int i,iglob, k,v0,v1,v2,v3;
  int icount,p,p2,rtag;
  int iloc,irem;

  array<int> matched(n_mpi_faces);
  array<int> old_f_mpi2f;

  old_f_mpi2f = inout_f_mpi2f;

  MPI_Status instatus;

  array<double> delta_zero(FlowSol->n_dims);
  for (int m=0;m<FlowSol->n_dims;m++)
    delta_zero(m) = 0.;

  // Calculate the centroid of each face
  array<double> loc_center_inter(FlowSol->n_dims,n_mpi_faces);

  for(i=0;i<n_mpi_faces;i++)
    {
      for (int m=0;m<FlowSol->n_dims;m++)
        loc_center_inter(m,i) = 0.;

      iglob = inout_f_mpi2f(i);
      for (k=0;k<in_f2nv(iglob);k++)
        for (int m=0;m<FlowSol->n_dims;m++)
          loc_center_inter(m,i) += in_xv(in_f2v(iglob,k),m)/in_f2nv(iglob);
    }

  // Initialize array matched with 0
  for(i=0;i<n_mpi_faces;i++)
    matched(i) = 0;

  //Initialize array mpifaces_part to 0
  for(i=0;i<FlowSol->nproc;i++)
    out_mpifaces_part(i) = 0;

  // Exchange the number of mpi_faces to receive
  // Create array mpfaces_from
  array<int> mpifaces_from(FlowSol->nproc);
  MPI_Allgather( &n_mpi_faces,1,MPI_INT,mpifaces_from.get_ptr_cpu(),1,MPI_INT,MPI_COMM_WORLD);

  int max_mpi_faces = 0;
  for(i=0;i<FlowSol->nproc;i++)
    if (mpifaces_from(i) >= max_mpi_faces) max_mpi_faces = mpifaces_from(i);

  // Allocate the xyz_cent with the max_mpi_faces size;
  array<double> in_loc_center_inter(FlowSol->n_dims,max_mpi_faces);
  array<double> loc_center_1(FlowSol->n_dims);
  array<double> loc_center_2(FlowSol->n_dims);

  // Begin the exchange
  icount = 0;
  for(p=0;p<FlowSol->nproc;p++) {
      if(p==FlowSol->rank) { // Send data
          for (p2=0;p2<FlowSol->nproc;p2++) {

              if (p2!=FlowSol->rank)  {
                  //MPI_Send to processor p2
                  MPI_Send(loc_center_inter.get_ptr_cpu(),FlowSol->n_dims*n_mpi_faces,MPI_DOUBLE,p2,100000*p+1000*p2,MPI_COMM_WORLD);
                }

            }
          out_mpifaces_part(p) = 0; // Processor p won't send any edges to himself
        }
      else // Receive data
        {
          //MPI_Recv from processor p
          MPI_Recv(in_loc_center_inter.get_ptr_cpu(),FlowSol->n_dims*mpifaces_from(p),MPI_DOUBLE,p,100000*p+1000*FlowSol->rank,MPI_COMM_WORLD,&instatus);

          if (p<FlowSol->rank)
            {
              // Loop over local mpi_edges
              for (iloc=0;iloc<n_mpi_faces;iloc++) {
                  if (!matched(iloc)) // if local edge hasn't been matched yet
                    {
                      // Loop over remote edges just received
                      for(irem=0;irem<mpifaces_from(p);irem++)
                        {
                          for (int m=0;m<FlowSol->n_dims;m++) {
                              loc_center_1(m) = in_loc_center_inter(m,irem);
                              loc_center_2(m) = loc_center_inter(m,iloc);
                            }

                          if (check_cyclic(delta_cyclic,loc_center_1,loc_center_2,tol,FlowSol) ||
                              check_cyclic(delta_zero  ,loc_center_1,loc_center_2,tol,FlowSol) )
                            {
                              matched(iloc) = 1;
                              out_mpifaces_part(p)++;
                              i = old_f_mpi2f(iloc);
                              inout_f_mpi2f(icount) = i;

                              icount++;
                              break;
                            }
                        }
                    }
                }
            }
          else // if p >= FlowSol->rank
            {
              // Loop over remote edges
              for (irem=0;irem<mpifaces_from(p);irem++)
                {
                  for (iloc=0;iloc<n_mpi_faces;iloc++)
                    {
                      if (!matched(iloc)) // if local edge hasn't been matched yet
                        {
                          for (int m=0;m<FlowSol->n_dims;m++) {
                              loc_center_1(m) = in_loc_center_inter(m,irem);
                              loc_center_2(m) = loc_center_inter(m,iloc);
                            }
                          // Check if it matches vertex iloc
                          if (check_cyclic(delta_cyclic,loc_center_1,loc_center_2,tol,FlowSol) ||
                              check_cyclic(delta_zero  ,loc_center_1,loc_center_2,tol,FlowSol))
                            {
                              matched(iloc) = 1;
                              out_mpifaces_part(p)++;
                              i = old_f_mpi2f(iloc);
                              inout_f_mpi2f(icount) = i;

                              icount++;
                              break;
                            }
                        }
                    }
                }
            }
        }
      MPI_Barrier(MPI_COMM_WORLD);
    }


  // Check that every edge has been matched
  for (i=0;i<n_mpi_faces;i++)
    {
      if (!matched(i))
        {
          cout << "Some mpi_faces were not matched!!! could try changing tol, exiting!" << endl;
          cout << "rank=" << FlowSol->rank << "i=" << i << endl;
          exit(1);
        }
    }
}

void find_rot_mpifaces(array<int> &in_f2v, array<int> &in_f2nv, array<double>& in_xv, array<int>& in_f_mpi2f, array<int> &out_rot_tag_mpi, array<int> &mpifaces_part, array<double> delta_cyclic, int n_mpi_faces, double tol, struct solution* FlowSol)
{

  int Nout,i,i_mpi,p,iglob,k;
  int n_vert_out;
  int count1,count2,count3;
  int found,rtag;

  array<double> xvert1(MAX_V_PER_F,FlowSol->n_dims); // 4 is maximum number of vertices per face
  array<double> xvert2(MAX_V_PER_F,FlowSol->n_dims);

  // Count the number of messages to send
  int request_count = 0;
  for (int p=0;p<FlowSol->nproc;p++) {
      if (mpifaces_part(p)!=0) request_count++;
    }

  MPI_Request* mpi_in_requests = (MPI_Request*) malloc(request_count*sizeof(MPI_Request));
  MPI_Request* mpi_out_requests = (MPI_Request*) malloc(request_count*sizeof(MPI_Request));

  // Count number of vertices to send
  n_vert_out=0;
  for(i_mpi=0;i_mpi<n_mpi_faces;i_mpi++)
    {
      iglob = in_f_mpi2f(i_mpi);
      for(k=0;k<in_f2nv(iglob);k++)
        n_vert_out++;
    }

  array<double> xyz_vert_out(FlowSol->n_dims,n_vert_out);
  array<double> xyz_vert_in(FlowSol->n_dims,n_vert_out);

  int Nmess = 0;
  int sk = 0;

  count2 = 0;
  count3 = 0;
  request_count=0;

  for (int p=0;p<FlowSol->nproc;p++)
    {
      count1 = 0;
      for(i=0;i<mpifaces_part(p);i++)
        {
          i_mpi = count2+i;
          iglob = in_f_mpi2f(i_mpi);
          for(k=0;k<in_f2nv(iglob);k++)
            {
              for (int m=0;m<FlowSol->n_dims;m++)
                xyz_vert_out(m,count3) = in_xv(in_f2v(iglob,k),m);
              count3++;
            }
          count1 += in_f2nv(iglob);
        }

      Nout = count1;
      count2 += mpifaces_part(p);

      if (Nout)
        {
          MPI_Isend(xyz_vert_out.get_ptr_cpu(0,sk),Nout*FlowSol->n_dims,MPI_DOUBLE,p,6666+p   ,MPI_COMM_WORLD,&mpi_out_requests[request_count]);
          MPI_Irecv(xyz_vert_in.get_ptr_cpu(0,sk),Nout*FlowSol->n_dims,MPI_DOUBLE,p,6666+FlowSol->rank,MPI_COMM_WORLD,&mpi_in_requests[request_count]);
          sk+=Nout;
          Nmess++;
          request_count++;
        }
    }

  MPI_Waitall(Nmess,mpi_in_requests,MPI_STATUSES_IGNORE);
  MPI_Waitall(Nmess,mpi_out_requests,MPI_STATUSES_IGNORE);

  MPI_Barrier(MPI_COMM_WORLD);

  array<double> loc_vert_0(MAX_V_PER_F,FlowSol->n_dims);
  array<double> loc_vert_1(MAX_V_PER_F,FlowSol->n_dims);

  count1 = 0;
  for(i_mpi=0;i_mpi<n_mpi_faces;i_mpi++)
    {
      iglob = in_f_mpi2f(i_mpi);
      for(k=0;k<in_f2nv(iglob);k++)
        {
          for (int m=0;m<FlowSol->n_dims;m++)
            {
              loc_vert_0(k,m) = xyz_vert_out(m,count1);
              loc_vert_1(k,m) = xyz_vert_in(m,count1);
            }
          count1++;
        }

      compare_mpi_faces(loc_vert_0,loc_vert_1,in_f2nv(iglob),rtag,delta_cyclic,tol,FlowSol);
      out_rot_tag_mpi(i_mpi) = rtag;
    }
}


// method to compare two faces and check if they match
void compare_mpi_faces(array<double> &xvert1, array<double> &xvert2, int& num_v_per_f, int& rtag, array<double> &delta_cyclic, double tol, struct solution* FlowSol)
{
  int found = 0;
  if (FlowSol->n_dims==2)
    {
      found = 1;
      rtag = 0;
    }
  else if (FlowSol->n_dims==3)
    {
      if(num_v_per_f==4) // quad face
        {
          // Determine rot_tag based on xvert1(0)
          // vert1(0) matches vert2(1), rot_tag=0
          if(
             (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(1,0))-delta_cyclic(0)) < tol
                  && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(1,1))-delta_cyclic(1)) < tol
                  && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(1,1))               ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(1,2))-delta_cyclic(2)) < tol )
             )
            {
              rtag = 0;
              found= 1;
            }
          // vert1(0) matches vert2(3), rot_tag=1
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(3,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(3,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(3,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(3,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(3,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(3,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(3,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(3,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(3,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(3,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(3,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(3,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 1;
              found= 1;
            }
          // vert1(0) matches vert2(0), rot_tag=2
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(0,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(0,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(0,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 2;
              found= 1;
            }
          // vert1(0) matches vert2(2), rot_tag=3
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 3;
              found= 1;
            }
        }
      else if(num_v_per_f==3) // tri face
        {
          // vert1(0) matches vert2(0), rot_tag=0
          if(
             (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(0,0))-delta_cyclic(0)) < tol
                  && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(0,1))-delta_cyclic(1)) < tol
                  && abs(abs(xvert1(0,2)-xvert2(0,2))          ) < tol ) ||
             (    abs(abs(xvert1(0,0)-xvert2(0,0))          ) < tol
                  && abs(abs(xvert1(0,1)-xvert2(0,1))          ) < tol
                  && abs(abs(xvert1(0,2)-xvert2(0,2))-delta_cyclic(2)) < tol )
             )
            {
              rtag = 0;
              found= 1;
            }
          // vert1(0) matches vert2(2), rot_tag=1
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(2,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(2,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(2,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 1;
              found= 1;
            }
          // vert1(0) matches vert2(1), rot_tag=2
          else if (
                   (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(1,0))-delta_cyclic(0)) < tol
                        && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(1,1))-delta_cyclic(1)) < tol
                        && abs(abs(xvert1(0,2)-xvert2(1,2))          ) < tol ) ||
                   (    abs(abs(xvert1(0,0)-xvert2(1,0))          ) < tol
                        && abs(abs(xvert1(0,1)-xvert2(1,1))          ) < tol
                        && abs(abs(xvert1(0,2)-xvert2(1,2))-delta_cyclic(2)) < tol )
                   )
            {
              rtag = 2;
              found= 1;
            }
        }
      else
        {
          FatalError("ERROR: Haven't implemented this face type in compare_cyclic_face yet....");
        }
    }

  if (found==0)
    FatalError("Could not match vertices in compare faces");

}

#endif

