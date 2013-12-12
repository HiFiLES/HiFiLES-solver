/*!
 * \file geometry.cpp
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

// used to switch between single- and multi-zone tecplot binary output
#define MULTI_ZONE
//#define SINGLE_ZONE

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
	if ((FlowSol->rank%2)==0) { cudaSetDevice(0); }
	if ((FlowSol->rank%2)==1) { cudaSetDevice(1); }
  
#endif
  
#endif
  
}

int get_bc_number(string& bcname) {
  
  int bcflag;
  
  if (!bcname.compare("Sub_In_Simp")) bcflag = 1;
	else if (!bcname.compare("Sub_Out_Simp")) bcflag = 2;
	else if (!bcname.compare("Sub_In_Char")) bcflag = 3;
	else if (!bcname.compare("Sub_Out_Char")) bcflag = 4;
	else if (!bcname.compare("Sup_In")) bcflag = 5;
	else if (!bcname.compare("Sup_Out")) bcflag = 6;
	else if (!bcname.compare("Slip_Wall")) bcflag = 7;
	else if (!bcname.compare("Cyclic")) bcflag = 9;
	else if (!bcname.compare("Isotherm_Fix")) bcflag = 11;
	else if (!bcname.compare("Adiabat_Fix")) bcflag = 12;
	else if (!bcname.compare("Isotherm_Move")) bcflag = 13;
	else if (!bcname.compare("Adiabat_Move")) bcflag = 14;
	else if (!bcname.compare("Char")) bcflag = 15;
	else if (!bcname.compare("Slip_Wall_Dual")) bcflag = 16;
	else if (!bcname.compare("Char_lala")) bcflag = 17;
	else if (!bcname.compare("AD_Wall")) bcflag = 50;
	else if (!bcname.compare("Bdy1")) bcflag = 100;
	else if (!bcname.compare("Bdy2")) bcflag = 200;
	else if (!bcname.compare("Bdy3")) bcflag = 300;
	else
  {
    cout << "Boundary=" << bcname << endl;
    FatalError("Boundary condition not recognized");
  }
  
  return bcflag;
}

void GeoPreprocess(int in_run_type, struct solution* FlowSol) {
  array<double> xv;
  array<int> c2v,c2n_v,ctype,bctype_c,ic2icg,iv2ivg;
  
  /*! Reading vertices and cells. */
  ReadMesh(run_input.mesh_file, xv, c2v, c2n_v, ctype, ic2icg, iv2ivg, FlowSol->num_eles, FlowSol->num_verts, FlowSol);
  
  if (in_run_type==1) // Plotting mode
  {
    /*! Store c2v, c2n_v, ctype. */
    FlowSol->ele2vert = c2v;
    FlowSol->ele2n_vert = c2n_v;
    FlowSol->ele_type = ctype;
  }
  
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
  
	// Initialize arrays to -1
	for (int i=0;i<max_inters;i++) {
		f2c(i,0) = f2c(i,1) = -1;
		f2loc_f(i,0) = f2loc_f(i,1) = -1;
	}
  
	for (int i=0;i<FlowSol->num_eles;i++)
		for(int k=0;k<MAX_F_PER_C;k++)
      c2f(i,k)=-1;
  
  array<int> icvsta, icvert;
  
  // Compute connectivity
	CompConectivity(c2v, c2n_v, ctype, c2f, c2e, f2c, f2loc_f, f2v, f2nv, rot_tag, unmatched_inters, n_unmatched_inters, icvsta, icvert, FlowSol->num_inters, FlowSol->num_edges, FlowSol);
  
  // Reading boundaries
  ReadBound(run_input.mesh_file,c2v,c2n_v,ctype,bctype_c,ic2icg,icvsta,icvert,iv2ivg,FlowSol->num_eles,FlowSol->num_verts, FlowSol);
  if (in_run_type==1)
  {
    // Store ele2face
    FlowSol->ele2face = c2f;
    FlowSol->ele2edge = c2e;
    FlowSol->inter2loc_inter = f2loc_f;
    FlowSol->inter2ele = f2c;
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
    cout << "Error in mesh reader, n_dims=2 and 3d elements exists" << endl;
    exit(1);
  }
  if (FlowSol->n_dims == 3 && (num_tris!= 0 || num_quads != 0)) {
    cout << "Error in mesh reader, n_dims=3 and 2d elements exists" << endl;
    exit(1);
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
  
  for (int i=0;i<FlowSol->n_ele_types;i++)
  {
    FlowSol->mesh_eles(i)->set_rank(FlowSol->rank);
  }
  
  if (FlowSol->rank==0) cout << "initializing elements" << endl;
  if (FlowSol->rank==0) cout << "tris" << endl;
	FlowSol->mesh_eles_tris.setup(num_tris,max_n_spts_per_tri,in_run_type);
  if (FlowSol->rank==0) cout << "quads" << endl;
	FlowSol->mesh_eles_quads.setup(num_quads,max_n_spts_per_quad,in_run_type);
  if (FlowSol->rank==0) cout << "tets" << endl;
	FlowSol->mesh_eles_tets.setup(num_tets,max_n_spts_per_tet,in_run_type);
  if (FlowSol->rank==0) cout << "pris" << endl;
	FlowSol->mesh_eles_pris.setup(num_pris,max_n_spts_per_pri,in_run_type);
  if (FlowSol->rank==0) cout << "hexas" << endl;
	FlowSol->mesh_eles_hexas.setup(num_hexas,max_n_spts_per_hexa,in_run_type);
  if (FlowSol->rank==0) cout << "done initializing elements" << endl;
  
  // Set shape for each cell
  array<int> local_c(FlowSol->num_eles);
  
  int tris_count = 0;
  int quads_count = 0;
  int tets_count = 0;
  int pris_count = 0;
  int hexas_count = 0;
  
  array<double> pos(FlowSol->n_dims);
  
  if (FlowSol->rank==0) cout << "setting elements shape" << endl;
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
  if (FlowSol->rank==0) cout << "done setting elements shape" << endl;
  
	// set transforms
	if (FlowSol->rank==0) cout << "setting element transforms ... " << endl;
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
		  FlowSol->mesh_eles(i)->set_transforms(in_run_type);
    }
  }
  
  // Set metrics at interface cubpts
	if (FlowSol->rank==0) cout << "setting element transforms at interface cubpts ... " << endl;
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
		  FlowSol->mesh_eles(i)->set_transforms_inters_cubpts();
    }
  }
  
  // Set metrics at volume cubpts
	if (FlowSol->rank==0) cout << "setting element transforms at volume cubpts ... " << endl;
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
		  FlowSol->mesh_eles(i)->set_transforms_vol_cubpts();
    }
  }
  
	// set on gpu (important - need to do this before we set connectivity, so that pointers point to GPU memory)
#ifdef _GPU
  if (in_run_type==0)
  {
	  for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
        
	      if (FlowSol->rank==0) cout << "Moving eles to GPU ... " << endl;
	  	  FlowSol->mesh_eles(i)->mv_all_cpu_gpu();
      }
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
  
	double tol = 1.e-8;
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
  
  FlowSol->mesh_mpi_inters(0).setup(n_seg_mpi_inters,0,in_run_type);
  FlowSol->mesh_mpi_inters(1).setup(n_tri_mpi_inters,1,in_run_type);
  FlowSol->mesh_mpi_inters(2).setup(n_quad_mpi_inters,2,in_run_type);
  
  array<int> mpifaces_part(FlowSol->nproc);
  
	// Call function that takes in f_mpi2f,f2v and returns a new array f_mpi2f, and an array mpiface_part
  // that contains the number of faces to send to each processor
	// the new array f_mpi2f is in good order i.e. proc1,proc2,....
  
  match_mpifaces(f2v,f2nv,xv,f_mpi2f,mpifaces_part,delta_cyclic,FlowSol->n_mpi_inters,tol,FlowSol);
  
  if (in_run_type==1)
  {
    FlowSol->inter_mpi2inter = f_mpi2f;
  }
  
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
      FlowSol->mesh_mpi_inters(0).set_mpi(i_seg_mpi,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),rot_tag_mpi(i_mpi),in_run_type,FlowSol);
      i_seg_mpi++;
    }
    else if (f2nv(i)==3) {
      FlowSol->mesh_mpi_inters(1).set_mpi(i_tri_mpi,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),rot_tag_mpi(i_mpi),in_run_type,FlowSol);
		  i_tri_mpi++;
    }
    else if (f2nv(i)==4) {
      FlowSol->mesh_mpi_inters(2).set_mpi(i_quad_mpi,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),rot_tag_mpi(i_mpi),in_run_type,FlowSol);
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
  if (in_run_type==0)
  {
	  for(int i=0;i<FlowSol->n_mpi_inter_types;i++)
	  	FlowSol->mesh_mpi_inters(i).mv_all_cpu_gpu();
  }
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
			if (bctype_f==0)	// internal interface
			{
				if (ic_r==-1)
				{
					cout << "Error: Interior interface has ic_r=-1. Should not be here, exiting" << endl;
					exit(1);
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
  FlowSol->mesh_int_inters(0).setup(n_seg_int_inters,0,in_run_type);
  FlowSol->mesh_int_inters(1).setup(n_tri_int_inters,1,in_run_type);
  FlowSol->mesh_int_inters(2).setup(n_quad_int_inters,2,in_run_type);
  
	FlowSol->n_bdy_inter_types=3;
	FlowSol->mesh_bdy_inters.setup(FlowSol->n_bdy_inter_types);
  FlowSol->mesh_bdy_inters(0).setup(n_seg_bdy_inters,0,in_run_type);
  FlowSol->mesh_bdy_inters(1).setup(n_tri_bdy_inters,1,in_run_type);
  FlowSol->mesh_bdy_inters(2).setup(n_quad_bdy_inters,2,in_run_type);
  
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
          FlowSol->mesh_int_inters(0).set_interior(i_seg_int,ctype(ic_l),ctype(ic_r),local_c(ic_l),local_c(ic_r),f2loc_f(i,0),f2loc_f(i,1),rot_tag(i),in_run_type,FlowSol);
          i_seg_int++;
        }
        if (f2nv(i)==3) {
          FlowSol->mesh_int_inters(1).set_interior(i_tri_int,ctype(ic_l),ctype(ic_r),local_c(ic_l),local_c(ic_r),f2loc_f(i,0),f2loc_f(i,1),rot_tag(i),in_run_type,FlowSol);
				  i_tri_int++;
        }
        if (f2nv(i)==4) {
          FlowSol->mesh_int_inters(2).set_interior(i_quad_int,ctype(ic_l),ctype(ic_r),local_c(ic_l),local_c(ic_r),f2loc_f(i,0),f2loc_f(i,1),rot_tag(i),in_run_type,FlowSol);
				  i_quad_int++;
        }
			}
			else // boundary face other than cyclic face
			{
				if (bctype_f!=99) //  Not a deleted cyclic face
				{
          if (f2nv(i)==2){
            FlowSol->mesh_bdy_inters(0).set_boundary(i_seg_bdy,bctype_f,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),in_run_type,FlowSol);
					  i_seg_bdy++;
          }
          else if (f2nv(i)==3){
            FlowSol->mesh_bdy_inters(1).set_boundary(i_tri_bdy,bctype_f,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),in_run_type,FlowSol);
					  i_tri_bdy++;
          }
          else if (f2nv(i)==4){
            FlowSol->mesh_bdy_inters(2).set_boundary(i_quad_bdy,bctype_f,ctype(ic_l),local_c(ic_l),f2loc_f(i,0),in_run_type,FlowSol);
					  i_quad_bdy++;
          }
				}
			}
		}
	}
  
	// set on GPU
#ifdef _GPU
  if (in_run_type==0)
  {
	  if (FlowSol->rank==0) cout << "Moving interfaces to GPU ... " << endl;
	  for(int i=0;i<FlowSol->n_int_inter_types;i++)
	  	FlowSol->mesh_int_inters(i).mv_all_cpu_gpu();
    
	  for(int i=0;i<FlowSol->n_bdy_inter_types;i++)
	  	FlowSol->mesh_bdy_inters(i).mv_all_cpu_gpu();
  }
#endif
  
}

void ReadMesh(string& in_file_name, array<double>& out_xv, array<int>& out_c2v, array<int>& out_c2n_v, array<int>& out_ctype, array<int>& out_ic2icg, array<int>& out_iv2ivg, int& out_n_cells, int& out_n_verts, struct solution* FlowSol) {
  
  if (FlowSol->rank==0) cout << "reading connectivity" << endl;
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
  
  if (run_input.mesh_format==0) { read_vertices_gambit(in_file_name, n_verts, out_iv2ivg, out_xv, FlowSol); }
  else if (run_input.mesh_format==1) { read_vertices_gmsh(in_file_name, n_verts, out_iv2ivg, out_xv, FlowSol); }
  else { FatalError("Mesh format not recognized"); }
  
  out_n_verts = n_verts;
  if (FlowSol->rank==0) cout << "done reading vertices" << endl;

}

void ReadBound(string& in_file_name, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_ctype, array<int>& out_bctype, array<int>& in_ic2icg, array<int>& in_icvsta, array<int>&in_icvert, array<int>& in_iv2ivg, int& in_n_cells, int& in_n_verts, struct solution* FlowSol)
{
  
  if (FlowSol->rank==0) cout << "reading boundary conditions" << endl;
  // Set the boundary conditions
  // HACK
  out_bctype.setup(in_n_cells,MAX_F_PER_C);
  
	// initialize to 0 (as interior edges)
	for (int i=0;i<in_n_cells;i++)
		for (int k=0;k<MAX_F_PER_C;k++)
      out_bctype(i,k) = 0;
  
  if (run_input.mesh_format==0) {
    read_boundary_gambit(in_file_name, in_n_cells, in_ic2icg, out_bctype);
  }
  else if (run_input.mesh_format==1) {
    read_boundary_gmsh(in_file_name, in_n_cells, in_ic2icg, in_c2v, in_c2n_v, out_bctype, in_iv2ivg, in_n_verts, in_ctype, in_icvsta, in_icvert, FlowSol);
    
  }
  else {
    FatalError("Mesh format not recognized");
  }
  
  if (FlowSol->rank==0) cout << "done reading boundary conditions" << endl;
}


// Method to read boundary edges in mesh file
void read_boundary_gambit(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& out_bctype)
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
  mesh_file       >> n_verts_global       // num vertices in mesh
  >> n_cells_global // num elements
  >> n_mats       // num material groups
  >> n_bcs        // num boundary groups
  >> dummy;        // num space dimensions
  
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
		if (3!=nread) {cout << "ERROR while reading Gambit file" << endl; cout << "nread =" << nread << endl; exit(1); }
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
  
  int bcNF, bcID, bcflag,icg,k, real_k, index;
  string bcname;
  char bcTXT[100];
  
	for (int i=0;i<n_bcs;i++)
	{
		mesh_file.getline(buf,BUFSIZ);	// Load ith boundary group
		if (strstr(buf,"ENDOFSECTION")){
			continue; // may not have a boundary section
		}
    
		sscanf(buf,"%s %d %d", bcTXT, &bcID, &bcNF);
    
		bcname.assign(bcTXT,0,14);
    bcflag = get_bc_number(bcname);
    
    int bdy_count = 0;
		for (int bf=0;bf<bcNF;bf++)
		{
			mesh_file >> icg >> dummy >> k;
			icg--;
			// Matching Gambit faces with face convention in code
      if (dummy==2 || dummy==3)
        real_k = k-1;
      // Hex
      else if (dummy==4)
			{
				if (k==1)
					real_k = 0;
				else if (k==2)
					real_k = 3;
				else if (k==3)
					real_k = 5;
				else if (k==4)
					real_k = 1;
				else if (k==5)
					real_k = 4;
				else if (k==6)
					real_k = 2;
			}
      // Tet
			else if (dummy==6)
			{
				if (k==1)
					real_k = 3;
				else if (k==2)
					real_k = 2;
				else if (k==3)
					real_k = 0;
				else if (k==4)
					real_k = 1;
			}
			else if (dummy==5)
			{
				if (k==1)
					real_k = 2;
				else if (k==2)
					real_k = 3;
				else if (k==3)
					real_k = 4;
				else if (k==4)
					real_k = 0;
				else if (k==5)
					real_k = 1;
			}
			else
			{
				cout << "ERROR: cannot handle other element type in readbnd" << endl;
				exit(1);
        
			}
			// Check if cell icg belongs to processor
			index = index_locate_int(icg,cell_list.get_ptr_cpu(),in_n_cells);
      
			// If it does, find local cell ic corresponding to icg
			if (index!=-1)
			{
        bdy_count++;
        out_bctype(index,real_k) = bcflag;
        /*
         // Loop over the array ic2icg and find ic
         for (int ic=0;ic<in_n_cells;ic++)
         {
         if (in_ic2icg(ic)==icg)
         {
         out_bctype(ic,real_k) = bcflag;
         cout << "ic=" << ic << endl;
         cout << "index=" << index << endl;
         }
         }
         */
        
			}
		}
    
		mesh_file.getline(buf,BUFSIZ); // Clear "end of line"
		mesh_file.getline(buf,BUFSIZ); // Skip "ENDOFSECTION"
		mesh_file.getline(buf,BUFSIZ); // Skip "Element group"
	}
  
  mesh_file.close();
}

void read_boundary_gmsh(string& in_file_name, int &in_n_cells, array<int>& in_ic2icg, array<int>& in_c2v, array<int>& in_c2n_v, array<int>& out_bctype, array<int>& in_iv2ivg, int in_n_verts, array<int>& in_ctype, array<int>& in_icvsta, array<int>& in_icvert, struct solution* FlowSol)
{
  string str;
  
  ifstream mesh_file;
  
	mesh_file.open(&in_file_name[0]);
	if (!mesh_file)
    FatalError("Unable to open mesh file");
  
  // Move cursor to $PhysicalNames
  while(1) {
    getline(mesh_file,str);
    if (str=="$PhysicalNames") break;
  }
  
	// Read number of boundaries and fields defined
  int dummy,dummy2;
  int id,elmtype,ntags,bcid,bcdim,bcflag;
  
	char buf[BUFSIZ]={""};
	char bcTXT[100][100];// can read up to 100 different boundary conditions
	char bc_txt_temp[100];
  
	mesh_file >> dummy;
	mesh_file.getline(buf,BUFSIZ);	// clear rest of line
	for(int i=0;i<dummy;i++)
	{
		mesh_file.getline(buf,BUFSIZ);
		sscanf(buf,"%d %d \"%s", &bcdim, &bcid, bc_txt_temp);
		strcpy(bcTXT[bcid],bc_txt_temp);
	}
  
  // Move cursor to $Elements
  while(1) {
    getline(mesh_file,str);
    if (str=="$Elements") break;
  }
  
  // Each processor reads number of entities
  int n_entities;
	// Read number of elements and bdys
  mesh_file 	>> n_entities; 	// num cells in mesh
	mesh_file.getline(buf,BUFSIZ);	// clear rest of line
  
  array<int> vlist_boun(4), vlist_cell(4);
  array<int> vlist_local(4);
  
  int found, num_v_per_f;
  int num_face_vert;
  
  string bcname;
  int bdy_count=0;
  
  int sta_ind,end_ind;
  
  for (int i=0;i<n_entities;i++)
  {
		mesh_file >> id >> elmtype >> ntags;
    
		if (ntags!=2)
      FatalError("ntags != 2,exiting");
    
		mesh_file >> bcid >> dummy;
    
		if (strstr(bcTXT[bcid],"FLUID"))
    {
	    mesh_file.getline(buf,BUFSIZ);	// skip line
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
      mesh_file >> vlist_boun(0) >> vlist_boun(1);
    }
    else if (elmtype==3) // Quad face
    {
      num_face_vert = 4;
      mesh_file >> vlist_boun(0) >> vlist_boun(1) >> vlist_boun(3) >> vlist_boun(2);
    }
    else
      FatalError("Boundary elmtype not recognized");
    
    // Shift by -1
    
    for (int j=0;j<num_face_vert;j++)
    {
      vlist_boun(j)--;
    }
    
	  mesh_file.getline(buf,BUFSIZ);	// Get rest of line
    
    // Check if two vertices belong to processor
    bool belong_to_proc = 1;
    for (int j=0;j<num_face_vert;j++)
    {
      vlist_local(j) = index_locate_int(vlist_boun(j),in_iv2ivg.get_ptr_cpu(),in_n_verts);
      if (vlist_local(j) == -1)
        belong_to_proc = 0;
    }
    
    if (belong_to_proc)
    {
      // Both vertices belong to processor
      // Try to find the cell that they belong to
      found=0;
      
      // Loop over cells touching that vertex
	    sta_ind = in_icvsta(vlist_local(0));
	    end_ind = in_icvsta(vlist_local(0)+1)-1;
      
      //for (int ic=0;ic<in_n_cells;ic++)
      for (int ind=sta_ind;ind<=end_ind;ind++)
      {
        int ic=in_icvert(ind);
        for (int k=0;k<FlowSol->num_f_per_c(in_ctype(ic));k++)
        {
          // Get local vertices of local face k of cell ic
          //cout << "ctype(ic)=" << in_ctype(ic) << "n_v=" << in_c2n_v(ic) << endl;
          get_vlist_loc_face(in_ctype(ic),in_c2n_v(ic),k,vlist_cell,num_v_per_f);
          
          //cout << "Before, vlist_cell(0)=" << vlist_cell(0) << endl;
          //cout << "Before, vlist_cell(1)=" << vlist_cell(1) << endl;
          
          if (num_v_per_f!= num_face_vert)
            continue;
          
          for (int j=0;j<num_v_per_f;j++)
          {
            //cout << "ic=" << ic << endl;
            vlist_cell(j) = in_c2v(ic,vlist_cell(j));
          }
          
          //cout << "vlist_cell(0)=" << vlist_cell(0) << endl;
          //cout << "vlist_cell(1)=" << vlist_cell(1) << endl;
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
        cout << "num_v_per_face=" << num_v_per_f << endl;
        cout << "vlist_boun(0)=" << vlist_boun(0) << " vlist_boun(1)=" << vlist_boun(1) << endl;
        cout << "vlist_boun(2)=" << vlist_boun(2) << " vlist_boun(3)=" << vlist_boun(3) << endl;
        cout << "Warning, all nodes of boundary face belong to processor but could not find the coresponding faces" << endl;
        exit(1);
      }
      
    } // If all vertices belong to processor
    
  } // Loop over entities
  
  
  /*
   for (int i=0;i<in_n_cells;i++)
   for (int j=0;j<MAX_F_PER_C;j++)
   cout << out_bctype(i,j) << endl;
   */
  
  mesh_file.close();
  
  cout << "bdy_count=" << bdy_count << endl;
  
}

void read_vertices_gambit(string& in_file_name, int in_n_verts, array<int> &in_iv2ivg, array<double> &out_xv, struct solution* FlowSol)
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
  int n_verts_global, dummy;
  mesh_file       >> n_verts_global // num vertices in mesh
  >> dummy 	// num elements
  >> dummy 	// num material groups
  >> dummy 	// num boundary groups
  >> FlowSol->n_dims;        // num space dimensions
  
  mesh_file.getline(buf,BUFSIZ);  // clear rest of line
  mesh_file.getline(buf,BUFSIZ);  // Skip 2 lines
  mesh_file.getline(buf,BUFSIZ);
  
  // Read the location of vertices
	int icount = 0;
  int id,index;
  double pos;
	for (int i=0;i<n_verts_global;i++)
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

void read_vertices_gmsh(string& in_file_name, int in_n_verts, array<int> &in_iv2ivg, array<double> &out_xv, struct solution* FlowSol)
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
    if (str=="$Nodes") break;
  }
  
  int n_verts_global;
  double pos;
  
  mesh_file       >> n_verts_global ;// num vertices in mesh
  mesh_file.getline(buf,BUFSIZ);
  
  int id;
  int index;
  
  for (int i=0;i<n_verts_global;i++)
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
	for (int i=0;i<6;i++) mesh_file.getline(buf,BUFSIZ);
  
  // Find number of vertices and number of cells
  mesh_file 	>> n_verts_global 	// num vertices in mesh
  >> n_cells_global	// num elements
  >> dummy        // num material groups
  >> dummy         // num boundary groups
  >> FlowSol->n_dims;	// num space dimensions
  
	mesh_file.getline(buf,BUFSIZ);	// clear rest of line
	mesh_file.getline(buf,BUFSIZ);	// Skip 2 lines
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
	for (int i=0;i<out_n_cells;i++)
	{
	  //  ctype is the element type:	1=edge, 2=quad, 3=tri, 4=brick, 5=wedge, 6=tet, 7=pyramid
		mesh_file >> out_ic2icg(i) >> dummy >> out_c2n_v(i);
    
    if (dummy==3) out_ctype(i)=0;
    else if (dummy==2) out_ctype(i)=1;
    else if (dummy==6) out_ctype(i)=2;
    else if (dummy==5) out_ctype(i)=3;
    else if (dummy==4) out_ctype(i)=4;
    
    // triangle
		if (out_ctype(i)==0)
		{
			if (out_c2n_v(i)==3) // linear triangle
				mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2);
			else if (out_c2n_v(i)==6) // quadratic triangle
			  mesh_file >> out_c2v(i,0) >> out_c2v(i,3) >>  out_c2v(i,1) >> out_c2v(i,4) >> out_c2v(i,2) >> out_c2v(i,5);
      else
        FatalError("triangle element type not implemented");
		}
    // quad
		else if (out_ctype(i)==1)
		{
			if (out_c2n_v(i)==4) // linear quadrangle
				mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,3) >> out_c2v(i,2);
			else if (out_c2n_v(i)==8)  // quadratic quad
			  mesh_file >> out_c2v(i,0) >> out_c2v(i,4) >> out_c2v(i,1) >> out_c2v(i,5) >> out_c2v(i,2) >> out_c2v(i,6) >> out_c2v(i,3) >> out_c2v(i,7);
      else
        FatalError("quad element type not implemented");
		}
    // tet
		else if (out_ctype(i)==2)
		{
			if (out_c2n_v(i)==4) // linear tets
      {
				mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3);
      }
      else if (out_c2n_v(i)==10) // quadratic tet
      {
				mesh_file >> out_c2v(i,0) >> out_c2v(i,4) >> out_c2v(i,1) >> out_c2v(i,5) >> out_c2v(i,7) >> out_c2v(i,2) >> out_c2v(i,6) >> out_c2v(i,9) >> out_c2v(i,8) >> out_c2v(i,3);
      }
      else
        FatalError("tet element type not implemented");
		}
    // prisms
		else if (out_ctype(i)==3)
		{
			if (out_c2n_v(i)==6) // linear prism
				mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,2) >> out_c2v(i,3) >> out_c2v(i,4) >> out_c2v(i,5);
			else if (out_c2n_v(i)==15) // quadratic prism
				mesh_file >> out_c2v(i,0) >> out_c2v(i,6) >> out_c2v(i,1) >> out_c2v(i,8) >> out_c2v(i,7) >> out_c2v(i,2) >> out_c2v(i,9) >> out_c2v(i,10) >> out_c2v(i,11) >> out_c2v(i,3) >> out_c2v(i,12) >> out_c2v(i,4) >> out_c2v(i,14) >> out_c2v(i,13) >> out_c2v(i,5) ;
      else
        FatalError("Prism element type not implemented");
		}
    // hexa
    else if (out_ctype(i)==4)
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
			cout << "Haven't implemented this element type in gambit_meshreader3, exiting " << endl;
			exit(1);
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
	int n_verts_global,n_cells_global;
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
  while(1) {
    getline(mesh_file,str);
    if (str=="$PhysicalNames") break;
  }
  
	// Read number of boundaries and fields defined
	mesh_file >> dummy;
	cout << "dummy "<<dummy << endl;
	mesh_file.getline(buf,BUFSIZ);	// clear rest of line
	for(int i=0;i<dummy;i++)
	{
		cout << "i "<<i << endl;
		mesh_file.getline(buf,BUFSIZ);
		sscanf(buf,"%d %d %s", &bcdim, &bcid, bc_txt_temp);
		strcpy(bcTXT[bcid],bc_txt_temp);
		cout << "bc_txt_temp " <<bc_txt_temp<< endl;
    if (strcmp(bc_txt_temp,"FLUID"))
      FlowSol->n_dims=bcdim;
	}
  
  // Move cursor to $Elements
  while(1) {
    getline(mesh_file,str);
    if (str=="$Elements") break;
  }
  
	// -------------------------------
	//  Read element connectivity
	//  ------------------------------
	
  // Each processor first reads number of global cells
  int n_entities;
	// Read number of elements and bdys
  mesh_file 	>> n_entities; 	// num cells in mesh
	mesh_file.getline(buf,BUFSIZ);	// clear rest of line
  
  int icount=0;
  
  for (int i=0;i<n_entities;i++)
  {
		mesh_file >> id >> elmtype >> ntags;
		if (ntags!=2)
      FatalError("ntags != 2,exiting");
    
		mesh_file >> bcid >> dummy;
		//cout << "bcid "<<bcid << endl;
		//cout << "strstr(bcTXT[bcid] "<<strstr(bcTXT[bcid]) << endl;
		if (strstr(bcTXT[bcid],"FLUID"))
      icount++;
    
	  mesh_file.getline(buf,BUFSIZ);	// clear rest of line
    
  }
  n_cells_global=icount;
  
  cout << "n_cell_global=" << n_cells_global << endl;
  
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
  //
  mesh_file.clear();
  mesh_file.seekg(0, ios::beg);
  while(1) {
    getline(mesh_file,str);
    if (str=="$Elements") break;
  }
  
  mesh_file 	>> n_entities; 	// num cells in mesh
	mesh_file.getline(buf,BUFSIZ);	// clear rest of line
  
  // Skip elements being read by other processors
  icount=0;
  int i=0;
  
  //cout << "out_n_cells=" << out_n_cells << endl;
  //cout << "k_start=" << kstart << endl;
  
  for (int k=0;k<n_entities;k++)
  {
		mesh_file >> id >> elmtype >> ntags;
    
    //cout << "id=" <<  id << endl;
    //cout << "k=" << k << endl;
    
		if (ntags!=2)
    {
      cout << "ntags=" << ntags << endl;
      FatalError("ntags != 2,exiting");
    }
    
    //cout << "elmtype=" << elmtype << endl;
    
		mesh_file >> bcid >> dummy;
		if (strstr(bcTXT[bcid],"FLUID"))
    {
      //cout << "icount=" << icount << endl;
      if (icount>=kstart && i< out_n_cells) // Read this cell
      {
        //cout << "i=" << i << endl;
        //cout << "bcid" << bcid << " dummy=" << dummy << endl;
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
            //cout << "here" << endl;
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
				    mesh_file >> out_c2v(i,0) >> out_c2v(i,2) >> out_c2v(i,8) >> out_c2v(i,6) >> out_c2v(i,1) >> out_c2v(i,5) >> out_c2v(i,7) >> out_c2v(i,3) >> out_c2v(i,4);
            //cout << "i=" << i << "id=" << id << endl;
            //cout << "out_c2v(i,0)=" << out_c2v(i,0) << endl;
            //cout << "out_c2v(i,1)=" << out_c2v(i,1) << endl;
            //cout << "out_c2v(i,2)=" << out_c2v(i,2) << endl;
            //cout << "out_c2v(i,3)=" << out_c2v(i,3) << endl;
            //cout << "out_c2v(i,4)=" << out_c2v(i,4) << endl;
          }
        }
        else if (elmtype==5) // Hexahedral
        {
          out_ctype(i) = 4;
          if (elmtype==5) // linear quadrangle
          {
            out_c2n_v(i) = 8;
					  mesh_file >> out_c2v(i,0) >> out_c2v(i,1) >> out_c2v(i,3) >> out_c2v(i,2);
					  mesh_file >> out_c2v(i,4) >> out_c2v(i,5) >> out_c2v(i,7) >> out_c2v(i,6);
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
            //cout << "elmtype=" << elmtype << endl;
            //cout << "out_c2v=" << out_c2v(i,k) << endl;
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
  
  //out_n_cells=icount;
  //cout << "n_cells=" << out_n_cells << endl;
  
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
			elmwgt[i] = 1.;
		else if (ctype_temp(i) == 1)
			elmwgt[i] = 1.;
		else if (ctype_temp(i) == 2)
			elmwgt[i] = 1.;
		else if (ctype_temp(i) == 3)
			elmwgt[i] = 1.;
		else if (ctype_temp(i) == 4)
			elmwgt[i] = 1.;
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
   tpwgts,
   ubvec,
   options,
   &edgecut,
   part,
   &comm);
  
	if (FlowSol->rank==0) cout << "After parmetis " << endl;
  
	// Printing results of parmetis
	//array<int> part_array(klocal);
	//for (i=0;i<klocal;i++)
	//{
	//	part_array(i) = part[i];
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

/*! method to create list of faces from the mesh */
void CompConectivity(array<int>& in_c2v, array<int>& in_c2n_v, array<int>& in_ctype, array<int>& out_c2f, array<int>& out_c2e, array<int>& out_f2c, array<int>& out_f2loc_f, array<int>& out_f2v, array<int>& out_f2nv, array<int>& out_rot_tag, array<int>& out_unmatched_faces, int& out_n_unmatched_faces, array<int>& out_icvsta, array<int>& out_icvert, int& out_n_faces, int& out_n_edges, struct solution* FlowSol)
{
  
	// inputs: 	in_c2v (clls to vertex) , in_ctype (type of cell)
	// outputs:	f2c (face to cell), c2f (cell to face), f2loc_f (face to local face index of right and left cells), rot_tag,  n_faces (number of faces in the mesh)
	// assumes that array f2c,f2f
  
	int sta_ind,end_ind;
	int n_cells,n_verts;
	int num_v_per_f,num_v_per_f2;
	int iface, iface_old;
	int found,rtag;
	int pause;
  
	n_cells = in_c2v.get_dim(0);
	n_verts = in_c2v.get_max()+1;
  
	//cout << "n_verts=" << n_verts << endl;
	
	//array<int> num_v_per_c(5); // for 5 element types
	array<int> vlist_loc(MAX_V_PER_F),vlist_loc2(MAX_V_PER_F),vlist_glob(MAX_V_PER_F),vlist_glob2(MAX_V_PER_F); // faces cannot have more than 4 vertices
  
	array<int> ifill,idummy;
  
	// Number of vertices for different type of cells
	//num_v_per_c(0) = 3; // tri
	//num_v_per_c(1) = 4; // quads
	//num_v_per_c(2) = 4; // tets
	//num_v_per_c(3) = 6; // pris
	//num_v_per_c(4) = 8; // hexas
  
	ifill.setup(n_verts);
	idummy.setup(n_verts);
	// Assumes there won't be more than X cells around 1 vertex
	int max_cells_per_vert = 50;
	out_icvert.setup(max_cells_per_vert*n_verts+1);
	out_icvsta.setup(n_verts+1);
  
	// Initialize arrays to zero
	for (int i=0;i<n_verts;i++) {
		ifill(i) = 0;
		idummy(i) = 0;
		out_icvsta(i) = 0;
	}
  
	out_icvsta(n_verts) = 0;
	for (int i=0;i<max_cells_per_vert*n_verts+1;i++)
		out_icvert(i) = 0;
  
	for(int i=0;i<MAX_V_PER_F;i++)
		vlist_loc(i) = vlist_loc2(i) = vlist_glob(i) = vlist_glob2(i) = 0;
  
	// Compute how many cells share one node
	for (int ic=0;ic<n_cells;ic++)
	  //for(int k=0;k<num_v_per_c(in_ctype(ic));k++)
	  for(int k=0;k<in_c2n_v(ic);k++)
 	    ifill(in_c2v(ic,k))++;
  
	int k=0;
	int max = 0;
	for(int iv=0;iv<n_verts;iv++)
	{
    if (ifill(iv)>max)
      max = ifill(iv);
		
    out_icvsta(iv) = k;
    idummy(iv) = out_icvsta(iv);
    k = k+ifill(iv);
	}
  
	//cout << "Maximum number of cells who share same vertex = " << max << endl;
	if (max>max_cells_per_vert)
		FatalError("ERROR: some vertices are shared by more than max_cells_per_vert");
  
	out_icvsta(n_verts) = out_icvsta(n_verts-1)+ifill(n_verts-1);
  
  int iv,ic2,k2;
	for(int ic=0;ic<n_cells;ic++)
	{
    //for(int k=0;k<num_v_per_c(in_ctype(ic));k++)
    for(int k=0;k<in_c2n_v(ic);k++)
    {
      iv = in_c2v(ic,k);
      out_icvert(idummy(iv)) = ic;
      idummy(iv)++;
    }
	}
  
  out_n_edges=-1;
  if (FlowSol->n_dims==3)
  {
    // Create array ic2e
    array<int> num_e_per_c(5);
    for (int i=0;i<n_cells;i++)
      for (int j=0;j<MAX_E_PER_C;j++)
        out_c2e(i,j) = -1;
    
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
        
        get_vlist_loc_edge(in_ctype(ic),in_c2n_v(ic),k,vlist_loc);
        for (int i=0;i<2;i++)
        {
          vlist_glob(i) = in_c2v(ic,vlist_loc(i));
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
			      // Get local vertices of local face k2 of cell ic2
			      get_vlist_loc_edge(in_ctype(ic2),in_c2n_v(ic),k2,vlist_loc2);
            
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
    out_n_edges++;
  } // if n_dims=3
  
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
      
      //cout << "ctype=" << in_ctype(ic) << "n_v=" << in_c2n_v(ic) << endl;
      // Get local vertices of local face k of cell ic
      //cout << "ic=" << ic << endl;
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
        
        //cout << "ic2=" << ic2 << endl;
        //cout << "ctype=" << in_ctype(ic2) << endl;
        // Loop over faces of cell ic2 touching vertex vlist_glob(0)
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
            // For 3D SHOULD RETURN THE ORIENTATION OF FACE2 WRT FACE
            compare_faces(vlist_glob,vlist_glob2,num_v_per_f,found,rtag);
            
            if (found==1) break;
          }
        }
        
        if (found==1) break;
      }
      
      if(found==1)
      {
        out_c2f(ic2,k2) = iface;
        
        out_f2c(iface,0) = ic;
        out_f2c(iface,1) = ic2;
        
        out_f2loc_f(iface,0) = k;
        out_f2loc_f(iface,1) = k2;
        
        out_c2f(ic,k) = iface;
        
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
      
    }	// end of loop over k
  } // end of loop over ic
  
  out_n_faces = iface;
  //cout << "n_faces = " << out_n_faces << endl;
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
		//cout << "CHECK ORIENTATION FOR TETS" << endl;
		num_v_per_f = 3;
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
	else if (in_ctype==3) // Prism
	{
		//cout << "CHECK ORIENTATION FOR TETS" << endl;
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
		cout << "ERROR: Haven't implemented other 3D Elements yet...." << endl;
	}
  
}

double compute_distance(array<double>& pos_0, array<double>& pos_1, int in_dims, struct solution* FlowSol)
{
  double dist = 0.;
  for (int m=0;m<FlowSol->n_dims;m++)
  {
    dist += (pos_0(m)-pos_1(m))*(pos_0(m)-pos_1(m));
  }
  dist = sqrt(dist);
  return dist;
}

void compare_faces(array<int>& vlist1, array<int>& vlist2, int& num_v_per_f, int& found, int& rtag)
{
  
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
		cout << "ERROR: Haven't implemented this face type in compare_face yet...." << endl;
		exit(1);
	}
  
}

void compare_faces_boundary(array<int>& vlist1, array<int>& vlist2, int& num_v_per_f, int& found)
{
  
  if ( !(num_v_per_f==2 || num_v_per_f==3 || num_v_per_f==4))
    FatalError("Face not recognized");
  
  int count = 0;
  for (int j=0;j<num_v_per_f;j++)
  {
    for (int k=0;k<num_v_per_f;k++)
    {
      if (vlist1(j)==vlist2(k))
      {
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
	  	cout << "ERROR: Haven't implemented this face type in compare_cyclic_face yet...." << endl;
	  	exit(1);
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

void exchange_plotq(struct solution* FlowSol)
{
  int counter=0;
  for (int i=0;i<FlowSol->n_mpi_pnodes;i++)
  {
    for (int j=0;j<run_input.n_plot_quantities;j++)
    {
      FlowSol->out_buffer_plotq(counter++) = FlowSol->plotq_pnodes(FlowSol->mpi_pnode2pnode(i),j);
    }
  }
  
  // Count the number of messages to send
  int request_count = 0;
  for (int p=0;p<FlowSol->nproc;p++) {
    if (FlowSol->mpi_pnodes_part(p)!=0) request_count++;
  }
  
  MPI_Request* mpi_in_requests = (MPI_Request*) malloc(request_count*sizeof(MPI_Request));
  MPI_Request* mpi_out_requests = (MPI_Request*) malloc(request_count*sizeof(MPI_Request));
  
	// Initiate mpi_send
	int Nmess = 0;
	int sk = 0;
  int Nout;
  request_count=0;
	for (int p=0;p<FlowSol->nproc;p++) {
		Nout = FlowSol->mpi_pnodes_part(p)*run_input.n_plot_quantities;
		if (Nout) {
			MPI_Isend(FlowSol->out_buffer_plotq.get_ptr_cpu(sk),Nout,MPI_DOUBLE,p,p   ,MPI_COMM_WORLD,&mpi_out_requests[request_count]);
			MPI_Irecv(FlowSol->in_buffer_plotq.get_ptr_cpu(sk),Nout,MPI_DOUBLE,p,FlowSol->rank,MPI_COMM_WORLD,&mpi_in_requests[request_count]);
			sk+=Nout;
			Nmess++;
      request_count++;
		}
	}
  
	MPI_Waitall(Nmess,mpi_in_requests,MPI_STATUSES_IGNORE);
	MPI_Waitall(Nmess,mpi_out_requests,MPI_STATUSES_IGNORE);
  
  int count=0;
  for (int i=0;i<FlowSol->n_mpi_pnodes;i++)
  {
    for (int j=0;j<run_input.n_plot_quantities;j++)
    {
      FlowSol->plotq_pnodes(FlowSol->mpi_pnode2pnode(i),j) += FlowSol->in_buffer_plotq(count++);
    }
  }
  
}

void update_factor_pnodes(struct solution* FlowSol)
{
  int counter=0;
  for (int i=0;i<FlowSol->n_mpi_pnodes;i++)
  {
    FlowSol->out_buffer_pnode(counter++) = FlowSol->factor_pnode(FlowSol->mpi_pnode2pnode(i));
  }
  
  // Count the number of messages to send
  int request_count = 0;
  for (int p=0;p<FlowSol->nproc;p++) {
    if (FlowSol->mpi_pnodes_part(p)!=0) request_count++;
  }
  
  MPI_Request* mpi_in_requests = (MPI_Request*) malloc(request_count*sizeof(MPI_Request));
  MPI_Request* mpi_out_requests = (MPI_Request*) malloc(request_count*sizeof(MPI_Request));
  
	// Initiate mpi_send
	int Nmess = 0;
	int sk = 0;
  int Nout;
  request_count=0;
	for (int p=0;p<FlowSol->nproc;p++) {
		Nout = FlowSol->mpi_pnodes_part(p);;
		if (Nout) {
			MPI_Isend(FlowSol->out_buffer_pnode.get_ptr_cpu(sk),Nout,MPI_INT,p,p   ,MPI_COMM_WORLD,&mpi_out_requests[request_count]);
			MPI_Irecv(FlowSol->in_buffer_pnode.get_ptr_cpu(sk),Nout,MPI_INT,p,FlowSol->rank,MPI_COMM_WORLD,&mpi_in_requests[request_count]);
			sk+=Nout;
			Nmess++;
      request_count++;
		}
	}
  
	MPI_Waitall(Nmess,mpi_in_requests,MPI_STATUSES_IGNORE);
	MPI_Waitall(Nmess,mpi_out_requests,MPI_STATUSES_IGNORE);
  
  for (int i=0;i<FlowSol->n_mpi_pnodes;i++)
  {
    FlowSol->factor_pnode(FlowSol->mpi_pnode2pnode(i)) += FlowSol->in_buffer_pnode(i);
  }
  
}

void match_mpipnodes(array<int>& in_out_mpi_pnode2pnode, int& in_out_n_mpi_pnodes, array<int> &out_mpi_pnodes_part, array<double>& delta_cyclic, double tol, struct solution* FlowSol)
{
  
  // Copying inputs
  array<int> old_mpi_pnode2pnode;
  old_mpi_pnode2pnode = in_out_mpi_pnode2pnode;
  int old_n_mpi_pnodes = in_out_n_mpi_pnodes;
  
  // Allocate storage for twice as many nodes
  in_out_mpi_pnode2pnode.setup(2*FlowSol->n_mpi_pnodes);
  
	MPI_Status instatus;
  array<double> delta_zero(FlowSol->n_dims);
  for (int m=0;m<FlowSol->n_dims;m++)
    delta_zero(m) = 0.;
  
  for(int i=0;i<FlowSol->nproc;i++)
    out_mpi_pnodes_part(i)=0;
  
  MPI_Barrier(MPI_COMM_WORLD);
  // Exchange number of mpi_pnodes to receive
  array<int> mpi_pnodes_from(FlowSol->nproc);
  MPI_Allgather( &old_n_mpi_pnodes,1,MPI_INT,mpi_pnodes_from.get_ptr_cpu(),1,MPI_INT,MPI_COMM_WORLD);
  
  int max_mpi_pnodes=0;
  for (int i=0;i<FlowSol->nproc;i++)
    if (mpi_pnodes_from(i) >= max_mpi_pnodes) max_mpi_pnodes=mpi_pnodes_from(i);
  
  array<double> in_loc_mpi_pnodes(FlowSol->n_dims,max_mpi_pnodes);
  array<double> loc_mpi_pnodes(FlowSol->n_dims,max_mpi_pnodes);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  for (int i=0;i<old_n_mpi_pnodes;i++) {
    for (int j=0;j<FlowSol->n_dims;j++)
    {
      int node = old_mpi_pnode2pnode(i);
      loc_mpi_pnodes(j,i) = FlowSol->pos_pnode(old_mpi_pnode2pnode(i))(j);
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  array<double> loc_1(FlowSol->n_dims);
  array<double> loc_2(FlowSol->n_dims);
  array<int> matched(old_n_mpi_pnodes);
  for (int i=0;i<old_n_mpi_pnodes;i++)
    matched(i) = 0;
  
  int i;
  
	// Begin the exchange
	int icount = 0;
  int p,p2;
	for(p=0;p<FlowSol->nproc;p++) {
		if(p==FlowSol->rank) { // Send data
			for (p2=0;p2<FlowSol->nproc;p2++) {
				if (p2!=FlowSol->rank)  {
          //MPI_Send to processor p2
					MPI_Send(loc_mpi_pnodes.get_ptr_cpu(),FlowSol->n_dims*old_n_mpi_pnodes,MPI_DOUBLE,p2,100000*p+1000*p2,MPI_COMM_WORLD);
        }
			}
			out_mpi_pnodes_part(p) = 0; // Processor p won't send any edges to himself
		}
		else // Receive data
		{
			//MPI_Recv from processor p
			MPI_Recv(in_loc_mpi_pnodes.get_ptr_cpu(),FlowSol->n_dims*mpi_pnodes_from(p),MPI_DOUBLE,p,100000*p+1000*FlowSol->rank,MPI_COMM_WORLD,&instatus);
      
			if (p<FlowSol->rank)
			{
			  // Loop over local mpi_edges
			  for (int iloc=0;iloc<old_n_mpi_pnodes;iloc++) {
          // Loop over remote edges just received
          for(int irem=0;irem<mpi_pnodes_from(p);irem++)
          {
            for (int m=0;m<FlowSol->n_dims;m++) {
              loc_1(m) = in_loc_mpi_pnodes(m,irem);
              loc_2(m) = loc_mpi_pnodes(m,iloc);
            }
            if (check_cyclic(delta_cyclic,loc_1,loc_2,tol,FlowSol) ||
                check_cyclic(delta_zero  ,loc_1,loc_2,tol,FlowSol) )
            {
              out_mpi_pnodes_part(p)++;
              i = old_mpi_pnode2pnode(iloc);
              in_out_mpi_pnode2pnode(icount) = i;
              
              
              matched(iloc)=1;
              icount++;
              //break;
            }
          }
			  }
			}
			else // if p >= FlowSol->rank
			{
			  // Loop over remote edges
			  for (int irem=0;irem<mpi_pnodes_from(p);irem++)
			  {
			  	for (int iloc=0;iloc<old_n_mpi_pnodes;iloc++)
			  	{
            for (int m=0;m<FlowSol->n_dims;m++) {
              loc_1(m) = in_loc_mpi_pnodes(m,irem);
              loc_2(m) = loc_mpi_pnodes(m,iloc);
            }
            
            // Check if it matches vertex iloc
            if (check_cyclic(delta_cyclic,loc_1,loc_2,tol,FlowSol) ||
                check_cyclic(delta_zero  ,loc_1,loc_2,tol,FlowSol))
            {
              out_mpi_pnodes_part(p)++;
              i = old_mpi_pnode2pnode(iloc);
              in_out_mpi_pnode2pnode(icount) = i;
              
              
              matched(iloc)=1;
              icount++;
              //break;
            }
			  	}
			  }
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
  
  for (int i=0;i<old_n_mpi_pnodes;i++)
  {
    if (matched(i)!=1)
    {
      FatalError("Problem in match_mpipnodes, one plotnode has not been matched");
    }
  }
  
  in_out_n_mpi_pnodes=icount;
  
}




void create_mpi_pnode2pnode(array<int>& in_out_mpi_pnode2pnode, int& out_n_mpi_pnodes, array<int>& in_f_mpi2f, array<int>& in_f2loc_f, array<int>& in_f2c, array<int>& in_ctype, int& n_mpi_inters, struct solution* FlowSol)
{
  int max_pnodes_per_inter = (FlowSol->p_res)*(FlowSol->p_res);
  int max_mpi_pnodes = n_mpi_inters*max_pnodes_per_inter;
  
  array<int> mpi_pnode_list(max_mpi_pnodes);
  array<int> inter_pnodes(max_pnodes_per_inter);
  
  int n_inter_pnodes;
  out_n_mpi_pnodes=0;
  
  //cout << "n_mpi_inters=" << FlowSol->n_mpi_inters << endl;
  for (int i=0;i<n_mpi_inters;i++)
  {
    int f = in_f_mpi2f(i);
    int loc_f = in_f2loc_f(f,0);
    int ic_l = in_f2c(f,0);
    int ctype = in_ctype(ic_l);
    
    // Get the list of pnodes and the number of pnodes for that face
    FlowSol->mesh_eles(ctype)->get_face_pnode_list(inter_pnodes,FlowSol->c2ctype_c(ic_l),loc_f,n_inter_pnodes);
    
    // Check if they have already been counted
    for (int j=0;j<n_inter_pnodes;j++)
    {
      
      int found =0;
      for (int k=0;k<out_n_mpi_pnodes;k++)
      {
        if (mpi_pnode_list(k)==inter_pnodes(j))
        {
          found=1;
          break;
        }
      }
      if (found==0) {
        mpi_pnode_list(out_n_mpi_pnodes)=inter_pnodes(j);
        out_n_mpi_pnodes++;
      }
      
      
    }
  }
  
  in_out_mpi_pnode2pnode.setup(out_n_mpi_pnodes);
  for (int i=0;i<out_n_mpi_pnodes;i++)
  {
    in_out_mpi_pnode2pnode(i) = mpi_pnode_list(i);
  }
  
}


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
	  	cout << "ERROR: Haven't implemented this face type in compare_cyclic_face yet...." << endl;
	  	exit(1);
	  }
  }
  
  if (found==0)
    FatalError("Could not match vertices in compare faces");
  
}

#endif

