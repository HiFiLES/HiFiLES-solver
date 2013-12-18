/*!
 * \file output.cpp
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

// Used for making sub-directories
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/input.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/funcs.h"
#include "../include/error.h"
#include "../include/solution.h"

// Use VTK routines to write out to paraview
//#ifdef _VTK
//#include "../include/vtk.h"
//#endif

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

//method to write out a continuous solution field
void plot_continuous(struct solution* FlowSol)
{
  // Compute and store position of plot points
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  FlowSol->mesh_eles(i)->set_pos_ppts();
      
    }
  }
  
  //CGL050412: need to do plotter_setup before connectivity
  plotter_setup(FlowSol);
  
  // Compute the plotting connectivity
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
      FlowSol->mesh_eles(i)->set_connectivity_plot();
      
    }
  }
  
  //CGL050412  plotter_setup();
  
  // First compute the fields at the plot points
  // Loop over all the eles and add their contribution to each plot node
  
  FlowSol->plotq_pnodes.setup(FlowSol->num_pnodes,run_input.n_plot_quantities);
  
  for (int i=0;i<FlowSol->num_pnodes;i++)
    for (int j=0;j<run_input.n_plot_quantities;j++)
      FlowSol->plotq_pnodes(i,j) = 0.;
  
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  FlowSol->mesh_eles(i)->add_contribution_to_pnodes(FlowSol->plotq_pnodes);
      
    }
  }
  
#ifdef _MPI
  
  // Create a list of pnodes on mpi faces
  create_mpi_pnode2pnode(FlowSol->mpi_pnode2pnode,FlowSol->n_mpi_pnodes,FlowSol->inter_mpi2inter,FlowSol->inter2loc_inter,FlowSol->inter2ele,FlowSol->ele_type,FlowSol->n_mpi_inters,FlowSol);
  
  // Match the mpi pnodes
  FlowSol->mpi_pnodes_part.setup(FlowSol->nproc);
  array<double> delta_cyclic(FlowSol->n_dims);
  delta_cyclic(0) = run_input.dx_cyclic;
  delta_cyclic(1) = run_input.dy_cyclic;
  if (FlowSol->n_dims==3) {
    delta_cyclic(2) = run_input.dz_cyclic;
  }
  
  cout << "n_mpi_inters=" << FlowSol->n_mpi_inters << endl;
  
  double tol = 1e-8;
  match_mpipnodes(FlowSol->mpi_pnode2pnode,FlowSol->n_mpi_pnodes,FlowSol->mpi_pnodes_part,delta_cyclic,tol,FlowSol);
  
  // Update the factor of pnodes
  FlowSol->out_buffer_pnode.setup(FlowSol->n_mpi_pnodes);
  FlowSol->in_buffer_pnode.setup(FlowSol->n_mpi_pnodes);
  update_factor_pnodes(FlowSol);
  
  // Exchange the plotq data
  FlowSol->out_buffer_plotq.setup(FlowSol->n_mpi_pnodes*run_input.n_plot_quantities);
  FlowSol->in_buffer_plotq.setup(FlowSol->n_mpi_pnodes*run_input.n_plot_quantities);
  exchange_plotq(FlowSol);
  
#endif
  
  for(int i=0;i<FlowSol->num_pnodes;i++)
  {
    for (int j=0;j<run_input.n_plot_quantities;j++)
    {
      FlowSol->plotq_pnodes(i,j) /= (1.0*FlowSol->factor_pnode(i));
    }
  }
  
  //if(FlowSol->write_type==0)
    //write_vtu_bin(FlowSol->ini_iter, FlowSol);
  //else if(FlowSol->write_type==1)
    //write_tec_bin(FlowSol->ini_iter, FlowSol);
  
}

void plotter_setup(struct solution* FlowSol)
{
  FlowSol->p_res = run_input.p_res;
  int gnid;
  FlowSol->num_pnodes=0;
  int max_num_pnodes;
  array<double> pos(FlowSol->n_dims);
  array<double> pos2(FlowSol->n_dims);
  
  double tol = 1e-10;
  
  // First count the maximum number of ppts
  max_num_pnodes=0;
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  max_num_pnodes += FlowSol->mesh_eles(i)->calc_num_ppts();
      
    }
  }
  
  FlowSol->pos_pnode.setup(max_num_pnodes);
  FlowSol->factor_pnode.setup(max_num_pnodes);
  
  for (int i=0;i<max_num_pnodes;i++)
  {
    FlowSol->pos_pnode(i).setup(FlowSol->n_dims);
    FlowSol->factor_pnode(i)=0;
  }
  
  int tris_count=0;
  int quads_count=0;
  int tets_count=0;
  int pris_count=0;
  int hexas_count=0;
  
  FlowSol->c2ctype_c.setup(FlowSol->num_eles);
  for (int i=0;i<FlowSol->num_eles;i++)
  {
    if (FlowSol->ele_type(i)==0)
      FlowSol->c2ctype_c(i) = tris_count++;
    if (FlowSol->ele_type(i)==1)
      FlowSol->c2ctype_c(i) = quads_count++;
    if (FlowSol->ele_type(i)==2)
      FlowSol->c2ctype_c(i) = tets_count++;
    if (FlowSol->ele_type(i)==3)
      FlowSol->c2ctype_c(i) = pris_count++;
    if (FlowSol->ele_type(i)==4)
      FlowSol->c2ctype_c(i) = hexas_count++;
  }
  
  // ---------------------------
  // Loop over mesh vertices
  // ---------------------------
  int vert;
  array<int> vert2pnode(FlowSol->num_verts);
  
  for (int i=0;i<FlowSol->num_verts;i++)
    vert2pnode(i) = -1;
  
  array<int> n_vert_per_type(5);
  n_vert_per_type(0)=3;
  n_vert_per_type(1)=4;
  n_vert_per_type(2)=4;
  n_vert_per_type(3)=6;
  n_vert_per_type(4)=8;
  
  int j_spt;
  for (int i=0;i<FlowSol->num_eles;i++)
  {
    int temp_ele_type=FlowSol->ele_type(i);
    for (int j=0;j<n_vert_per_type(temp_ele_type);j++)
    {
      get_vert_loc(temp_ele_type,FlowSol->ele2n_vert(i),j,j_spt);
      vert = FlowSol->ele2vert(i,j_spt);
      if (vert2pnode(vert)==-1) // Haven't counted that vertex
      {
        vert2pnode(vert)=FlowSol->num_pnodes;
        
        // TODO
        pos = FlowSol->mesh_eles(temp_ele_type)->calc_pos_pnode_vert(FlowSol->c2ctype_c(i),j);
        FlowSol->pos_pnode(FlowSol->num_pnodes) = pos;
        FlowSol->num_pnodes++;
      }
      
      gnid = vert2pnode(vert);
      FlowSol->factor_pnode(gnid)++;
      // TODO
      FlowSol->mesh_eles(temp_ele_type)->set_pnode_vert(FlowSol->c2ctype_c(i),j,gnid);
    }
  }
  
  // ---------------------------
  // Then loop mesh edges
  // ---------------------------
  
  if (FlowSol->n_dims==3)
  {
    int edge;
    
    array<int> is_edge_node(FlowSol->num_edges);
    array<int> gedgenode(FlowSol->p_res-2,FlowSol->num_edges);
    
    for (int i=0;i<FlowSol->num_edges;i++)
    {
      is_edge_node(i)=0;
    }
    
    array<int> n_edge_per_type(5);
    n_edge_per_type(2)=6;
    n_edge_per_type(3)=9;
    n_edge_per_type(4)=12;
    
    for (int i=0;i<FlowSol->num_eles;i++)
    {
      int temp_ele_type=FlowSol->ele_type(i);
      for (int j=0;j<n_edge_per_type(temp_ele_type);j++)
      {
        edge = FlowSol->ele2edge(i,j);
        for (int k=0;k<FlowSol->p_res-2;k++)
        {
          // TODO
          pos = FlowSol->mesh_eles(temp_ele_type)->calc_pos_pnode_edge(FlowSol->c2ctype_c(i),j,k);
          
          if (is_edge_node(edge)==1)
          {
            int check_match=0;
            // Find the corresponding pnode
            for (int l=0;l<FlowSol->p_res-2;l++)
            {
              gnid = gedgenode(l,edge);
              pos2 = FlowSol->pos_pnode(gnid);
              if (compute_distance(pos,pos2,FlowSol->n_dims,FlowSol) < tol)
              {
                FlowSol->mesh_eles(temp_ele_type)->set_pnode_edge(FlowSol->c2ctype_c(i),j,k,gnid);
                FlowSol->factor_pnode(gnid)++;
                check_match=1;
                break;
              }
            }
            if (check_match==0)
            {
              cout << "edge=" << edge << endl;
              FatalError("Something wrong when nodalizing edges");
            }
          }
          else
          {
            FlowSol->pos_pnode(FlowSol->num_pnodes) = pos;
            FlowSol->mesh_eles(temp_ele_type)->set_pnode_edge(FlowSol->c2ctype_c(i),j,k,FlowSol->num_pnodes);
            gedgenode(k,edge) = FlowSol->num_pnodes;
            FlowSol->factor_pnode(FlowSol->num_pnodes)++;
            FlowSol->num_pnodes++;
          }
          
        }
        if (is_edge_node(edge)==0)
          is_edge_node(edge)=1;
      } // loop over edge
    } // loop over cells
  } // end if n_dims==3
  
  // ----------------------
  // Loop over inters
  // ----------------------
  //
  int face;
  
  array<int> is_face_node(FlowSol->num_inters);
  
  // Get maximum number of ppts on face
  
  int max_n_ppts_per_face = 0;
  for (int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
      int  temp = FlowSol->mesh_eles(i)->get_max_n_ppts_per_face();
      if (temp > max_n_ppts_per_face)
        max_n_ppts_per_face = temp;
      
    }
  }
  
  array<int> gfacenode(max_n_ppts_per_face,FlowSol->num_inters);
  
  for (int i=0;i<FlowSol->num_inters;i++)
  {
    is_face_node(i)=0;
  }
  
  array<int> n_face_per_type(5);
  n_face_per_type(0)=3;
  n_face_per_type(1)=4;
  n_face_per_type(2)=4;
  n_face_per_type(3)=5;
  n_face_per_type(4)=6;
  
  for (int i=0;i<FlowSol->num_eles;i++)
  {
    int temp_ele_type=FlowSol->ele_type(i);
    for (int j=0;j<n_face_per_type(temp_ele_type);j++)
    {
      face = FlowSol->ele2face(i,j);
      int n_ppts_per_face = FlowSol->mesh_eles(temp_ele_type)->get_n_ppts_per_face(j);
      for (int k=0;k<n_ppts_per_face;k++)
      {
        // TODO
        pos = FlowSol->mesh_eles(temp_ele_type)->calc_pos_pnode_face(FlowSol->c2ctype_c(i),j,k);
        if (is_face_node(face)==1)
        {
          int check_match=0;
          for (int l=0;l<n_ppts_per_face;l++)
          {
            gnid = gfacenode(l,face);
            pos2 = FlowSol->pos_pnode(gnid);
            if (compute_distance(pos,pos2,FlowSol->n_dims,FlowSol) < tol)
            {
              FlowSol->mesh_eles(temp_ele_type)->set_pnode_face(FlowSol->c2ctype_c(i),j,k,gnid);
              FlowSol->factor_pnode(gnid)++;
              check_match=1;
              break;
            }
          }
          if (check_match==0)
          {
            FatalError("Something wrong when nodelizing faces ");
          }
        }
        else
        {
          FlowSol->pos_pnode(FlowSol->num_pnodes) = pos;
          FlowSol->mesh_eles(temp_ele_type)->set_pnode_face(FlowSol->c2ctype_c(i),j,k,FlowSol->num_pnodes);
          gfacenode(k,face) = FlowSol->num_pnodes;
          FlowSol->factor_pnode(FlowSol->num_pnodes)++;
          FlowSol->num_pnodes++;
        }
      }
      if (is_face_node(face)==0)
        is_face_node(face)=1;
      
    } // Loop over faces
  } // Loop over cells
  
  // ----------------------
  // Interior plot points
  // ---------------------
  
  for (int i=0;i<FlowSol->num_eles;i++)
  {
    int temp_ele_type=FlowSol->ele_type(i);
    int n_interior_ppts;
    
    n_interior_ppts = FlowSol->mesh_eles(temp_ele_type)->get_n_interior_ppts();
    
    for (int j=0;j<n_interior_ppts; j++)
    {
      FlowSol->mesh_eles(temp_ele_type)->set_pnode_interior(FlowSol->c2ctype_c(i),j,FlowSol->num_pnodes);
      pos = FlowSol->mesh_eles(temp_ele_type)->calc_pos_pnode_interior(FlowSol->c2ctype_c(i),j);
      FlowSol->pos_pnode(FlowSol->num_pnodes) = pos;
      FlowSol->factor_pnode(FlowSol->num_pnodes)++;
      FlowSol->num_pnodes++;
    }
  }
  
}

// method to write out a tecplot file
void write_tec(int in_file_num, struct solution* FlowSol) // TODO: Tidy this up
{
	int i,j,k,l,m;
  
	// copy solution to cpu
#ifdef _GPU
	for(i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
      
    }
  }
#endif
	
	int vertex_0, vertex_1, vertex_2, vertex_3, vertex_4, vertex_5, vertex_6, vertex_7;
	
	int p_res=run_input.p_res; // HACK
	
	array<double> pos_ppts_temp;
	array<double> plotq_ppts_temp;
  int n_plot_data;
  
  char  file_name_s[50] ;
  char *file_name;
  ofstream write_tec;
  write_tec.precision(15);
  
#ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
	sprintf(file_name_s,"Mesh_%.09d_p%.04d.plt",in_file_num,FlowSol->rank);
	if (FlowSol->rank==0) cout << "Writing Tecplot file number " << in_file_num << " ...." << endl;
#else
	sprintf(file_name_s,"Mesh_%.09d_p%.04d.plt",in_file_num,0);
	cout << "Writing Tecplot file number " << in_file_num << " on rank " << FlowSol->rank << endl;
#endif
  
  file_name = &file_name_s[0];
  write_tec.open(file_name);
  
	// write header
	write_tec << "Title = \"SD++ Solution\"" << endl;
	
	// write variable names
  if (run_input.equation==0)
  {
	  if(FlowSol->n_dims==2)
	  {
	  	write_tec << "Variables = \"x\", \"y\", \"rho\", \"mom_x\", \"mom_y\", \"ene\"" << endl;
	  }
	  else if(FlowSol->n_dims==3)
	  {
	  	write_tec << "Variables = \"x\", \"y\", \"z\", \"rho\", \"mom_x\", \"mom_y\", \"mom_z\", \"ene\"" << endl;
	  }
	  else
	  {
	  	cout << "ERROR: Invalid number of dimensions ... " << endl;
	  }
  }
  else if (run_input.equation==1)
  {
	  if(FlowSol->n_dims==2)
	  {
		  write_tec << "Variables = \"x\", \"y\", \"rho\"" << endl;
	  }
	  else if(FlowSol->n_dims==3)
	  {
	  	write_tec << "Variables = \"x\", \"y\", \"z\", \"rho\"" << endl;
	  }
	  else
	  {
	  	cout << "ERROR: Invalid number of dimensions ... " << endl;
	  }
  }
  
	int time_iter = 0;
  
	for(i=0;i<FlowSol->n_ele_types;i++)
	{
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
      n_plot_data = FlowSol->mesh_eles(i)->get_n_fields();
      
		  pos_ppts_temp.setup(FlowSol->mesh_eles(i)->get_n_ppts_per_ele(),FlowSol->mesh_eles(i)->get_n_dims());
		  plotq_ppts_temp.setup(FlowSol->mesh_eles(i)->get_n_ppts_per_ele(),n_plot_data);
      
      int num_pts= (FlowSol->mesh_eles(i)->get_n_eles())*(FlowSol->mesh_eles(i)->get_n_ppts_per_ele());
      int num_elements = (FlowSol->mesh_eles(i)->get_n_eles())*(FlowSol->mesh_eles(i)->get_n_peles_per_ele());
      
		  // write element specific header
		  if(FlowSol->mesh_eles(i)->get_ele_type()==0) // tri
		  {
		  	write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FETRIANGLE" << endl;
		  }
		  else if(FlowSol->mesh_eles(i)->get_ele_type()==1) // quad
		  {
		  	write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FEQUADRILATERAL" << endl;
		  }
      else if (FlowSol->mesh_eles(i)->get_ele_type()==2) // tet
      {
		  	write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FETETRAHEDRON" << endl;
      }
      else if (FlowSol->mesh_eles(i)->get_ele_type()==3) // prisms
      {
		  	write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FEBRICK" << endl;
      }
		  else if(FlowSol->mesh_eles(i)->get_ele_type()==4) // hexa
		  {
		  	write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FEBRICK" << endl;
		  }
		  else
		  {
        FatalError("Invalid element type");
		  }
		  
			if(time_iter == 0)
			{
				write_tec <<"SolutionTime=" << FlowSol->time << endl;
				time_iter = 1;
			}
      
		  // write element specific data
		  
		  for(j=0;j<FlowSol->mesh_eles(i)->get_n_eles();j++)
		  {
		  	FlowSol->mesh_eles(i)->calc_pos_ppts(j,pos_ppts_temp);
		  	FlowSol->mesh_eles(i)->calc_disu_ppts(j,plotq_ppts_temp);
		  	
		  	for(k=0;k<FlowSol->mesh_eles(i)->get_n_ppts_per_ele();k++)
		  	{
		  		for(l=0;l<FlowSol->mesh_eles(i)->get_n_dims();l++)
		  		{
		  			write_tec << pos_ppts_temp(k,l) << " ";
		  		}
		  		
		  		for(l=0;l<n_plot_data;l++)
		  		{
            if ( isnan(plotq_ppts_temp(k,l))) {
              FatalError("Nan in tecplot file, exiting");
            }
            else {
              write_tec << plotq_ppts_temp(k,l) << " ";
            }
		  		}
		  		
		  		write_tec << endl;
		  	}
		  }
		  
		  // write element specific connectivity
		  
		  if(FlowSol->mesh_eles(i)->get_ele_type()==0) // tri
		  {
		  	for (j=0;j<FlowSol->mesh_eles(i)->get_n_eles();j++)
		  	{
		      for(k=0;k<p_res-1;k++) // look to right from each point
		      {
		      	for(l=0;l<p_res-k-1;l++)
		      	{
		      		vertex_0=l+(k*(p_res+1))-((k*(k+1))/2);
		      		vertex_1=vertex_0+1;
		      		vertex_2=l+((k+1)*(p_res+1))-(((k+1)*(k+2))/2);
		      		vertex_0+=j*(p_res*(p_res+1)/2);
		      		vertex_1+=j*(p_res*(p_res+1)/2);
		      		vertex_2+=j*(p_res*(p_res+1)/2);
              
		      		write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1  << endl;
		      	}
		      }
          
		      for(k=0;k<p_res-2;k++) //  look to left from each point
		      {
		      	for(l=1;l<p_res-k-1;l++)
		      	{
		      		vertex_0=l+(k*(p_res+1))-((k*(k+1))/2);
		      		vertex_1=l+((k+1)*(p_res+1))-(((k+1)*(k+2))/2);
		      		vertex_2=l-1+((k+1)*(p_res+1))-(((k+1)*(k+2))/2);
		      		
		      		vertex_0+=j*(p_res*(p_res+1)/2);
		      		vertex_1+=j*(p_res*(p_res+1)/2);
		      		vertex_2+=j*(p_res*(p_res+1)/2);
              
		      		write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1  << endl;
		      	}
		      }
        }
        
		  }
		  else if(FlowSol->mesh_eles(i)->get_ele_type()==1) // quad
		  {
		  	for (j=0;j<FlowSol->mesh_eles(i)->get_n_eles();j++)
		  	{
		  		for(k=0;k<p_res-1;k++)
		  		{
		  			for(l=0;l<p_res-1;l++)
		  			{
		  				vertex_0=l+(p_res*k);
		  				vertex_1=vertex_0+1;
		  				vertex_2=vertex_0+p_res+1;
		  				vertex_3=vertex_0+p_res;
              
		  				vertex_0 += j*p_res*p_res;
		  				vertex_1 += j*p_res*p_res;
		  				vertex_2 += j*p_res*p_res;
		  				vertex_3 += j*p_res*p_res;
              
		  				write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1 << " " <<  vertex_3+1  << endl;
		  			}
		  		}
		  	}
		  }
      else if (FlowSol->mesh_eles(i)->get_ele_type()==2) // tet
      {
	      int temp = (p_res)*(p_res+1)*(p_res+2)/6;
	      
	      for (int m=0;m<FlowSol->mesh_eles(i)->get_n_eles();m++)
	      {
          
	      	for(int k=0;k<p_res-1;k++)
	      	{
            for(int j=0;j<p_res-1-k;j++)
            {
              for(int i=0;i<p_res-1-k-j;i++)
              {
                
                vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i;
                
                vertex_1 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i + 1;
                
                vertex_2 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i;
                
                vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + j*(p_res-(k+1)) - (j-1)*j/2 + i;
                
                vertex_0+=m*temp;
                vertex_1+=m*temp;
                vertex_2+=m*temp;
                vertex_3+=m*temp;
                
                write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1  << " " << vertex_3+1 << endl;
                
              }
            }
	      	}
	      	
	      	for(int k=0;k<p_res-2;k++)
	      	{
            for(int j=0;j<p_res-2-k;j++)
            {
              for(int i=0;i<p_res-2-k-j;i++)
              {
                
                vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i + 1;
                
                vertex_1 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i + 1;
                vertex_2 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + j*(p_res-(k+1)) - (j-1)*j/2 + i + 1;
                vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + (i-1) + 1;
                vertex_4 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j)*(p_res-(k+1)) - (j-1)*(j)/2 + (i-1) + 1;
                vertex_5 = temp - (p_res-(k))*(p_res+1-(k))*(p_res+2-(k))/6 + (j+1)*(p_res-(k)) - (j)*(j+1)/2 + (i-1) + 1;
                
                vertex_0+=m*temp;
                vertex_1+=m*temp;
                vertex_2+=m*temp;
                vertex_3+=m*temp;
                vertex_4+=m*temp;
                vertex_5+=m*temp;
                
                write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1  << " " << vertex_5+1 << endl;
                write_tec << vertex_0+1  << " " <<  vertex_2+1  << " " <<  vertex_4+1  << " " << vertex_5+1 << endl;
                write_tec << vertex_2+1  << " " <<  vertex_3+1  << " " <<  vertex_4+1  << " " << vertex_5+1 << endl;
                write_tec << vertex_1+1  << " " <<  vertex_2+1  << " " <<  vertex_3+1  << " " << vertex_5+1 << endl;
              }
            }
	      	}
          
	      	for(int k=0;k<p_res-3;k++)
	      	{
            for(int j=0;j<p_res-3-k;j++)
            {
              for(int i=0;i<p_res-3-k-j;i++)
              {
                
                vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i + 1;
                vertex_1 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j)*(p_res-(k+1)) - (j-1)*(j)/2 + i + 1;
                vertex_2 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + i ;
                vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + i + 1;
                
                vertex_0+=m*temp;
                vertex_1+=m*temp;
                vertex_2+=m*temp;
                vertex_3+=m*temp;
                
                write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1  << " " << vertex_3+1 << endl;
              }
            }
	      	}
	      }
        
      }
      else if (FlowSol->mesh_eles(i)->get_ele_type()==3) // prisms
      {
	      int temp = (p_res)*(p_res+1)/2;
        
	      for (int m=0;m<FlowSol->mesh_eles(i)->get_n_eles();m++)
	      {
          
	  	    for (int l=0;l<p_res-1;l++)
	  	    {
	  	      for(int j=0;j<p_res-1;j++) // look to right from each point
	  	      {
	  	      	for(int k=0;k<p_res-j-1;k++)
	  	      	{
	  	      		vertex_0=k+(j*(p_res+1))-((j*(j+1))/2) + l*temp;
	  	      		vertex_1=vertex_0+1;
	  	      		vertex_2=k+((j+1)*(p_res+1))-(((j+1)*(j+2))/2) + l*temp;
                
	  	      		vertex_3 = vertex_0 + temp;
	  	      		vertex_4 = vertex_1 + temp;
	  	      		vertex_5 = vertex_2 + temp;
                
	  	      		vertex_0+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_1+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_2+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_3+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_4+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_5+=m*(p_res*(p_res+1)/2*p_res);
                
	  	      		write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1  << " " << vertex_2+1 << " " << vertex_3+1 << " " << vertex_4+1 << " " << vertex_5+1 << " " << vertex_5+1 << endl;
	  	      	}
	  	      }
	  	    }
          
	  	    for (int l=0;l<p_res-1;l++)
	  	    {
	  	      for(int j=0;j<p_res-2;j++) //  look to left from each point
	  	      {
	  	      	for(int k=1;k<p_res-j-1;k++)
	  	      	{
	  	      		vertex_0=k+(j*(p_res+1))-((j*(j+1))/2) + l*temp;
	  	      		vertex_1=k+((j+1)*(p_res+1))-(((j+1)*(j+2))/2) + l*temp;
	  	      		vertex_2=k-1+((j+1)*(p_res+1))-(((j+1)*(j+2))/2) + l*temp;
                
	  	      		vertex_3 = vertex_0 + temp;
	  	      		vertex_4 = vertex_1 + temp;
	  	      		vertex_5 = vertex_2 + temp;
                
	  	      		vertex_0+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_1+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_2+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_3+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_4+=m*(p_res*(p_res+1)/2*p_res);
	  	      		vertex_5+=m*(p_res*(p_res+1)/2*p_res);
                
	  	      		write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1  << " " << vertex_2+1 << " " << vertex_3+1 << " " << vertex_4+1 << " " << vertex_5+1 << " " << vertex_5+1 << endl;
	  	      	}
	  	      }
	  	    }
        }
      }
		  else if(FlowSol->mesh_eles(i)->get_ele_type()==4) // hexa
		  {
		  	for (int j=0;j<FlowSol->mesh_eles(i)->get_n_eles();j++)
		  	{
		  		for(int k=0;k<p_res-1;k++)
		  		{
		  			for(int l=0;l<p_res-1;l++)
		  			{
		  				for(int m=0;m<p_res-1;m++)
		  				{
		  					vertex_0=m+(p_res*l)+(p_res*p_res*k);
		  					vertex_1=vertex_0+1;
		  					vertex_2=vertex_0+p_res+1;
		  					vertex_3=vertex_0+p_res;
                
		  					vertex_4=vertex_0+p_res*p_res;
		  					vertex_5=vertex_4+1;
		  					vertex_6=vertex_4+p_res+1;
		  					vertex_7=vertex_4+p_res;
                
		  					vertex_0 += j*p_res*p_res*p_res;
		  					vertex_1 += j*p_res*p_res*p_res;
		  					vertex_2 += j*p_res*p_res*p_res;
		  					vertex_3 += j*p_res*p_res*p_res;
		  					vertex_4 += j*p_res*p_res*p_res;
		  					vertex_5 += j*p_res*p_res*p_res;
		  					vertex_6 += j*p_res*p_res*p_res;
		  					vertex_7 += j*p_res*p_res*p_res;
                
		  					write_tec << vertex_0+1  << " " <<  vertex_1+1  << " " <<  vertex_2+1 << " " <<  vertex_3+1  << " " << vertex_4+1 << " " << vertex_5+1 << " " << vertex_6+1 << " " << vertex_7+1 <<endl;
		  				}
		  			}
		  		}
		  	}
		  }
		  else
		  {
		  	FatalError("ERROR: Invalid element type ... ");
		  }
    }
	}
	
	write_tec.close();
	
#ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
	if (FlowSol->rank==0) cout << "Done writing Tecplot file number " << in_file_num << " ...." << endl;
#else
	cout << "Done writing Tecplot file number " << in_file_num << " ...." << endl;
#endif
	
}

// method to write out a tecplot file
 void write_tec_bin(int in_file_num) // TODO: Tidy this up
 {
#ifdef _TECIO
 int i,j,k,l,m;
 
 int vertex_0, vertex_1, vertex_2, vertex_3, vertex_4, vertex_5, vertex_6, vertex_7;
 
 int p_res=run_input.p_res; // HACK
 
 array<double> pos_ppts_temp;
 array<double> plotq_ppts_temp;
 int n_plot_data;
 
 char  file_name_s[50] ;
 char *file_name;
 
 #ifdef _MPI
 MPI_Barrier(MPI_COMM_WORLD);
 sprintf(file_name_s,"Mesh_%.09d_p%.04d.plt",in_file_num,FlowSol->rank);
 if (FlowSol->rank==0) cout << "Writing Tecplot file number " << in_file_num << " ...." << endl;
 #else
 sprintf(file_name_s,"Mesh_%.09d_p%.04d.plt",in_file_num,0);
 cout << "Writing Tecplot file number " << in_file_num << " ...." << endl;
 #endif
 
 // File name
 file_name = &file_name_s[0];
 
 // write header
 string title;
 string variable_names;
 string scratch_dir;
 
 title= "SD++ Solution";
 
 // write variable names
 if(FlowSol->n_dims==2)
 variable_names = "x, y, ";
 else if(FlowSol->n_dims==3)
 variable_names = "x, y, z, ";
 
 for (int i=0;i<run_input.n_plot_quantities;i++)
 {
 variable_names += run_input.plot_quantities(i);
 if (i!= run_input.n_plot_quantities-1)
 variable_names += ", ";
 }
 
 scratch_dir = "./";
 int zero=0;
 int one=1;
 int two=2;
 
 int success;
 success = TECINI111(&title[0],&variable_names[0],file_name,&scratch_dir[0],&zero,&zero,&one);
 
 string zone_title;
 int zone_type;
 int num_pts;
 int num_elements;
 int n_data_per_ppt;
 
 n_data_per_ppt = n_dims+run_input.n_plot_quantities;
 
 array<double> plot_data;
 plot_data.setup(n_data_per_ppt);
 
 array<int> act_pas_var;
 act_pas_var.setup(n_data_per_ppt);
 for (int i=0;i<n_data_per_ppt;i++)
 act_pas_var(i) = 1;
 
 for(i=0;i<n_ele_types;i++)
 {
 n_plot_data = run_input.n_plot_quantities;
 
 if (mesh_eles(i)->get_n_eles()!=0) {
 
 pos_ppts_temp.setup(mesh_eles(i)->get_n_ppts_per_ele(),mesh_eles(i)->get_n_dims());
 plotq_ppts_temp.setup(mesh_eles(i)->get_n_ppts_per_ele(),n_plot_data);
 
 // write element specific header
 num_pts= (mesh_eles(i)->get_n_eles())*(mesh_eles(i)->get_n_ppts_per_ele());
 num_elements = (mesh_eles(i)->get_n_eles())*(mesh_eles(i)->get_n_peles_per_ele());
 
 if(mesh_eles(i)->get_ele_type()==0) // tri
 {
 zone_title= "triangles";
 zone_type=2;
 }
 else if(mesh_eles(i)->get_ele_type()==1) // quad
 {
 zone_title= "quads";
 zone_type=3;
 }
 else if (mesh_eles(i)->get_ele_type()==2) // tet
 {
 zone_title= "tets";
 zone_type=4;
 }
 else if (mesh_eles(i)->get_ele_type()==3) // prisms
 {
 zone_title= "prism";
 zone_type=5;
 }
 else if(mesh_eles(i)->get_ele_type()==4) // hexa
 {
 zone_title= "hexas";
 zone_type=5;
 }
 else
 {
 FatalError("Invalid element type");
 }
 
 //success = TECZNE111(&zone_title[0],&zone_type,&num_pts,&num_elements,&zero,&zero,&zero,&zero,&time,&zero,&zero,&zero,&zero,&zero,NULL,NULL,NULL,act_pas_var.get_ptr_cpu(),NULL,NULL,&zero);
 success = TECZNE111(&zone_title[0],&zone_type,&num_pts,&num_elements,&zero,&zero,&zero,&zero,&time,&zero,&zero,&zero,&zero,&zero,NULL,NULL,NULL,NULL,NULL,NULL,&zero);
 
 // write element specific data
 
 for(j=0;j<mesh_eles(i)->get_n_eles();j++)
 {
 mesh_eles(i)->get_pos_ppts(j,pos_ppts_temp);
 mesh_eles(i)->get_plotq_ppts(j,plotq_ppts_temp,FlowSol->plotq_pnodes);
 
 for(k=0;k<mesh_eles(i)->get_n_ppts_per_ele();k++)
 {
 int count=0;
 for(l=0;l<mesh_eles(i)->get_n_dims();l++)
 {
 plot_data(count++) = pos_ppts_temp(k,l);
 }
 
 for(l=0;l<n_plot_data;l++)
 {
 if ( isnan(plotq_ppts_temp(k,l))) {
 FatalError("Nan in tecplot file, exiting");
 }
 else {
 plot_data(count++) = plotq_ppts_temp(k,l);
 }
 }
 
 //write_tec << pos_ppts_temp(k,l) << " ";
 TECDAT111(&n_data_per_ppt,plot_data.get_ptr_cpu(),&one);
 }
 }
 
 // write element specific connectivity
 success = TECNOD111(mesh_eles(i)->get_connectivity_plot_ptr());
 
 } // if n_eles!=0
 } // loop over ele types
 
 TECEND111();
 
 #ifdef _MPI
 MPI_Barrier(MPI_COMM_WORLD);
 if (FlowSol->rank==0) cout << "Done writing Tecplot file number " << in_file_num << " ...." << endl;
 #else
 cout << "Done writing Tecplot file number " << in_file_num << " ...." << endl;
 #endif
 
#endif
 }


/*! Method to write out a Paraview .vtu file.
Used in run mode.
input: in_file_num																						current timestep
input: FlowSol																								solution structure
output: Mesh_<in_file_num>.vtu																(serial) data file
output: Mesh_<in_file_num>/Mesh_<in_file_num>_<rank>.vtu			(parallel) data file containing portion of domain owned by current node. Files contained in directory Mesh_<in_file_num>.
output: Mesh_<in_file_num>.pvtu																(parallel) file stitching together all .vtu files (written by master node)
*/

void write_vtu(int in_file_num, struct solution* FlowSol) // TODO: Tidy this up
{
	int i,j,k,l,m,count;
	/*! Current rank */
	int my_rank = 0;
	/*! No. of processes */
	int n_proc = 1;
	/*! No. of output fields */
	int n_fields;
	/*! No. of dimensions */
	int n_dims;
	/*! No. of elements */
	int n_eles;
	/*! Number of plot points in element */
	int n_points;
	/*! Number of plot sub-elements in element */
  int n_cells;
	/*! No. of vertices per element */
	int n_verts;
	/*! Element type */
	int ele_type;

	/*! Plot point coordinates */
	array<double> pos_ppts_temp;
	/*! Solution data at plot points */
	array<double> disu_ppts_temp;

	/*! Plot sub-element connectivity array (node IDs) */
	array<int> con;

	/*! VTK element types (different to HiFiLES element type) */
	/*! tri, quad, tet, prism (undefined), hex */
	/*! See vtkCellType.h for full list */
	array<int> vtktypes(5);
	vtktypes(0) = 5;
	vtktypes(1) = 9;
	vtktypes(2) = 10;
	vtktypes(3) = 0;
	vtktypes(4) = 12;

	/*! File names */
	char vtu_s[50];
	char dumpnum_s[50];
	char pvtu_s[50];
	/*! File name pointers needed for opening files */
	char *vtu;
	char *pvtu;
	char *dumpnum;

	/*! Output files */
	ofstream write_vtu;
	write_vtu.precision(15);
	ofstream write_pvtu;
	write_pvtu.precision(15);

	/*! copy solution to cpu before beginning */
#ifdef _GPU
	for(i=0;i<FlowSol->n_ele_types;i++)
		FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
#endif

#ifdef _MPI
	/*! Get rank of each process */
	my_rank = FlowSol->rank;
	n_proc   = FlowSol->nproc;
	/*! Dump number */
	sprintf(dumpnum_s,"Mesh_%.09d",in_file_num,my_rank);
	/*! Each rank writes a .vtu file in a subdirectory named 'dumpnum_s' created by master process */
	sprintf(vtu_s,"Mesh_%.09d/Mesh_%.09d_%d.vtu",in_file_num,in_file_num,my_rank,my_rank);
	/*! On rank 0, write a .pvtu file to gather data from all .vtu files */
	sprintf(pvtu_s,"Mesh_%.09d.pvtu",in_file_num,0);
#else
	/*! Only write a vtu file in serial */
	sprintf(vtu_s,"Mesh_%.09d.vtu",in_file_num,0);
#endif

	/*! Point to names */
  vtu = &vtu_s[0];
  pvtu = &pvtu_s[0];
  dumpnum = &dumpnum_s[0];

#ifdef _MPI
	/*! Master node creates a subdirectory to store .vtu files */
	if (my_rank == 0) {
		struct stat st = {0};
		if (stat(dumpnum, &st) == -1) {
	    mkdir(dumpnum, 0755);
		}
		/*! Delete old .vtu files from directory */
		//remove(strcat(dumpnum,"/*.vtu"));
	}

	/*! Master node writes the .pvtu file */
	if (my_rank == 0) {
		cout << "Writing Paraview dump number " << dumpnum << " ...." << endl;

	  write_pvtu.open(pvtu);
  	write_pvtu << "<?xml version=\"1.0\" ?>" << endl;
		write_pvtu << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << endl;
		write_pvtu << "	<PUnstructuredGrid GhostLevel=\"1\">" << endl;

		/*! Write point data */
		write_pvtu << "		<PPointData Scalars=\"Density\" Vectors=\"Velocity\">" << endl;
		write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Density\" />" << endl;
		write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" />" << endl;
		write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Energy\" />" << endl;
		write_pvtu << "		</PPointData>" << endl;

		/*! Write points */
		write_pvtu << "		<PPoints>" << endl;
		write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" />" << endl;
		write_pvtu << "		</PPoints>" << endl;

		/*! Write names of source .vtu files to include */
		for (i=0;i<n_proc;++i) {
			write_pvtu << "		<Piece Source=\"" << dumpnum << "/" << dumpnum <<"_" << i << ".vtu" << "\" />" << endl;
		}

		/*! Write footer */
		write_pvtu << "	</PUnstructuredGrid>" << endl;
		write_pvtu << "</VTKFile>" << endl;
		write_pvtu.close();
	}
#else
	/*! In serial, don't write a .pvtu file. */
	cout << "Writing Paraview dump number " << dumpnum << " ...." << endl;
#endif

#ifdef _MPI
	/*! Wait for all processes to get to this point, otherwise there won't be a directory to put .vtus into */
  MPI_Barrier(MPI_COMM_WORLD);
#endif

	/*! Each process writes its own .vtu file */
	write_vtu.open(vtu);
	/*! File header */
 	write_vtu << "<?xml version=\"1.0\" ?>" << endl;
	write_vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << endl;
	write_vtu << "	<UnstructuredGrid>" << endl;

  /*! Loop over element types */
	for(i=0;i<FlowSol->n_ele_types;i++)
	{
		/*! no. of elements of type i */
		n_eles = FlowSol->mesh_eles(i)->get_n_eles();
		/*! Only proceed if there any elements of type i */
		if (n_eles!=0) {
			/*! element type */
			ele_type = FlowSol->mesh_eles(i)->get_ele_type();

			/*! no. of plot points per ele */
			n_points = FlowSol->mesh_eles(i)->get_n_ppts_per_ele();

			/*! no. of plot sub-elements per ele */
			n_cells  = FlowSol->mesh_eles(i)->get_n_peles_per_ele();

			/*! no. of vertices per ele */
			n_verts  = FlowSol->mesh_eles(i)->get_n_verts_per_ele();

			/*! no. of fields */
			n_fields = FlowSol->mesh_eles(i)->get_n_fields();

			/*! no. of dimensions */
			n_dims = FlowSol->mesh_eles(i)->get_n_dims();

			/*! Temporary array of plot point coordinates */
			pos_ppts_temp.setup(n_points,n_dims);

			/*! Temporary solution array at plot points */
			disu_ppts_temp.setup(n_points,n_fields);

			con.setup(n_verts,n_cells);
			con = FlowSol->mesh_eles(i)->get_connectivity_plot();

			/*! Loop over individual elements and write their data as a separate VTK DataArray */
			for(j=0;j<n_eles;j++)
			{
				write_vtu << "		<Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\"" << n_cells << "\">" << endl;

				/*! Calculate the solution at the plot points */
				FlowSol->mesh_eles(i)->calc_disu_ppts(j,disu_ppts_temp);

				/*! write out solution to file */
				write_vtu << "			<PointData>" << endl;

				/*! density */
				write_vtu << "				<DataArray type= \"Float32\" Name=\"Density\" format=\"ascii\">" << endl;
				for(k=0;k<n_points;k++)
				{
					write_vtu << disu_ppts_temp(k,0) << " ";
				}
				write_vtu << endl;
				write_vtu << "				</DataArray>" << endl;
      
				/*! velocity */
				write_vtu << "				<DataArray type= \"Float32\" NumberOfComponents=\"3\" Name=\"Velocity\" format=\"ascii\">" << endl;
				for(k=0;k<n_points;k++)
				{
					/*! Divide momentum components by density to obtain velocity components */
					write_vtu << disu_ppts_temp(k,1)/disu_ppts_temp(k,0) << " " << disu_ppts_temp(k,2)/disu_ppts_temp(k,0) << " ";

					/*! In 2D the z-component of velocity is not stored, but Paraview needs it so write a 0. */
					if(n_fields==4)
					{
						write_vtu << 0.0 << " ";
					}
					/*! In 3D just write the z-component of velocity */
					else
					{
						write_vtu << disu_ppts_temp(k,3)/disu_ppts_temp(k,0) << " ";
					}
				}
				write_vtu << endl;
				write_vtu << "				</DataArray>" << endl;
			
				/*! energy */
				write_vtu << "				<DataArray type= \"Float32\" Name=\"Energy\" format=\"ascii\">" << endl;
				for(k=0;k<n_points;k++)
				{
					/*! In 2D energy is the 4th solution component */
					if(n_fields==4)
					{
						write_vtu << disu_ppts_temp(k,3)/disu_ppts_temp(k,0) << " ";
					}
					/*! In 3D energy is the 5th solution component */
					else
					{
						write_vtu << disu_ppts_temp(k,4)/disu_ppts_temp(k,0) << " ";
					}
				}
				/*! End the line and finish writing DataArray and PointData objects */
				write_vtu << endl;
				write_vtu << "				</DataArray>" << endl;
				write_vtu << "			</PointData>" << endl;
      
				/*! Calculate the plot coordinates */
				FlowSol->mesh_eles(i)->calc_pos_ppts(j,pos_ppts_temp);
			
				/*! write out the plot coordinates */
				write_vtu << "			<Points>" << endl;
				write_vtu << "				<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;

				/*! Loop over plot points in element */
				for(k=0;k<n_points;k++)
				{
					for(l=0;l<n_dims;l++)
					{
						write_vtu << pos_ppts_temp(k,l) << " ";
					}

					/*! If 2D, write a 0 as the z-component */
					if(n_dims==2)
					{
						write_vtu << "0 ";
					}
				}
			
				write_vtu << endl;
				write_vtu << "				</DataArray>" << endl;
				write_vtu << "			</Points>" << endl;
      
				/*! write out Cell data: connectivity, offsets, element types */
				write_vtu << "			<Cells>" << endl;

				/*! Write connectivity array */
				write_vtu << "				<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << endl;

				for(k=0;k<n_cells;k++)
				{
					for(l=0;l<n_verts;l++)
					{
						write_vtu << con(l,k) << " ";
					}
					write_vtu << endl;
				}
				write_vtu << "				</DataArray>" << endl;

				/*! Write cell numbers */
				write_vtu << "				<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << endl;
				for(k=0;k<n_cells;k++)
				{
					write_vtu << (k+1)*n_verts << " ";
				}
				write_vtu << endl;
				write_vtu << "				</DataArray>" << endl;

				/*! Write VTK element type */
				write_vtu << "				<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << endl;
				for(k=0;k<n_cells;k++)
				{
					write_vtu << vtktypes(i) << " ";
				}
				write_vtu << endl;
  	    write_vtu << "				</DataArray>" << endl;

				/*! Write cell and piece footers */
				write_vtu << "			</Cells>" << endl;
				write_vtu << "		</Piece>" << endl;
			}
		}
	}

  /*! Write footer of file */
	write_vtu << "	</UnstructuredGrid>" << endl;
  write_vtu << "</VTKFile>" << endl;

	/*! Close the .vtu file */
	write_vtu.close();
}


//CGL adding binary output for Paraview ************************************ BEGIN
void write_vtu_bin(int in_file_num, struct solution* FlowSol) { // 3D (MPI) mixed elements
  
  // a sequential Paraview file writing...
  
  char  file_name_s[50] ;
  char *file_name;
  sprintf(file_name_s,"Mesh_%.09d.vtu",in_file_num);
  file_name = &file_name_s[0];
  
#ifdef _MPI
  int dummy_in = 0, dummy_out = 0;
  MPI_Request request_in, request_out;
  MPI_Barrier(MPI_COMM_WORLD);
  if(FlowSol->rank==0) cout << "Writing Paraview file number " << in_file_num << endl;
#else
  cout << "Writing Paraview file number " << in_file_num << endl;
#endif
  
  // step 1: count the elements (total and by type)
  
  int my_buf[7];
  my_buf[0] = FlowSol->num_pnodes; // local node count
  my_buf[1] = 0; // local eles count
  my_buf[2] = 0; // local hex count
  my_buf[3] = 0; // local prism count
  my_buf[4] = 0; // local tet count
  my_buf[5] = 0; // local quad count 2D
  my_buf[6] = 0; // local tri count  2D
  
  for(int i=0;i<FlowSol->n_ele_types;++i){
    if(FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
      int num_eles = (FlowSol->mesh_eles(i)->get_n_eles())*(FlowSol->mesh_eles(i)->get_n_peles_per_ele());
      my_buf[1] += num_eles;
      
      // element specific counters
      if(FlowSol->mesh_eles(i)->get_ele_type()==0)      my_buf[6] += num_eles; // tri
      else if(FlowSol->mesh_eles(i)->get_ele_type()==1) my_buf[5] += num_eles; // quad
      else if(FlowSol->mesh_eles(i)->get_ele_type()==2) my_buf[4] += num_eles; // tet
      else if(FlowSol->mesh_eles(i)->get_ele_type()==3) my_buf[3] += num_eles; // prisms
      else if(FlowSol->mesh_eles(i)->get_ele_type()==4) my_buf[2] += num_eles; // hexa
      else{
        FatalError("Invalid element type");
      }
    }
  }
  
  // figure out our nodal displacement for connectivities
  int displ=0;
#ifdef _MPI
  MPI_Scan(&(my_buf[0]),&displ,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  displ -= my_buf[0];
#endif
  
  // send our counts to rank 0...
#ifdef _MPI
  int buf[7];
  MPI_Reduce(my_buf,buf,7,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
#else
  int buf[7];
  for(int i=0;i<7;++i) buf[i] = my_buf[i];
#endif
  
  // Let's start writing...
  int my_rank = 0;
#ifdef _MPI
  my_rank = FlowSol->rank;
#endif
  if (my_rank == 0) {
    
    // rank 0 writes the header...
    
    // Checking Endian-ness of the machine
    const char *Endian[] = { "BigEndian", "LittleEndian" };
    unsigned char EndianTest[2] = {1,0};
    short tmp = *(short *)EndianTest;
    if( tmp != 1 ) tmp = 0;
    
    FILE * fp;
    if ( (fp=fopen(file_name,"w"))==NULL )
      FatalError("ERROR: cannot open output file");
    
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" ");
    fprintf(fp, "version=\"0.1\" ");
    fprintf(fp, "byte_order=\"%s\">\n",Endian[tmp]);
    fprintf(fp, "<UnstructuredGrid>\n");
    fprintf(fp, "<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",
            buf[0],              buf[1]);
    
    int offset = 0;
    fprintf(fp, "<Points>\n");
    fprintf(fp, "<DataArray type=\"Float64\" "
            "NumberOfComponents=\"3\" format=\"appended\" "
            "offset=\"%d\">\n", offset);
    offset += sizeof(int)+3*buf[0]*sizeof(double);
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Points>\n");
    
    // add header for data... PointData only for now
    fprintf(fp, "<PointData>\n");
    
    //fprintf(fp, "<DataArray type=\"Float64\" "
    //            "Name=\"%s\" NumberOfComponents=\"%d\" "
    //            "format=\"appended\" "
    //            "offset=\"%d\">\n",run_input.plot_quantities(i).c_str(),n_cmp,offset);
    
    for (int i=0;i<run_input.n_plot_quantities;++i){
      int n_cmp=1, dims=1;
      if(run_input.plot_quantities(i)=="u"){
        n_cmp = 3;
        if(run_input.plot_quantities(i+1)=="v") ++dims;
        if(run_input.plot_quantities(i+2)=="w") ++dims;
        
        fprintf(fp, "<DataArray type=\"Float64\" "
                "Name=\"%s\" NumberOfComponents=\"3\" "
                "format=\"appended\" "
                "offset=\"%d\">\n",run_input.plot_quantities(i).c_str(),offset);
      }
      else{
        fprintf(fp, "<DataArray type=\"Float64\" "
                "Name=\"%s\" format=\"appended\" "
                "offset=\"%d\">\n",run_input.plot_quantities(i).c_str(),offset);
      }
      offset += sizeof(int)+n_cmp*buf[0]*sizeof(double);
      fprintf(fp, "</DataArray>\n");
      i += dims-1;
    }
    fprintf(fp, "</PointData>\n");
    
    // Writing Cells section declaration
    // buf[2]: global hex count     VTK_HEXAHEDRON (=12)
    // buf[3]: global prism count   VTK_WEDGE      (=13)
    // buf[4]: global tet count     VTK_TETRA      (=10)
    // buf[5]: global quad count    VTK_QUAD       (=9)
    // buf[6]: global tri count     VTK_TRIANGLE   (=5)
    fprintf(fp, "<Cells>\n");
    fprintf(fp, "<DataArray type=\"Int32\" "
            "Name=\"connectivity\" format=\"appended\" "
            "offset=\"%d\">\n", offset);
    offset += sizeof(int)+(8*buf[2]+6*buf[3]+4*buf[4]+4*buf[5]+3*buf[6])*sizeof(int);
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray type=\"Int32\" "
            "Name=\"offsets\" format=\"appended\" "
            "offset=\"%d\">\n", offset);
    offset += sizeof(int)+buf[1]*sizeof(int);
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray type=\"Int32\" "
            "Name=\"types\" format=\"appended\" "
            "offset=\"%d\">\n", offset);
    offset += sizeof(int)+buf[1]*sizeof(int);
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Cells>\n");
    
    fprintf(fp, "</Piece>\n");
    fprintf(fp, "</UnstructuredGrid>\n");
    fclose(fp);
    
  }
  
  // Writing raw binary AppendedData section
  // ------------------------------------------------------
  // x,y,z...
  // ------------------------------------------------------
  
#ifdef _MPI
  if (my_rank > 0) {
    // rank 0 writes right away, other processes wait for previous to finish
    MPI_Status status;
    MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
  }
#endif
  
  FILE * fp;
  if ( (fp=fopen(file_name,"a"))==NULL )
    FatalError("ERROR: cannot open output file");
  
  if (my_rank == 0) {
    fprintf(fp, "<AppendedData encoding=\"raw\">\n_");
    int bytes = sizeof(int)+3*buf[0]*sizeof(double);
    fwrite(&bytes, sizeof(int), 1, fp);
  }
  
  // writing mesh point coordinates
  double x_ppt[3] = {0.0,0.0,0.0};
  for(int i=0;i<FlowSol->num_pnodes;++i){
    array<double> pos(FlowSol->n_dims);
    pos = FlowSol->pos_pnode(i);
    for(int j=0;j<FlowSol->n_dims;++j) x_ppt[j] = pos(j);
    fwrite(x_ppt, sizeof(double), 3, fp);
    //fwrite(pos.get_ptr_cpu(), sizeof(double), 3, fp);
  }
  
#ifdef _MPI
  if (my_rank < FlowSol->nproc-1) {
    // for ranks that are not last, just close and send a message on...
    fclose(fp);
    MPI_Send(&dummy_out,1,MPI_INT,my_rank+1,2222,MPI_COMM_WORLD);
  }
  else {
    // the last rank closes and sends a message to the first...
    fclose(fp);
    MPI_Isend(&dummy_out,1,MPI_INT,0,1111,MPI_COMM_WORLD,&request_out);
    //MPI_Send(&dummy_int,1,MPI_INT,0,1111,MPI_COMM_WORLD);
  }
#else
  fclose(fp);
#endif
  
  // ------------------------------------------------------
  // scalar and vector data...
  // ------------------------------------------------------
  
  for (int i=0;i<run_input.n_plot_quantities;++i){
#ifdef _MPI
    if (my_rank == 0) {
      MPI_Status status;
      MPI_Irecv(&dummy_in,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&request_in);
      MPI_Wait(&request_in,&status);
      //MPI_Recv(&dummy_int,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&status);
    }
    else {
      // processes wait for previous to finish
      MPI_Status status;
      MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
    }
#endif
    
    FILE * fp;
    if ( (fp=fopen(file_name,"a"))==NULL )
      FatalError("ERROR: cannot open output file");
    
    int n_cmp=1, dims=1;
    if(run_input.plot_quantities(i)=="u"){
      n_cmp = 3;
      if(run_input.plot_quantities(i+1)=="v") ++dims;
      if(run_input.plot_quantities(i+2)=="w") ++dims;
      //dims = n_dims;
    }
    
    if(my_rank == 0){
      int bytes = sizeof(int)+n_cmp*buf[0]*sizeof(double);
      fwrite(&bytes, sizeof(int), 1, fp);
    }
    
    for(int j=0;j<FlowSol->num_pnodes;++j){
      double phi[3] = {0.0,0.0,0.0};
      for(int k=0;k<dims;++k) phi[k] = FlowSol->plotq_pnodes(j,i+k);
      fwrite(phi, sizeof(double), n_cmp, fp);
    }
    //for(int j=0;j<num_pnodes;++j){
    //  double phi = FlowSol->plotq_pnodes(j,0);
    //  fwrite(&phi, sizeof(double), 1, fp);
    //}
    
    i += dims-1;
    
#ifdef _MPI
    if (my_rank < FlowSol->nproc-1) {
      // for ranks that are not last, just close and send a message on...
      fclose(fp);
      MPI_Send(&dummy_out,1,MPI_INT,my_rank+1,2222,MPI_COMM_WORLD);
    }
    else {
      // the last rank closes and sends a message to the first...
      fclose(fp);
      MPI_Isend(&dummy_out,1,MPI_INT,0,1111,MPI_COMM_WORLD,&request_out);
      //MPI_Send(&dummy_int,1,MPI_INT,0,1111,MPI_COMM_WORLD);
    }
#else
    fclose(fp);
#endif
  }
  
  // ------------------------------------------------------
  // connectivity...
  // ------------------------------------------------------
  
#ifdef _MPI
  if (my_rank == 0) {
    MPI_Status status;
    MPI_Irecv(&dummy_in,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&request_in);
    MPI_Wait(&request_in,&status);
    //MPI_Recv(&dummy_int,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&status);
  }
  else {
    // processes wait for previous to finish
    MPI_Status status;
    MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
  }
#endif
  
  //FILE * fp;
  if ( (fp=fopen(file_name,"a"))==NULL )
    FatalError("ERROR: cannot open output file");
  
  if (my_rank == 0) {
    // buf[2]: global hex count     VTK_HEXAHEDRON (=12)
    // buf[3]: global prism count   VTK_WEDGE      (=13)
    // buf[4]: global tet count     VTK_TETRA      (=10)
    // buf[5]: global quad count    VTK_QUAD       (=9)
    // buf[6]: global tri count     VTK_TRIANGLE   (=5)
    int bytes = sizeof(int)+(8*buf[2]+6*buf[3]+4*buf[4]+4*buf[5]+3*buf[6])*sizeof(int);
    fwrite(&bytes, sizeof(int), 1, fp);
  }
  
  for(int i=0;i<FlowSol->n_ele_types;++i) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      int cbuf[8];
      int count=0,npt;
      int ele_type=FlowSol->mesh_eles(i)->get_ele_type();
      
      switch (ele_type) {
        case 4:
          npt=8;  // hex...
          for(int j=0; j<my_buf[2]; ++j){
            for(int k=0; k<npt; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            fwrite(cbuf, sizeof(int), npt, fp);   // 0-indexed
          }
          break;
        case 3:
          npt=6;  // prism...
          for(int j=0; j<my_buf[3]; ++j){
            // need to skip 4th and 8th components in connectivity
            for(int k=0; k<3; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            ++count;
            for(int k=3; k<npt; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            ++count;
            fwrite(cbuf, sizeof(int), npt, fp);   // 0-indexed
          }
          break;
        case 2:
          npt=4;  // tet...
          for(int j=0; j<my_buf[4]; ++j){
            for(int k=0; k<npt; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            fwrite(cbuf, sizeof(int), npt, fp);   // 0-indexed
          }
          break;
        case 1:
          npt=4;  // quad...
          for(int j=0; j<my_buf[5]; ++j){
            for(int k=0; k<npt; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            fwrite(cbuf, sizeof(int), npt, fp);   // 0-indexed
          }
          break;
        case 0:
          npt=3;  // tri...
          for(int j=0; j<my_buf[6]; ++j){
            for(int k=0; k<npt; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            fwrite(cbuf, sizeof(int), npt, fp);   // 0-indexed
          }
          break;
        default:
          FatalError("Invalid element type");
      }
    }
  }
  
#ifdef _MPI
  if (my_rank < FlowSol->nproc-1) {
    // for ranks that are not last, just close and send a message on...
    fclose(fp);
    MPI_Send(&dummy_out,1,MPI_INT,my_rank+1,2222,MPI_COMM_WORLD);
  }
  else {
    // the last rank closes and sends a message to the first...
    fclose(fp);
    MPI_Isend(&dummy_out,1,MPI_INT,0,1111,MPI_COMM_WORLD,&request_out);
    //MPI_Send(&dummy_int,1,MPI_INT,0,1111,MPI_COMM_WORLD);
  }
#else
  fclose(fp);
#endif
  
  // ------------------------------------------------------
  // offsets...
  // ------------------------------------------------------
  
  int cls_offset = 0;
  
#ifdef _MPI
  if (my_rank == 0) {
    MPI_Status status;
    MPI_Irecv(&dummy_in,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&request_in);
    MPI_Wait(&request_in,&status);
    //MPI_Recv(&dummy_int,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&status);
  }
  else {
    // processes wait for previous to finish, and receive cls_offset...
    MPI_Status status;
    MPI_Recv(&cls_offset,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
  }
#endif
  
  //FILE * fp;
  if ( (fp=fopen(file_name,"a"))==NULL )
    FatalError("ERROR: cannot open output file");
  
  if (my_rank == 0) {
    int bytes = sizeof(int)+buf[1]*sizeof(int);
    fwrite(&bytes, sizeof(int), 1, fp);
  }
  
  for(int i=0;i<FlowSol->n_ele_types;++i) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      int npt;
      int ele_type=FlowSol->mesh_eles(i)->get_ele_type();
      
      switch (ele_type) {
        case 4:
          npt=8;  // hex...
          for(int j=0; j<my_buf[2]; ++j){
            cls_offset += npt;
            fwrite(&cls_offset, sizeof(int), 1, fp);
          }
          break;
        case 3:
          npt=6;  // prism...
          for(int j=0; j<my_buf[3]; ++j){
            cls_offset += npt;
            fwrite(&cls_offset, sizeof(int), 1, fp);
          }
          break;
        case 2:
          npt=4;  // tet...
          for(int j=0; j<my_buf[4]; ++j){
            cls_offset += npt;
            fwrite(&cls_offset, sizeof(int), 1, fp);
          }
          break;
        case 1:
          npt=4;  // quad...
          for(int j=0; j<my_buf[5]; ++j){
            cls_offset += npt;
            fwrite(&cls_offset, sizeof(int), 1, fp);
          }
          break;
        case 0:
          npt=3;  // tri...
          for(int j=0; j<my_buf[6]; ++j){
            cls_offset += npt;
            fwrite(&cls_offset, sizeof(int), 1, fp);
          }
          break;
        default:
          FatalError("Invalid element type");
      }
    }
  }
  
#ifdef _MPI
  if (my_rank < FlowSol->nproc-1) {
    // for ranks that are not last, just close and send a cls_offset on...
    fclose(fp);
    MPI_Send(&cls_offset,1,MPI_INT,my_rank+1,2222,MPI_COMM_WORLD);
  }
  else {
    // the last rank closes and sends a message to the first...
    fclose(fp);
    MPI_Isend(&dummy_out,1,MPI_INT,0,1111,MPI_COMM_WORLD,&request_out);
    //MPI_Send(&dummy_int,1,MPI_INT,0,1111,MPI_COMM_WORLD);
  }
#else
  fclose(fp);
#endif
  
  // ------------------------------------------------------
  // types...
  // ------------------------------------------------------
  
#ifdef _MPI
  if (my_rank == 0) {
    MPI_Status status;
    MPI_Irecv(&dummy_in,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&request_in);
    MPI_Wait(&request_in,&status);
    //MPI_Recv(&dummy_int,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&status);
  }
  else {
    // processes wait for previous to finish
    MPI_Status status;
    MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
  }
#endif
  
  if ( (fp=fopen(file_name,"a"))==NULL )
    FatalError("ERROR: cannot open output file");
  
  if (my_rank == 0) {
    int bytes = sizeof(int)+buf[1]*sizeof(int);
    fwrite(&bytes, sizeof(int), 1, fp);
  }
  
  // buf[2]: global hex count     VTK_HEXAHEDRON (=12)
  // buf[3]: global prism count   VTK_WEDGE      (=13)
  // buf[4]: global tet count     VTK_TETRA      (=10)
  // buf[5]: global quad count    VTK_QUAD       (=9)
  // buf[6]: global tri count     VTK_TRIANGLE   (=5)
  for(int i=0;i<FlowSol->n_ele_types;++i) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      int ctype;
      int ele_type=FlowSol->mesh_eles(i)->get_ele_type();
      
      switch (ele_type) {
        case 4:
          ctype=12;  // hex...
          for(int j=0; j<my_buf[2]; ++j)
            fwrite(&ctype, sizeof(int), 1, fp);
          break;
        case 3:
          ctype=13;  // prism...
          for(int j=0; j<my_buf[3]; ++j)
            fwrite(&ctype, sizeof(int), 1, fp);
          break;
        case 2:
          ctype=10;  // tet...
          for(int j=0; j<my_buf[4]; ++j)
            fwrite(&ctype, sizeof(int), 1, fp);
          break;
        case 1:
          ctype=9;  // quad...
          for(int j=0; j<my_buf[5]; ++j)
            fwrite(&ctype, sizeof(int), 1, fp);
          break;
        case 0:
          ctype=5;  // tri...
          for(int j=0; j<my_buf[6]; ++j)
            fwrite(&ctype, sizeof(int), 1, fp);
          break;
        default:
          FatalError("Invalid element type");
      }
    }
  }
  
#ifdef _MPI
  if ( my_rank == FlowSol->nproc-1 ) {
    fprintf(fp, "</AppendedData>\n");
    fprintf(fp, "</VTKFile>\n");
  }
  fclose(fp);
  
  if ( my_rank < FlowSol->nproc-1 ) {
    MPI_Send(&dummy_out,1,MPI_INT,FlowSol->rank+1,2222,MPI_COMM_WORLD);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
#else
  fprintf(fp, "</AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
#endif
  
}

#ifdef SINGLE_ZONE
void write_tec_bin(int in_file_num, struct solution* FlowSol) { // 3D (MPI) mixed elements
  
  // a sequential Tecplot file writing...
  
  char  file_name_s[50] ;
  char *file_name;
  sprintf(file_name_s,"Mesh_%.09d.plt",in_file_num);
  file_name = &file_name_s[0];
  
#ifdef _MPI
  int dummy_in = 0, dummy_out = 0;
  MPI_Request request_in, request_out;
  MPI_Barrier(MPI_COMM_WORLD);
  if(FlowSol->rank==0) cout << "Writing Tecplot file number " << in_file_num << endl;
#else
  cout << "Writing Tecplot file number " << in_file_num << endl;
#endif
  
  // step 1: count the elements (total and by type)
  
  int my_buf[7];
  my_buf[0] = num_pnodes; // local node count
  my_buf[1] = 0; // local eles count
  my_buf[2] = 0; // local hex count
  my_buf[3] = 0; // local prism count
  my_buf[4] = 0; // local tet count
  my_buf[5] = 0; // local quad count 2D
  my_buf[6] = 0; // local tri count  2D
  
  for(int i=0;i<n_ele_types;++i){
    if(mesh_eles(i)->get_n_eles()!=0) {
      
      int num_eles = (mesh_eles(i)->get_n_eles())*(mesh_eles(i)->get_n_peles_per_ele());
      my_buf[1] += num_eles;
      
      // element specific counters
      if(mesh_eles(i)->get_ele_type()==0)      my_buf[6] += num_eles; // tri
      else if(mesh_eles(i)->get_ele_type()==1) my_buf[5] += num_eles; // quad
      else if(mesh_eles(i)->get_ele_type()==2) my_buf[4] += num_eles; // tet
      else if(mesh_eles(i)->get_ele_type()==3) my_buf[3] += num_eles; // prisms
      else if(mesh_eles(i)->get_ele_type()==4) my_buf[2] += num_eles; // hexa
      else{
        FatalError("Invalid element type");
      }
    }
  }
  
  // figure out our nodal displacement for connectivities
  int displ=0;
#ifdef _MPI
  MPI_Scan(&(my_buf[0]),&displ,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  displ -= my_buf[0];
#endif
  
  // send our counts to rank 0...
#ifdef _MPI
  int buf[7];
  MPI_Reduce(my_buf,buf,7,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
#else
  int buf[7];
  for(int i=0;i<7;++i) buf[i] = my_buf[i];
#endif
  
  // ------------------------------------------------------
  // find min/max pairs for each variable
  // ------------------------------------------------------
  
  int NumVar = n_dims + run_input.n_plot_quantities;
  array<double> my_MinVal(NumVar);
  array<double> my_MaxVal(NumVar);
  for(int k=0;k<NumVar;++k){
    my_MinVal(k) =  1.7e+308;
    my_MaxVal(k) = -1.7e+308;
  }
  for(int k=0;k<FlowSol->n_dims;++k){
    for(int i=0;i<num_pnodes;++i){
      double x_ppt = *(pos_pnode(i).get_ptr_cpu() + k);
      my_MinVal(k) = min(x_ppt,my_MinVal(k));
      my_MaxVal(k) = max(x_ppt,my_MaxVal(k));
    }
  }
  for (int k=0;k<run_input.n_plot_quantities;++k){
    for(int i=0;i<num_pnodes;++i){
      double phi = FlowSol->plotq_pnodes(i,k);
      my_MinVal(k+n_dims) = min(phi,my_MinVal(k+n_dims));
      my_MaxVal(k+n_dims) = max(phi,my_MaxVal(k+n_dims));
    }
  }
  
#ifdef _MPI
  array<double> MinVal(NumVar);
  array<double> MaxVal(NumVar);
  MPI_Reduce(my_MinVal.get_ptr_cpu(),MinVal.get_ptr_cpu(),NumVar,
             MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(my_MaxVal.get_ptr_cpu(),MaxVal.get_ptr_cpu(),NumVar,
             MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
#else
  array<double> MinVal(NumVar);
  array<double> MaxVal(NumVar);
  for(int k=0;k<NumVar;++k){
    MinVal(k) = my_MinVal(k);
    MaxVal(k) = my_MaxVal(k);
  }
#endif
  
  // Let's start writing...
  int zero = 0;
  int one  = 1;
  int value;
  int VarType;
  
  int my_rank = 0;
#ifdef _MPI
  my_rank = FlowSol->rank;
#endif
  
  // ------------------------------------------------------
  // rank 0 writes the header...
  // ------------------------------------------------------
  
  if (my_rank == 0) {
    
    FILE * fp;
    if ( (fp=fopen(file_name,"w"))==NULL )
      FatalError("ERROR: cannot open output file");
    
    fprintf(fp, "#!TDV111");           // Magic number, Version number
    
    fwrite(&one,  sizeof(int), 1, fp); // used to determine Endian-ness of the machine
    fwrite(&zero, sizeof(int), 1, fp); // 0 = FULL, 1 = GRID, 2 = SOLUTION
    
    // writing title
    string title = "SD++ Solution";
    for(int c=0;c<=title.size();++c){
      int chr =  int(title[c]);
      fwrite(&chr, sizeof(int), 1, fp);
    }
    
    // write number of variables and variable names
    fwrite(&NumVar, sizeof(int), 1, fp);
    
    string VarName;
    VarName = "x";
    for(int c=0;c<=VarName.size();++c){
      int chr =  int(VarName[c]);
      fwrite(&chr, sizeof(int), 1, fp);
    }
    VarName = "y";
    for(int c=0;c<=VarName.size();++c){
      int chr =  int(VarName[c]);
      fwrite(&chr, sizeof(int), 1, fp);
    }
    if(FlowSol->n_dims==3){
      VarName = "z";
      for(int c=0;c<=VarName.size();++c){
        int chr =  int(VarName[c]);
        fwrite(&chr, sizeof(int), 1, fp);
      }
    }
    for (int i=0;i<run_input.n_plot_quantities;++i){
      VarName = run_input.plot_quantities(i);
      for(int c=0;c<=VarName.size();++c){
        int chr =  int(VarName[c]);
        fwrite(&chr, sizeof(int), 1, fp);
      }
    }
    
    // write zone
    float marker = 299.0;
    fwrite(&marker, sizeof(float), 1, fp);
    
    string ZoneName = "2D_mesh";
    if(FlowSol->n_dims==3) ZoneName = "3D_mesh";
    for(int c=0;c<=ZoneName.size();++c){
      int chr =  int(ZoneName[c]);
      fwrite(&chr, sizeof(int), 1, fp);
    }
    
    value = -1;
    fwrite(&value, sizeof(int), 1, fp);     // ParentZone
    value = -1;
    fwrite(&value, sizeof(int), 1, fp);     // StrandID
    fwrite(&time, sizeof(double), 1, fp);   // solution time
    value = -1;
    fwrite(&value, sizeof(int), 1, fp);     // Zone Color
    
    if(FlowSol->n_dims==2)
      value = 3;  //FEQUADRILATERAL
    else if(FlowSol->n_dims==3)
      value = 5;  //FEBRICK
    fwrite(&value, sizeof(int), 1, fp);     // ZoneType
    fwrite(&zero, sizeof(int), 1, fp);      // DataPacking
    fwrite(&zero, sizeof(int), 1, fp);      // Var Location (not specified => nodes)
    fwrite(&zero, sizeof(int), 1, fp);      // No face neighbors supplied
    fwrite(&zero, sizeof(int), 1, fp);      // # misc. user-defined face connections
    
    fwrite(&buf[0], sizeof(int), 1, fp);    // NumPts
    fwrite(&buf[1], sizeof(int), 1, fp);    // NumElements
    
    fwrite(&zero, sizeof(int), 1, fp);      // ICellDim (unused)
    fwrite(&zero, sizeof(int), 1, fp);      // JCellDim (unused)
    fwrite(&zero, sizeof(int), 1, fp);      // KCellDim (unused)
    fwrite(&zero, sizeof(int), 1, fp);      // No more Auxiliary name/value pairs
    
    marker = 357.0;
    fwrite(&marker, sizeof(float), 1, fp);  // EOHMARKER
    
    marker = 299.0;
    fwrite(&marker, sizeof(float), 1, fp);  // Zone marker
    
    for(int i=0;i<NumVar;++i){              // Variable data format
      //VarType = 1;  //Float
      VarType = 2;  //Double
      //VarType = 3;  //LongInt
      //VarType = 4;  //ShortInt
      //VarType = 5;  //Byte
      //VarType = 6;  //Bit
      fwrite(&VarType, sizeof(int), 1, fp);
    }
    
    fwrite(&zero, sizeof(int), 1, fp);      // Passive variables
    fwrite(&zero, sizeof(int), 1, fp);      // Variables sharing
    value = -1;
    fwrite(&value, sizeof(int), 1, fp);     // No share connectivity
    
    for(int k=0;k<NumVar;++k){
      double MinMax[2];
      MinMax[0] = MinVal(k);
      MinMax[1] = MaxVal(k);
      fwrite(MinMax, sizeof(double), 2, fp);
      //fwrite(MinVal.get_ptr_cpu(k), sizeof(double), 1, fp);
    }
    
    fclose(fp);
    
  }
  
  // Writing zone's data section
  // ------------------------------------------------------
  // x,y,z...
  // ------------------------------------------------------
  
  for(int k=0;k<FlowSol->n_dims;++k){
#ifdef _MPI
    if (my_rank > 0) {
      // rank 0 writes right away, other processes wait for previous to finish
      MPI_Status status;
      MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
    }
#endif
    
    FILE * fp;
    if ( (fp=fopen(file_name,"a"))==NULL )
      FatalError("ERROR: cannot open output file");
    
    for(int i=0;i<num_pnodes;++i){
      if(VarType == 1){
        float x_ppt = (float)(*(pos_pnode(i).get_ptr_cpu() + k));
        fwrite(&x_ppt, sizeof(float), 1, fp);
      }
      else{
        double x_ppt = *(pos_pnode(i).get_ptr_cpu() + k);
        fwrite(&x_ppt, sizeof(double), 1, fp);
      }
    }
    
#ifdef _MPI
    if (my_rank < FlowSol->nproc-1) {
      // for ranks that are not last, just close and send a message on...
      fclose(fp);
      MPI_Send(&dummy_out,1,MPI_INT,my_rank+1,2222,MPI_COMM_WORLD);
    }
    else {
      // the last rank closes and sends a message to the first...
      fclose(fp);
      MPI_Isend(&dummy_out,1,MPI_INT,0,1111,MPI_COMM_WORLD,&request_out);
    }
#else
    fclose(fp);
#endif
  }
  
  // ------------------------------------------------------
  // scalar and vector data...
  // ------------------------------------------------------
  for (int i=0;i<run_input.n_plot_quantities;++i){
#ifdef _MPI
    if (my_rank == 0) {
      MPI_Status status;
      MPI_Irecv(&dummy_in,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&request_in);
      MPI_Wait(&request_in,&status);
    }
    else {
      // processes wait for previous to finish
      MPI_Status status;
      MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
    }
#endif
    
    FILE * fp;
    if ( (fp=fopen(file_name,"a"))==NULL )
      FatalError("ERROR: cannot open output file");
    
    for(int j=0;j<num_pnodes;++j){
      if(VarType == 1){
        float phi = (float)(FlowSol->plotq_pnodes(j,i));
        fwrite(&phi, sizeof(float), 1, fp);
      }
      else{
        double phi = FlowSol->plotq_pnodes(j,i);
        fwrite(&phi, sizeof(double), 1, fp);
      }
    }
    
#ifdef _MPI
    if (my_rank < FlowSol->nproc-1) {
      // for ranks that are not last, just close and send a message on...
      fclose(fp);
      MPI_Send(&dummy_out,1,MPI_INT,my_rank+1,2222,MPI_COMM_WORLD);
    }
    else {
      // the last rank closes and sends a message to the first...
      fclose(fp);
      MPI_Isend(&dummy_out,1,MPI_INT,0,1111,MPI_COMM_WORLD,&request_out);
    }
#else
    fclose(fp);
#endif
  }
  
  // ------------------------------------------------------
  // connectivity...
  // ------------------------------------------------------
#ifdef _MPI
  if (my_rank == 0) {
    MPI_Status status;
    MPI_Irecv(&dummy_in,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&request_in);
    MPI_Wait(&request_in,&status);
  }
  else {
    // processes wait for previous to finish
    MPI_Status status;
    MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
  }
#endif
  
  FILE * fp;
  if ( (fp=fopen(file_name,"a"))==NULL )
    FatalError("ERROR: cannot open output file");
  
  // buf[2]: global hex count
  // buf[3]: global prism count
  // buf[4]: global tet count
  // buf[5]: global quad count
  // buf[6]: global tri count
  
  for(int i=0;i<n_ele_types;++i) {
    if (mesh_eles(i)->get_n_eles()!=0) {
      int cbuf[8];
      int count=0;
      int ele_type=mesh_eles(i)->get_ele_type();
      
      switch (ele_type) {
        case 4:   // hex...
          for(int j=0; j<my_buf[2]; ++j){
            for(int k=0; k<8; ++k){
              cbuf[k] = *(mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            fwrite(cbuf, sizeof(int), 8, fp);   // 0-indexed
          }
          break;
        case 3:   // prism...
          for(int j=0; j<my_buf[3]; ++j){
            for(int k=0; k<8; ++k){
              cbuf[k] = *(mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            fwrite(cbuf, sizeof(int), 8, fp);   // 0-indexed
          }
          break;
        case 2:   // tet...
          for(int j=0; j<my_buf[4]; ++j){
            int * cptr = mesh_eles(i)->get_connectivity_plot_ptr() + count;
            cbuf[0] = *(cptr + 0) + displ;
            cbuf[1] = *(cptr + 1) + displ;
            cbuf[2] = *(cptr + 2) + displ;
            cbuf[3] = cbuf[2];
            cbuf[4] = *(cptr + 3) + displ;
            cbuf[5] = cbuf[4];
            cbuf[6] = cbuf[4];
            cbuf[7] = cbuf[4];
            count += 4;
            fwrite(cbuf, sizeof(int), 8, fp);   // 0-indexed
          }
          break;
        case 1:   // quad...
          for(int j=0; j<my_buf[5]; ++j){
            for(int k=0; k<4; ++k){
              cbuf[k] = *(mesh_eles(i)->get_connectivity_plot_ptr() + count) + displ;
              ++count;
            }
            fwrite(cbuf, sizeof(int), 4, fp);   // 0-indexed
          }
          break;
        case 0:   // tri...
          for(int j=0; j<my_buf[6]; ++j){
            int * cptr = mesh_eles(i)->get_connectivity_plot_ptr() + count;
            cbuf[0] = *(cptr + 0) + displ;
            cbuf[1] = *(cptr + 1) + displ;
            cbuf[2] = *(cptr + 2) + displ;
            cbuf[3] = cbuf[2];
            count += 3;
            fwrite(cbuf, sizeof(int), 4, fp);   // 0-indexed
          }
          break;
        default:
          FatalError("Invalid element type");
      }
    }
  }
  
#ifdef _MPI
  fclose(fp);
  
  if ( my_rank < FlowSol->nproc-1 ) {
    MPI_Send(&dummy_out,1,MPI_INT,FlowSol->rank+1,2222,MPI_COMM_WORLD);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
#else
  fclose(fp);
#endif
  
}

#else  //SINGLE_ZONE
void write_tec_bin(int in_file_num, struct solution* FlowSol) { // 3D (MPI) mixed elements
  
  // a sequential Tecplot file writing...
  
  char  file_name_s[50] ;
  char *file_name;
  sprintf(file_name_s,"Mesh_%.09d.plt",in_file_num);
  file_name = &file_name_s[0];
  
#ifdef _MPI
  int dummy_in = 0, dummy_out = 0;
  MPI_Request request_in, request_out;
  MPI_Barrier(MPI_COMM_WORLD);
  if(FlowSol->rank==0) cout << "Writing Tecplot file number " << in_file_num << endl;
#else
  cout << "Writing Tecplot file number " << in_file_num << endl;
#endif
  
  // step 1: count the elements (total and by type)
  
  int my_buf[7];
  my_buf[0] = FlowSol->num_pnodes; // local node count
  my_buf[1] = 0; // local eles count
  my_buf[2] = 0; // local hex count
  my_buf[3] = 0; // local prism count
  my_buf[4] = 0; // local tet count
  my_buf[5] = 0; // local quad count 2D
  my_buf[6] = 0; // local tri count  2D
  
  for(int i=0;i<FlowSol->n_ele_types;++i){
    if(FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
      int num_eles = (FlowSol->mesh_eles(i)->get_n_eles())*(FlowSol->mesh_eles(i)->get_n_peles_per_ele());
      my_buf[1] += num_eles;
      
      // element specific counters
      if(FlowSol->mesh_eles(i)->get_ele_type()==0)      my_buf[6] += num_eles; // tri
      else if(FlowSol->mesh_eles(i)->get_ele_type()==1) my_buf[5] += num_eles; // quad
      else if(FlowSol->mesh_eles(i)->get_ele_type()==2) my_buf[4] += num_eles; // tet
      else if(FlowSol->mesh_eles(i)->get_ele_type()==3) my_buf[3] += num_eles; // prisms
      else if(FlowSol->mesh_eles(i)->get_ele_type()==4) my_buf[2] += num_eles; // hexa
      else{
        FatalError("Invalid element type");
      }
    }
  }
  
  // Let's start writing...
  int zero = 0;
  int one  = 1;
  int value;
  int VarType;
  int NumVar = FlowSol->n_dims + run_input.n_plot_quantities;
  
  int my_rank = 0;
#ifdef _MPI
  my_rank = FlowSol->rank;
#endif
  
  // ------------------------------------------------------
  // rank 0 writes the header...
  // ------------------------------------------------------
  
  if (my_rank == 0) {
    
    FILE * fp;
    if ( (fp=fopen(file_name,"w"))==NULL )
      FatalError("ERROR: cannot open output file");
    
    fprintf(fp, "#!TDV111");           // Magic number, Version number
    
    fwrite(&one,  sizeof(int), 1, fp); // used to determine Endian-ness of the machine
    fwrite(&zero, sizeof(int), 1, fp); // 0 = FULL, 1 = GRID, 2 = SOLUTION
    
    // writing title
    string title = "SD++ Solution";
    for(int c=0;c<=title.size();++c){
      int chr =  int(title[c]);
      fwrite(&chr, sizeof(int), 1, fp);
    }
    
    // write number of variables and variable names
    fwrite(&NumVar, sizeof(int), 1, fp);
    
    string VarName;
    VarName = "x";
    for(int c=0;c<=VarName.size();++c){
      int chr =  int(VarName[c]);
      fwrite(&chr, sizeof(int), 1, fp);
    }
    VarName = "y";
    for(int c=0;c<=VarName.size();++c){
      int chr =  int(VarName[c]);
      fwrite(&chr, sizeof(int), 1, fp);
    }
    if(FlowSol->n_dims==3){
      VarName = "z";
      for(int c=0;c<=VarName.size();++c){
        int chr =  int(VarName[c]);
        fwrite(&chr, sizeof(int), 1, fp);
      }
    }
    for (int i=0;i<run_input.n_plot_quantities;++i){
      VarName = run_input.plot_quantities(i);
      for(int c=0;c<=VarName.size();++c){
        int chr =  int(VarName[c]);
        fwrite(&chr, sizeof(int), 1, fp);
      }
    }
    
    
    fclose(fp);
    
  }
  
  // ------------------------------------------------------
  // zone header
  // ------------------------------------------------------
#ifdef _MPI
  if (my_rank > 0) {
    // rank 0 writes right away, other processes wait for previous to finish
    MPI_Status status;
    MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
  }
#endif
  
  FILE * fp;
  if ( (fp=fopen(file_name,"a"))==NULL )
    FatalError("ERROR: cannot open output file");
  
  float marker = 299.0;
  fwrite(&marker, sizeof(float), 1, fp);
  
  ostringstream tmpStr;
  tmpStr << my_rank;
  string ZoneName = "rank_" + tmpStr.str();
  for(int c=0;c<=ZoneName.size();++c){
    int chr =  int(ZoneName[c]);
    fwrite(&chr, sizeof(int), 1, fp);
  }
  
  value = -1;
  fwrite(&value, sizeof(int), 1, fp);     // ParentZone
  value = -1;
  fwrite(&value, sizeof(int), 1, fp);     // StrandID
  fwrite(&FlowSol->time, sizeof(double), 1, fp);   // solution time
  value = -1;
  fwrite(&value, sizeof(int), 1, fp);     // Zone Color
  
  if(FlowSol->n_dims==2)
    value = 3;  //FEQUADRILATERAL
  else if(FlowSol->n_dims==3)
    value = 5;  //FEBRICK
  fwrite(&value, sizeof(int), 1, fp);     // ZoneType
  fwrite(&zero, sizeof(int), 1, fp);      // DataPacking
  fwrite(&zero, sizeof(int), 1, fp);      // Var Location (not specified => nodes)
  fwrite(&zero, sizeof(int), 1, fp);      // No face neighbors supplied
  fwrite(&zero, sizeof(int), 1, fp);      // # misc. user-defined face connections
  
  fwrite(&my_buf[0], sizeof(int), 1, fp); // NumPts
  fwrite(&my_buf[1], sizeof(int), 1, fp); // NumElements
  
  fwrite(&zero, sizeof(int), 1, fp);      // ICellDim (unused)
  fwrite(&zero, sizeof(int), 1, fp);      // JCellDim (unused)
  fwrite(&zero, sizeof(int), 1, fp);      // KCellDim (unused)
  fwrite(&zero, sizeof(int), 1, fp);      // No more Auxiliary name/value pairs
  
#ifdef _MPI
  if (my_rank < FlowSol->nproc-1) {
    // for ranks that are not last, just close and send a message on...
    fclose(fp);
    MPI_Send(&dummy_out,1,MPI_INT,my_rank+1,2222,MPI_COMM_WORLD);
  }
  else {
    // the last rank closes and sends a message to the first...
    fclose(fp);
    MPI_Isend(&dummy_out,1,MPI_INT,0,1111,MPI_COMM_WORLD,&request_out);
  }
#else
  fclose(fp);
#endif
  
  // ------------------------------------------------------
  // Writing zone's data section
  // ------------------------------------------------------
#ifdef _MPI
  if (my_rank == 0) {
    MPI_Status status;
    MPI_Irecv(&dummy_in,1,MPI_INT,FlowSol->nproc-1,1111,MPI_COMM_WORLD,&request_in);
    MPI_Wait(&request_in,&status);
  }
  else {
    // processes wait for previous to finish
    MPI_Status status;
    MPI_Recv(&dummy_in,1,MPI_INT,my_rank-1,2222,MPI_COMM_WORLD,&status);
  }
#endif
  
  //FILE * fp;
  if ( (fp=fopen(file_name,"a"))==NULL )
    FatalError("ERROR: cannot open output file");
  
  if(my_rank == 0) {
    marker = 357.0;
    fwrite(&marker, sizeof(float), 1, fp);  // EOHMARKER
  }
  
  marker = 299.0;
  fwrite(&marker, sizeof(float), 1, fp);  // Zone marker
  
  for(int i=0;i<NumVar;++i){              // Variable data format
    //VarType = 1;  //Float
    VarType = 2;  //Double
    //VarType = 3;  //LongInt
    //VarType = 4;  //ShortInt
    //VarType = 5;  //Byte
    //VarType = 6;  //Bit
    fwrite(&VarType, sizeof(int), 1, fp);
  }
  
  fwrite(&zero, sizeof(int), 1, fp);      // Passive variables
  fwrite(&zero, sizeof(int), 1, fp);      // Variables sharing
  value = -1;
  fwrite(&value, sizeof(int), 1, fp);     // No share connectivity
  
  // ------------------------------------------------------
  // min/max pairs for each variable
  // ------------------------------------------------------
  
  for(int k=0;k<FlowSol->n_dims;++k){
    double MinVal =  1.7e+308;
    double MaxVal = -1.7e+308;
    for(int i=0;i<FlowSol->num_pnodes;++i){
      double x_ppt = *(FlowSol->pos_pnode(i).get_ptr_cpu() + k);
      MinVal = min(x_ppt,MinVal);
      MaxVal = max(x_ppt,MaxVal);
    }
    fwrite(&MinVal, sizeof(double), 1, fp);
    fwrite(&MaxVal, sizeof(double), 1, fp);
  }
  
  for (int i=0;i<run_input.n_plot_quantities;++i){
    double MinVal =  1.7e+308;
    double MaxVal = -1.7e+308;
    for(int j=0;j<FlowSol->num_pnodes;++j){
      double phi = FlowSol->plotq_pnodes(j,i);
      MinVal = min(phi,MinVal);
      MaxVal = max(phi,MaxVal);
    }
    fwrite(&MinVal, sizeof(double), 1, fp);
    fwrite(&MaxVal, sizeof(double), 1, fp);
  }
  
  // ------------------------------------------------------
  // x,y,z...
  // ------------------------------------------------------
  
  for(int k=0;k<FlowSol->n_dims;++k){
    for(int i=0;i<FlowSol->num_pnodes;++i){
      if(VarType == 1){
        float x_ppt = (float)(*(FlowSol->pos_pnode(i).get_ptr_cpu() + k));
        fwrite(&x_ppt, sizeof(float), 1, fp);
      }
      else{
        double x_ppt = *(FlowSol->pos_pnode(i).get_ptr_cpu() + k);
        fwrite(&x_ppt, sizeof(double), 1, fp);
      }
    }
  }
  
  // ------------------------------------------------------
  // scalar and vector data...
  // ------------------------------------------------------
  for (int i=0;i<run_input.n_plot_quantities;++i){
    for(int j=0;j<FlowSol->num_pnodes;++j){
      if(VarType == 1){
        float phi = (float)(FlowSol->plotq_pnodes(j,i));
        fwrite(&phi, sizeof(float), 1, fp);
      }
      else{
        double phi = FlowSol->plotq_pnodes(j,i);
        fwrite(&phi, sizeof(double), 1, fp);
      }
    }
  }
  
  // ------------------------------------------------------
  // connectivity...
  // ------------------------------------------------------
  // buf[2]: global hex count
  // buf[3]: global prism count
  // buf[4]: global tet count
  // buf[5]: global quad count
  // buf[6]: global tri count
  
  for(int i=0;i<FlowSol->n_ele_types;++i) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      int cbuf[8];
      int count=0;
      int ele_type=FlowSol->mesh_eles(i)->get_ele_type();
      
      switch (ele_type) {
        case 4:   // hex...
          for(int j=0; j<my_buf[2]; ++j){
            for(int k=0; k<8; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count);
              ++count;
            }
            fwrite(cbuf, sizeof(int), 8, fp);   // 0-indexed
          }
          break;
        case 3:   // prism...
          for(int j=0; j<my_buf[3]; ++j){
            for(int k=0; k<8; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count);
              ++count;
            }
            fwrite(cbuf, sizeof(int), 8, fp);   // 0-indexed
          }
          break;
        case 2:   // tet...
          for(int j=0; j<my_buf[4]; ++j){
            int * cptr = FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count;
            cbuf[0] = *(cptr + 0);
            cbuf[1] = *(cptr + 1);
            cbuf[2] = *(cptr + 2);
            cbuf[3] = cbuf[2];
            cbuf[4] = *(cptr + 3);
            cbuf[5] = cbuf[4];
            cbuf[6] = cbuf[4];
            cbuf[7] = cbuf[4];
            count += 4;
            fwrite(cbuf, sizeof(int), 8, fp);   // 0-indexed
          }
          break;
        case 1:   // quad...
          for(int j=0; j<my_buf[5]; ++j){
            for(int k=0; k<4; ++k){
              cbuf[k] = *(FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count);
              ++count;
            }
            fwrite(cbuf, sizeof(int), 4, fp);   // 0-indexed
          }
          break;
        case 0:   // tri...
          for(int j=0; j<my_buf[6]; ++j){
            int * cptr = FlowSol->mesh_eles(i)->get_connectivity_plot_ptr() + count;
            cbuf[0] = *(cptr + 0);
            cbuf[1] = *(cptr + 1);
            cbuf[2] = *(cptr + 2);
            cbuf[3] = cbuf[2];
            count += 3;
            fwrite(cbuf, sizeof(int), 4, fp);   // 0-indexed
          }
          break;
        default:
          FatalError("Invalid element type");
      }
    }
  }
  
#ifdef _MPI
  fclose(fp);
  
  if ( my_rank < FlowSol->nproc-1 ) {
    MPI_Send(&dummy_out,1,MPI_INT,FlowSol->rank+1,2222,MPI_COMM_WORLD);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
#else
  fclose(fp);
#endif
  
}
#endif //SINGLE_ZONE
//CGL adding binary output for Paraview ************************************** END


void write_restart(int in_file_num, struct solution* FlowSol)
{
  
	// copy solution to cpu
#ifdef _GPU
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
      
    }
  }
#endif
  
	char file_name_s[50];
	char *file_name;
	ofstream restart_file;
	restart_file.precision(15);
  
  
#ifdef _MPI
	sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,FlowSol->rank);
	if (FlowSol->rank==0) cout << "Writing Restart file number " << in_file_num << " ...." << endl;
#else
	sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,0);
	cout << "Writing Restart file number " << in_file_num << " ...." << endl;
#endif
  
  
	file_name = &file_name_s[0];
	restart_file.open(file_name);
  
  restart_file << time << endl;
	//header
  for (int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
      FlowSol->mesh_eles(i)->write_restart_info(restart_file);
      FlowSol->mesh_eles(i)->write_restart_data(restart_file);
      
    }
  }
  
	restart_file.close();
  
}

void compute_forces(int in_file_num, double in_time,struct solution* FlowSol)
{
  
	char file_name_s[50];
	char *file_name;
  ofstream cp_file;
  
#ifdef _MPI
	sprintf(file_name_s,"cp_%.09d_p%.04d.dat",in_file_num,FlowSol->rank);
#else
	sprintf(file_name_s,"cp_%.09d_p%.04d.dat",in_file_num,0);
#endif
  
	file_name = &file_name_s[0];
  cp_file.open(file_name);
  
	// copy solution to cpu
#ifdef _GPU
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
  		FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
      if (FlowSol->viscous==1)
      {
  		  FlowSol->mesh_eles(i)->cp_grad_disu_upts_gpu_cpu();
      }
    }
  }
#endif
  
  array<double> inv_force(FlowSol->n_dims),temp_inv_force(FlowSol->n_dims);
  array<double> vis_force(FlowSol->n_dims),temp_vis_force(FlowSol->n_dims);
  
  for (int m=0;m<FlowSol->n_dims;m++)
  {
    inv_force(m) = 0.;
    vis_force(m) = 0.;
  }
  
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  FlowSol->mesh_eles(i)->compute_wall_forces(temp_inv_force,temp_vis_force,cp_file);
      
      for (int m=0;m<FlowSol->n_dims;m++) {
        inv_force(m) += temp_inv_force(m);
        vis_force(m) += temp_vis_force(m);
      }
    }
  }
  
#ifdef _MPI
  
  array<double> inv_force_global(FlowSol->n_dims);
  array<double> vis_force_global(FlowSol->n_dims);
  
  for (int m=0;m<FlowSol->n_dims;m++) {
    inv_force_global(m) = 0.;
    vis_force_global(m) = 0.;
    MPI_Reduce(&inv_force(m),&inv_force_global(m),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&vis_force(m),&vis_force_global(m),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }
  
  for (int m=0;m<FlowSol->n_dims;m++)
  {
    inv_force(m) = inv_force_global(m);
    vis_force(m) = vis_force_global(m);
  }
#endif
  
  // Calculate body forcing, if running periodic channel, and add to viscous flux
	if(run_input.equation==0 and run_input.run_type==0 and run_input.forcing==1 and FlowSol->n_dims==3)
	{
		for(int i=0;i<FlowSol->n_ele_types;i++)
			FlowSol->mesh_eles(i)->calc_body_force_upts(vis_force, FlowSol->body_force);
	}
  
  if (FlowSol->rank==0)
  {
    sprintf(file_name_s,"force000.dat",FlowSol->rank);
    file_name = &file_name_s[0];
    ofstream write_force;
	  write_force.open(file_name,ios::app);
    
    write_force << scientific << setprecision(7) <<  in_time << " , ";
    write_force << setw(10) << scientific << setprecision(7) << inv_force(0)+vis_force(0) << " , " << scientific << setprecision(7) << setw(10) << inv_force(1)+vis_force(1);
    if (FlowSol->n_dims==3)
      write_force << " , " << scientific << setprecision(7) << setw(10) << inv_force(2)+vis_force(2) ;
    write_force << endl;
    write_force.close();
    
    //cout <<scientific << "    fx= " << setprecision(13) << inv_force(0)+vis_force(0) << "    fy=" << inv_force(1)+vis_force(1);
    cout <<scientific << "    fx_i= " << setprecision(7) << inv_force(0) << "    fx_v= " << setprecision(7) << vis_force(0);
    cout <<scientific << "    fy_i= " << setprecision(7) << inv_force(1) << "    fy_v= " << setprecision(7) << vis_force(1);
    if (FlowSol->n_dims==3)
      cout <<scientific << "    fz_i= " << setprecision(7) << inv_force(2) << "    fz_v= " << setprecision(7) << vis_force(2) << "    time= " << setprecision(7) << in_time;
  }
  
  cp_file.close();
  
}

// Calculate global diagnostic quantities
void CalcDiagnostics(int in_file_num, double in_time, struct solution* FlowSol)
{
  
	char file_name_s[50];
	char *file_name;
  
	// copy solution to cpu
#ifdef _GPU
  
	for(int i=0;i<FlowSol->n_ele_types;i++)
	{
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
		{
  		FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
      if (FlowSol->viscous==1)
      {
  		  FlowSol->mesh_eles(i)->cp_grad_disu_upts_gpu_cpu();
      }
    }
  }
  
#endif
  
  int ndiags = run_input.n_diagnostics;
  array <double> diagnostics(ndiags);
	for(int j=0;j<ndiags;++j)
		diagnostics(j) = 0.;
  
	// Loop over element types
	for(int i=0;i<FlowSol->n_ele_types;i++)
	{
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
		{
		  FlowSol->mesh_eles(i)->CalcDiagnostics(ndiags, diagnostics);
    }
  }
  
#ifdef _MPI
  
  array<double> diagnostics_global(ndiags);
	for(int j=0;j<ndiags;++j)
	{
		diagnostics_global(j) = 0.0;
	  MPI_Reduce(&diagnostics(j),&diagnostics_global(j),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		diagnostics(j) = diagnostics_global(j);
	}
#endif
  
  if (FlowSol->rank==0)
  {
    sprintf(file_name_s,"statfile.dat",FlowSol->rank);
    file_name = &file_name_s[0];
    ofstream write_diagnostics;
	  write_diagnostics.open(file_name,ios::app);
    
    write_diagnostics << scientific << setprecision(7) <<  in_time << " ";
		for(int j=0;j<ndiags;++j)
		{
    	write_diagnostics << setw(10) << scientific << setprecision(7) << diagnostics(j) << " ";
		}
    write_diagnostics << endl;
    write_diagnostics.close();
  }
}

void compute_error(int in_file_num, struct solution* FlowSol)
{
  int n_fields;
	
  //HACK (assume same number of fields for all elements)
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      n_fields = FlowSol->mesh_eles(i)->get_n_fields();
    }
  }
  
  array<double> error(2,n_fields);
  array<double> temp_error(2,n_fields);
  
  for (int i=0; i<n_fields; i++)
  {
    error(0,i) = 0.;
    error(1,i) = 0.;
  }
	
  // copy solution to cpu
#ifdef _GPU
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
      if (FlowSol->viscous==1)
      {
  		  FlowSol->mesh_eles(i)->cp_grad_disu_upts_gpu_cpu();
      }
    }
  }
#endif
  
  
  //Compute the error
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
		  temp_error = FlowSol->mesh_eles(i)->compute_error(run_input.error_norm_type,FlowSol->time);
      
      for(int j=0;j<n_fields; j++) {
        error(0,j) += temp_error(0,j);
        if(FlowSol->viscous) {
          error(1,j) += temp_error(1,j);
        }
      }
    }
  }
  
#ifdef _MPI
	int n_err_vals = 2*n_fields;
  
	array<double> error_global(2,n_fields);
  for (int i=0; i<n_fields; i++)
  {
    error_global(0,i) = 0.;
    error_global(1,i) = 0.;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(error.get_ptr_cpu(),error_global.get_ptr_cpu(),n_err_vals,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  
  error = error_global;
#endif
  
  if (FlowSol->rank==0)
  {
    if (run_input.error_norm_type==1) // L1 norm
    {
      error = error;
    }
    else if (run_input.error_norm_type==2) // L2 norm
    {
      for(int j=0;j<n_fields; j++) {
        error(0,j) = sqrt(error(0,j));
        if(FlowSol->viscous) {
          error(1,j) = sqrt(error(1,j));
        }
      }
    }
    
    //cout << scientific << "    error= " << setprecision(13) << error << endl;
    
    for(int j=0;j<n_fields; j++) {
      cout << scientific << " sol error, field " << j << " = " << setprecision(13) << error(0,j) << endl;
    }
    if(FlowSol->viscous)
    {
      for(int j=0;j<n_fields; j++) {
        cout << scientific << " grad error, field " << j << " = " << setprecision(13) << error(1,j) << endl;
      }
    }
    
  }
  
  // Writing error to file
  
  char  file_name_s[50] ;
  char *file_name;
  int r_flag;
  
	if (FlowSol->rank==0)
  {
    sprintf(file_name_s,"error000.dat",FlowSol->rank);
    file_name = &file_name_s[0];
    ofstream write_error;
    
	  write_error.open(file_name,ios::app);
	  write_error << in_file_num << ", ";
	  write_error <<  run_input.order << ", ";
	  write_error <<  scientific << run_input.c_tet << ", ";
	  write_error << run_input.mesh_file << ", ";
	  write_error << run_input.upts_type_tri << ", ";
	  write_error << run_input.upts_type_quad << ", ";
	  write_error << run_input.fpts_type_tri << ", ";
	  write_error << run_input.adv_type << ", ";
	  write_error << run_input.riemann_solve_type << ", ";
	  write_error << scientific << run_input.error_norm_type  << ", " ;
	  
    for(int j=0;j<n_fields; j++) {
      write_error << scientific << error(0,j);
      if((j == (n_fields-1)) && FlowSol->viscous==0)
      {
        write_error << endl;
      }
      else
      {
        write_error <<", ";
      }
    }
    
    if(FlowSol->viscous) {
      for(int j=0;j<n_fields; j++) {
        write_error << scientific << error(1,j);
        if(j == (n_fields-1))
        {
          write_error << endl;
        }
        else
        {
          write_error <<", ";
        }
      }
    }
    
	  write_error.close();
		
    double etol = 1.0e-5;
    
    r_flag = 0;
    
    //HACK
    /*
     if( ((abs(ene_hist - error(0,n_fields-1))/ene_hist) < etol && (abs(grad_ene_hist - error(1,n_fields-1))/grad_ene_hist) < etol) || (abs(error(0,n_fields-1)) > abs(ene_hist)) )
     {
     r_flag = 1;
     }
     */
    
    FlowSol->ene_hist = error(0,n_fields-1);
    FlowSol->grad_ene_hist = error(1,n_fields-1);
  }
	
	//communicate exit_state across processors
#ifdef _MPI
  MPI_Bcast(&r_flag,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
  
#ifdef _MPI
	if(r_flag)
	{
		MPI_Finalize();
	}
#endif
  
	if(r_flag)
	{
		cout << "Tolerance achieved " << endl;
		exit(0);
	}
  
}

int monitor_residual(int in_file_num, struct solution* FlowSol) {
  
  int i, j, n_upts = 0, n_fields;
  double sum[5] = {0.0, 0.0, 0.0, 0.0, 0.0}, norm[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  bool write_heads = ((((in_file_num % (run_input.monitor_res_freq*20)) == 0)) || (in_file_num == 1));
  
  
  if (FlowSol->n_dims==2) n_fields = 4;
  else n_fields = 5;
  
#ifdef _GPU
	// copy residual to cpu
	for(i=0; i<FlowSol->n_ele_types; i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
		  FlowSol->mesh_eles(i)->cp_div_tconf_upts_gpu_cpu();
    }
  }
#endif
  
	for(i=0; i<FlowSol->n_ele_types; i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles() != 0) {
      n_upts += FlowSol->mesh_eles(i)->get_n_eles()*FlowSol->mesh_eles(i)->get_n_upts_per_ele();
      for(j=0; j<n_fields; j++)
        sum[j] += FlowSol->mesh_eles(i)->compute_res_upts(run_input.res_norm_type, j);
    }
  }
  
#ifdef _MPI
  
  int n_upts_global = 0;
  double sum_global[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  MPI_Reduce(&n_upts, &n_upts_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(sum, sum_global, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  n_upts = n_upts_global;
  for(i=0; i<n_fields; i++) sum[i] = sum_global[i];
  
#endif
  
  if (FlowSol->rank == 0) {
    
    // Compute the norm
    for(i=0; i<n_fields; i++) {
      if (run_input.res_norm_type==1) { norm[i] = sum[i] / n_upts; } // L1 norm
      else if (run_input.res_norm_type==2) { norm[i] = sqrt(sum[i]) / n_upts; } // L2 norm
      else FatalError("norm_type not recognized");
      
      if (isnan(norm[i])) {
        cout << "NaN residual at iteration " << in_file_num << ". Exiting" << endl;
        return 1;
      }
    }
    
    // Write the header
    if (write_heads) {
      if (FlowSol->n_dims==2) cout << "\n  Iter       Res[Rho]   Res[RhoVelx]   Res[RhoVely]      Res[RhoE]" << endl;
      else cout <<  "\n  Iter       Res[Rho]   Res[RhoVelx]   Res[RhoVely]   Res[RhoVelz]      Res[RhoE]" << endl;
    }
    
    // Screen output
    cout.precision(8);
    cout.setf(ios::fixed, ios::floatfield);
    cout.width(6); cout << in_file_num;
    for(i=0; i<n_fields; i++) { cout.width(15); cout << norm[i]; }
    
  }
  
  return 0;
}


void check_stability(struct solution* FlowSol)
{
  int n_plot_data;
  int bisect_ind, file_lines;
  
  double c_now, dt_now;
  double a_temp, b_temp;
  double c_file, a_file, b_file;
	
  array<double> plotq_ppts_temp;
  
  int r_flag = 0;
  double i_tol    = 1.0e-4;
  double e_thresh = 1.5;
  
  bisect_ind = run_input.bis_ind;
  file_lines = run_input.file_lines;
  
	// copy solution to cpu
#ifdef _GPU
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
		  FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
      
    }
  }
#endif
  
  // check element specific data
  
	for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      
      n_plot_data = FlowSol->mesh_eles(i)->get_n_fields();
      
		  plotq_ppts_temp.setup(FlowSol->mesh_eles(i)->get_n_ppts_per_ele(),n_plot_data);
      
      for(int j=0;j<FlowSol->mesh_eles(i)->get_n_eles();j++)
	    {
		    FlowSol->mesh_eles(i)->calc_disu_ppts(j,plotq_ppts_temp);
        
		    for(int k=0;k<FlowSol->mesh_eles(i)->get_n_ppts_per_ele();k++)
		    {
			    for(int l=0;l<n_plot_data;l++)
			    {
            if ( isnan(plotq_ppts_temp(k,l)) || (abs(plotq_ppts_temp(k,l))> e_thresh) ) {
              r_flag = 1;
            }
			    }
		    }
	    }
    }
  }
  
  //HACK
  c_now   = run_input.c_tet;
  dt_now  = run_input.dt;
  
  
  if( r_flag==0 )
  {
    a_temp = dt_now;
    b_temp = run_input.b_init;
  }
  else
  {
    a_temp = run_input.a_init;
    b_temp = dt_now;
  }
  
  
  //file input
  ifstream read_time;
  read_time.open("time_step.dat",ios::in);
  read_time.precision(12);
  
  //file output
  ofstream write_time;
  write_time.open("temp.dat",ios::out);
  write_time.precision(12);
  
  if(bisect_ind > 0)
  {
    for(int i=0; i<file_lines; i++)
    {
      read_time >> c_file >> a_file >> b_file;
      
      cout << c_file << " " << a_file << " " << b_file << endl;
      
      if(i == (file_lines-1))
      {
        cout << "Writing to time step file ..." << endl;
        write_time << c_now << " " << a_temp << " " << b_temp << endl;
        
        read_time.close();
        write_time.close();
        
        remove("time_step.dat");
        rename("temp.dat","time_step.dat");
      }
      else
      {
        write_time << c_file << " " << a_file << " " << b_file << endl;
      }
    }
  }
  
  
  if(bisect_ind==0)
  {
    for(int i=0; i<file_lines; i++)
    {
      read_time >> c_file >> a_file >> b_file;
      write_time << c_file << " " << a_file << " " << b_file << endl;
    }
    
    cout << "Writing to time step file ..." << endl;
    write_time << c_now << " " << a_temp << " " << b_temp << endl;
    
    read_time.close();
    write_time.close();
    
    remove("time_step.dat");
    rename("temp.dat","time_step.dat");
  }
  
  if( (abs(b_temp - a_temp)/(0.5*(b_temp + a_temp))) < i_tol )
    exit(1);
  
  if(r_flag>0)
    exit(0);
  
}

