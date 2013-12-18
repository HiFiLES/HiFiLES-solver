/*!
 * \file eles_pris.cpp
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

#include <iomanip>
#include <iostream>
#include <cmath>

#include "../include/global.h"
#include "../include/eles.h"
#include "../include/eles_pris.h"
#include "../include/array.h"
#include "../include/funcs.h"
#include "../include/error.h"
#include "../include/cubature_tri.h"
#include "../include/cubature_quad.h"

using namespace std;

// #### constructors ####

// default constructor

eles_pris::eles_pris()
{	
}

// #### methods ####

void eles_pris::setup_ele_type_specific(int in_run_type)
{

#ifndef _MPI
  cout << "Initializing pris" << endl;
#endif

	ele_type=3;
	n_dims=3;

  if (run_input.equation==0)
	  n_fields=5;
  else if (run_input.equation==1)
    n_fields=1;
  else 
    FatalError("Equation not supported");

	n_inters_per_ele=5;

	n_upts_per_ele=(order+2)*(order+1)*(order+1)/2;
	upts_type_pri_tri = run_input.upts_type_pri_tri;
	upts_type_pri_1d = run_input.upts_type_pri_1d;
	set_loc_upts();
  set_vandermonde_tri();  

  set_inters_cubpts();

	n_ppts_per_ele=(p_res+1)*(p_res)*(p_res)/2;
  n_peles_per_ele=( (p_res-1)*(p_res-1)*(p_res-1) );
	set_loc_ppts();
	set_opp_p();

  if (in_run_type==0)
  {
	  n_fpts_per_inter.setup(5);

	  n_fpts_per_inter(0)=(order+2)*(order+1)/2;
	  n_fpts_per_inter(1)=(order+2)*(order+1)/2;
	  n_fpts_per_inter(2)=(order+1)*(order+1);
	  n_fpts_per_inter(3)=(order+1)*(order+1);
	  n_fpts_per_inter(4)=(order+1)*(order+1);
	  
	  n_fpts_per_ele=3*(order+1)*(order+1)+(order+2)*(order+1);

    // Check consistency between tet-pri interface
    if (upts_type_pri_tri != run_input.fpts_type_tet)
      FatalError("upts_type_pri_tri != fpts_type_tet");

    // Check consistency between hex-pri interface
    if (upts_type_pri_1d != run_input.upts_type_hexa)
      FatalError("upts_type_pri_1d != upts_type_hexa");

	  set_tloc_fpts();

	  set_tnorm_fpts();
	  
	  set_opp_0(run_input.sparse_pri);
	  set_opp_1(run_input.sparse_pri);
	  set_opp_2(run_input.sparse_pri);
	  set_opp_3(run_input.sparse_pri);
	  
	  if(viscous)
	  {
	  	set_opp_4(run_input.sparse_pri);
	  	set_opp_5(run_input.sparse_pri);
	  	set_opp_6(run_input.sparse_pri);
	  
	  	temp_grad_u.setup(n_fields,n_dims);
	  }
	  
	  temp_u.setup(n_fields);
	  temp_f.setup(n_fields,n_dims);
  //}  
  //else
  //{
    if (viscous==1)
    {
	  	set_opp_4(run_input.sparse_pri);
    }

    n_verts_per_ele = 6;
    n_edges_per_ele = 9; 

    n_ppts_per_edge = p_res-2;

    // Number of plot points per face, excluding points on vertices or edges
    n_ppts_per_face.setup(n_inters_per_ele);
    n_ppts_per_face(0) = (p_res-3)*(p_res-2)/2; 
    n_ppts_per_face(1) = (p_res-3)*(p_res-2)/2; 
    n_ppts_per_face(2) = (p_res-2)*(p_res-2);
    n_ppts_per_face(3) = (p_res-2)*(p_res-2); 
    n_ppts_per_face(4) = (p_res-2)*(p_res-2); 

    n_ppts_per_face2.setup(n_inters_per_ele);
    n_ppts_per_face2(0) = (p_res+1)*(p_res)/2; 
    n_ppts_per_face2(1) = (p_res+1)*(p_res)/2; 
    n_ppts_per_face2(2) = (p_res)*(p_res);
    n_ppts_per_face2(3) = (p_res)*(p_res); 
    n_ppts_per_face2(4) = (p_res)*(p_res); 

    max_n_ppts_per_face = n_ppts_per_face(2);

    // Number of plot points not on faces, edges or vertices
    n_interior_ppts = n_ppts_per_ele-6-2*n_ppts_per_face(0)-3*n_ppts_per_face(2)
                      -9*n_ppts_per_edge; 

    vert_to_ppt.setup(n_verts_per_ele);
    edge_ppt_to_ppt.setup(n_edges_per_ele,n_ppts_per_edge);
    face_ppt_to_ppt.setup(n_inters_per_ele);
    for (int i=0;i<n_inters_per_ele;i++)
      face_ppt_to_ppt(i).setup(n_ppts_per_face(i));

    face2_ppt_to_ppt.setup(n_inters_per_ele);
    for (int i=0;i<n_inters_per_ele;i++)
      face2_ppt_to_ppt(i).setup(n_ppts_per_face2(i));

    interior_ppt_to_ppt.setup(n_interior_ppts);

    create_map_ppt();  

    /*
    cout << "vert_ppt" << endl << endl;
    vert_to_ppt.print();
    cout << "edge_ppt" << endl << endl;
    edge_ppt_to_ppt.print();
    cout << "face_ppt" << endl << endl;
    for (int i=0;i<n_inters_per_ele;i++) {
      cout << "face=" << i<< endl;
      face_ppt_to_ppt(i).print();
    }
    cout << "interior_ppt" << endl << endl;
    interior_ppt_to_ppt.print();

    loc_ppts.print();
    */

  }
}

// set shape

/*
void eles_pris::set_shape(int in_s_order)
{
	// fill in
}
*/


void eles_pris::create_map_ppt(void)
{

	int i,j,k,index;
  int vert_ppt_count = 0;
  int interior_ppt_count = 0;

  array<int> edge_ppt_count(n_edges_per_ele);
  array<int> face_ppt_count(n_inters_per_ele);
  array<int> face2_ppt_count(n_inters_per_ele);
  for (int i=0;i<n_edges_per_ele;i++)
    edge_ppt_count(i)=0;

  for (int i=0;i<n_inters_per_ele;i++)
  {
    face_ppt_count(i)=0;
    face2_ppt_count(i)=0;
  }


	for(k=0;k<p_res;k++)
	{
	for(j=0;j<p_res;j++)
	{
	for(i=0;i<p_res-j;i++)
	{
    index = (p_res*(p_res+1)/2)*k + (i+(j*(p_res+1))-((j*(j+1))/2));

    if ( (k==0 || k==p_res-1) && ( (i==0 && j==0) || i==p_res-1 || j==p_res-1) )
    {
      vert_to_ppt(vert_ppt_count++)=index;
      //cout << "vert" << endl;
    }
    else if (k==0 && j==0) {
      edge_ppt_to_ppt(0,edge_ppt_count(0)++) = index;
      //cout << "edge 0" << endl;
    }
    else if (k==0 && i==p_res-j-1) {
      edge_ppt_to_ppt(1,edge_ppt_count(1)++) = index;
      //cout << "edge 1" << endl;
    }
    else if (k==0 && i==0) {
      edge_ppt_to_ppt(2,edge_ppt_count(2)++) = index;
      //cout << "edge 2" << endl;
    }
    else if (k==p_res-1 && j==0) {
      edge_ppt_to_ppt(3,edge_ppt_count(3)++) = index;
      //cout << "edge 0" << endl;
    }
    else if (k==p_res-1 && i==p_res-j-1) {
      edge_ppt_to_ppt(4,edge_ppt_count(4)++) = index;
      //cout << "edge 1" << endl;
    }
    else if (k==p_res-1 && i==0) {
      edge_ppt_to_ppt(5,edge_ppt_count(5)++) = index;
      //cout << "edge 2" << endl;
    }
    else if (i==0 && j==0) {
      edge_ppt_to_ppt(6,edge_ppt_count(6)++) = index;
      //cout << "edge 0" << endl;
    }
    else if (i==p_res-1) {
      edge_ppt_to_ppt(7,edge_ppt_count(7)++) = index;
      //cout << "edge 1" << endl;
    }
    else if (j==p_res-1) {
      edge_ppt_to_ppt(8,edge_ppt_count(8)++) = index;
      //cout << "edge 2" << endl;
    }
    else if (k==0) {
      face_ppt_to_ppt(0)(face_ppt_count(0)++) = index;
      //cout << "face 0" << endl;
    }
    else if (k==p_res-1) {
      face_ppt_to_ppt(1)(face_ppt_count(1)++) = index;
      //cout << "face 1" << endl;
    }
    else if (j==0) {
      face_ppt_to_ppt(2)(face_ppt_count(2)++) = index;
      //cout << "face 2" << endl;
    }
    else if (i==p_res-j-1) {
      face_ppt_to_ppt(3)(face_ppt_count(3)++) = index;
      //cout << "face 3" << endl;
    }
    else if (i==0) {
      face_ppt_to_ppt(4)(face_ppt_count(4)++) = index;
      //cout << "face 3" << endl;
    }
    else
      interior_ppt_to_ppt(interior_ppt_count++) = index;

			//loc_ppts(0,index)=-1.0+((2.0*i)/(1.0*(p_res-1)));
			//loc_ppts(1,index)=-1.0+((2.0*j)/(1.0*(p_res-1)));
			//loc_ppts(2,index)=-1.0+((2.0*k)/(1.0*(p_res-1)));
    if (k==0) {
      face2_ppt_to_ppt(0)(face2_ppt_count(0)++) = index;
      //cout << "face 0" << endl;
    }
    if (k==p_res-1) {
      face2_ppt_to_ppt(1)(face2_ppt_count(1)++) = index;
      //cout << "face 1" << endl;
    }
    if (j==0) {
      face2_ppt_to_ppt(2)(face2_ppt_count(2)++) = index;
      //cout << "face 2" << endl;
    }
    if (i==p_res-j-1) {
      face2_ppt_to_ppt(3)(face2_ppt_count(3)++) = index;
      //cout << "face 3" << endl;
    }
    if (i==0) {
      face2_ppt_to_ppt(4)(face2_ppt_count(4)++) = index;
      //cout << "face 3" << endl;
    }


  }
	}
	}
}

void eles_pris::set_connectivity_plot()
{
  int vertex_0,vertex_1,vertex_2,vertex_3,vertex_4,vertex_5;
  int count=0;
  int temp = (p_res)*(p_res+1)/2;

  for (int l=0;l<p_res-1;++l){
    for(int j=0;j<p_res-1;++j){ // look to right from each point
      for(int k=0;k<p_res-j-1;++k){

        vertex_0=k+(j*(p_res+1))-((j*(j+1))/2) + l*temp;
        vertex_1=vertex_0+1;
        vertex_2=k+((j+1)*(p_res+1))-(((j+1)*(j+2))/2) + l*temp;

        vertex_3 = vertex_0 + temp;
        vertex_4 = vertex_1 + temp;
        vertex_5 = vertex_2 + temp;

        connectivity_plot(0) = vertex_0;
        connectivity_plot(1) = vertex_1;
        connectivity_plot(2) = vertex_2;
        connectivity_plot(3) = vertex_2;
        connectivity_plot(4) = vertex_3;
        connectivity_plot(5) = vertex_4;
        connectivity_plot(6) = vertex_5;
        connectivity_plot(7) = vertex_5;
        count++;
      }
    }
  }
  for (int l=0;l<p_res-1;++l){
    for(int j=0;j<p_res-2;++j){ //  look to left from each point
      for(int k=1;k<p_res-j-1;++k){

        vertex_0=k+(j*(p_res+1))-((j*(j+1))/2) + l*temp;
        vertex_1=k+((j+1)*(p_res+1))-(((j+1)*(j+2))/2) + l*temp;
        vertex_2=k-1+((j+1)*(p_res+1))-(((j+1)*(j+2))/2) + l*temp;

        vertex_3 = vertex_0 + temp;
        vertex_4 = vertex_1 + temp;
        vertex_5 = vertex_2 + temp;

        connectivity_plot(0) = vertex_0;
        connectivity_plot(1) = vertex_1;
        connectivity_plot(2) = vertex_2;
        connectivity_plot(3) = vertex_2;
        connectivity_plot(4) = vertex_3;
        connectivity_plot(5) = vertex_4;
        connectivity_plot(6) = vertex_5;
        connectivity_plot(7) = vertex_5;
        count++;
      }
    }	
  }
}




// set location of solution points in standard element

void eles_pris::set_loc_upts(void)
{

  int get_order=order;
  loc_upts.setup(n_dims,n_upts_per_ele);

  n_upts_tri = (order+1)*(order+2)/2;
  n_upts_1d = order+1;

  loc_upts_pri_1d.setup(n_upts_1d);
  loc_upts_pri_tri.setup(2,n_upts_tri);

  if (upts_type_pri_1d == 0)
  {
    // 1D: gauss
		array<double> loc_1d_gauss_pts(order+1);
		#include "../data/loc_1d_gauss_pts.dat"
		loc_upts_pri_1d=loc_1d_gauss_pts;
  }
  else if (upts_type_pri_1d == 1) {
    // 1D: gauss-lobatto
		array<double> loc_1d_gauss_lobatto_pts(order+1);
		#include "../data/loc_1d_gauss_lobatto_pts.dat"
		loc_upts_pri_1d=loc_1d_gauss_lobatto_pts;
  }
  else {
		FatalError("ERROR: Unknown fpts_type_hexa .... ");
  }

  if (upts_type_pri_tri==0) // tri: inter
  {
		array<double> loc_inter_pts(n_upts_tri,2);
		#include "../data/loc_tri_inter_pts.dat"
    for (int i=0;i<n_upts_tri;i++) {
      loc_upts_pri_tri(0,i) = loc_inter_pts(i,0);
      loc_upts_pri_tri(1,i) = loc_inter_pts(i,1);
    }  
  }
  else if (upts_type_pri_tri == 1) // tri: alpha
  {
		array<double> loc_alpha_pts(n_upts_tri,2);
		#include "../data/loc_tri_alpha_pts.dat"
    for (int i=0;i<n_upts_tri;i++) {
      loc_upts_pri_tri(0,i) = loc_alpha_pts(i,0);
      loc_upts_pri_tri(1,i) = loc_alpha_pts(i,1);
    }  
  }
  else {
    FatalError("Unknown upts_type_pri_tri");
  }

  // Now set loc_upts
  for (int i=0;i<n_upts_1d;i++) {
    for (int j=0;j<n_upts_tri;j++) {
       loc_upts(0,n_upts_tri*i+j) = loc_upts_pri_tri(0,j);
       loc_upts(1,n_upts_tri*i+j) = loc_upts_pri_tri(1,j);
       loc_upts(2,n_upts_tri*i+j) = loc_upts_pri_1d(i);
    } 
  }

}

// set location of flux points in standard element

void eles_pris::set_tloc_fpts(void)
{

  tloc_fpts.setup(n_dims,n_fpts_per_ele);

  int get_order = order;

  array<double> loc_tri_fpts( (order+1)*(order+2)/2,2);
  loc_1d_fpts.setup(order+1);

  // Triangular Faces
	if (upts_type_pri_tri==0) // internal points
	{
		array<double> loc_inter_pts(n_fpts_per_inter(0),2);
		#include "../data/loc_tri_inter_pts.dat"
		loc_tri_fpts = loc_inter_pts;
	}
  else if(upts_type_pri_tri==1) // alpha optimized
	{
		array<double> loc_alpha_pts(n_fpts_per_inter(0),2);
		#include "../data/loc_tri_alpha_pts.dat"
		loc_tri_fpts = loc_alpha_pts;
	}
	else
	{
    FatalError("Unknown fpts type pri tri");
	}	

  // Quad faces
  if(upts_type_pri_1d==0) // gauss
	{
		array<double> loc_1d_gauss_pts(order+1);
		#include "../data/loc_1d_gauss_pts.dat"

    loc_1d_fpts = loc_1d_gauss_pts;
  }	
  else if(upts_type_pri_1d==1) // gauss lobatto
	{
		array<double> loc_1d_gauss_lobatto_pts(order+1);
		#include "../data/loc_1d_gauss_lobatto_pts.dat"

    loc_1d_fpts = loc_1d_gauss_lobatto_pts;
	}
	else
	{
		FatalError("ERROR: Unknown edge flux point location type.... ");
	}	

	// Now need to map these points on faces of prisms
	// Inter 0
	for (int i=0;i<n_fpts_per_inter(0);i++)
	{
		tloc_fpts(0,i) = loc_tri_fpts(i,1);
		tloc_fpts(1,i) = loc_tri_fpts(i,0);
		tloc_fpts(2,i) = -1.;
	}	
	
	// Inter 1
	for (int i=0;i<n_fpts_per_inter(1);i++)
	{
		tloc_fpts(0,n_fpts_per_inter(0)+i) = loc_tri_fpts(i,0);
		tloc_fpts(1,n_fpts_per_inter(0)+i) = loc_tri_fpts(i,1);
		tloc_fpts(2,n_fpts_per_inter(0)+i) = 1.;
	}	

	// Inters 2,3,4
  int offset = n_fpts_per_inter(0)*2;
	for (int face=0;face<3;face++) {
		for (int i=0;i<order+1;i++) {
			for (int j=0;j<order+1;j++) {

        if (face==0) {
				  tloc_fpts(0,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(j);
				  tloc_fpts(1,offset+face*(order+1)*(order+1)+i*(order+1)+j) = -1;;
        }
        else if (face==1) {
				  tloc_fpts(0,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(order-j);
				  tloc_fpts(1,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(j);
        }
        else if (face==2) {
				  tloc_fpts(0,offset+face*(order+1)*(order+1)+i*(order+1)+j) = -1.;
				  tloc_fpts(1,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(order-j);;
        }

				tloc_fpts(2,offset+face*(order+1)*(order+1)+i*(order+1)+j) = loc_1d_fpts(i);
			}
		}	
	}

}


void eles_pris::set_inters_cubpts(void)
{

  n_cubpts_per_inter.setup(n_inters_per_ele);
  loc_inters_cubpts.setup(n_inters_per_ele);
  weight_inters_cubpts.setup(n_inters_per_ele);
  tnorm_inters_cubpts.setup(n_inters_per_ele);

  cubature_tri cub_tri(inters_cub_order);
  cubature_quad cub_quad(inters_cub_order);

  int n_cubpts_tri = cub_tri.get_n_pts();
  int n_cubpts_quad = cub_quad.get_n_pts();

  for (int i=0;i<n_inters_per_ele;i++)
  {
    if (i==0 || i==1) {
      n_cubpts_per_inter(i) = n_cubpts_tri;
    }
    else if (i==2 || i==3 || i==4) {
      n_cubpts_per_inter(i) = n_cubpts_quad;
    }

  }

  for (int i=0;i<n_inters_per_ele;i++) {

    loc_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));
    weight_inters_cubpts(i).setup(n_cubpts_per_inter(i));
    tnorm_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));

    for (int j=0;j<n_cubpts_per_inter(i);j++) {

      if (i==0) {
	  	  loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
	  	  loc_inters_cubpts(i)(2,j)=-1.;
      }
      else if (i==1) {
	  	  loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
	  	  loc_inters_cubpts(i)(2,j)=1.;
      }
      else if (i==2) {
	  	  loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=-1.;
	  	  loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
      }
      else if (i==3) {
	  	  loc_inters_cubpts(i)(0,j)=cub_quad.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=-cub_quad.get_r(j);
	  	  loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
      }
      else if (i==4) {
	  	  loc_inters_cubpts(i)(0,j)=-1.;
	  	  loc_inters_cubpts(i)(1,j)=cub_quad.get_r(j);
	  	  loc_inters_cubpts(i)(2,j)=cub_quad.get_s(j);
      }

      if (i==0 || i==1)
        weight_inters_cubpts(i)(j) = cub_tri.get_weight(j);
      else if (i==2 || i==3 || i==4)
        weight_inters_cubpts(i)(j) = cub_quad.get_weight(j);

      if (i==0) {
	  	  tnorm_inters_cubpts(i)(0,j)= 0.;
	  	  tnorm_inters_cubpts(i)(1,j)= 0.;
	  	  tnorm_inters_cubpts(i)(2,j)= -1.;
      }
      else if (i==1) {
	  	  tnorm_inters_cubpts(i)(0,j)= 0.;
	  	  tnorm_inters_cubpts(i)(1,j)= 0.;
	  	  tnorm_inters_cubpts(i)(2,j)= 1.;
      }
      else if (i==2) {
	  	  tnorm_inters_cubpts(i)(0,j)= 0.;
	  	  tnorm_inters_cubpts(i)(1,j)= -1.;
	  	  tnorm_inters_cubpts(i)(2,j)= 0.;
      }
      else if (i==3) {
	  	  tnorm_inters_cubpts(i)(0,j)= 1./sqrt(2.);
	  	  tnorm_inters_cubpts(i)(1,j)= 1./sqrt(2.);
	  	  tnorm_inters_cubpts(i)(2,j)= 0.;
      }
      else if (i==4) {
	  	  tnorm_inters_cubpts(i)(0,j)= -1.;
	  	  tnorm_inters_cubpts(i)(1,j)= 0.;
	  	  tnorm_inters_cubpts(i)(2,j)= 0.;
      }

    }
  }
  set_opp_inters_cubpts();

}


// Compute the surface jacobian determinant on a face
double eles_pris::compute_inter_detjac_inters_cubpts(int in_inter,array<double> d_pos)
{
  double output = 0.;
  double xr, xs, xt;
  double yr, ys, yt;
  double zr, zs, zt;
  double temp0,temp1,temp2;

  xr = d_pos(0,0);
  xs = d_pos(0,1);
  xt = d_pos(0,2);
  yr = d_pos(1,0);
  ys = d_pos(1,1);
  yt = d_pos(1,2);
  zr = d_pos(2,0);
  zs = d_pos(2,1);
  zt = d_pos(2,2);

  double xu=0.;
  double yu=0.;
  double zu=0.;
  double xv=0.;
  double yv=0.;
  double zv=0.;

  // From calculus, for a surface parameterized by two parameters
  // u and v, than jacobian determinant is
  //
  // || (xu i + yu j + zu k) cross ( xv i + yv j + zv k)  ||

  if (in_inter==0) // u=r, v=s
  {
    xu = xr;
    yu = yr;
    zu = zr;

    xv = xs;
    yv = ys;
    zv = zs;
  }
  else if (in_inter==1) // u=s, v=s
  {
    xu = xr;
    yu = yr;
    zu = zr;

    xv = xs;
    yv = ys;
    zv = zs;
  }
  else if (in_inter==2) //u=r, v=t
  {
    xu = xr;
    yu = yr;
    zu = zr;

    xv = xt;
    yv = yt;
    zv = zt;
  }
  else if (in_inter==3) //r=u,t=v,s=1-u
  {
    xu = xr-xs;
    yu = yr-ys;
    zu = zr-zs;

    xv = xt;
    yv = yt;
    zv = zt;
  }
  else if (in_inter==4) //u=s,v=t
  {
    xu = xs;
    yu = ys;
    zu = zs;

    xv = xt;
    yv = yt;
    zv = zt;
  }


  temp0 = yu*zv-zu*yv;
  temp1 = zu*xv-xu*zv;
  temp2 = xu*yv-yu*xv;

  output = sqrt(temp0*temp0+temp1*temp1+temp2*temp2);

  return output;
}




// set location of plot points in standard element

void eles_pris::set_loc_ppts(void)
{
	int i,j,k,index;
	
	loc_ppts.setup(3,p_res*(p_res+1)/2*p_res);

	for(k=0;k<p_res;k++)
	{
	for(j=0;j<p_res;j++)
	{
		for(i=0;i<p_res-j;i++)
		{
			index = (p_res*(p_res+1)/2)*k + (i+(j*(p_res+1))-((j*(j+1))/2));

			loc_ppts(0,index)=-1.0+((2.0*i)/(1.0*(p_res-1)));
			loc_ppts(1,index)=-1.0+((2.0*j)/(1.0*(p_res-1)));
			loc_ppts(2,index)=-1.0+((2.0*k)/(1.0*(p_res-1)));
		}
	}
	}
}



// set location of shape points in standard element
/*
void eles_pris::set_loc_spts(void)
{
	// fill in
}
*/

// set transformed normal at flux points

void eles_pris::set_tnorm_fpts(void)
{

	tnorm_fpts.setup(n_dims,n_fpts_per_ele);

  int fpt = -1;
  for (int i=0;i<n_inters_per_ele;i++)
  {
    for (int j=0;j<n_fpts_per_inter(i);j++)
    {
      fpt++;
      if (i==0) {
        tnorm_fpts(0,fpt) = 0.;
        tnorm_fpts(1,fpt) = 0.;
        tnorm_fpts(2,fpt) = -1.;
      }
      else if (i==1) {
        tnorm_fpts(0,fpt) = 0.;
        tnorm_fpts(1,fpt) = 0.;
        tnorm_fpts(2,fpt) = 1.;
      }
      else if (i==2) {
        tnorm_fpts(0,fpt) = 0.;
        tnorm_fpts(1,fpt) = -1.;
        tnorm_fpts(2,fpt) = 0.;
      }
      else if (i==3) {
        tnorm_fpts(0,fpt) = 1./sqrt(2.);
        tnorm_fpts(1,fpt) = 1./sqrt(2.);
        tnorm_fpts(2,fpt) = 0.;
      }
      else if (i==4) {
        tnorm_fpts(0,fpt) = -1.;
        tnorm_fpts(1,fpt) = 0.;
        tnorm_fpts(2,fpt) = 0.;
      }
    }
  }
  //cout << "tnorm_fpts" << endl;
  //tnorm_fpts.print();
}

//#### helper methods ####

// initialize the vandermonde matrix
void eles_pris::set_vandermonde_tri()
{
  vandermonde_tri.setup(n_upts_tri,n_upts_tri);

	// create the vandermonde matrix
	for (int i=0;i<n_upts_tri;i++)
		for (int j=0;j<n_upts_tri;j++) 
			vandermonde_tri(i,j) = eval_dubiner_basis_2d(loc_upts_pri_tri(0,i),loc_upts_pri_tri(1,i),j,order);

	// Store its inverse
	inv_vandermonde_tri = inv_array(vandermonde_tri);
}

// initialize the vandermonde matrix
void eles_pris::set_vandermonde_tri_restart()
{
  array<double> vandermonde_tri_rest;
  vandermonde_tri_rest.setup(n_upts_tri_rest,n_upts_tri_rest);

	// create the vandermonde matrix
	for (int i=0;i<n_upts_tri_rest;i++)
		for (int j=0;j<n_upts_tri_rest;j++) 
			vandermonde_tri_rest(i,j) = eval_dubiner_basis_2d(loc_upts_pri_tri_rest(0,i),loc_upts_pri_tri_rest(1,i),j,order_rest);

	// Store its inverse
	inv_vandermonde_tri_rest = inv_array(vandermonde_tri_rest);
}

int eles_pris::read_restart_info(ifstream& restart_file)
{

  string str;
  // Move to triangle element
  while(1) {
    getline(restart_file,str);
    if (str=="PRIS") break;

    if (restart_file.eof()) return 0;
  }

  getline(restart_file,str);
  restart_file >> order_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_per_ele_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_tri_rest;
  getline(restart_file,str);
  getline(restart_file,str);

  loc_upts_pri_1d_rest.setup(order_rest+1);
  loc_upts_pri_tri_rest.setup(2,n_upts_tri_rest);

  for (int i=0;i<order_rest+1;i++) {
      restart_file >> loc_upts_pri_1d_rest(i);
  }
  getline(restart_file,str);
  getline(restart_file,str);

  for (int i=0;i<n_upts_tri_rest;i++) {
    for (int j=0;j<2;j++) {
      restart_file >> loc_upts_pri_tri_rest(j,i);
    }
  }

  set_vandermonde_tri_restart();
  set_opp_r();
  
  return 1;

}

void eles_pris::write_restart_info(ofstream& restart_file)        
{
  restart_file << "PRIS" << endl;

  restart_file << "Order" << endl;
  restart_file << order << endl;

  restart_file << "Number of solution points per prismatic element" << endl; 
  restart_file << n_upts_per_ele << endl;

  restart_file << "Number of solution points in triangle" << endl; 
  restart_file << n_upts_tri << endl;

  restart_file << "Location of solution points in 1D" << endl;
  for (int i=0;i<order+1;i++) {
      restart_file << loc_upts_pri_1d(i) << " ";
  }
  restart_file << endl;

  restart_file << "Location of solution points in triangle" << endl;
  for (int i=0;i<n_upts_tri;i++) {
    for (int j=0;j<2;j++) {
      restart_file << loc_upts_pri_tri(j,i) << " ";
    }
    restart_file << endl;
  }
}

// evaluate nodal basis

double eles_pris::eval_nodal_basis(int in_index, array<double> in_loc)
{
  double oned_nodal_basis_at_loc;
	double tri_nodal_basis_at_loc;

  int index_tri = in_index%n_upts_tri;
  int index_1d = in_index/n_upts_tri;

  // 1. First evaluate the triangular nodal basis at loc(0) and loc(1)

	// First evaluate the normalized Dubiner basis at position in_loc	
 	array<double> dubiner_basis_at_loc(n_upts_tri);
	for (int i=0;i<n_upts_tri;i++) 
		dubiner_basis_at_loc(i) = eval_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);

	// From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
	tri_nodal_basis_at_loc = 0.;
	for (int i=0;i<n_upts_tri;i++)
		tri_nodal_basis_at_loc += inv_vandermonde_tri(i,index_tri)*dubiner_basis_at_loc(i);

  // 2. Now evaluate the 1D lagrange basis at loc(2)
  oned_nodal_basis_at_loc = eval_lagrange(in_loc(2),index_1d,loc_upts_pri_1d);

  return (tri_nodal_basis_at_loc*oned_nodal_basis_at_loc);

}

// evaluate nodal basis for restart

double eles_pris::eval_nodal_basis_restart(int in_index, array<double> in_loc)
{
  double oned_nodal_basis_at_loc;
	double tri_nodal_basis_at_loc;

  int index_tri = in_index%n_upts_tri_rest;
  int index_1d = in_index/n_upts_tri_rest;

  // 1. First evaluate the triangular nodal basis at loc(0) and loc(1)

	// First evaluate the normalized Dubiner basis at position in_loc	
 	array<double> dubiner_basis_at_loc(n_upts_tri_rest);
	for (int i=0;i<n_upts_tri_rest;i++) 
		dubiner_basis_at_loc(i) = eval_dubiner_basis_2d(in_loc(0),in_loc(1),i,order_rest);

	// From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
	tri_nodal_basis_at_loc = 0.;
	for (int i=0;i<n_upts_tri_rest;i++)
		tri_nodal_basis_at_loc += inv_vandermonde_tri_rest(i,index_tri)*dubiner_basis_at_loc(i);

  // 2. Now evaluate the 1D lagrange basis at loc(2)
  oned_nodal_basis_at_loc = eval_lagrange(in_loc(2),index_1d,loc_upts_pri_1d_rest);

  return (tri_nodal_basis_at_loc*oned_nodal_basis_at_loc);
}

// evaluate derivative of nodal basis

double eles_pris::eval_d_nodal_basis(int in_index, int in_cpnt, array<double> in_loc)
{
	double out_d_nodal_basis_at_loc;

  int index_tri = in_index%n_upts_tri;
  int index_1d = in_index/n_upts_tri;

  if (in_cpnt == 0 || in_cpnt == 1)
  {
    double d_tri_nodal_basis_at_loc;
    double oned_nodal_basis_at_loc;

	  // 1. Evaluate the derivative of triangular nodal basis at loc(0) and loc(1)
    
    // Evalute the derivative normalized Dubiner basis at position in_loc	
	  array<double> d_dubiner_basis_at_loc(n_upts_per_ele);
	  for (int i=0;i<n_upts_tri;i++) {
      if (in_cpnt==0)
	  	 d_dubiner_basis_at_loc(i) = eval_dr_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);
      else if (in_cpnt==1)
	  	 d_dubiner_basis_at_loc(i) = eval_ds_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);
    }

	  // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
	  d_tri_nodal_basis_at_loc = 0.;
	  for (int i=0;i<n_upts_tri;i++)
	  	d_tri_nodal_basis_at_loc += inv_vandermonde_tri(i,index_tri)*d_dubiner_basis_at_loc(i);

    // 2. Evaluate the 1d nodal basis at loc(2)
    oned_nodal_basis_at_loc = eval_lagrange(in_loc(2),index_1d,loc_upts_pri_1d);

    out_d_nodal_basis_at_loc = d_tri_nodal_basis_at_loc*oned_nodal_basis_at_loc;
  }
  else if (in_cpnt==2)
  {

    double tri_nodal_basis_at_loc;
    double d_oned_nodal_basis_at_loc;

    // 1. First evaluate the triangular nodal basis at loc(0) and loc(1)

	  // Evaluate the normalized Dubiner basis at position in_loc	
 	  array<double> dubiner_basis_at_loc(n_upts_tri);
	  for (int i=0;i<n_upts_tri;i++) 
	  	dubiner_basis_at_loc(i) = eval_dubiner_basis_2d(in_loc(0),in_loc(1),i,order);

	  // From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
	  tri_nodal_basis_at_loc = 0.;
	  for (int i=0;i<n_upts_tri;i++)
	  	tri_nodal_basis_at_loc += inv_vandermonde_tri(i,index_tri)*dubiner_basis_at_loc(i);

    // 2. Then evaluate teh derivative of 1d nodal basis at loc(2)
    d_oned_nodal_basis_at_loc = eval_d_lagrange(in_loc(2),index_1d,loc_upts_pri_1d);
   
   out_d_nodal_basis_at_loc = tri_nodal_basis_at_loc*d_oned_nodal_basis_at_loc; 
    
  }


	return out_d_nodal_basis_at_loc;	
	// fill in
}

// evaluate nodal shape basis

double eles_pris::eval_nodal_s_basis(int in_index, array<double> in_loc, int in_n_spts)
{

  double nodal_s_basis;

  if (in_n_spts==6) {
    if (in_index==0) 
      nodal_s_basis =  1./4.*(in_loc(0)+in_loc(1)) *(in_loc(2)-1.);
    else if (in_index==1) 
      nodal_s_basis = -1./4.*(in_loc(0)+1.)*(in_loc(2)-1.);
    else if (in_index==2) 
      nodal_s_basis = -1./4.*(in_loc(1)+1.)*(in_loc(2)-1.);
    else if (in_index==3) 
      nodal_s_basis = -1./4.*(in_loc(0)+in_loc(1))*(in_loc(2)+1.);
    else if (in_index==4) 
      nodal_s_basis =  1./4.*(in_loc(0)+1.)*(in_loc(2)+1.);
    else if (in_index==5) 
      nodal_s_basis =  1./4.*(in_loc(1)+1.)*(in_loc(2)+1.);
  }
  else if (in_n_spts==15) {
    if (in_index==0) 
	    nodal_s_basis = (1./4*(in_loc(0)+in_loc(1)))*(in_loc(0)+in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
    else if (in_index==1) 
	    nodal_s_basis = (1./4)*in_loc(0)*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)-1.);
    else if (in_index==2) 
	    nodal_s_basis = (1./4)*in_loc(1)*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
    else if (in_index==3) 
	    nodal_s_basis = (1./4*(in_loc(0)+in_loc(1)))*(in_loc(0)+in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
    else if (in_index==4) 
	    nodal_s_basis = (1./4)*in_loc(0)*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)+1.);
    else if (in_index==5) 
	    nodal_s_basis = (1./4)*in_loc(1)*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
    else if (in_index==6) 
	    nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)-1.);
    else if (in_index==7) 
	    nodal_s_basis = (1./2*(in_loc(0)+1.))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
    else if (in_index==8) 
	    nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)-1.);
    else if (in_index==9) 
	    nodal_s_basis = (1./2*(in_loc(0)+in_loc(1)))*(in_loc(2)*in_loc(2)-1.);
    else if (in_index==10) 
	    nodal_s_basis = -(1./2*(in_loc(0)+1.))*(in_loc(2)*in_loc(2)-1.);
    else if (in_index==11) 
	    nodal_s_basis = -(1./2*(in_loc(1)+1.))*(in_loc(2)*in_loc(2)-1.);
    else if (in_index==12) 
	    nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(0)+1.)*in_loc(2)*(in_loc(2)+1.);
    else if (in_index==13) 
	    nodal_s_basis = (1./2*(in_loc(0)+1.))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
    else if (in_index==14) 
	    nodal_s_basis = -(1./2*(in_loc(0)+in_loc(1)))*(in_loc(1)+1.)*in_loc(2)*(in_loc(2)+1.);
  }
  else
  {
    FatalError("Shape order not implemented yet, exiting");
  }
  return nodal_s_basis;

}

// evaluate derivative of nodal shape basis

void eles_pris::eval_d_nodal_s_basis(array<double> &d_nodal_s_basis, array<double> in_loc, int in_n_spts)
{

  if (in_n_spts==6) {
	  d_nodal_s_basis(0,0) =  1./4.*(in_loc(2)-1.); 
	  d_nodal_s_basis(1,0) = -1./4.*(in_loc(2)-1.); 
	  d_nodal_s_basis(2,0) = 0;
	  d_nodal_s_basis(3,0) = -1./4.*(in_loc(2)+1.); 
	  d_nodal_s_basis(4,0) =  1./4.*(in_loc(2)+1.); 
	  d_nodal_s_basis(5,0) =  0.; 

	  d_nodal_s_basis(0,1) =  1./4.*(in_loc(2)-1.); 
	  d_nodal_s_basis(1,1) = 0.; 
	  d_nodal_s_basis(2,1) = -1./4.*(in_loc(2)-1.); 
	  d_nodal_s_basis(3,1) = -1./4.*(in_loc(2)+1.); 
	  d_nodal_s_basis(4,1) =  0.; 
	  d_nodal_s_basis(5,1) =  1./4.*(in_loc(2)+1.); 

	  d_nodal_s_basis(0,2) =  1./4.*(in_loc(0)+in_loc(1)); 
	  d_nodal_s_basis(1,2) = -1./4.*(in_loc(0)+1.); 
	  d_nodal_s_basis(2,2) = -1./4.*(in_loc(1)+1.); 
	  d_nodal_s_basis(3,2) = -1./4.*(in_loc(0)+in_loc(1)); 
	  d_nodal_s_basis(4,2) =  1./4.*(in_loc(0)+1.); 
	  d_nodal_s_basis(5,2) =  1./4.*(in_loc(1)+1.); 
  }
  else if (in_n_spts==15) {

	  d_nodal_s_basis(0 ,0) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+2*in_loc(1)+1.);
	  d_nodal_s_basis(1 ,0) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+1.); 
	  d_nodal_s_basis(2 ,0) = 0.;
	  d_nodal_s_basis(3 ,0) = (1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+2*in_loc(1)+1.);
	  d_nodal_s_basis(4 ,0) =(1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+1.);
	  d_nodal_s_basis(5 ,0) = 0.;
	  d_nodal_s_basis(6 ,0) = -(1./2)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+1.+in_loc(1));
	  d_nodal_s_basis(7 ,0) = (1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)-1.);
	  d_nodal_s_basis(8 ,0) = -(1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)-1.);
	  d_nodal_s_basis(9 ,0) = (1./2)*in_loc(2)*in_loc(2)-1./2;
	  d_nodal_s_basis(10,0) = -(1./2)*in_loc(2)*in_loc(2)+1./2;
	  d_nodal_s_basis(11,0) = 0.;
	  d_nodal_s_basis(12,0) = -(1./2)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+1.+in_loc(1));
	  d_nodal_s_basis(13,0) = (1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)+1.);
	  d_nodal_s_basis(14,0) = -(1./2*(in_loc(1)+1.))*in_loc(2)*(in_loc(2)+1.);


	  d_nodal_s_basis(0 ,1) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(0)+2*in_loc(1)+1.);
	  d_nodal_s_basis(1 ,1) = 0.;
	  d_nodal_s_basis(2 ,1) = (1./4)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(1)+1.);
	  d_nodal_s_basis(3 ,1) = (1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(0)+2*in_loc(1)+1.);
	  d_nodal_s_basis(4 ,1) = 0.;
	  d_nodal_s_basis(5 ,1) = (1./4)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(1)+1.);
	  d_nodal_s_basis(6 ,1) = -(1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)-1.);
	  d_nodal_s_basis(7 ,1) = (1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)-1.);
	  d_nodal_s_basis(8 ,1) = -(1./2)*in_loc(2)*(in_loc(2)-1.)*(2*in_loc(1)+1.+in_loc(0));
	  d_nodal_s_basis(9 ,1) = (1./2)*in_loc(2)*in_loc(2)-1./2;
	  d_nodal_s_basis(10,1) = 0.;
	  d_nodal_s_basis(11,1) = -(1./2)*in_loc(2)*in_loc(2)+1./2;
	  d_nodal_s_basis(12,1) = -(1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)+1.);
	  d_nodal_s_basis(13,1) = (1./2*(in_loc(0)+1.))*in_loc(2)*(in_loc(2)+1.);
	  d_nodal_s_basis(14,1) = -(1./2)*in_loc(2)*(in_loc(2)+1.)*(2*in_loc(1)+1.+in_loc(0));

	  d_nodal_s_basis(0 ,2) = (1./4*(in_loc(0)+in_loc(1)+1.))*(in_loc(0)+in_loc(1))*(2*in_loc(2)-1.);
	  d_nodal_s_basis(1 ,2) = (1./4)*in_loc(0)*(2*in_loc(2)-1.)*(in_loc(0)+1.);
	  d_nodal_s_basis(2 ,2) = (1./4)*in_loc(1)*(2*in_loc(2)-1.)*(in_loc(1)+1.);
	  d_nodal_s_basis(3 ,2) = (1./4*(in_loc(0)+in_loc(1)+1.))*(in_loc(0)+in_loc(1))*(2*in_loc(2)+1.);
	  d_nodal_s_basis(4 ,2) = (1./4)*in_loc(0)*(2*in_loc(2)+1.)*(in_loc(0)+1.);
	  d_nodal_s_basis(5 ,2) = (1./4)*in_loc(1)*(2*in_loc(2)+1.)*(in_loc(1)+1.);
	  d_nodal_s_basis(6 ,2) = -(1./2*(2*in_loc(2)-1.))*(in_loc(0)+1.)*(in_loc(0)+in_loc(1));
	  d_nodal_s_basis(7 ,2) = (1./2*(2*in_loc(2)-1.))*(in_loc(1)+1.)*(in_loc(0)+1.);
	  d_nodal_s_basis(8 ,2) = -(1./2*(2*in_loc(2)-1.))*(in_loc(1)+1.)*(in_loc(0)+in_loc(1));
	  d_nodal_s_basis(9 ,2) = in_loc(2)*(in_loc(0)+in_loc(1));
	  d_nodal_s_basis(10,2) = -in_loc(2)*(in_loc(0)+1.);
	  d_nodal_s_basis(11,2) = -in_loc(2)*(in_loc(1)+1.);
	  d_nodal_s_basis(12,2) = -(1./2*(2*in_loc(2)+1.))*(in_loc(0)+1.)*(in_loc(0)+in_loc(1));
	  d_nodal_s_basis(13,2) = (1./2*(2*in_loc(2)+1.))*(in_loc(1)+1.)*(in_loc(0)+1.);
	  d_nodal_s_basis(14,2) = -(1./2*(2*in_loc(2)+1.))*(in_loc(1)+1.)*(in_loc(0)+in_loc(1));
  }
  else
  {
    FatalError("Shape order not implemented yet, exiting");
  }
}

// evaluate second derivative of nodal shape basis

void eles_pris::eval_dd_nodal_s_basis(array<double> &dd_nodal_s_basis, array<double> in_loc, int in_n_spts)
{

  if (in_n_spts==6) 
  {
    dd_nodal_s_basis(0,0) = 0.;
    dd_nodal_s_basis(1,0) = 0.;
    dd_nodal_s_basis(2,0) = 0.;
    dd_nodal_s_basis(3,0) = 0.;
    dd_nodal_s_basis(4,0) = 0.;
    dd_nodal_s_basis(5,0) = 0.;

    dd_nodal_s_basis(0,1) = 0.;
    dd_nodal_s_basis(1,1) = 0.;
    dd_nodal_s_basis(2,1) = 0.;
    dd_nodal_s_basis(3,1) = 0.;
    dd_nodal_s_basis(4,1) = 0.;
    dd_nodal_s_basis(5,1) = 0.;

    dd_nodal_s_basis(0,2) = 0.;
    dd_nodal_s_basis(1,2) = 0.;
    dd_nodal_s_basis(2,2) = 0.;
    dd_nodal_s_basis(3,2) = 0.;
    dd_nodal_s_basis(4,2) = 0.;
    dd_nodal_s_basis(5,2) = 0.;

    dd_nodal_s_basis(0,3) = 0.;
    dd_nodal_s_basis(1,3) = 0.;
    dd_nodal_s_basis(2,3) = 0.;
    dd_nodal_s_basis(3,3) = 0.;
    dd_nodal_s_basis(4,3) = 0.;
    dd_nodal_s_basis(5,3) = 0.;

    dd_nodal_s_basis(0,4) = 0.25;
    dd_nodal_s_basis(1,4) = -0.25;
    dd_nodal_s_basis(2,4) = 0.;
    dd_nodal_s_basis(3,4) = -0.25;
    dd_nodal_s_basis(4,4) = 0.25;
    dd_nodal_s_basis(5,4) = 0.;

    dd_nodal_s_basis(0,5) = 0.25;
    dd_nodal_s_basis(1,5) = 0.;
    dd_nodal_s_basis(2,5) = -0.25;
    dd_nodal_s_basis(3,5) = -0.25;
    dd_nodal_s_basis(4,5) = 0.;
    dd_nodal_s_basis(5,5) = 0.25;

  }
  else if(in_n_spts==15) 
  {
    dd_nodal_s_basis(0 ,0) = 0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(1 ,0) = 0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(2 ,0) = 0.;
    dd_nodal_s_basis(3 ,0) = 0.5*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(4 ,0) = 0.5*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(5 ,0) = 0.;
    dd_nodal_s_basis(6 ,0) = -1.*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(7 ,0) = 0.;
    dd_nodal_s_basis(8 ,0) = 0.;
    dd_nodal_s_basis(9 ,0) = 0.;
    dd_nodal_s_basis(10,0) = 0.;
    dd_nodal_s_basis(11,0) = 0.;
    dd_nodal_s_basis(12,0) = -1.*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(13,0) = 0.;
    dd_nodal_s_basis(14,0) = 0.;

    dd_nodal_s_basis(0 ,1) = 0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(1 ,1) = 0.;
    dd_nodal_s_basis(2 ,1) = 0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(3 ,1) = 0.5*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(4 ,1) = 0.;
    dd_nodal_s_basis(5 ,1) = 0.5*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(6 ,1) = 0.;
    dd_nodal_s_basis(7 ,1) = 0.;
    dd_nodal_s_basis(8 ,1) = -1.*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(9 ,1) = 0.;
    dd_nodal_s_basis(10,1) = 0.;
    dd_nodal_s_basis(11,1) = 0.;
    dd_nodal_s_basis(12,1) = 0.;
    dd_nodal_s_basis(13,1) = 0.;
    dd_nodal_s_basis(14,1) = -1.*in_loc(2)*(1. + in_loc(2));

    dd_nodal_s_basis(0 ,2) = 0.5*(in_loc(0) + in_loc(1))*(1. + in_loc(0) + in_loc(1));
    dd_nodal_s_basis(1 ,2) = 0.5*in_loc(0)*(1. + in_loc(0));
    dd_nodal_s_basis(2 ,2) = 0.5*in_loc(1)*(1. + in_loc(1));
    dd_nodal_s_basis(3 ,2) = 0.5*(in_loc(0) + in_loc(1))*(1. + in_loc(0) + in_loc(1));
    dd_nodal_s_basis(4 ,2) = 0.5*in_loc(0)*(1. + in_loc(0));
    dd_nodal_s_basis(5 ,2) = 0.5*in_loc(1)*(1. + in_loc(1));
    dd_nodal_s_basis(6 ,2) = -1.*(1. + in_loc(0))*(in_loc(0) + in_loc(1));
    dd_nodal_s_basis(7 ,2) = 1.*(1. + in_loc(0))*(1. + in_loc(1));
    dd_nodal_s_basis(8 ,2) = -1.*(1. + in_loc(1))*(in_loc(0) + in_loc(1));
    dd_nodal_s_basis(9 ,2) = 1.*(in_loc(0) + in_loc(1));
    dd_nodal_s_basis(10,2) = -1.*(1. + in_loc(0));
    dd_nodal_s_basis(11,2) = -1.*(1. + in_loc(1));
    dd_nodal_s_basis(12,2) = -1.*(1. + in_loc(0))*(in_loc(0) + in_loc(1));
    dd_nodal_s_basis(13,2) = 1.*(1. + in_loc(0))*(1. + in_loc(1));
    dd_nodal_s_basis(14,2) = -1.*(1. + in_loc(1))*(in_loc(0) + in_loc(1));

    dd_nodal_s_basis(0 ,3) = 0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(1 ,3) = 0.;
    dd_nodal_s_basis(2 ,3) = 0.;
    dd_nodal_s_basis(3 ,3) = 0.5*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(4 ,3) = 0.;
    dd_nodal_s_basis(5 ,3) = 0.;
    dd_nodal_s_basis(6 ,3) = -0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(7 ,3) = 0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(8 ,3) = -0.5*(-1. + in_loc(2))*in_loc(2);
    dd_nodal_s_basis(9 ,3) = 0.;
    dd_nodal_s_basis(10,3) = 0.;
    dd_nodal_s_basis(11,3) = 0.;
    dd_nodal_s_basis(12,3) = -0.5*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(13,3) = 0.5*in_loc(2)*(1. + in_loc(2));
    dd_nodal_s_basis(14,3) = -0.5*in_loc(2)*(1. + in_loc(2));

    dd_nodal_s_basis(0 ,4) = 0.25*(in_loc(0) + in_loc(1))*(-1. + in_loc(2)) + 0.25*(1. + in_loc(0) + in_loc(1))*(-1. + in_loc(2)) + 0.25*(in_loc(0) + in_loc(1))*in_loc(2) + 0.25*(1. + in_loc(0) + in_loc(1))*in_loc(2);
    dd_nodal_s_basis(1 ,4) = 0.25*in_loc(0)*(-1. + in_loc(2)) + 0.25*(1. + in_loc(0))*(-1. + in_loc(2)) + 0.25*in_loc(0)*in_loc(2) + 0.25*(1. + in_loc(0))*in_loc(2);
    dd_nodal_s_basis(2 ,4) = 0.;
    dd_nodal_s_basis(3 ,4) = 0.25*(in_loc(0) + in_loc(1))*in_loc(2) + 0.25*(1. + in_loc(0) + in_loc(1))*in_loc(2) + 0.25*(in_loc(0) + in_loc(1))*(1. + in_loc(2)) + 0.25*(1. + in_loc(0) + in_loc(1))*(1. + in_loc(2));
    dd_nodal_s_basis(4 ,4) = 0.25*in_loc(0)*in_loc(2) + 0.25*(1. + in_loc(0))*in_loc(2) + 0.25*in_loc(0)*(1. + in_loc(2)) + 0.25*(1. + in_loc(0))*(1. + in_loc(2));
    dd_nodal_s_basis(5 ,4) = 0.;
    dd_nodal_s_basis(6 ,4) = -0.5*(1. + in_loc(0))*(-1. + in_loc(2)) - 0.5*(in_loc(0) + in_loc(1))*(-1. + in_loc(2)) - 0.5*(1. + in_loc(0))*in_loc(2) - 0.5*(in_loc(0) + in_loc(1))*in_loc(2);
    dd_nodal_s_basis(7 ,4) = 0.5*(1. + in_loc(1))*(-1. + in_loc(2)) + 0.5*(1. + in_loc(1))*in_loc(2);
    dd_nodal_s_basis(8 ,4) = -0.5*(1. + in_loc(1))*(-1. + in_loc(2)) - 0.5*(1. + in_loc(1))*in_loc(2);
    dd_nodal_s_basis(9 ,4) = 1.*in_loc(2);
    dd_nodal_s_basis(10,4) = -1.*in_loc(2);
    dd_nodal_s_basis(11,4) = 0.;
    dd_nodal_s_basis(12,4) = -0.5*(1. + in_loc(0))*in_loc(2) - 0.5*(in_loc(0) + in_loc(1))*in_loc(2) - 0.5*(1. + in_loc(0))*(1. + in_loc(2)) - 0.5*(in_loc(0) + in_loc(1))*(1. + in_loc(2));
    dd_nodal_s_basis(13,4) = 0.5*(1. + in_loc(1))*in_loc(2) + 0.5*(1. + in_loc(1))*(1. + in_loc(2));
    dd_nodal_s_basis(14,4) = -0.5*(1. + in_loc(1))*in_loc(2) - 0.5*(1. + in_loc(1))*(1. + in_loc(2));

    dd_nodal_s_basis(0 ,5) = 0.25*(in_loc(0) + in_loc(1))*(-1. + in_loc(2)) + 0.25*(1. + in_loc(0) + in_loc(1))*(-1. + in_loc(2)) + 0.25*(in_loc(0) + in_loc(1))*in_loc(2) + 0.25*(1. + in_loc(0) + in_loc(1))*in_loc(2);
    dd_nodal_s_basis(1 ,5) = 0.;
    dd_nodal_s_basis(2 ,5) = 0.25*in_loc(1)*(-1. + in_loc(2)) + 0.25*(1. + in_loc(1))*(-1. + in_loc(2)) + 0.25*in_loc(1)*in_loc(2) + 0.25*(1. + in_loc(1))*in_loc(2);
    dd_nodal_s_basis(3 ,5) = 0.25*(in_loc(0) + in_loc(1))*in_loc(2) + 0.25*(1. + in_loc(0) + in_loc(1))*in_loc(2) + 0.25*(in_loc(0) + in_loc(1))*(1. + in_loc(2)) + 0.25*(1. + in_loc(0) + in_loc(1))*(1. + in_loc(2));
    dd_nodal_s_basis(4 ,5) = 0.;
    dd_nodal_s_basis(5 ,5) = 0.25*in_loc(1)*in_loc(2) + 0.25*(1. + in_loc(1))*in_loc(2) + 0.25*in_loc(1)*(1. + in_loc(2)) + 0.25*(1. + in_loc(1))*(1. + in_loc(2));
    dd_nodal_s_basis(6 ,5) = -0.5*(1. + in_loc(0))*(-1. + in_loc(2)) - 0.5*(1. + in_loc(0))*in_loc(2);
    dd_nodal_s_basis(7 ,5) = 0.5*(1. + in_loc(0))*(-1. + in_loc(2)) + 0.5*(1. + in_loc(0))*in_loc(2);
    dd_nodal_s_basis(8 ,5) = -0.5*(1. + in_loc(1))*(-1. + in_loc(2)) - 0.5*(in_loc(0) + in_loc(1))*(-1. + in_loc(2)) - 0.5*(1. + in_loc(1))*in_loc(2) - 0.5*(in_loc(0) + in_loc(1))*in_loc(2);
    dd_nodal_s_basis(9 ,5) = 1.*in_loc(2);
    dd_nodal_s_basis(10,5) = 0.;
    dd_nodal_s_basis(11,5) = -1.*in_loc(2);
    dd_nodal_s_basis(12,5) = -0.5*(1. + in_loc(0))*in_loc(2) - 0.5*(1. + in_loc(0))*(1. + in_loc(2));
    dd_nodal_s_basis(13,5) = 0.5*(1. + in_loc(0))*in_loc(2) + 0.5*(1. + in_loc(0))*(1. + in_loc(2));
    dd_nodal_s_basis(14,5) = -0.5*(1. + in_loc(1))*in_loc(2) - 0.5*(in_loc(0) + in_loc(1))*in_loc(2) - 0.5*(1. + in_loc(1))*(1. + in_loc(2)) - 0.5*(in_loc(0) + in_loc(1))*(1. + in_loc(2));
  }
  else
  {
    FatalError("Shape order not implemented yet, exiting");
  }

}

void eles_pris::fill_opp_3(array<double>& opp_3)
{

  array<double> loc(3);  
  array<double> opp_3_tri(n_upts_tri,3*(order+1));
  get_opp_3_tri(opp_3_tri,loc_upts_pri_tri,loc_1d_fpts,vandermonde_tri, inv_vandermonde_tri,n_upts_tri,order,run_input.c_tri,run_input.vcjh_scheme_tri);

  // Compute value of eta
  double eta;
  if (run_input.vcjh_scheme_pri_1d == 0)
    eta = run_input.eta_pri;
  else
    eta = compute_eta(run_input.vcjh_scheme_pri_1d,order);

  for (int upt=0;upt<n_upts_per_ele;upt++)
  {
    loc(0)=loc_upts(0,upt);
    loc(1)=loc_upts(1,upt);
    loc(2)=loc_upts(2,upt);

    int upt_1d = upt/n_upts_tri;
    int upt_tri = upt%n_upts_tri;

    for (int in_index=0;in_index<n_fpts_per_ele;in_index++)
    {
      // Face 0
      if (in_index < n_fpts_per_inter(0))
      {
        int face_fpt = in_index;
        if (face0_map(face_fpt)==upt_tri)
          opp_3(upt,in_index)= -eval_d_vcjh_1d(loc(2),0,order,eta);
        else 
          opp_3(upt,in_index)= 0.;
      }
      // Face 1
      else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1))
      {
        int face_fpt = in_index-n_fpts_per_inter(0);
        if (face_fpt == upt_tri)
          opp_3(upt,in_index)= eval_d_vcjh_1d(loc(2),1,order,eta);
        else
          opp_3(upt,in_index)= 0.;
      }
      // face 2
      else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2))
      {
        int face_fpt = in_index-2*n_fpts_per_inter(0);
        int edge_fpt = face_fpt%(order+1);
        int edge = 0;

        if ( face_fpt/(order+1)==upt_1d)
          //opp_3(upt,in_index)= eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
          opp_3(upt,in_index)= opp_3_tri(upt_tri,edge*(order+1)+edge_fpt);
        else
          opp_3(upt,in_index)= 0.;
      }
      // face 3
      else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3))
      {
        int face_fpt = in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2);
        int edge_fpt = face_fpt%(order+1);
        int edge = 1;

        if (face_fpt/(order+1) == upt_1d)
          //opp_3(upt,in_index)= eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
          opp_3(upt,in_index)= opp_3_tri(upt_tri,edge*(order+1)+edge_fpt);
        else 
          opp_3(upt,in_index)= 0.;
      }
      // face 4
      else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3)+n_fpts_per_inter(4))
      {
        int face_fpt = (in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2)-n_fpts_per_inter(3));
        int edge_fpt = face_fpt%(order+1);
        int edge = 2;

        if (face_fpt/(order+1) == upt_1d)
          //opp_3(upt,in_index)= eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
          opp_3(upt,in_index)= opp_3_tri(upt_tri,edge*(order+1)+edge_fpt);
        else 
          opp_3(upt,in_index)= 0.;
      }
    }
  }


}

// evaluate divergence of vcjh basis

double eles_pris::eval_div_vcjh_basis(int in_index, array<double>& loc)
{
  double div_vcjh_basis;
  double tol = 1e-12;
  double eta;

  // Check that loc is at one of the solution points, otherwise procedure doesn't work
  int flag = 1;
  int upt;
  for (int i=0;i<n_upts_per_ele;i++) {
    if (   abs(loc(0)-loc_upts(0,i)) < tol
        && abs(loc(1)-loc_upts(1,i)) < tol
        && abs(loc(2)-loc_upts(2,i)) < tol) {
      flag = 0;
      upt = i;
      break;
    }
  }
  if (flag==1) FatalError("eval_div_vcjh_basis is not at solution point, exiting");

  int upt_1d = upt/n_upts_tri;
  int upt_tri = upt%n_upts_tri;

  // Compute value of eta
  if (run_input.vcjh_scheme_pri_1d == 0)
    eta = run_input.eta_pri;
  else
    eta = compute_eta(run_input.vcjh_scheme_pri_1d,order);

  // Compute value of c_tri
  double c_tri =  0.; // HACK

  // Face 0
  if (in_index < n_fpts_per_inter(0))
  {
    int face_fpt = in_index;
    if (face0_map(face_fpt)==upt_tri)
      div_vcjh_basis = -eval_d_vcjh_1d(loc(2),0,order,eta);
    else 
      div_vcjh_basis = 0.;
  }
  // Face 1
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1))
  {
    int face_fpt = in_index-n_fpts_per_inter(0);
    if (face_fpt == upt_tri)
      div_vcjh_basis = eval_d_vcjh_1d(loc(2),1,order,eta);
    else
      div_vcjh_basis = 0.;
  }
  // face 2
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2))
  {
    int face_fpt = in_index-2*n_fpts_per_inter(0);
    int edge_fpt = face_fpt%(order+1);
    int edge = 0;

    if ( face_fpt/(order+1)==upt_1d)
      div_vcjh_basis = eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
    else
      div_vcjh_basis = 0.;
  }
  // face 3
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3))
  {
    int face_fpt = in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2);
    int edge_fpt = face_fpt%(order+1);
    int edge = 1;

    if (face_fpt/(order+1) == upt_1d)
      div_vcjh_basis = eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
    else 
      div_vcjh_basis = 0.;
  }
  // face 4
  else if (in_index < n_fpts_per_inter(0)+n_fpts_per_inter(1)+n_fpts_per_inter(2)+n_fpts_per_inter(3)+n_fpts_per_inter(4))
  {
    int face_fpt = (in_index-2*n_fpts_per_inter(0)-n_fpts_per_inter(2)-n_fpts_per_inter(3));
    int edge_fpt = face_fpt%(order+1);
    int edge = 2;

    if (face_fpt/(order+1) == upt_1d)
      div_vcjh_basis = eval_div_dg_tri(loc,edge,edge_fpt,order,loc_upts_pri_1d);
    else 
      div_vcjh_basis = 0.;
  }
 
  return div_vcjh_basis; 
}

int eles_pris::face0_map(int index)
{

  int k;
  for(int j=0;j<(order+1);j++)
  {
    for (int i=0;i<(order+1)-j;i++)
    {
      k= j*(order+1) -(j-1)*j/2+i;
      if (k==index)
      {
        return (i*(order+1) - (i-1)*i/2+j);
      }
    }
  }
  cout << "Should not be here in face0_map, exiting" << endl;
  exit(1);
}


/*! Calculate element volume */
double eles_pris::calc_ele_vol(double& detjac)
{
	
}

