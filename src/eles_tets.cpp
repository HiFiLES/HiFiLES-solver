/*!
 * \file eles_tets.cpp
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

#include "../include/global.h"
#include "../include/eles.h"
#include "../include/eles_tets.h"
#include "../include/array.h"
#include "../include/funcs.h"
#include "../include/error.h"
#include "../include/cubature_tri.h"
#include "../include/cubature_tet.h"

using namespace std;

// #### constructors ####

// default constructor

eles_tets::eles_tets()
{	
}

// #### methods ####


void eles_tets::setup_ele_type_specific(int in_run_type)
{
#ifndef _MPI
  cout << "Initializing tets" << endl;
#endif

	ele_type=2;
	n_dims=3;

  if (run_input.equation==0)
	  n_fields=5;
  else if (run_input.equation==1)
    n_fields=1;
  else 
    FatalError("Equation not supported");

	n_inters_per_ele=4;

	n_upts_per_ele=(order+3)*(order+2)*(order+1)/6;
	upts_type=run_input.upts_type_tet;
	set_loc_upts();
  set_vandermonde();

	n_ppts_per_ele=(p_res+2)*(p_res+1)*p_res/6;
  n_peles_per_ele = (p_res-1)*(p_res)*(p_res+1)/6 + 4*(p_res-2)*(p_res-1)*(p_res)/6 +(p_res-3)*(p_res-2)*(p_res-1)/6;

	set_loc_ppts();
	set_opp_p();

  set_inters_cubpts();
  set_volume_cubpts();
  set_opp_volume_cubpts();

  if (in_run_type==0)
  {
	  n_fpts_per_inter.setup(4);

	  n_fpts_per_inter(0)=(order+2)*(order+1)/2;
	  n_fpts_per_inter(1)=(order+2)*(order+1)/2;
	  n_fpts_per_inter(2)=(order+2)*(order+1)/2;
	  n_fpts_per_inter(3)=(order+2)*(order+1)/2;

	  n_fpts_per_ele=n_inters_per_ele*(order+2)*(order+1)/2;

    fpts_type=run_input.fpts_type_tet;	

	  set_tloc_fpts();

	  //set_loc_spts();
	  
	  set_tnorm_fpts();
	  
	  set_opp_0(run_input.sparse_tet);
	  set_opp_1(run_input.sparse_tet);
	  set_opp_2(run_input.sparse_tet);
	  set_opp_3(run_input.sparse_tet);
	  
	  if(viscous)
	  {
	  	set_opp_4(run_input.sparse_tet);
	  	set_opp_5(run_input.sparse_tet);
	  	set_opp_6(run_input.sparse_tet);
	  
	  	temp_grad_u.setup(n_fields,n_dims);
			if(run_input.LES)
			{
				temp_sgsf.setup(n_fields,n_dims);
				// Compute tri filter matrix
				compute_filter_upts();
			}
	  }
	  
	  
	  temp_u.setup(n_fields);
	  temp_f.setup(n_fields,n_dims);
  //}
  //else
  //{

    if (viscous==1)
    {
	  	set_opp_4(run_input.sparse_tet);
    }

    n_verts_per_ele = 4;
    n_edges_per_ele = 6; 

    n_ppts_per_edge = p_res-2;

    // Number of plot points per face, excluding points on vertices or edges
    n_ppts_per_face.setup(n_inters_per_ele);
    for (int i=0;i<n_inters_per_ele;i++)
      n_ppts_per_face(i) = (p_res-3)*(p_res-2)/2;

    n_ppts_per_face2.setup(n_inters_per_ele);
    for (int i=0;i<n_inters_per_ele;i++)
      n_ppts_per_face2(i) = (p_res+1)*(p_res)/2;

    max_n_ppts_per_face = n_ppts_per_face(0);

    // Number of plot points not on faces, edges or vertices
    n_interior_ppts = n_ppts_per_ele-4-4*n_ppts_per_face(0)-6*n_ppts_per_edge; 

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

void eles_tets::create_map_ppt(void)
{
  int index;
  int vert_ppt_count = 0;
  int interior_ppt_count = 0;

  array<int> edge_ppt_count(n_edges_per_ele);
  array<int> face_ppt_count(n_inters_per_ele);
  array<int> face2_ppt_count(n_inters_per_ele);
  for (int i=0;i<n_edges_per_ele;i++)
    edge_ppt_count(i)=0;
  for (int i=0;i<n_inters_per_ele;i++) {
    face_ppt_count(i)=0;
    face2_ppt_count(i)=0;
  }

	for(int k=0;k<p_res;k++)
	{
	for(int j=0;j<p_res-k;j++)
	{
	for(int i=0;i<p_res-k-j;i++)
	{

		index = (p_res)*(p_res+1)*(p_res+2)/6 -
			(p_res-k)*(p_res+1-k)*(p_res+2-k)/6 +
			j*(p_res-k)-(j-1)*j/2
			+ i;
    //cout << "------------" << endl;
    //cout << "index=" << index << endl;
    //cout << "x=" << loc_ppts(0,index) << "y=" << loc_ppts(1,index) << "z=" << loc_ppts(2,index) << endl;
    if (i==p_res-1 || j==p_res-1 || k==p_res-1 || (i==0 && j==0 && k==0))
    {
      vert_to_ppt(vert_ppt_count++)=index;
      //cout << "vert" << endl;
    }
    else if (j==0 && k==0) {
      edge_ppt_to_ppt(0,edge_ppt_count(0)++) = index;
      //cout << "edge 0" << endl;
    }
    else if (i==0 && k==0) {
      edge_ppt_to_ppt(1,edge_ppt_count(1)++) = index;
      //cout << "edge 1" << endl;
    }
    else if (i==0 && j==0) {
      edge_ppt_to_ppt(2,edge_ppt_count(2)++) = index;
      //cout << "edge 2" << endl;
    }
    else if (j==0 && i==(p_res-k-1)) {
      edge_ppt_to_ppt(3,edge_ppt_count(3)++) = index;
      //cout << "edge 3" << endl;
    }
    else if (k==0 && i==(p_res-j-1)) {
      edge_ppt_to_ppt(4,edge_ppt_count(4)++) = index;
      //cout << "edge 4" << endl;
    }
    else if (i==0 && j==(p_res-k-1)) {
      edge_ppt_to_ppt(5,edge_ppt_count(5)++) = index;
      //cout << "edge 5" << endl;
    }
    else if (i==p_res-k-j-1) {
      face_ppt_to_ppt(0)(face_ppt_count(0)++) = index;
      //cout << "face 0" << endl;
    }
    else if (i==0) {
      face_ppt_to_ppt(1)(face_ppt_count(1)++) = index;
      //cout << "face 1" << endl;
    }
    else if (j==0) {
      face_ppt_to_ppt(2)(face_ppt_count(2)++) = index;
      //cout << "face 2" << endl;
    }
    else if (k==0) {
      face_ppt_to_ppt(3)(face_ppt_count(3)++) = index;
      //cout << "face 3" << endl;
    }
    else
      interior_ppt_to_ppt(interior_ppt_count++) = index;


    if (i==p_res-k-j-1) {
      face2_ppt_to_ppt(0)(face2_ppt_count(0)++) = index;
    }
    if (i==0) {
      face2_ppt_to_ppt(1)(face2_ppt_count(1)++) = index;
    }
    if (j==0) {
      face2_ppt_to_ppt(2)(face2_ppt_count(2)++) = index;
    }
    if (k==0) {
      face2_ppt_to_ppt(3)(face2_ppt_count(3)++) = index;
    }



	}
	}
	}
}


void eles_tets::set_connectivity_plot()
{
  int vertex_0,vertex_1,vertex_2,vertex_3,vertex_4,vertex_5;
  int count=0;
  int temp = (p_res)*(p_res+1)*(p_res+2)/6;	

	/*! Loop over the plot sub-elements. */
	/*! For tets there are 3 sets with different orientations */
	/*! First set */
  for(int k=0;k<p_res-1;++k){
    for(int j=0;j<p_res-1-k;++j){
      for(int i=0;i<p_res-1-k-j;++i){
	  
        vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i;
        vertex_1 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i + 1;
        vertex_2 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i;
        vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + j*(p_res-(k+1)) - (j-1)*j/2 + i;

        connectivity_plot(0,count) = vertex_0;
        connectivity_plot(1,count) = vertex_1;
        connectivity_plot(2,count) = vertex_2;
        connectivity_plot(3,count) = vertex_3;
        count++;
      }
    }
  }

	/*! Second set */
  for(int k=0;k<p_res-2;++k){
    for(int j=0;j<p_res-2-k;++j){
      for(int i=0;i<p_res-2-k-j;++i){
        vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + j*(p_res-k) - (j-1)*j/2 + i + 1;
        vertex_1 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i + 1;
        vertex_2 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + j*(p_res-(k+1)) - (j-1)*j/2 + i + 1;
        vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + (i-1) + 1;
        vertex_4 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j)*(p_res-(k+1)) - (j-1)*(j)/2 + (i-1) + 1;
        vertex_5 = temp - (p_res-(k))*(p_res+1-(k))*(p_res+2-(k))/6 + (j+1)*(p_res-(k)) - (j)*(j+1)/2 + (i-1) + 1;


        connectivity_plot(0,count) = vertex_0;
        connectivity_plot(1,count) = vertex_2;
        connectivity_plot(2,count) = vertex_1;
        connectivity_plot(3,count) = vertex_4;
        count++;

        connectivity_plot(0,count) = vertex_2;
        connectivity_plot(1,count) = vertex_3;
        connectivity_plot(2,count) = vertex_1;
        connectivity_plot(3,count) = vertex_4;
        count++;

        connectivity_plot(0,count) = vertex_5;
        connectivity_plot(1,count) = vertex_1;
        connectivity_plot(2,count) = vertex_3;
        connectivity_plot(3,count) = vertex_4;
        count++;

        connectivity_plot(0,count) = vertex_0;
        connectivity_plot(1,count) = vertex_4;
        connectivity_plot(2,count) = vertex_1;
        connectivity_plot(3,count) = vertex_5;
        count++;
      }
    }
  }

	/*! Third set */
  for(int k=0;k<p_res-3;++k){
    for(int j=0;j<p_res-3-k;++j){
      for(int i=0;i<p_res-3-k-j;++i){

        vertex_0 = temp - (p_res-k)*(p_res+1-k)*(p_res+2-k)/6 + (j+1)*(p_res-k) - (j)*(j+1)/2 + i + 1;
        vertex_1 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j)*(p_res-(k+1)) - (j-1)*(j)/2 + i + 1;
        vertex_2 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + i ;
        vertex_3 = temp - (p_res-(k+1))*(p_res+1-(k+1))*(p_res+2-(k+1))/6 + (j+1)*(p_res-(k+1)) - (j)*(j+1)/2 + i + 1;

        connectivity_plot(0,count) = vertex_0;
        connectivity_plot(1,count) = vertex_1;
        connectivity_plot(2,count) = vertex_2;
        connectivity_plot(3,count) = vertex_3;
        count++;
      }
    }
  }
}



// set location of solution points in standard element

void eles_tets::set_loc_upts(void)
{
	int get_order=order;
	loc_upts.setup(n_dims,n_upts_per_ele);

	if (upts_type==0) // internal points (good quadrature points)
	{
		array<double> loc_inter_pts(n_upts_per_ele,3);
		#include "../data/loc_tet_inter_pts.dat"

    for (int i=0;i<n_upts_per_ele;i++)
    {
      loc_upts(0,i) = loc_inter_pts(i,0);
      loc_upts(1,i) = loc_inter_pts(i,1);
      loc_upts(2,i) = loc_inter_pts(i,2);
    }  

	}
	else if (upts_type==1) // alpha optimized
	{
		array<double> loc_alpha_pts(n_upts_per_ele,3);
		#include "../data/loc_tet_alpha_pts.dat"

    for (int i=0;i<n_upts_per_ele;i++)
    {
      loc_upts(0,i) = loc_alpha_pts(i,0);
      loc_upts(1,i) = loc_alpha_pts(i,1);
      loc_upts(2,i) = loc_alpha_pts(i,2);
    }  
	}
	else
	{
		cout << "Error: Unknown solution points location type...." << endl;
		exit(1);
	}

}

// set location of flux points in standard element

void eles_tets::set_tloc_fpts(void)
{

  int i,j,fpt;
  int get_order=order;
	tloc_fpts.setup(n_dims,n_fpts_per_ele);

	array<double> loc_tri_fpts(n_fpts_per_inter(0),2);

	if (fpts_type==0) // internal points
	{
		array<double> loc_inter_pts(n_fpts_per_inter(0),2);
		#include "../data/loc_tri_inter_pts.dat"
		loc_tri_fpts = loc_inter_pts;
	}
  else if(fpts_type==1) // alpha optimized
	{
		array<double> loc_alpha_pts(n_fpts_per_inter(0),2);
		#include "../data/loc_tri_alpha_pts.dat"
		loc_tri_fpts = loc_alpha_pts;
	}
	else
	{
    FatalError("Unknown tet fpts type");
	}	

  // Now map these points to 3D

  int i_tri_fpts, i_alpha;
	for(j=0;j<order+1;j++)
	{
		for (i=0;i<order+1-j;i++)
		{

	 		i_tri_fpts = j*(order+1) -(j-1)*j/2+i;
			i_alpha = j*(order+1) - (j-1)*j/2 + (order-j-i);
			tloc_fpts(0,i_tri_fpts) = loc_tri_fpts(i_alpha,0);

			i_alpha = i_tri_fpts;
			tloc_fpts(1,i_tri_fpts) = loc_tri_fpts(i_alpha,0);
			tloc_fpts(2,i_tri_fpts) = loc_tri_fpts(i_alpha,1);

			tloc_fpts(0,n_fpts_per_inter(0)+i_tri_fpts) = -1;
			tloc_fpts(1,n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,1);
			tloc_fpts(2,n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,0);

			tloc_fpts(0,2*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,0);
			tloc_fpts(1,2*n_fpts_per_inter(0)+i_tri_fpts) = -1;
			tloc_fpts(2,2*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,1);
	
			tloc_fpts(0,3*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,1);
			tloc_fpts(1,3*n_fpts_per_inter(0)+i_tri_fpts) = loc_tri_fpts(i_alpha,0);
			tloc_fpts(2,3*n_fpts_per_inter(0)+i_tri_fpts) = -1;
		}
	}	

}

// set location of plot points in standard element

void eles_tets::set_loc_ppts(void)
{
  loc_ppts.setup(n_dims,n_ppts_per_ele);
  int index;

	for(int k=0;k<p_res;k++)
	{
	for(int j=0;j<p_res-k;j++)
	{
	for(int i=0;i<p_res-k-j;i++)
	{

		index = (p_res)*(p_res+1)*(p_res+2)/6 -
			(p_res-k)*(p_res+1-k)*(p_res+2-k)/6 +
			j*(p_res-k)-(j-1)*j/2
			+ i;

		 loc_ppts(0,index) = -1.0+(2.0*i/(p_res-1));
		 loc_ppts(1,index) = -1.0+(2.0*j/(p_res-1));
		 loc_ppts(2,index) = -1.0+(2.0*k/(p_res-1));
	}
	}
	}
}

void eles_tets::set_inters_cubpts(void)
{


  n_cubpts_per_inter.setup(n_inters_per_ele);
  loc_inters_cubpts.setup(n_inters_per_ele);
  weight_inters_cubpts.setup(n_inters_per_ele);
  tnorm_inters_cubpts.setup(n_inters_per_ele);

  cubature_tri cub_tri(inters_cub_order);
  int n_cubpts_tri = cub_tri.get_n_pts();

  for (int i=0;i<n_inters_per_ele;i++)
    n_cubpts_per_inter(i) = n_cubpts_tri;

  for (int i=0;i<n_inters_per_ele;i++) {

    loc_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));
    weight_inters_cubpts(i).setup(n_cubpts_per_inter(i));
    tnorm_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));

    for (int j=0;j<n_cubpts_tri;j++) {

      if (i==0) {
	  	  loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
	  	  loc_inters_cubpts(i)(2,j)=-1.-cub_tri.get_r(j)-cub_tri.get_s(j);
      }
      else if (i==1) {
	  	  loc_inters_cubpts(i)(0,j)=-1;
	  	  loc_inters_cubpts(i)(1,j)=cub_tri.get_r(j);
	  	  loc_inters_cubpts(i)(2,j)=cub_tri.get_s(j);
      }
      else if (i==2) {
	  	  loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=-1.;
	  	  loc_inters_cubpts(i)(2,j)=cub_tri.get_s(j);
      }
      else if (i==3) {
	  	  loc_inters_cubpts(i)(0,j)=cub_tri.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=cub_tri.get_s(j);
	  	  loc_inters_cubpts(i)(2,j)=-1.;
      }

      weight_inters_cubpts(i)(j) = cub_tri.get_weight(j);

      if (i==0) {
	  	  tnorm_inters_cubpts(i)(0,j)= 1./sqrt(3.);
	  	  tnorm_inters_cubpts(i)(1,j)= 1./sqrt(3.);
	  	  tnorm_inters_cubpts(i)(2,j)= 1./sqrt(3.);
      }
      else if (i==1) {
	  	  tnorm_inters_cubpts(i)(0,j)= -1.0;
	  	  tnorm_inters_cubpts(i)(1,j)= 0.;
	  	  tnorm_inters_cubpts(i)(2,j)= 0.;
      }
      else if (i==2) {
	  	  tnorm_inters_cubpts(i)(0,j)= 0.;
	  	  tnorm_inters_cubpts(i)(1,j)= -1.;
	  	  tnorm_inters_cubpts(i)(2,j)= 0.;
      }
      else if (i==3) {
	  	  tnorm_inters_cubpts(i)(0,j)= 0.;
	  	  tnorm_inters_cubpts(i)(1,j)= 0.;
	  	  tnorm_inters_cubpts(i)(2,j)= -1.;
      }

    }
  }
  set_opp_inters_cubpts();

}


void eles_tets::set_volume_cubpts(void)
{
  cubature_tet cub_tet(volume_cub_order);
  int n_cubpts_tet = cub_tet.get_n_pts();
  n_cubpts_per_ele = n_cubpts_tet;

  loc_volume_cubpts.setup(n_dims,n_cubpts_tet);
  weight_volume_cubpts.setup(n_cubpts_tet);

  for (int i=0;i<n_cubpts_tet;i++)
  {
    loc_volume_cubpts(0,i) = cub_tet.get_r(i);
    loc_volume_cubpts(1,i) = cub_tet.get_s(i);
    loc_volume_cubpts(2,i) = cub_tet.get_t(i);

    weight_volume_cubpts(i) = cub_tet.get_weight(i);
  }
}


// Compute the surface jacobian determinant on a face
double eles_tets::compute_inter_detjac_inters_cubpts(int in_inter,array<double> d_pos)
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

  if (in_inter==0) // r=u,s=v,t=-u-v;
  {
    xu = xr-xt;
    yu = yr-yt;
    zu = zr-zt;

    xv = xs-xt;
    yv = ys-yt;
    zv = zs-zt;
    //cout << "xu=" << xu << "yu=" << yu << "zu=" << zu << endl;
    //cout << "xv=" << xv << "yv=" << yv << "zv=" << zv << endl;
  }
  else if (in_inter==1) // u=s, v=t
  {
    xu = xr;
    yu = yr;
    zu = zr;

    xv = xt;
    yv = yt;
    zv = zt;
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
  else if (in_inter==3) //u=r,v=s
  {
    xu = xr;
    yu = yr;
    zu = zr;

    xv = xs;
    yv = ys;
    zv = zs;
  }

  temp0 = yu*zv-zu*yv;
  temp1 = zu*xv-xu*zv;
  temp2 = xu*yv-yu*xv;

  output = sqrt(temp0*temp0+temp1*temp1+temp2*temp2);
  //cout << "temp0=" << temp0 << "temp1" << temp1 << "temp2" << temp2 << endl;
  //cout << "in_inter=" << in_inter << "output=" << output << endl;


  return output;
}

// set location of shape points in standard element
/*
void eles_tets::set_loc_spts(void)
{

// First order

	// Node 0 at (-1,-1,-1)
	// Node 1 at (1,-1,-1)
	// Node 2 at (-1,1,-1)
	// Node 3 at (-1,-1,1)
		
// 	z	
//	|      y
// 	      /
//  3
//  |   2
//  |  / 
//	| /   
//	|/        
//  0--------1    ----> x

// Second order

// Node 0 at (-1,-1,-1)
// Node 1 at (1,-1,-1)
// Node 2 at (-1,1,-1)
// Node 3 at (-1,-1,1)
// Node 4 at (0,-1,-1)
// Node 5 at (-1,0,-1)
// Node 6 at (-1,-1,0)
// Node 7 at (0,0,-1)
// Node 8 at (-1,0,0)
// Node 9 at (0,-1,0)
		
// 	z	
//	|      y
// 	      /
//  3
//  |   2
//  6  / \ 
//	| 5   7  
//	|/     \    
//  0---4----1    ----> x

}
*/

// set transformed normal at flux points

void eles_tets::set_tnorm_fpts(void)
{
  int i,j,fpt;
	tnorm_fpts.setup(n_dims,n_fpts_per_ele);

  for (i=0;i<n_inters_per_ele;i++)  
  {
	  for(j=0;j<(order+1)*(order+2)/2;j++)
	  {
      fpt = (order+1)*(order+2)/2*i+j;

      if (i==0) {
	  	  tnorm_fpts(0,fpt)= 1./sqrt(3.);
	  	  tnorm_fpts(1,fpt)= 1./sqrt(3.);
	  	  tnorm_fpts(2,fpt)= 1./sqrt(3.);
      }
      else if (i==1) {
	  	  tnorm_fpts(0,fpt)= -1.0;
	  	  tnorm_fpts(1,fpt)= 0.;
	  	  tnorm_fpts(2,fpt)= 0.;
      }
      else if (i==2) {
	  	  tnorm_fpts(0,fpt)= 0.;
	  	  tnorm_fpts(1,fpt)= -1.0;
	  	  tnorm_fpts(2,fpt)= 0.;
      }
      else if (i==3) {
	  	  tnorm_fpts(0,fpt)= 0.;
	  	  tnorm_fpts(1,fpt)= 0.;
	  	  tnorm_fpts(2,fpt)= -1.0;
      }
	  }
  }
}

// Filtering operators for use in subgrid-scale modelling
void eles_tets::compute_filter_upts(void)
{
	printf("\nEntering filter computation function\n");
	int i,j,k,l,N,N2;
	double dlt, k_c, sum, vol, norm;
	N = n_upts_per_ele;

	array<double> X(n_dims,N);
	array<double> beta(N,N);
	array<double> B(N);

	filter_upts.setup(N,N);

	X = loc_upts;
	printf("\n3D solution point coordinates:\n");
	X.print();

	N2 = N/2;
	// If N is odd, round up N/2
	if(N % 2 != 0){N2 += 1;}
	// Cutoff wavenumber
	k_c = 1.0/run_input.filter_ratio;

	// Approx resolution in element (assumes uniform point spacing)
	dlt = 2.0/order;
	printf("\nN, N2, dlt, k_c:\n");
	cout << N << ", " << N2 << ", " << dlt << ", " << k_c << endl;

	// Normalised solution point separation: r = sqrt((x_a-x_b)^2 + (y_a-y_b)^2)
	for (i=0;i<N;i++)
		for (j=i;j<N;j++)
			beta(i,j) = sqrt(pow(X(0,i)-X(0,j),2) + pow(X(1,i)-X(1,j),2) + pow(X(2,i)-X(2,j),2))/dlt;
	for (i=0;i<N;i++)
		for (j=0;j<i;j++)
			beta(i,j) = beta(j,i);

	printf("\nNormalised solution point separation beta:\n");
	beta.print();

	if(run_input.filter_type==0) // Vasilyev filter
	{
		printf("Vasilyev filters not implemented for tris. Exiting.");
		exit(1);
	}
	else if(run_input.filter_type==1) // Discrete Gaussian filter
	{
		printf("\nBuilding discrete Gaussian filter\n");

		if(N != n_cubpts_per_ele)
		{
			cout<<"WARNING: Gaussian filter cannot be built for tets since n_upts_per_ele != n_cubpts_per_ele for any order. Exiting"<<endl;
			exit(1);
		}
	}
	else if(run_input.filter_type==2) // Modal coefficient filter
	{
		printf("\nBuilding modal filter\n");

		// Compute modal filter
		compute_modal_filter(filter_upts, vandermonde, inv_vandermonde, N);

		printf("\nFilter:\n");
		filter_upts.print();

	}
	else // Simple average for low order
	{
		printf("\nBuilding average filter\n");
		sum=0;
		for(i=0;i<N;i++)
		{
			for(j=0;j<N;j++)
			{
				filter_upts(i,j) = 1.0/N;
				sum+=1.0/N;
			}
		}
	}

	// Ensure symmetry and normalisation
	for(i=0;i<N2;i++)
	{
		for(j=0;j<N;j++)
		{
			filter_upts(i,j) = 0.5*filter_upts(i,j) + filter_upts(N-i-1,N-j-1);
			filter_upts(N-i-1,N-j-1) = filter_upts(i,j);
		}
	}
	printf("\nFilter after symmetrising:\n");
	filter_upts.print();
	for(i=0;i<N2;i++)
	{
		norm = 0.0;
		for(j=0;j<N;j++)
			norm += filter_upts(i,j); // or abs(filter_upts(i,j))?
		for(j=0;j<N;j++)
			filter_upts(i,j) /= norm;
		for(j=0;j<N;j++)
			filter_upts(N-i-1,N-j-1) = filter_upts(i,j);
	}
	sum = 0;
	for(i=0;i<N;i++)
		for(j=0;j<N;j++)
			sum+=filter_upts(i,j);

	printf("\nFilter after normalising:\n");
	filter_upts.print();
	cout<<"coeff sum " << sum << endl;

	printf("\nLeaving filter computation function\n");
}


//#### helper methods ####

// initialize the vandermonde matrix
void eles_tets::set_vandermonde(void)
{
  vandermonde.setup(n_upts_per_ele,n_upts_per_ele);

	// create the vandermonde matrix
	for (int i=0;i<n_upts_per_ele;i++)
		for (int j=0;j<n_upts_per_ele;j++) 
			vandermonde(i,j) = eval_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order);

	// Store its inverse
	inv_vandermonde = inv_array(vandermonde);
}

// initialize the vandermonde matrix
void eles_tets::set_vandermonde_restart()
{
  vandermonde.setup(n_upts_per_ele_rest,n_upts_per_ele_rest);

	// create the vandermonde matrix
	for (int i=0;i<n_upts_per_ele_rest;i++)
		for (int j=0;j<n_upts_per_ele_rest;j++) 
			vandermonde(i,j) = eval_dubiner_basis_3d(loc_upts_rest(0,i),loc_upts_rest(1,i),loc_upts_rest(2,i),j,order_rest);

	// Store its inverse
	inv_vandermonde_rest = inv_array(vandermonde);
}

int eles_tets::read_restart_info(ifstream& restart_file)
{
  string str;
  // Move to triangle element
  while(1) {
    getline(restart_file,str);
    if (str=="TETS") break;
    if (restart_file.eof()) return 0;
  }

  getline(restart_file,str);
  restart_file >> order_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_per_ele_rest;
  getline(restart_file,str);
  getline(restart_file,str);

  loc_upts_rest.setup(n_dims,n_upts_per_ele_rest);

  for (int i=0;i<n_upts_per_ele_rest;i++) {
    for (int j=0;j<n_dims;j++) {
      restart_file >> loc_upts_rest(j,i);
    }
  }

  set_vandermonde_restart();
  set_opp_r();

  return 1;
}

// write restart info
void eles_tets::write_restart_info(ofstream& restart_file)        
{
  restart_file << "TETS" << endl;

  restart_file << "Order" << endl;
  restart_file << order << endl;

  restart_file << "Number of solution points per element" << endl; 
  restart_file << n_upts_per_ele << endl;

  restart_file << "Location of solution points in tetrahedral elements" << endl;
  for (int i=0;i<n_upts_per_ele;i++) {
    for (int j=0;j<n_dims;j++) {
      restart_file << loc_upts(j,i) << " ";
    }
    restart_file << endl;
  }

}


// evaluate nodal basis

double eles_tets::eval_nodal_basis(int in_index, array<double> in_loc)
{
 	array<double> dubiner_basis_at_loc(n_upts_per_ele);
	double out_nodal_basis_at_loc;

	// First evaluate the normalized Dubiner basis at position in_loc	
	for (int i=0;i<n_upts_per_ele;i++) 
		dubiner_basis_at_loc(i) = eval_dubiner_basis_3d(in_loc(0),in_loc(1),in_loc(2),i,order);

	// From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
	out_nodal_basis_at_loc = 0.;
	for (int i=0;i<n_upts_per_ele;i++)
		out_nodal_basis_at_loc += inv_vandermonde(i,in_index)*dubiner_basis_at_loc(i);
	
	return out_nodal_basis_at_loc;	
}

// evaluate nodal basis

double eles_tets::eval_nodal_basis_restart(int in_index, array<double> in_loc)
{
 	array<double> dubiner_basis_at_loc(n_upts_per_ele_rest);
	double out_nodal_basis_at_loc;

	// First evaluate the normalized Dubiner basis at position in_loc	
	for (int i=0;i<n_upts_per_ele_rest;i++) 
		dubiner_basis_at_loc(i) = eval_dubiner_basis_3d(in_loc(0),in_loc(1),in_loc(2),i,order_rest);

	// From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
	out_nodal_basis_at_loc = 0.;
	for (int i=0;i<n_upts_per_ele_rest;i++)
		out_nodal_basis_at_loc += inv_vandermonde_rest(i,in_index)*dubiner_basis_at_loc(i);
	
	return out_nodal_basis_at_loc;	
}

// evaluate derivative of nodal basis

double eles_tets::eval_d_nodal_basis(int in_index, int in_cpnt, array<double> in_loc)
{
	array<double> d_dubiner_basis_at_loc(n_upts_per_ele);
	double out_d_nodal_basis_at_loc;

	// First evaluate the derivative normalized Dubiner basis at position in_loc	
	for (int i=0;i<n_upts_per_ele;i++) 
		 d_dubiner_basis_at_loc(i) = eval_grad_dubiner_basis_3d(in_loc(0),in_loc(1),in_loc(2),i,order,in_cpnt);

	// From Hesthaven, equation 3.3, V^T * l = P, or l = (V^-1)^T P
	out_d_nodal_basis_at_loc = 0.;
	for (int i=0;i<n_upts_per_ele;i++)
		out_d_nodal_basis_at_loc += inv_vandermonde(i,in_index)*d_dubiner_basis_at_loc(i);
	
	return out_d_nodal_basis_at_loc;	
}

// evaluate nodal shape basis

double eles_tets::eval_nodal_s_basis(int in_index, array<double> in_loc, int in_n_spts)
{
  double nodal_s_basis;

  if (in_n_spts==4) {
    if (in_index==0) 
      nodal_s_basis = -0.5*(in_loc(0)+in_loc(1)+in_loc(2)+1.);
    else if (in_index==1) 
      nodal_s_basis = 0.5*(in_loc(0)+1.);
    else if (in_index==2) 
      nodal_s_basis = 0.5*(in_loc(1)+1.);
    else if (in_index==3) 
      nodal_s_basis = 0.5*(in_loc(2)+1.);
  }
  else if (in_n_spts==10) {
    if (in_index==0) 
	    nodal_s_basis = (1./2.*(2.+in_loc(0)+in_loc(1)+in_loc(2)))*(in_loc(0)+1.+in_loc(1)+in_loc(2));
    else if (in_index==1) 
	    nodal_s_basis = (1./2.)*in_loc(0)*(in_loc(0)+1.);
    else if (in_index==2) 
	    nodal_s_basis = (1./2.)*in_loc(1)*(in_loc(1)+1.);
    else if (in_index==3) 
	    nodal_s_basis = (1./2.)*in_loc(2)*(in_loc(2)+1.);
    else if (in_index==4) 
	    nodal_s_basis = -(in_loc(0)+1.+in_loc(1)+in_loc(2))*(in_loc(0)+1.);
    else if (in_index==5) 
	    nodal_s_basis = -(in_loc(0)+1.+in_loc(1)+in_loc(2))*(in_loc(1)+1.);
    else if (in_index==6) 
	    nodal_s_basis = -(in_loc(0)+1.+in_loc(1)+in_loc(2))*(in_loc(2)+1.);
    else if (in_index==7) 
	    nodal_s_basis = (in_loc(0)+1.)*(in_loc(1)+1.);
    else if (in_index==8) 
	    nodal_s_basis = (in_loc(1)+1.)*(in_loc(2)+1.);
    else if (in_index==9) 
	    nodal_s_basis = (in_loc(2)+1.)*(in_loc(0)+1.);
  }
  else
  {
    cout << "Shape order not implemented yet, exiting" << endl;
    cout << "n_spt = " << in_n_spts << endl;
    exit(1);
  }
  return nodal_s_basis;
}

// evaluate derivative of nodal shape basis

void eles_tets::eval_d_nodal_s_basis(array<double> &d_nodal_s_basis, array<double> in_loc, int in_n_spts)
{

  if (in_n_spts==4) {
    d_nodal_s_basis(0,0) = -0.5;
    d_nodal_s_basis(1,0) = 0.5;
    d_nodal_s_basis(2,0) = 0.;
    d_nodal_s_basis(3,0) = 0.;

    d_nodal_s_basis(0,1) = -0.5;
    d_nodal_s_basis(1,1) = 0.;
    d_nodal_s_basis(2,1) = 0.5;
    d_nodal_s_basis(3,1) = 0.;

    d_nodal_s_basis(0,2) = -0.5;
    d_nodal_s_basis(1,2) = 0.;
    d_nodal_s_basis(2,2) = 0.;
    d_nodal_s_basis(3,2) = 0.5;
  }
  else if (in_n_spts==10) {

    d_nodal_s_basis(0,0) = 1.5+in_loc(0)+in_loc(1)+in_loc(2);
	  d_nodal_s_basis(1,0) = in_loc(0)+0.5;
	  d_nodal_s_basis(2,0) = 0.;
	  d_nodal_s_basis(3,0) = 0.;
	  d_nodal_s_basis(4,0) = -2.*in_loc(0)-2.-in_loc(1)-in_loc(2);
	  d_nodal_s_basis(5,0) = -in_loc(1)-1.;
	  d_nodal_s_basis(6,0) = -in_loc(2)-1.;
	  d_nodal_s_basis(7,0) = in_loc(1)+1.;
	  d_nodal_s_basis(8,0) = 0.;
	  d_nodal_s_basis(9,0) = in_loc(2)+1.;

	  d_nodal_s_basis(0,1) = 1.5+in_loc(0)+in_loc(1)+in_loc(2);
	  d_nodal_s_basis(1,1) = 0.;
	  d_nodal_s_basis(2,1) = in_loc(1)+0.5;
	  d_nodal_s_basis(3,1) = 0.;
	  d_nodal_s_basis(4,1) = -in_loc(0)-1.;
	  d_nodal_s_basis(5,1) = -2.*in_loc(1)-2.-in_loc(0)-in_loc(2);
	  d_nodal_s_basis(6,1) = -in_loc(2)-1.;
	  d_nodal_s_basis(7,1) = in_loc(0)+1.;
	  d_nodal_s_basis(8,1) = in_loc(2)+1.;
	  d_nodal_s_basis(9,1) = 0.;

	  d_nodal_s_basis(0,2)= 1.5+in_loc(0)+in_loc(1)+in_loc(2);
	  d_nodal_s_basis(1,2)= 0.;
	  d_nodal_s_basis(2,2)= 0.;
	  d_nodal_s_basis(3,2)= in_loc(2)+0.5;
	  d_nodal_s_basis(4,2)= -in_loc(0)-1.;
	  d_nodal_s_basis(5,2)= -in_loc(1)-1.;
	  d_nodal_s_basis(6,2)= -2.*in_loc(2)-2.-in_loc(0)-in_loc(1);
	  d_nodal_s_basis(7,2)= 0.;
	  d_nodal_s_basis(8,2)= in_loc(1)+1.;
	  d_nodal_s_basis(9,2)= in_loc(0)+1.;
  } 
  else
  {
    cout << "Shape order not implemented yet, exiting" << endl;
    cout << "n_spt = " << in_n_spts << endl;
    exit(1);
  }

}

// evaluate second derivative of nodal shape basis

void eles_tets::eval_dd_nodal_s_basis(array<double> &dd_nodal_s_basis, array<double> in_loc, int in_n_spts)
{

  if (in_n_spts==4) 
  {
    for (int i=0;i<in_n_spts;i++)
      for (int j=0;j<6;j++)
        dd_nodal_s_basis(i,j) = 0.;
  }
  else if (in_n_spts==10) {
    dd_nodal_s_basis(0,0) = 1.;
    dd_nodal_s_basis(1,0) = 1.;
    dd_nodal_s_basis(2,0) = 0.;
    dd_nodal_s_basis(3,0) = 0.;
    dd_nodal_s_basis(4,0) = -2.;
    dd_nodal_s_basis(5,0) = 0.;
    dd_nodal_s_basis(6,0) = 0.;
    dd_nodal_s_basis(7,0) = 0.;
    dd_nodal_s_basis(8,0) = 0.;
    dd_nodal_s_basis(9,0) = 0.;

    dd_nodal_s_basis(0,1) = 1.;
    dd_nodal_s_basis(1,1) = 0.;
    dd_nodal_s_basis(2,1) = 1.;
    dd_nodal_s_basis(3,1) = 0.;
    dd_nodal_s_basis(4,1) = 0.;
    dd_nodal_s_basis(5,1) = -2.;
    dd_nodal_s_basis(6,1) = 0.;
    dd_nodal_s_basis(7,1) = 0.;
    dd_nodal_s_basis(8,1) = 0.;
    dd_nodal_s_basis(9,1) = 0.;

    dd_nodal_s_basis(0,2) = 1.;
    dd_nodal_s_basis(1,2) = 0.;
    dd_nodal_s_basis(2,2) = 0.;
    dd_nodal_s_basis(3,2) = 1.;
    dd_nodal_s_basis(4,2) = 0.;
    dd_nodal_s_basis(5,2) = 0.;
    dd_nodal_s_basis(6,2) = -2.;
    dd_nodal_s_basis(7,2) = 0.;
    dd_nodal_s_basis(8,2) = 0.;
    dd_nodal_s_basis(9,2) = 0.;

    dd_nodal_s_basis(0,3) = 1.;
    dd_nodal_s_basis(1,3) = 0.;
    dd_nodal_s_basis(2,3) = 0.;
    dd_nodal_s_basis(3,3) = 0.;
    dd_nodal_s_basis(4,3) = -1.;
    dd_nodal_s_basis(5,3) = -1.;
    dd_nodal_s_basis(6,3) = 0.;
    dd_nodal_s_basis(7,3) = 1.;
    dd_nodal_s_basis(8,3) = 0.;
    dd_nodal_s_basis(9,3) = 0.;

    dd_nodal_s_basis(0,4) = 1.;
    dd_nodal_s_basis(1,4) = 0.;
    dd_nodal_s_basis(2,4) = 0.;
    dd_nodal_s_basis(3,4) = 0.;
    dd_nodal_s_basis(4,4) = -1.;
    dd_nodal_s_basis(5,4) = 0.;
    dd_nodal_s_basis(6,4) = -1.;
    dd_nodal_s_basis(7,4) = 0.;
    dd_nodal_s_basis(8,4) = 0.;
    dd_nodal_s_basis(9,4) = 1.;

    dd_nodal_s_basis(0,5) = 1.;
    dd_nodal_s_basis(1,5) = 0.;
    dd_nodal_s_basis(2,5) = 0.;
    dd_nodal_s_basis(3,5) = 0.;
    dd_nodal_s_basis(4,5) = 0.;
    dd_nodal_s_basis(5,5) = -1.;
    dd_nodal_s_basis(6,5) = -1.;
    dd_nodal_s_basis(7,5) = 0.;
    dd_nodal_s_basis(8,5) = 1.;
    dd_nodal_s_basis(9,5) = 0.;
  } 
  else {
    cout << "Shape order not implemented yet, exiting" << endl;
    cout << "n_spt = " << in_n_spts << endl;
    exit(1);
  }

}

void eles_tets::fill_opp_3(array<double>& opp_3)
{

  array <double> Filt(n_upts_per_ele,n_upts_per_ele);
  array <double> opp_3_dg(n_upts_per_ele, n_fpts_per_ele);
  array <double> m_temp(n_upts_per_ele, n_fpts_per_ele);

  compute_filt_matrix_tet(Filt,run_input.vcjh_scheme_tet, run_input.c_tet);

  get_opp_3_dg_tet(opp_3_dg);

  //cout << "opp_3_dg" << endl;
  //opp_3_dg.print();
  cout << endl;

  m_temp = mult_arrays(Filt,opp_3_dg);

  //cout << "opp_3_vcjh" << endl;
  //m_temp.print();
  cout << endl;
  opp_3 = array<double>(m_temp);
}


void eles_tets::get_opp_3_dg_tet(array<double>& opp_3_dg)
{
	int i,j,k;
	array<double> loc(n_dims);

	for(i=0;i<n_fpts_per_ele;i++)		
	{			
		for(j=0;j<n_upts_per_ele;j++)
		{
			for(k=0;k<n_dims;k++)
			{
				loc(k)=loc_upts(k,j);	
			}
			
			opp_3_dg(j,i)=eval_div_dg_tet(i,loc);
		}	
	}
}


// evaluate divergence of dg basis

double eles_tets::eval_div_dg_tet(int in_index, array<double>& loc)
{
  int face, face_fpt;  
  double r,s,t;
  double r_face,s_face,face_jac;
  double integral, edge_length, gdotn_at_cubpt;
  double div_vcjh_basis;

  array<double> mtemp_0(n_fpts_per_inter(0),n_fpts_per_inter(0));
  array<double> gdotn(n_fpts_per_inter(0),1);
  array<double> coeff_gdotn(n_fpts_per_inter(0),1);
  array<double> coeff_divg(n_upts_per_ele,1);

	face = in_index/n_fpts_per_inter(0);
	face_fpt = in_index-(n_fpts_per_inter(0)*face);

  // Compute the coefficients of vjch basis in Dubiner basis
  // i.e. sigma_i = integral of (h cdot n)*L_i over the edge


  // 1. Construct a polynomial for g*n over the edge
  //    Store the coefficient of g*n in terms of jacobi basis in coeff_gdotn
  for (int i=0;i<n_fpts_per_inter(0);i++) {

    if (i==face_fpt)
      gdotn(i,0) = 1.;
    else
      gdotn(i,0) = 0.;

    r = tloc_fpts(0,face*n_fpts_per_inter(0)+i);
    s = tloc_fpts(1,face*n_fpts_per_inter(0)+i);
    t = tloc_fpts(2,face*n_fpts_per_inter(0)+i);

    if (face==0) {
		  r_face = r;
		  s_face = t;
    }
    else if (face==1) {
		  r_face = t;
		  s_face = s;
    }
    else if (face==2) {
		  r_face = r;
		  s_face = t;
    }
    else if (face==3) {
		   r_face = s;
		   s_face = r;
    }

    for (int j=0;j<n_fpts_per_inter(0);j++) 
      mtemp_0(i,j) = eval_dubiner_basis_2d(r_face,s_face,j,order);
  }

  mtemp_0 = inv_array(mtemp_0);
  coeff_gdotn = mult_arrays(mtemp_0,gdotn);

  if (isnan(coeff_gdotn(0,0)))
    exit(1);

  // 2. Perform the edge integrals to obtain coefficients sigma_i
  for (int i=0;i<n_upts_per_ele;i++)
  {
	  cubature_tri cub2d(12); //TODO: Check if strong enough
    integral = 0.;

    for (int j=0;j<cub2d.get_n_pts();j++)
    {
      r_face = cub2d.get_r(j);
      s_face = cub2d.get_s(j);

		  // Get the position along the edge
      if (face==0) {
        face_jac = sqrt(3.);
        r = r_face;
        t = s_face;
        s = -1. -t -r;
      }
      else if (face==1) {
        face_jac = 1.;
		    r = -1.0;
		    s = s_face;
		    t = r_face;
      }
      else if (face==2) {
        face_jac = 1.;
		    r = r_face;
		    s = -1.0;
		    t = s_face;
      }
      else if (face==3) {
        face_jac = 1.;
			  r = s_face;
			  s = r_face;
			  t = -1.0;
      }

      gdotn_at_cubpt = 0.;
      for (int k=0;k<n_fpts_per_inter(0);k++)
        gdotn_at_cubpt += coeff_gdotn(k,0)*eval_dubiner_basis_2d(r_face,s_face,k,order);

		  integral += cub2d.get_weight(j)*eval_dubiner_basis_3d(r,s,t,i,order)*gdotn_at_cubpt;
    }
    coeff_divg(i,0) = integral*face_jac;
  }

  div_vcjh_basis = 0.;
  for (int i=0;i<n_upts_per_ele;i++)
    div_vcjh_basis += coeff_divg(i,0)*eval_dubiner_basis_3d(loc(0),loc(1),loc(2),i,order);

  return div_vcjh_basis;

}
  

void eles_tets::compute_filt_matrix_tet(array<double>& Filt, int vcjh_scheme_tet, double c_tet)
{

  // -----------------
  // VCJH Filter
  // -----------------
  int Ncoeff, indx;
  double ap;
  double c_plus;
  double c_plus_1d, c_sd_1d, c_hu_1d;
  
  Ncoeff = (order+1)*(order+2)/2;

  array <double> c_coeff(Ncoeff);
  array <double> mtemp_0, mtemp_1;
  array <double> K(n_upts_per_ele,n_upts_per_ele);
  array <double> Identity(n_upts_per_ele,n_upts_per_ele);
  array <double> Filt_dubiner(n_upts_per_ele,n_upts_per_ele);
  array <double> Dr(n_upts_per_ele,n_upts_per_ele);
  array <double> Ds(n_upts_per_ele,n_upts_per_ele);
  array <double> Dt(n_upts_per_ele,n_upts_per_ele);
  array <double> tempr(n_upts_per_ele,n_upts_per_ele);
  array <double> temps(n_upts_per_ele,n_upts_per_ele);
  array <double> tempt(n_upts_per_ele,n_upts_per_ele);
  array <double> D_high_order_trans(n_upts_per_ele,n_upts_per_ele);
  array <double> vandermonde_trans(n_upts_per_ele,n_upts_per_ele);

  array<array <double> > D_high_order;
  array<array <double> > D_T_D;
  
  // 1D prep
  ap = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
 
  c_sd_1d = (2*order)/((2*order+1)*(order+1)*(factorial(order)*ap)*(factorial(order)*ap));
  c_hu_1d = (2*(order+1))/((2*order+1)*order*(factorial(order)*ap)*(factorial(order)*ap));
  
  if(vcjh_scheme_tet>1)
  {
    //1D c+
    if (order==2)
      c_plus_1d = 0.206;
    else if (order==3)
      c_plus_1d = 3.80e-3;
    else if (order==4)
      c_plus_1d = 4.67e-5;
    else if (order==5)
      c_plus_1d = 4.28e-7;
    else
      FatalError("C_plus scheme not implemented for this order");

    //3D c+
    if (order==2)
      c_plus = 3.07e-2;
    else if (order==3)
      c_plus = 5.44e-4;
    else if (order==4)
      c_plus = 9.92e-6;
    else if (order==5)
      c_plus = 1.10e-7;
    else
        FatalError("C_plus scheme not implemented for this order");
  }

  
  if (vcjh_scheme_tet==0)
  {
    //c_tet set by user
  }
  else if (vcjh_scheme_tet==1) // DG
  {
    c_tet = 0.;
  }
  else if (vcjh_scheme_tet==2) // SD-like
  {
    c_tet = (c_sd_1d/c_plus_1d)*c_plus;
  }
  else if (vcjh_scheme_tet==3) // HU-like
  {
    c_tet = (c_hu_1d/c_plus_1d)*c_plus;
  }
  else if (vcjh_scheme_tet==4) // Cplus scheme
  {
    c_tet = c_plus;
  }
  else
    FatalError("VCJH tetrahedral scheme not recognized");
  
  cout << "c_tet " << c_tet << endl;

  run_input.c_tet = c_tet;

	// Evaluate the derivative normalized of Dubiner basis at position in_loc
	for (int i=0;i<n_upts_per_ele;i++) {
    for (int j=0;j<n_upts_per_ele;j++) {
      tempr(i,j) = eval_grad_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order,0);
      temps(i,j) = eval_grad_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order,1);
      tempt(i,j) = eval_grad_dubiner_basis_3d(loc_upts(0,i),loc_upts(1,i),loc_upts(2,i),j,order,2);
    }
  }

  //Convert to nodal derivatives
	Dr = mult_arrays(tempr,inv_vandermonde);
	Ds = mult_arrays(temps,inv_vandermonde);
	Dt = mult_arrays(tempt,inv_vandermonde);

  //cout << "Dr nodal" << endl;
  //Dr.print();
  //cout << endl;
  //cout << "Dr dubiner" << endl;
  //(Dr*vandermonde).print();
  //cout << endl;

  //cout << "Ds nodal" << endl;
  //Ds.print();
  //cout << endl;
  //cout << "Ds dubiner" << endl;
  //(Ds*vandermonde).print();
  //cout << endl;
  
  //cout << "Dt nodal" << endl;
  //Dt.print();
  //cout << endl;
  //cout << "Dt dubiner" << endl;
  //(Dt*vandermonde).print();
  //cout << endl;

	//Create identity matrix
	zero_array(Identity);

  for (int i=0;i<n_upts_per_ele;i++)
    Identity(i,i) = 1.;

	// Set array with trinomial coefficients multiplied by value of c
  indx = 0;
	for(int v=1; v<=(order+1); v++) {
    for(int w=1; w<=v; w++) {
		  c_coeff(indx) = (1./n_upts_per_ele)*(factorial(order)/( factorial(v-1)*factorial(order-(v-1)) ))*(factorial(v-1)/(factorial(w-1)*factorial((v-1)-(w-1))));
      //cout << "v=" << v << " w=" << w << " indx=" << indx << " coeff= " << c_coeff(indx) << endl;
      indx++;
    }
  }

  // Initialize K to zero
  zero_array(K);
  
  // Compute D_transpose*D
  D_high_order.setup(Ncoeff);
  D_T_D.setup(Ncoeff);

  indx = 0;
	for(int v=1; v<=(order+1); v++) 
  {
    for(int w=1; w<=v; w++)
    {
      D_high_order(indx) = array<double>(Identity);
    
      for (int i=1; i<=(order-v+1); i++)
        D_high_order(indx) = mult_arrays(D_high_order(indx),Dr);
      for (int i=1; i<=(v-w); i++)
        D_high_order(indx) = mult_arrays(D_high_order(indx),Ds);
      for (int i=1; i<=(w-1); i++)
        D_high_order(indx) = mult_arrays(D_high_order(indx),Dt);

      D_high_order_trans = transpose_array(D_high_order(indx));
      D_T_D(indx) = mult_arrays(D_high_order_trans,D_high_order(indx));

      //cout << "indx=" << indx << endl;
      //(D_high_order(indx)*vandermonde).print();

      //cout << endl;
      //mtemp_2 = vandermonde.get_trans()*D_T_D(indx)*vandermonde;
      //mtemp_2.print();
      //cout << endl;

      // Scale by c_coeff
      for (int i=0;i<n_upts_per_ele;i++) {
        for (int j=0;j<n_upts_per_ele;j++) {
          D_T_D(indx)(i,j) = c_tet*c_coeff(indx)*D_T_D(indx)(i,j); 
          K(i,j) += D_T_D(indx)(i,j); //without jacobian scaling
        }
      }
      indx++;
    }  
  }

  //mass matrix
  vandermonde_trans = transpose_array(vandermonde);
  mtemp_0 = mult_arrays(vandermonde,vandermonde_trans);

  //filter
  mtemp_1 = array<double>(mtemp_0);
  mtemp_1 = mult_arrays(mtemp_1,K);

  for (int i=0;i<n_upts_per_ele;i++)
    for (int j=0;j<n_upts_per_ele;j++)
      mtemp_1(i,j) += Identity(i,j);

  Filt = inv_array(mtemp_1);
  Filt_dubiner = mult_arrays(inv_vandermonde,Filt);
  Filt_dubiner = mult_arrays(Filt_dubiner,vandermonde);

  //cout << "Filt" << endl;
  //Filt.print();
  //cout << endl;
    
  //cout << "Filt_dubiner" << endl;
  //Filt_dubiner.print();


  /*
  // ------------------------
  // Diagonal filter
  // ------------------------
  matrix Filt_dubiner(n_upts_per_ele,n_upts_per_ele);
  int n_upts_lower = (order+1)*order/2;

  double frac;

  if (vcjh_scheme_tri==0)
  {
    double c_1d = c_tet*2*order;
    double cp = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
    double kappa = (2*order+1)/2*(factorial(order)*cp)*(factorial(order)*cp);
    frac = 1./ (1+c_1d*kappa);
  }
  else if (vcjh_scheme_tri==1) // DG
  {
    frac = 1.0;
  }
  else if (vcjh_scheme_tri==2) // SD-like
  {
    frac = (order+1.)/(2.*order+1);
  }
  else if (vcjh_scheme_tri==3) // HU-like
  {
    frac = (order)/(2.*order+1);
  }
  else if (vcjh_scheme_tri==4) // Cplus scheme
  {
    if (order==2)
      c_tet = 4.3e-2;
    else if (order==3)
      c_tet = 6.4e-4;
    else if (order==4)
      c_tet = 5.3e-6;
    else
      FatalError("C_plus scheme not implemented for this order");

    double c_1d = c_tet*2*order;
    double cp = 1./pow(2.0,order)*factorial(2*order)/ (factorial(order)*factorial(order));
    double kappa = (2*order+1)/2*(factorial(order)*cp)*(factorial(order)*cp);
    frac = 1./ (1+c_1d*kappa);
  }
  else
    FatalError("VCJH triangular scheme not recognized");

  cout << "Filtering fraction=" << frac << endl;

  for (int j=0;j<n_upts_per_ele;j++) {
    for (int k=0;k<n_upts_per_ele;k++) {
      if (j==k) {
        if (j < n_upts_lower)
          Filt_dubiner(j,k) = 1.;
        else
          Filt_dubiner(j,k) = frac;
      }
      else {
        Filt_dubiner(j,k) = 0.;
      }
    }
  }

  Filt = vandermonde_tri*Filt_dubiner*inv_vandermonde_tri;

  //cout << "Filt_dubiner_diag" << endl;
  //Filt_dubiner.print();

  cout << "Filt_diag" << endl;
  Filt.print();
  */

}

/*! Calculate element volume */
double eles_tets::calc_ele_vol(double& detjac)
{
	double vol;
	// Element volume = |Jacobian|*1/6*width*height*depth of reference element
	vol = detjac*8./6.;
	return vol;
}

