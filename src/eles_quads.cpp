/*!
 * \file eles_quads.cpp
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
#include "../include/eles_quads.h"
#include "../include/array.h"
#include "../include/funcs.h"
#include "../include/cubature_1d.h"
#include "../include/cubature_quad.h"

using namespace std;

// #### constructors ####

// default constructor

eles_quads::eles_quads()
{	

}

// #### methods ####

void eles_quads::setup_ele_type_specific(int in_run_type)
{

#ifndef _MPI
  cout << "Initializing quads" << endl;
#endif

	ele_type=1;
	n_dims=2;

  if (run_input.equation==0)
	  n_fields=4;
  else if (run_input.equation==1)
    n_fields=1;
  else 
    FatalError("Equation not supported");

	n_inters_per_ele=4;

	n_upts_per_ele=(order+1)*(order+1);
	upts_type=run_input.upts_type_quad;
	set_loc_1d_upts();
	set_loc_upts();
  set_vandermonde();

	n_ppts_per_ele=p_res*p_res;
	n_peles_per_ele=(p_res-1)*(p_res-1);
	set_loc_ppts();
	set_opp_p();

  set_inters_cubpts();
  set_volume_cubpts();
  set_opp_volume_cubpts();

  if (in_run_type==0)
  {
	  n_fpts_per_inter.setup(4);
	  
	  n_fpts_per_inter(0)=(order+1);
	  n_fpts_per_inter(1)=(order+1);
	  n_fpts_per_inter(2)=(order+1);
	  n_fpts_per_inter(3)=(order+1);
	  
	  n_fpts_per_ele=n_inters_per_ele*(order+1);
	  
	  set_tloc_fpts();

	  set_tnorm_fpts();
	  
	  set_opp_0(run_input.sparse_quad);
	  set_opp_1(run_input.sparse_quad);
	  set_opp_2(run_input.sparse_quad);
	  set_opp_3(run_input.sparse_quad);
	  
	  if(viscous)
	  {
	  	set_opp_4(run_input.sparse_quad);
	  	set_opp_5(run_input.sparse_quad);
	  	set_opp_6(run_input.sparse_quad);
	  	
	  	temp_grad_u.setup(n_fields,n_dims);
		if(run_input.LES)
		{
			temp_sgsf.setup(n_fields,n_dims);

			// Compute quad filter matrix
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
	  	set_opp_4(run_input.sparse_quad);
    }

    n_verts_per_ele = 4;
    n_edges_per_ele = 0; 
    n_ppts_per_edge = 0;

    // Number of plot points per face, excluding points on vertices or edges
    n_ppts_per_face.setup(n_inters_per_ele);
    for (int i=0;i<n_inters_per_ele;i++)
      n_ppts_per_face(i) = (p_res-2);

    n_ppts_per_face2.setup(n_inters_per_ele);
    for (int i=0;i<n_inters_per_ele;i++)
      n_ppts_per_face2(i) = (p_res);

    max_n_ppts_per_face = n_ppts_per_face(0);

    // Number of plot points not on faces, edges or vertices
    n_interior_ppts = n_ppts_per_ele-4-4*n_ppts_per_face(0); 

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

  }

}

void eles_quads::create_map_ppt(void)
{

	int i,j;
	int index;
  int vert_ppt_count = 0;
  int interior_ppt_count = 0;
  array<int> face_ppt_count(n_inters_per_ele);
  array<int> face2_ppt_count(n_inters_per_ele);

  for (i=0;i<n_inters_per_ele;i++) {
    face_ppt_count(i)=0;
    face2_ppt_count(i)=0;
  }

	for(j=0;j<p_res;j++)
	{
  for(i=0;i<p_res;i++)
	{
		index=i+(p_res*j);

		if (i==0 && j==0)
      vert_to_ppt(0)=index;
    else if (i==p_res-1 && j==0)
      vert_to_ppt(1)=index;
    else if (i==p_res-1 && j==p_res-1)
      vert_to_ppt(2)=index;
    else if (i==0 && j==p_res-1)
      vert_to_ppt(3)=index;
    else if (j==0) {
      face_ppt_to_ppt(0)(face_ppt_count(0)++) = index;
      //cout << "face 0" << endl;
    }
    else if (i==p_res-1) {
      face_ppt_to_ppt(1)(face_ppt_count(1)++) = index;
      //cout << "face 1" << endl;
    }
    else if (j==p_res-1) {
      face_ppt_to_ppt(2)(face_ppt_count(2)++) = index;
      //cout << "face 2" << endl;
    }
    else if (i==0) {
      face_ppt_to_ppt(3)(face_ppt_count(3)++) = index;
      //cout << "face 3" << endl;
    }
    else
      interior_ppt_to_ppt(interior_ppt_count++) = index;

    // Creating face 2 array
    if (j==0) {
      face2_ppt_to_ppt(0)(face2_ppt_count(0)++) = index;
      //cout << "face 0" << endl;
    }
    if (i==p_res-1) {
      face2_ppt_to_ppt(1)(face2_ppt_count(1)++) = index;
      //cout << "face 1" << endl;
    }
    if (j==p_res-1) {
      face2_ppt_to_ppt(2)(face2_ppt_count(2)++) = index;
      //cout << "face 2" << endl;
    }
    if (i==0) {
      face2_ppt_to_ppt(3)(face2_ppt_count(3)++) = index;
      //cout << "face 3" << endl;
    }



  }
	}

}

void eles_quads::set_connectivity_plot()
{
  int vertex_0,vertex_1,vertex_2,vertex_3;
  int count=0;

  for(int k=0;k<p_res-1;++k){
    for(int l=0;l<p_res-1;++l){

      vertex_0=l+(p_res*k);
      vertex_1=vertex_0+1;
      vertex_2=vertex_0+p_res+1;
      vertex_3=vertex_0+p_res;
  
      connectivity_plot(0) = vertex_0;
      connectivity_plot(1) = vertex_1;
      connectivity_plot(2) = vertex_2;
      connectivity_plot(3) = vertex_3;
      count++;
    }
  }
}



// set shape

/*
void eles_quads::set_shape(array<int> &in_n_spts_per_ele)
{
  //TODO: this is inefficient, copies by value
  n_spts_per_ele = in_n_spts_per_ele;

  // Computing maximum number of spts per ele for all elements
  int max_n_spts_per_ele = 0;
  for (int i=0;i<n_eles;i++) {
    if (n_spts_per_ele(i) > max_n_spts_per_ele) 
      max_n_spts_per_ele = n_spts_per_ele(i);
  }

	shape.setup(n_dims,max_n_spts_per_ele,n_eles);
}
*/


// set location of 1d solution points in standard interval (required for tensor product elements)

void eles_quads::set_loc_1d_upts(void)
{
	if(upts_type==0)
	{
		int get_order=order;
		
		array<double> loc_1d_gauss_pts(order+1);
		
		#include "../data/loc_1d_gauss_pts.dat"
	
		loc_1d_upts=loc_1d_gauss_pts;
	}
	else if(upts_type==1)
	{
		int get_order=order;
		
		array<double> loc_1d_gauss_lobatto_pts(order+1);
		
		#include "../data/loc_1d_gauss_lobatto_pts.dat"
		
		loc_1d_upts=loc_1d_gauss_lobatto_pts;
	}
	else
	{
		cout << "ERROR: Unknown solution point type.... " << endl;
	}
}

// set location of 1d shape points in standard interval (required for tensor product element)

void eles_quads::set_loc_1d_spts(array<double> &loc_1d_spts, int in_n_1d_spts)
{
	int i;
	
	for(i=0;i<in_n_1d_spts;i++)
	{
		loc_1d_spts(i)=-1.0+((2.0*i)/(1.0*(in_n_1d_spts-1)));
	}
}




// set location of solution points in standard element

void eles_quads::set_loc_upts(void)
{
	int i,j;
	
	int upt;

	loc_upts.setup(n_dims,n_upts_per_ele);
	
	for(i=0;i<(order+1);i++)
	{
		for(j=0;j<(order+1);j++)
		{
			upt=j+((order+1)*i);
			
			loc_upts(0,upt)=loc_1d_upts(j);
			loc_upts(1,upt)=loc_1d_upts(i);
		}
	}
}

// set location of flux points in standard element

void eles_quads::set_tloc_fpts(void)
{
	int i,j;
	
	int fpt;
	
	tloc_fpts.setup(n_dims,n_fpts_per_ele);
	
	for(i=0;i<n_inters_per_ele;i++)
	{
		for(j=0;j<(order+1);j++)
		{
			fpt=j+((order+1)*i);
				
			// for tensor product elements flux point location depends on solution point location
			
			if(i==0)
			{
				tloc_fpts(0,fpt)=loc_1d_upts(j);     
				tloc_fpts(1,fpt)=-1.0;     
			}
			else if(i==1)
			{
				tloc_fpts(0,fpt)=1.0;
				tloc_fpts(1,fpt)=loc_1d_upts(j);  
			}
			else if(i==2)
			{
				tloc_fpts(0,fpt)=loc_1d_upts(order-j);
				tloc_fpts(1,fpt)=1.0;   
			}
			else if(i==3)
			{
				tloc_fpts(0,fpt)=-1.0;
				tloc_fpts(1,fpt)=loc_1d_upts(order-j);   
			}
		}
	}
}

// set location and weights of interface cubature points in standard element

void eles_quads::set_inters_cubpts(void)
{

  n_cubpts_per_inter.setup(n_inters_per_ele);
  loc_inters_cubpts.setup(n_inters_per_ele);
  weight_inters_cubpts.setup(n_inters_per_ele);
  tnorm_inters_cubpts.setup(n_inters_per_ele);

  cubature_1d cub_1d(inters_cub_order);
  int n_cubpts_1d = cub_1d.get_n_pts();

  for (int i=0;i<n_inters_per_ele;i++)
    n_cubpts_per_inter(i) = n_cubpts_1d;

  for (int i=0;i<n_inters_per_ele;i++) {

    loc_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));
    weight_inters_cubpts(i).setup(n_cubpts_per_inter(i));
    tnorm_inters_cubpts(i).setup(n_dims,n_cubpts_per_inter(i));

    for (int j=0;j<n_cubpts_1d;j++) {

      if (i==0) {
	  	  loc_inters_cubpts(i)(0,j)=cub_1d.get_r(j);
	  	  loc_inters_cubpts(i)(1,j)=-1.;
      }
      else if (i==1) {
	  	  loc_inters_cubpts(i)(0,j)=1.;
	  	  loc_inters_cubpts(i)(1,j)=cub_1d.get_r(j);
      }
      else if (i==2) {
	  	  loc_inters_cubpts(i)(0,j)=cub_1d.get_r(n_cubpts_1d-j-1);
	  	  loc_inters_cubpts(i)(1,j)=1.0;
      }
      else if (i==3) {
	  	  loc_inters_cubpts(i)(0,j)=-1.0;
	  	  loc_inters_cubpts(i)(1,j)=cub_1d.get_r(n_cubpts_1d-j-1);
      }

      weight_inters_cubpts(i)(j) = cub_1d.get_weight(j);

      if (i==0) {
	  	  tnorm_inters_cubpts(i)(0,j)= 0.;
	  	  tnorm_inters_cubpts(i)(1,j)= -1.;
      }
      else if (i==1) {
	  	  tnorm_inters_cubpts(i)(0,j)= 1.;
	  	  tnorm_inters_cubpts(i)(1,j)= 0.;
      }
      else if (i==2) {
	  	  tnorm_inters_cubpts(i)(0,j)= 0.;
	  	  tnorm_inters_cubpts(i)(1,j)= 1.;
      }
      else if (i==3) {
	  	  tnorm_inters_cubpts(i)(0,j)= -1.;
	  	  tnorm_inters_cubpts(i)(1,j)= 0.;
      }

    }
  }

  set_opp_inters_cubpts();


}

void eles_quads::set_volume_cubpts(void)
{
  cubature_quad cub_quad(volume_cub_order);
  int n_cubpts_quad = cub_quad.get_n_pts();
  n_cubpts_per_ele = n_cubpts_quad;
  loc_volume_cubpts.setup(n_dims,n_cubpts_quad);
  weight_volume_cubpts.setup(n_cubpts_quad);

  for (int i=0;i<n_cubpts_quad;i++)
  {
    loc_volume_cubpts(0,i) = cub_quad.get_r(i);
    loc_volume_cubpts(1,i) = cub_quad.get_s(i);
		//cout << "x=" << loc_volume_cubpts(0,i) << endl;
		//cout << "y=" << loc_volume_cubpts(1,i) << endl;
    weight_volume_cubpts(i) = cub_quad.get_weight(i);
		//cout<<"wgt=" << weight_volume_cubpts(i) << endl;
  }
}

// Compute the surface jacobian determinant on a face
double eles_quads::compute_inter_detjac_inters_cubpts(int in_inter,array<double> d_pos)
{
  double output = 0.;
  double xr, xs;
  double yr, ys;

  xr = d_pos(0,0);
  xs = d_pos(0,1);

  yr = d_pos(1,0);
  ys = d_pos(1,1);

  if (in_inter==0) 
  {
     output = sqrt(xr*xr+yr*yr);
  }
  else if (in_inter==1) 
  {
     output = sqrt(xs*xs+ys*ys);
  }
  else if (in_inter==2)
  {
     output = sqrt(xr*xr+yr*yr);
  }
  else if (in_inter==3)
  {
     output = sqrt(xs*xs+ys*ys);
  }

  return output;
}

// set location of plot points in standard element

void eles_quads::set_loc_ppts(void)
{
	int i,j;
	
	int ppt;
	
	loc_ppts.setup(n_dims,n_ppts_per_ele);

	for(j=0;j<p_res;j++)
	{
		for(i=0;i<p_res;i++)
		{
			ppt=i+(p_res*j);
			
			loc_ppts(0,ppt)=-1.0+((2.0*i)/(1.0*(p_res-1)));
			loc_ppts(1,ppt)=-1.0+((2.0*j)/(1.0*(p_res-1)));
		}
	}
}

// set transformed normal at flux points

void eles_quads::set_tnorm_fpts(void)
{
	int i,j;
	
	int fpt;

	tnorm_fpts.setup(n_dims,n_fpts_per_ele);
	
	for(i=0;i<n_inters_per_ele;i++)
	{
		for(j=0;j<(order+1);j++)
		{
			fpt=j+((order+1)*i);
				
			if(i==0)
			{   
				tnorm_fpts(0,fpt)=0.0;     
				tnorm_fpts(1,fpt)=-1.0;	
			}
			else if(i==1)
			{
				tnorm_fpts(0,fpt)=1.0;  
				tnorm_fpts(1,fpt)=0.0;	
			}
			else if(i==2)
			{
				tnorm_fpts(0,fpt)=0.0;
				tnorm_fpts(1,fpt)=1.0;	
			}
			else if(i==3)
			{
				tnorm_fpts(0,fpt)=-1.0;
				tnorm_fpts(1,fpt)=0.0;		
			}
		}
	}	
}

// Filtering operators for use in subgrid-scale modelling
void eles_quads::compute_filter_upts(void)
{
	printf("\nEntering filter computation function\n");
	int i,j,k,l,N,N2;
	double dlt, k_c, sum, norm;
	N = order+1; // order is of basis polynomials NOT truncation error!

	array<double> X(N),B(N);
	array<double> beta(N,N);

	filter_upts_1D.setup(N,N);

	X = loc_1d_upts;
	printf("\n1D solution point coordinates:\n");
	X.print();

	N2 = N/2;
	if(N % 2 != 0){N2 += 1;}
	// Cutoff wavenumber
	k_c = 1.0/run_input.filter_ratio;
	// Approx resolution in element (assumes uniform point spacing)
	// Interval is [-1:1]
	dlt = 2.0/order;
	printf("\nN, N2, dlt, k_c:\n");
	cout << N << ", " << N2 << ", " << dlt << ", " << k_c << endl;

	// Normalised solution point separation
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
			beta(j,i) = (X(j)-X(i))/dlt;

	printf("\nNormalised solution point separation beta:\n");
	beta.print();

	// Build high-order-commuting Vasilyev filter
	// Only use high-order filters for high enough order
	if(run_input.filter_type==0 and N>=3)
	{
		printf("\nBuilding high-order-commuting Vasilyev filter\n");
		array<double> C(N);
		array<double> A(N,N);

		for (i=0;i<N;i++)
		{
			B(i) = 0.0;
			C(i) = 0.0;
			for (j=0;j<N;j++)
				A(i,j) = 0.0;

		}
		// Populate coefficient matrix
		for (i=0;i<N;i++)
		{
			// Populate constraints matrix
			B(0) = 1.0;
				// Gauss filter weights
			B(1) =  exp(-pow(pi,2)/24.0);
			B(2) = -B(1)*pow(pi,2)/k_c/12.0;

			if(N % 2 == 1 && i+1 == N2)
				B(2) = 0.0;

			for (j=0;j<N;j++)
			{
				A(j,0) = 1.0;
				A(j,1) = cos(pi*k_c*beta(j,i));
				A(j,2) = -beta(j,i)*pi*sin(pi*k_c*beta(j,i));

				if(N % 2 == 1 && i+1 == N2)
					A(j,2) = pow(beta(j,i),3);

			}

			// Enforce filter moments
			for (k=3;k<N;k++)
			{
				for (j=0;j<N;j++)
					A(j,k) = pow(beta(j,i),k+1);

				B(k) = 0.0;
			}
			
			// Solve linear system by inverting A using
			// Gauss-Jordan method
			gaussj(N,A,B);
			sum=0;
			for (j=0;j<N;j++)
				filter_upts_1D(j,i) = B(j);

		}
	}
	else if(run_input.filter_type==1) // Discrete Gaussian filter
	{
		printf("\nBuilding discrete Gaussian filter\n");
		int ctype,index;
		double k_R, k_L, coeff;
		double res_0, res_L, res_R;
		array<double> alpha(N);
	  cubature_1d cub_1d(inters_cub_order);
  	int n_cubpts_1d = cub_1d.get_n_pts();
		array<double> wf(n_cubpts_1d);

		if(N != n_cubpts_1d)
		{
			cout<<"WARNING: To build Gaussian filter, the interface cubature order must equal solution order, e.g. inters_cub_order=9 if order=4, inters_cub_order=7 if order=3, inters_cub_order=5 if order=2. Exiting"<<endl;
			cout<<"order: "<<order<<", inters_cub_order: "<<inters_cub_order<<endl;
			exit(1);
		}
		for (j=0;j<n_cubpts_1d;++j)
      wf(j) = cub_1d.get_weight(j);

		cout<<setprecision(10)<<"1D weights:"<<endl;
		wf.print();
		// Determine corrected filter width for skewed quadrature points
		// using iterative constraining procedure
		// ctype options: (-1) no constraining, (0) constrain moment, (1) constrain cutoff frequency
		ctype = -1;
		if(ctype>=0)
		{
			cout<<"Iterative cutoff procedure"<<endl;
			for (i=0;i<N2;i++)
			{
				for (j=0;j<N;j++)
					B(j) = beta(j,i);

				k_L = 0.1; k_R = 1.0;
				res_L = flt_res(N,wf,B,k_L,k_c,ctype);
				res_R = flt_res(N,wf,B,k_R,k_c,ctype);
				alpha(i) = 0.5*(k_L+k_R);
				for (j=0;j<1000;j++)
				{
					res_0 = flt_res(N,wf,B,k_c,alpha(i),ctype);
					if(abs(res_0)<1.e-12) return;
					if(res_0*res_L>0.0)
					{
						k_L = alpha(i);
						res_L = res_0;
					}
					else
					{
						k_R = alpha(i);
						res_R = res_0;
					}
					if(j==999)
					{
						alpha(i) = k_c;
						ctype = -1;
					}
				}
				alpha(N-i-1) = alpha(i);
			}
		}

		else if(ctype==-1) // no iterative constraining
			for (i=0;i<N;i++)
				alpha(i) = k_c;

		cout<<"alpha: "<<endl;
		alpha.print();

		sum = 0.0;
		for (i=0;i<N;i++)
		{
			norm = 0.0;
			for (j=0;j<N;j++)
			{
				filter_upts_1D(i,j) = wf(j)*exp(-6.0*pow(alpha(i)*beta(i,j),2));
				norm += filter_upts_1D(i,j);
			}
			for (j=0;j<N;j++)
			{
				filter_upts_1D(i,j) /= norm;
				sum += filter_upts_1D(i,j);
			}
		}
	}
	else if(run_input.filter_type==2) // Modal coefficient filter
	{
		printf("\nBuilding modal filter\n");

		// Compute modal filter
		compute_modal_filter(filter_upts_1D, vandermonde, inv_vandermonde, N);

		sum = 0;
		for(i=0;i<N;i++)
			for(j=0;j<N;j++)
				sum+=filter_upts_1D(i,j);
	}
	else // Simple average
	{
		printf("\nBuilding average filter\n");
		sum=0;
		for (i=0;i<N;i++)
		{
			for (j=0;j<N;j++)
			{
				filter_upts_1D(i,j) = 1.0/N;
				sum+=1.0/N;
			}
		}
	}

	printf("\n1D filter:\n");
	filter_upts_1D.print();
	cout<<"coeff sum " << sum << endl;

	// Build 2D filter on ideal (reference) element.
	int ii=0;
	filter_upts.setup(n_upts_per_ele,n_upts_per_ele);
	sum=0;
	for (i=0;i<N;i++)
		{
		for (j=0;j<N;j++)
			{
			int jj=0;
			for (k=0;k<N;k++)
				{
				for (l=0;l<N;l++)
					{
					filter_upts(ii,jj) = filter_upts_1D(j,l)*filter_upts_1D(i,k);
					sum+=filter_upts(ii,jj);
					++jj;
					}
				}
			++ii;
			}
		}
	printf("\n2D filter:\n");
	filter_upts.print();
	cout<<"2D coeff sum " << sum << endl;
	printf("\nLeaving filter computation function\n");
}


//#### helper methods ####

int eles_quads::read_restart_info(ifstream& restart_file)
{

  string str;
  // Move to triangle element
  while(1) {
    getline(restart_file,str);
    if (str=="QUADS") break;

    if (restart_file.eof()) return 0;
  }

  getline(restart_file,str);
  restart_file >> order_rest;
  getline(restart_file,str);
  getline(restart_file,str);
  restart_file >> n_upts_per_ele_rest;
  getline(restart_file,str);
  getline(restart_file,str);

  loc_1d_upts_rest.setup(order_rest+1);

  for (int i=0;i<order_rest+1;i++) 
      restart_file >> loc_1d_upts_rest(i);

  set_opp_r();

  return 1;
}

//
void eles_quads::write_restart_info(ofstream& restart_file)        
{
  restart_file << "QUADS" << endl;

  restart_file << "Order" << endl;
  restart_file << order << endl;

  restart_file << "Number of solution points per quadrilateral element" << endl; 
  restart_file << n_upts_per_ele << endl;

  restart_file << "Location of solution points in 1D" << endl;
  for (int i=0;i<order+1;i++) {
      restart_file << loc_1d_upts(i) << " ";
  }
  restart_file << endl;

}

// initialize the vandermonde matrix
void eles_quads::set_vandermonde(void)
{
  vandermonde.setup(order+1,order+1);

	for (int i=0;i<order+1;i++)
		for (int j=0;j<order+1;j++)
			vandermonde(i,j) = eval_legendre(loc_1d_upts(i),j);

	// Store its inverse
	inv_vandermonde = inv_array(vandermonde);
}

// evaluate nodal basis

double eles_quads::eval_nodal_basis(int in_index, array<double> in_loc)
{
	int i,j;
	
	double nodal_basis;

	i=in_index/(order+1);
	j=in_index-((order+1)*i);
	
	nodal_basis=eval_lagrange(in_loc(0),j,loc_1d_upts)*eval_lagrange(in_loc(1),i,loc_1d_upts);

	return nodal_basis;
}


// evaluate nodal basis using restart points

double eles_quads::eval_nodal_basis_restart(int in_index, array<double> in_loc)
{
	int i,j;
	
	double nodal_basis;

	i=in_index/(order_rest+1);
	j=in_index-((order_rest+1)*i);
	
	nodal_basis=eval_lagrange(in_loc(0),j,loc_1d_upts_rest)*eval_lagrange(in_loc(1),i,loc_1d_upts_rest);
 	
	return nodal_basis;
}

// evaluate derivative of nodal basis

double eles_quads::eval_d_nodal_basis(int in_index, int in_cpnt, array<double> in_loc)
{
	int i,j;
	
	double d_nodal_basis;
	
	i=in_index/(order+1);
	j=in_index-((order+1)*i);
	
	if(in_cpnt==0)
	{
		d_nodal_basis=eval_d_lagrange(in_loc(0),j,loc_1d_upts)*eval_lagrange(in_loc(1),i,loc_1d_upts);
	}
	else if(in_cpnt==1)
	{
		d_nodal_basis=eval_lagrange(in_loc(0),j,loc_1d_upts)*eval_d_lagrange(in_loc(1),i,loc_1d_upts);
	}
	else
	{
		cout << "ERROR: Invalid component requested ... " << endl;
	}
	
	return d_nodal_basis;
}

// evaluate nodal shape basis

double eles_quads::eval_nodal_s_basis(int in_index, array<double> in_loc, int in_n_spts)
{
	int i,j;
	double nodal_s_basis;

  if (is_perfect_square(in_n_spts))
  {
    int n_1d_spts = round(sqrt(1.0*in_n_spts));
    array<double> loc_1d_spts(n_1d_spts);
    set_loc_1d_spts(loc_1d_spts,n_1d_spts);

	  j=in_index/n_1d_spts;
	  i=in_index-(n_1d_spts*j);
	  nodal_s_basis=eval_lagrange(in_loc(0),i,loc_1d_spts)*eval_lagrange(in_loc(1),j,loc_1d_spts);
  }
  else if (in_n_spts==8)
  {
    if (in_index==0)
	    nodal_s_basis = -0.25*(1.-in_loc(0))*(1.-in_loc(1))*(1.+in_loc(0)+in_loc(1));
    else if (in_index==1)
	    nodal_s_basis = -0.25*(1.+in_loc(0))*(1.-in_loc(1))*(1.-in_loc(0)+in_loc(1));
    else if (in_index==2)
	    nodal_s_basis = -0.25*(1.+in_loc(0))*(1.+in_loc(1))*(1.-in_loc(0)-in_loc(1));
    else if (in_index==3)
    	nodal_s_basis = -0.25*(1.-in_loc(0))*(1.+in_loc(1))*(1.+in_loc(0)-in_loc(1));
    else if (in_index==4)
	    nodal_s_basis = 0.5*(1.-in_loc(0))*(1.+in_loc(0))*(1.-in_loc(1));
    else if (in_index==5)
	    nodal_s_basis = 0.5*(1.+in_loc(0))*(1.+in_loc(1))*(1.-in_loc(1));
    else if (in_index==6)
	    nodal_s_basis = 0.5*(1.-in_loc(0))*(1.+in_loc(0))*(1.+in_loc(1));
    else if (in_index==7)
	    nodal_s_basis = 0.5*(1.-in_loc(0))*(1.+in_loc(1))*(1.-in_loc(1));
  }
  else
  {
    cout << "Shape basis not implemented yet, exiting" << endl;
    exit(1);
  }
 	
	return nodal_s_basis;
}

// evaluate derivative of nodal shape basis

void eles_quads::eval_d_nodal_s_basis(array<double> &d_nodal_s_basis, array<double> in_loc, int in_n_spts)
{
	int i,j;

  if (is_perfect_square(in_n_spts))
  {
    int n_1d_spts = round(sqrt(1.0*in_n_spts));
    array<double> loc_1d_spts(n_1d_spts);
    set_loc_1d_spts(loc_1d_spts,n_1d_spts);

    for (int k=0;k<in_n_spts;k++)
    {
	    i=k/n_1d_spts;
	    j=k-(n_1d_spts*i);

  		d_nodal_s_basis(k,0)=eval_d_lagrange(in_loc(0),j,loc_1d_spts)*eval_lagrange(in_loc(1),i,loc_1d_spts);
  		d_nodal_s_basis(k,1)=eval_lagrange(in_loc(0),j,loc_1d_spts)*eval_d_lagrange(in_loc(1),i,loc_1d_spts);
    }
  }
  else if (in_n_spts==8)
  {
    d_nodal_s_basis(0,0) = -0.25*(-1.+in_loc(1))*(2.*in_loc(0)+in_loc(1));
    d_nodal_s_basis(1,0) = 0.25*(-1.+in_loc(1))*(in_loc(1)-2.*in_loc(0));
    d_nodal_s_basis(2,0)	 = 0.25*(1.+in_loc(1))*(2.*in_loc(0)+in_loc(1));
    d_nodal_s_basis(3,0)	 = -0.25*(1.+in_loc(1))*(in_loc(1)-2.*in_loc(0));
	  d_nodal_s_basis(4,0) = in_loc(0)*(-1.+in_loc(1));
	  d_nodal_s_basis(5,0) = -0.5*(1.+in_loc(1))*(-1.+in_loc(1));
	  d_nodal_s_basis(6,0) = -in_loc(0)*(1.+in_loc(1));
	  d_nodal_s_basis(7,0) = 0.5*(1.+in_loc(1))*(-1.+in_loc(1));


    d_nodal_s_basis(0,1) = -0.25*(-1.+in_loc(0))*(in_loc(0)+ 2.*in_loc(1));
    d_nodal_s_basis(1,1) = 0.25*(1.+in_loc(0))*(2.*in_loc(1)-in_loc(0));
    d_nodal_s_basis(2,1) = 0.25*(1.+in_loc(0))*(in_loc(0)+2.*in_loc(1));
    d_nodal_s_basis(3,1) = -0.25*(-1.+in_loc(0))*(2.*in_loc(1)-in_loc(0));
    d_nodal_s_basis(4,1) = 0.5*(1.+in_loc(0))*(-1.+in_loc(0));
    d_nodal_s_basis(5,1) = -in_loc(1)*(1.+in_loc(0));
    d_nodal_s_basis(6,1) = -0.5*(1.+in_loc(0))*(-1.+in_loc(0));
	  d_nodal_s_basis(7,1) = in_loc(1)*(-1.+in_loc(0));
  }
  else
  {
    cout << "Shape basis not implemented yet, exiting" << endl;
    exit(1);
  }

}

// evaluate second derivative of nodal shape basis

void eles_quads::eval_dd_nodal_s_basis(array<double> &dd_nodal_s_basis, array<double> in_loc, int in_n_spts)
{
	int i,j;

  if (is_perfect_square(in_n_spts))
  {
    int n_1d_spts = round(sqrt(1.0*in_n_spts));
    array<double> loc_1d_spts(n_1d_spts);
    set_loc_1d_spts(loc_1d_spts,n_1d_spts);
	
    for (int k=0;k<in_n_spts;k++)
    {
	    i=k/n_1d_spts;
	    j=k-(n_1d_spts*i);
	
	  	dd_nodal_s_basis(k,0) = eval_dd_lagrange(in_loc(0),j,loc_1d_spts)*eval_lagrange(in_loc(1),i,loc_1d_spts);
	  	dd_nodal_s_basis(k,1) = eval_lagrange(in_loc(0),j,loc_1d_spts)*eval_dd_lagrange(in_loc(1),i,loc_1d_spts);
	  	dd_nodal_s_basis(k,2) = eval_d_lagrange(in_loc(0),j,loc_1d_spts)*eval_d_lagrange(in_loc(1),i,loc_1d_spts);
    }
  }
  else if (in_n_spts==8)
  {
    dd_nodal_s_basis(0,0) = -0.5*(in_loc(1)-1.);
    dd_nodal_s_basis(1,0) = -0.5*(in_loc(1)-1.);
    dd_nodal_s_basis(2,0) = 0.5*(in_loc(1)+1.);
    dd_nodal_s_basis(3,0) = 0.5*(in_loc(1)+1.);
    dd_nodal_s_basis(4,0) = (in_loc(1)-1.);
    dd_nodal_s_basis(5,0) = 0.;
    dd_nodal_s_basis(6,0) = -(in_loc(1)+1.);
    dd_nodal_s_basis(7,0) = 0.;

    dd_nodal_s_basis(0,1) = -0.5*(in_loc(0)-1.);
    dd_nodal_s_basis(1,1) = 0.5*(in_loc(0)+1.);
    dd_nodal_s_basis(2,1) = 0.5*(in_loc(0)+1.);
    dd_nodal_s_basis(3,1) = -0.5*(in_loc(0)-1.);
    dd_nodal_s_basis(4,1) = 0.;
    dd_nodal_s_basis(5,1) = -(in_loc(0)+1.);
    dd_nodal_s_basis(6,1) = 0.;
    dd_nodal_s_basis(7,1) = (in_loc(0)-1.);

    dd_nodal_s_basis(0,2) = 0.25*(1.-2.*in_loc(0)-2.*in_loc(1));
    dd_nodal_s_basis(1,2) = 0.25*(2.*in_loc(1)-2.*in_loc(0)-1.);
    dd_nodal_s_basis(2,2) = 0.25*(2.*in_loc(0)+2.*in_loc(1)+1.);
    dd_nodal_s_basis(3,2) = 0.25*(2.*in_loc(0)-2.*in_loc(1)-1.);
    dd_nodal_s_basis(4,2) = in_loc(0);
    dd_nodal_s_basis(5,2) = -in_loc(1);
    dd_nodal_s_basis(6,2) = -in_loc(0);
    dd_nodal_s_basis(7,2) = in_loc(1);
  }
  else
  {
    cout << "Shape basis not implemented yet in dd_nodal_s_basis, exiting" << endl;
    exit(1);
  }
}

void eles_quads::fill_opp_3(array<double>& opp_3)
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
			
			opp_3(j,i)=eval_div_vcjh_basis(i,loc);
		}	
	}
}

// evaluate divergence of vcjh basis

double eles_quads::eval_div_vcjh_basis(int in_index, array<double>& loc)
{
	int i,j;
	double eta;

  if (run_input.vcjh_scheme_quad==0)
    eta = run_input.eta_quad;
  else
    eta = compute_eta(run_input.vcjh_scheme_quad,order);

	double div_vcjh_basis;
	
	i=in_index/n_fpts_per_inter(0);
	j=in_index-(n_fpts_per_inter(0)*i);
	
	if(i==0)
	{
		div_vcjh_basis=-eval_lagrange(loc(0),j,loc_1d_upts)*eval_d_vcjh_1d(loc(1),0,order,eta);
	}
	else if(i==1)
	{
		div_vcjh_basis=eval_d_vcjh_1d(loc(0),1,order,eta)*eval_lagrange(loc(1),j,loc_1d_upts);
	}
	else if(i==2)
	{
		div_vcjh_basis=eval_lagrange(loc(0),order-j,loc_1d_upts)*eval_d_vcjh_1d(loc(1),1,order,eta);
	}
	else if(i==3)
	{
		div_vcjh_basis=-eval_d_vcjh_1d(loc(0),0,order,eta)*eval_lagrange(loc(1),order-j,loc_1d_upts);
	}

	return div_vcjh_basis;
}

// Get position of 1d solution point
double eles_quads::get_loc_1d_upt(int in_index)
{
  return loc_1d_upts(in_index);
}

/*! Calculate element volume */
double eles_quads::calc_ele_vol(double& detjac)
{
	double vol;
	// Element volume = |Jacobian|*width*height of reference element
	vol = detjac*4.;
	return vol;
}

