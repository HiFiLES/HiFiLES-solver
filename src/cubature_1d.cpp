/*!
 * \file cubature_1d.cpp
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
#include <cmath>
#include <string>

#include "../include/cubature_1d.h"

using namespace std;

// #### constructors ####

// default constructor

cubature_1d::cubature_1d()
{	
	order=0;
	n_pts=0;
	locs.setup(0);
	weights.setup(0);
}

// constructor 1

cubature_1d::cubature_1d(int in_order) // set by number of points
{	
	order=in_order;
  n_pts = (order+1)/2;

	#include "../data/cubature_1d.dat"
}

// copy constructor

cubature_1d::cubature_1d(const cubature_1d& in_cubature_1d)
{
	order=in_cubature_1d.order;	
	n_pts=in_cubature_1d.n_pts;
	locs=in_cubature_1d.locs;
	weights=in_cubature_1d.weights;
}

// assignment

cubature_1d& cubature_1d::operator=(const cubature_1d& in_cubature_1d)
{
	// check for self asignment
	if(this == &in_cubature_1d)
	{
		return (*this);
	}
	else
	{
		order=in_cubature_1d.order;	
		n_pts=in_cubature_1d.n_pts;
		locs=in_cubature_1d.locs;
		weights=in_cubature_1d.weights;
	}
}

// destructor

cubature_1d::~cubature_1d()
{
	
}

// #### methods ####
	
// method to get number of cubature_1d points

double cubature_1d::get_n_pts(void)
{
	return n_pts;
}

// method to get r location of cubature_1d point

double cubature_1d::get_r(int in_pos)
{
	return locs(in_pos);
}

// method to get weight location of cubature_1d point

double cubature_1d::get_weight(int in_pos)
{
	return weights(in_pos);
}
