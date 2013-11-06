/*!
 * \file cubature_quad.cpp
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

#include "../include/cubature_quad.h"

using namespace std;

// #### constructors ####

// default constructor

cubature_quad::cubature_quad()
{	
	rule=0;
	n_pts=0;
	locs.setup(0,0);
	weights.setup(0);
}

// constructor 1

cubature_quad::cubature_quad(int in_rule) // set by rule
{	
	rule=in_rule;
	
	#include "../data/cubature_quad.dat"
}

// copy constructor

cubature_quad::cubature_quad(const cubature_quad& in_cubature)
{
	rule=in_cubature.rule;	
	n_pts=in_cubature.n_pts;
	locs=in_cubature.locs;
	weights=in_cubature.weights;
}

// assignment

cubature_quad& cubature_quad::operator=(const cubature_quad& in_cubature)
{
	// check for self asignment
	if(this == &in_cubature)
	{
		return (*this);
	}
	else
	{
		rule=in_cubature.rule;	
		n_pts=in_cubature.n_pts;
		locs=in_cubature.locs;
		weights=in_cubature.weights;
	}
}

// destructor

cubature_quad::~cubature_quad()
{
	
}

// #### methods ####
	
// method to set a cubature rule

void cubature_quad::set_rule(int in_rule)
{
	rule=in_rule;
	
	#include "../data/cubature_quad.dat"	
}

// method to get number of cubature points

double cubature_quad::get_n_pts(void)
{
	return n_pts;
}

// method to get r location of cubature point

double cubature_quad::get_r(int in_pos)
{
	return locs(in_pos,0);
}

// method to get s location of cubature point

double cubature_quad::get_s(int in_pos)
{
	return locs(in_pos,1);
}

// method to get weight location of cubature point

double cubature_quad::get_weight(int in_pos)
{
	return weights(in_pos);
}
