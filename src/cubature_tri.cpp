/*!
 * \file cubature_tri.cpp
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

#include "../include/cubature_tri.h"

using namespace std;

// #### constructors ####

// default constructor

cubature_tri::cubature_tri()
{	
  order=0;
  n_pts=0;
  locs.setup(0,0);
  weights.setup(0);
}

// constructor 1

cubature_tri::cubature_tri(int in_order) // set by order
{	
  order=in_order;

#include "../data/cubature_tri.dat"
}

// copy constructor

cubature_tri::cubature_tri(const cubature_tri& in_cubature_tri)
{
  order=in_cubature_tri.order;
  n_pts=in_cubature_tri.n_pts;
  locs=in_cubature_tri.locs;
  weights=in_cubature_tri.weights;
}

// assignment

cubature_tri& cubature_tri::operator=(const cubature_tri& in_cubature_tri)
{
  // check for self asignment
  if(this == &in_cubature_tri)
    {
      return (*this);
    }
  else
    {
      order=in_cubature_tri.order;
      n_pts=in_cubature_tri.n_pts;
      locs=in_cubature_tri.locs;
      weights=in_cubature_tri.weights;
    }
}

// destructor

cubature_tri::~cubature_tri()
{

}

// #### methods ####

// method to set a cubature_tri rule

void cubature_tri::set_order(int in_order)
{
  order=in_order;

#include "../data/cubature_tri.dat"
}

// method to get number of cubature_tri points

int cubature_tri::get_n_pts(void)
{
  return n_pts;
}

// method to get r location of cubature_tri point

double cubature_tri::get_r(int in_pos)
{
  return locs(in_pos,0);
}

// method to get s location of cubature_tri point

double cubature_tri::get_s(int in_pos)
{
  return locs(in_pos,1);
}

// method to get weight location of cubature_tri point

double cubature_tri::get_weight(int in_pos)
{
  return weights(in_pos);
}
