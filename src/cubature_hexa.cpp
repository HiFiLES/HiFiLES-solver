/*!
 * \file cubature_hexa.cpp
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

#include "../include/cubature_hexa.h"

using namespace std;

// #### constructors ####

// default constructor

cubature_hexa::cubature_hexa()
{	
  rule=0;
  n_pts=0;
  locs.setup(0,0);
  weights.setup(0);
}

// constructor 1

cubature_hexa::cubature_hexa(int in_rule) // set by rule
{	
  rule=in_rule;

#include "../data/cubature_hexa.dat"

}

// copy constructor

cubature_hexa::cubature_hexa(const cubature_hexa& in_cubature)
{
  rule=in_cubature.rule;
  n_pts=in_cubature.n_pts;
  locs=in_cubature.locs;
  weights=in_cubature.weights;
}

// assignment

cubature_hexa& cubature_hexa::operator=(const cubature_hexa& in_cubature)
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

cubature_hexa::~cubature_hexa()
{

}

// #### methods ####

// method to set a cubature rule

void cubature_hexa::set_rule(int in_rule)
{
  rule=in_rule;

#include "../data/cubature_hexa.dat"
}

// method to get number of cubature points

double cubature_hexa::get_n_pts(void)
{
  return n_pts;
}

// method to get r location of cubature point

double cubature_hexa::get_r(int in_pos)
{
  return locs(in_pos,0);
}

// method to get s location of cubature point
double cubature_hexa::get_s(int in_pos)
{
  return locs(in_pos,1);
}

// method to get s location of cubature point
double cubature_hexa::get_t(int in_pos)
{
  return locs(in_pos,2);
}

// method to get weight location of cubature point

double cubature_hexa::get_weight(int in_pos)
{
  return weights(in_pos);
}
