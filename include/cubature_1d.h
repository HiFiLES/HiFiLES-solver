/*!
 * \file cubature_1d.h
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

#pragma once

#include <string>
#include "array.h"

class cubature_1d
{
public:

  // #### constructors ####

  // default constructor
  cubature_1d();

  // constructor 1
  cubature_1d(int in_order); // set by order

  // copy constructor
  cubature_1d(const cubature_1d& in_cubature_1d);

  // assignment
  cubature_1d& operator=(const cubature_1d& in_cubature_1d);

  // destructor
  ~cubature_1d();

  // #### methods ####

  // method to get number of cubature_1d points
  int get_n_pts(void);

  // method to get r location of cubature_1d point
  double get_r(int in_pos);

  // method to get weight location of cubature_1d point
  double get_weight(int in_pos);

  // #### members ####

  // cubature_1d order
  int order;

  // number of cubature_1d points
  int n_pts;

  // location of cubature_1d points
  array<double> locs;

  // weight of cubature_1d points
  array<double> weights;
  
  // cubature data file
  ifstream datfile;
};
