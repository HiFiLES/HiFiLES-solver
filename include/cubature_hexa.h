/*!
 * \file cubature_hexa.h
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

class cubature_hexa
{
public:

  // #### constructors ####

  // default constructor
  cubature_hexa();

  // constructor 1
  cubature_hexa(int in_rule); // set by order

  // copy constructor
  cubature_hexa(const cubature_hexa& in_cubature);

  // assignment
  cubature_hexa& operator=(const cubature_hexa& in_cubature);

  // destructor
  ~cubature_hexa();

  // #### methods ####

  // method to get number of cubature points
  int get_n_pts(void);

  // method to get r location of cubature point
  double get_r(int in_pos);

  // method to get s location of cubature point
  double get_s(int in_pos);

  // method to get s location of cubature point
  double get_t(int in_pos);

  // method to get weight location of cubature point
  double get_weight(int in_pos);

  // #### members ####

  // cubature rule
  int rule;

  // number of cubature points
  int n_pts;

  // location of cubature points
  array<double> locs;

  // weight of cubature points
  array<double> weights;
  
  // cubature data file
  ifstream datfile;
};
