/*!
 * \file linear_solvers_structure.inl
 * \brief inline subroutines of the <i>linear_solvers_structure.hpp</i> file.
 * \author - Original Author: Aerospace Design Laboratory (Stanford University) <http://su2.stanford.edu>.
           - Current development: Aerospace Computing Laboratory (ACL)
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

inline double CSysSolve::sign(const double & x, const double & y) const {
  if (y == 0.0)
    return 0.0;
  else {
    return (y < 0 ? -fabs(x) : fabs(x));
  }
}
