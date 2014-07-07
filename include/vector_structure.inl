/*!
 * \file vector_structure.inl
 * \brief inline subroutines of the <i>vector_structure.hpp</i> file.
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

inline void CSysVector::SetValZero(void) { 
  for (unsigned long i = 0; i < nElm; i++)
		vec_val[i] = 0.0;
}

inline unsigned long CSysVector::GetLocSize() const { return nElm; }

inline unsigned long CSysVector::GetSize() const {
#ifdef _MPI
  return nElmGlobal;
#else
  return (unsigned long)nElm;
#endif
}

inline unsigned short CSysVector::GetNVar() const { return nVar; }

inline unsigned long CSysVector::GetNBlk() const { return nBlk; }

inline unsigned long CSysVector::GetNBlkDomain() const { return nBlkDomain; }

inline double & CSysVector::operator[](const unsigned long & i) { return vec_val[i]; }

inline const double & CSysVector::operator[](const unsigned long & i) const { return vec_val[i]; }
