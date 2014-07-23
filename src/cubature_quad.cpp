/*!
 * \file cubature_quad.cpp
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

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>

#include "../include/global.h"
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
  ifstream datfile;
  char buf[BUFSIZ]={""};
  char section_TXT[100], param_TXT[100];
  char* f;
  string filename, param_name, param, ord;
  istringstream strbuf;
  int rule_file;

  rule=in_rule;
  n_pts = rule*rule;
  
  locs.setup(n_pts,2);
  weights.setup(n_pts);

  if(rule < 21) {
    
    if (HIFILES_DIR == NULL)
      FatalError("environment variable HIFILES_HOME is undefined");
    
    filename = HIFILES_DIR;
    filename += "/data/cubature_quad.dat";
    f = (char*)filename.c_str();
    datfile.open(f, ifstream::in);
    if (!datfile) FatalError("Unable to open cubature file");

    // read data from file to arrays
    while(datfile.getline(buf,BUFSIZ))
    {
      sscanf(buf,"%s",section_TXT);
      param_name.assign(section_TXT,0,99);
      
      if(!param_name.compare(0,4,"rule"))
      {
        // get no. of pts
        ord = param_name.substr(5);
        stringstream str(ord);
        str >> rule_file;
        
        // if pts matches order, read locs and weights
        if (rule_file == rule) {
          
          // skip next line
          datfile.getline(buf,BUFSIZ);
          
          for(int i=0;i<n_pts;++i) {
            datfile.getline(buf,BUFSIZ);
            sscanf(buf,"%s",param_TXT);
            param.assign(param_TXT,0,99);
            strbuf.str(param);
            locs(i,0) = atof(param.c_str());
          }
          
          // skip next line
          datfile.getline(buf,BUFSIZ);
          
          for(int i=0;i<n_pts;++i) {
            datfile.getline(buf,BUFSIZ);
            sscanf(buf,"%s",param_TXT);
            param.assign(param_TXT,0,99);
            strbuf.str(param);
            locs(i,1) = atof(param.c_str());
          }
          
          // skip next line
          datfile.getline(buf,BUFSIZ);
          
          for(int i=0;i<n_pts;++i) {
            datfile.getline(buf,BUFSIZ);
            sscanf(buf,"%s",param_TXT);
            param.assign(param_TXT,0,99);
            strbuf.str(param);
            weights(i) = atof(param.c_str());
          }
          break;
        }
      }
    }
  }
  else { FatalError("cubature rule not implemented."); }
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

// method to get number of cubature points

int cubature_quad::get_n_pts(void)
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

