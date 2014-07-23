/*!
 * \file cubature_tri.cpp
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
  ifstream datfile;
  char buf[BUFSIZ]={""};
  char section_TXT[100], param_TXT[100];
  char* f;
  string filename, param_name, param, ord;
  istringstream strbuf;
  int order_file;
  
  order=in_order;

  if (order==2)
    n_pts=3;
  else if (order==3)
    n_pts=6;
  else if (order==4)
    n_pts=6;
  else if (order==5)
    n_pts=7;
  else if (order==6)
    n_pts=12;
  else if (order==7)
    n_pts=15;
  else if (order==8)
    n_pts=16;
  else if (order==9)
    n_pts=19;
  else if (order==10)
    n_pts=25;
  else if (order==11)
    n_pts=28;
  else if (order==12)
    n_pts=36;
  else if (order==13)
    n_pts=40;
  else if (order==14)
    n_pts=46;
  else if (order==15)
    n_pts=54;
  else if (order==16)
    n_pts=58;
  else if (order==17)
    n_pts=66;
  else if (order==18)
    n_pts=73;
  else if (order==19)
    n_pts=82;
  else if (order==20)
    n_pts=85;
  else {
    order=0;
    n_pts=0;
    locs.setup(0,0);
    weights.setup(0);
    
    FatalError("ERROR: Order of cubature rule currently not implemented ....");
  }
  
  locs.setup(n_pts,2);
  weights.setup(n_pts);
  
  
  if(order < 21) {
    
    if (HIFILES_DIR == NULL)
      FatalError("environment variable HIFILES_HOME is undefined");
    
    filename = HIFILES_DIR;
    filename += "/data/cubature_tri.dat";
    f = (char*)filename.c_str();
    datfile.open(f, ifstream::in);
    if (!datfile) FatalError("Unable to open cubature file");
    
    // read data from file to arrays
    while(datfile.getline(buf,BUFSIZ))
    {
      sscanf(buf,"%s",section_TXT);
      param_name.assign(section_TXT,0,99);
      
      if(!param_name.compare(0,5,"order"))
      {
        // get no. of pts
        ord = param_name.substr(6);
        stringstream str(ord);
        str >> order_file;
        
        // if pts matches order, read locs and weights
        if (order_file == order) {
          
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


