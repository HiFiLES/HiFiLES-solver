/*!
 * \file mesh.cpp
 * \brief  - Handle mesh motion using linear elasticity and other methods
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
#include "../include/mesh.h"
#include "../include/geometry.h"
#include "../include/cuda_kernels.h"
#include "../include/solver.h"
#include <string>
using namespace std;

template <typename T>
void displayMatrix(array<T> matrix) {
  int i,j;
  for (i=0; i<matrix.get_dim(0); i++) {
    for (j=0; j<matrix.get_dim(1); j++) {
      cout << matrix(i,j) << " ";
    }
    cout << endl;
  }
}

mesh::mesh(void)
{
  start = true;
  n_eles = 0;
  n_verts = 0;
  n_dims = 2;
  n_verts_global = 0;
  n_cells_global = 0;
  n_bnds = 0;
  n_faces = 0;
  LinSolIters = 0;
  failedIts = 0;
  min_vol = DBL_MAX;
  min_length = DBL_MAX;
  solver_tolerance = 1E-4;

  iter = 0;

  bc_num["Sub_In_Simp"] = 1;
  bc_num["Sub_Out_Simp"] = 2;
  bc_num["Sub_In_Char"] = 3;
  bc_num["Sub_Out_Char"] = 4;
  bc_num["Sup_In"] = 5;
  bc_num["Sup_Out"] = 6;
  bc_num["Slip_Wall"] = 7;
  bc_num["Cyclic"] = 9;
  bc_num["Isotherm_Fix"] = 11;
  bc_num["Adiabat_Fix"] = 12;
  bc_num["Isotherm_Move"] = 13;
  bc_num["Adiabat_Move"] = 14;
  bc_num["Char"] = 15;
  bc_num["Slip_Wall_Dual"] = 16;
  bc_num["AD_Wall"] = 50;

  bc_string[1] = "Sub_In_Simp";
  bc_string[2] = "Sub_Out_Simp";
  bc_string[3] = "Sub_In_Char";
  bc_string[4] = "Sub_Out_Char";
  bc_string[5] = "Sup_In";
  bc_string[6]= "Sup_Out";
  bc_string[7]= "Slip_Wall";
  bc_string[9]= "Cyclic";
  bc_string[11]= "Isotherm_Fix";
  bc_string[12]= "Adiabat_Fix";
  bc_string[13]= "Isotherm_Move";
  bc_string[14]= "Adiabat_Move";
  bc_string[15]= "Char";
  bc_string[16]= "Slip_Wall_Dual";
  bc_string[50]= "AD_Wall";
}

mesh::~mesh(void)
{
  // not currently needed
}

void mesh::setup(struct solution *in_FlowSol,array<double> &in_xv,array<int> &in_c2v,array<int> &in_c2n_v,array<int> &in_iv2ivg,array<int> &in_ctype)
{
  FlowSol = in_FlowSol;
  n_dims = FlowSol->n_dims;
  n_eles = FlowSol->num_eles;
  n_verts = FlowSol->num_verts;
  n_cells_global = FlowSol->num_cells_global;

  // Setup for 4th-order backward difference
  xv.setup(5);
  xv(0) = in_xv;
  xv(1) = in_xv;
  xv(2) = in_xv;
  xv(3) = in_xv;
  xv(4) = in_xv;

  xv_0 = in_xv;

  c2v = in_c2v;
  c2n_v = in_c2n_v;
  iv2ivg = in_iv2ivg;
  ctype = in_ctype;

  vel_old.setup(in_xv.get_dim(0),n_dims);
  vel_new.setup(in_xv.get_dim(0),n_dims);
  vel_old.initialize_to_zero();
  vel_new.initialize_to_zero();
//  grid_vel.setup(2);
//  grid_vel(0).setup(xv.get_dim(0),Mesh.n_dims);
//  grid_vel(1).setup(xv.get_dim(0),Mesh.n_dims);
//  grid_vel(0).initialize_to_zero();
//  grid_vel(1).initialize_to_zero();

  if (run_input.motion==LINEAR_ELASTICITY) {
    initialize_displacement();

    n_moving_bnds = run_input.n_moving_bnds;
    motion_params.setup(n_moving_bnds);
    for (int i=0; i<n_moving_bnds; i++)
      motion_params(i) = run_input.bound_vel_simple(i);
  }

  if (run_input.adv_type==0) {
    RK_a.setup(1);
    RK_c.setup(1);
    RK_b.setup(1);
    RK_a(0) = 0.0;
    RK_b(0) = 0.0;
    RK_c(0) = 0.0;
  }else if (run_input.adv_type==3) {
    RK_a.setup(5);
    RK_a(0) = 0.0;
    RK_a(1) = -0.417890474499852;
    RK_a(2) = -1.192151694642677;
    RK_a(3) = -1.697784692471528;
    RK_a(4) = -1.514183444257156;

    RK_b.setup(5);
    RK_b(0) = 0.149659021999229;
    RK_b(1) = 0.379210312999627;
    RK_b(2) = 0.822955029386982;
    RK_b(3) = 0.699450455949122;
    RK_b(4) = 0.153057247968152;

    RK_c.setup(5);
    RK_c(0) = 0.0;
    RK_c(1) = 1432997174477/9575080441755;
    RK_c(2) = 2526269341429/6820363962896;
    RK_c(3) = 2006345519317/3224310063776;
    RK_c(4) = 2802321613138/2924317926251;
  }
}

void mesh::move(int _iter, int in_rk_step) {
  iter = _iter;
  rk_step = in_rk_step;
  time = FlowSol->time;
  if (in_rk_step>0)
    rk_time = time+run_input.dt*RK_c(rk_step);
  else
    rk_time = time;
  run_input.rk_time = rk_time;

  if (run_input.motion == 1) {
    deform();
  }else if (run_input.motion == 2) {
    rigid_move();
  }else if (run_input.motion == 3) {
    perturb();
  }else{
    // Do Nothing
  }
}

/*! This will occur after the linear-elasticity problem has been solved */
void mesh::deform(void)
{
  for (int i=0; i<run_input.elas_max_iter; i++) {
    /*! Calculate position of boundary nodes & transfer to eles classes */
    //set_boundary_displacements_eles();

    /*! First steps of FR process up to boundary conditions */
    CalcResidualElasticity_start(FlowSol);

    /*! Apply the boundary conditions to the eles classes */
    //apply_boundary_displacements_fpts();

    /*! Remainder of FR process after boundary conditions */
    CalcResidualElasticity_finish(FlowSol);

    for(int i=0; i<FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->AdvanceSolutionElasticity(rk_step, FlowSol->adv_type);
  }

  /*! Extrapolate the displacement to the shape points */
  for(int i=0; i<FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->extrapolate_solution_spts_elasticity();

  /*! Using the final displacement solution, average values to the mesh vertices */
  update_displacements();

  /*! Update the nodes in the mesh given the final displacements */
  update_mesh_nodes();

  /*! Calculate new grid velocities, transformations, etc. */
  update();

}


void mesh::set_min_length(void)
{
  unsigned int n_edges = e2v.get_dim(0);
  double length2;
  double min_length2 = DBL_MAX;

  for (int i=0; i<n_edges; i++) {
    length2 = pow((xv(0)(e2v(i,0),0)-xv(0)(e2v(i,1),0)),2) + pow((xv(0)(e2v(i,0),1)-xv(0)(e2v(i,1),1)),2);
    min_length2 = fmin(min_length2,length2);
  }

  min_length = sqrt(min_length2);
}

void mesh::set_grid_velocity(solution* FlowSol, double dt)
{

  if (run_input.motion == 3) {
    /// Analytic solution for perturb test-case
    for (int i=0; i<n_verts; i++) {
      vel_new(i,0) = 4*pi/10*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/10)*cos(2*pi*rk_time/10); // from Kui
      vel_new(i,1) = 4*pi/10*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/10)*cos(2*pi*rk_time/10);
    }
  }
  else if (run_input.motion == 2) {
    for (int i=0; i<n_verts; i++) {
      for (int j=0; j<n_dims; j++) {
        vel_new(i,j) = run_input.bound_vel_simple(0)(2*j  )*run_input.bound_vel_simple(0)(6+j)*sin(run_input.bound_vel_simple(0)(6+j)*rk_time);
        vel_new(i,j)+= run_input.bound_vel_simple(0)(2*j+1)*run_input.bound_vel_simple(0)(6+j)*cos(run_input.bound_vel_simple(0)(6+j)*rk_time);
      }
    }
  }
  else
  {
    /// calculate velocity using backward difference formula
    for (int i=0; i<n_verts; i++) {
      for (int j=0; j<n_dims; j++) {
        //vel_new(i,j) = (xv(0)(i,j) - xv(1)(i,j))/dt;  // using simple backward-Euler
        vel_new(i,j) = 25/12*xv(0)(i,j) - 4*xv(1)(i,j) + 3*xv(2)(i,j) - 4/3*xv(3)(i,j) + 1/4*xv(4)(i,j); // 4th-order backward difference
        vel_new(i,j) /= run_input.dt;
      }
    }
  }

  // Apply velocity to the eles classes at the shape points
  int local_ic;
  array<double> vel(n_dims);
  for (int ic=0; ic<n_eles; ic++) {
    for (int j=0; j<c2n_v(ic); j++) {
      for (int idim=0; idim<n_dims; idim++) {
        vel(idim) = vel_new(iv2ivg(c2v(ic,j)),idim);
      }
      local_ic = ic2loc_c(ic);
      FlowSol->mesh_eles(ctype(ic))->set_grid_vel_spt(local_ic,j,vel);
    }
  }

  // Interpolate grid vel @ spts to fpts & upts
  for (int i=0; i<FlowSol->n_ele_types; i++) {
    FlowSol->mesh_eles(i)->set_grid_vel_fpts(rk_step);
    FlowSol->mesh_eles(i)->set_grid_vel_upts(rk_step);
  }
}

void mesh::update(void)
{
  // Update grid velocity & transfer to upts, fpts

  set_grid_velocity(FlowSol,run_input.dt);

  // Update element shape points

  int ele_type, local_id;
  array<double> pos(n_dims);

  for (int ic=0; ic<n_eles; ic++) {
    ele_type = ctype(ic);
    local_id = ic2loc_c(ic);
    for (int iv=0; iv<c2n_v(ic); iv++) {
      for (int k=0; k<n_dims; k++) {
        pos(k) = xv(0)(c2v(ic,iv),k);
      }
      FlowSol->mesh_eles(ele_type)->set_dynamic_shape_node(iv,local_id,pos);
    }
  }

#ifdef _GPU
  //FlowSol->mesh_eles(ele_type)->set_dynamic_shape_nodes_kernel_wrapper(n_dims, n_verts, n_eles, max_n_)
#endif

  // Update element transforms

  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      FlowSol->mesh_eles(i)->set_transforms_dynamic();
    }
  }

  /// if (iter%FlowSol->plot_freq == 0 || iter%FlowSol->restart_dump_freq == 0) {
//    // Set metrics at interface cubpts
//    //if (FlowSol->rank==0) cout << "Deform: setting element transforms at interface cubature points ... " << endl;
//    for(int i=0;i<FlowSol->n_ele_types;i++) {
//      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
//        FlowSol->mesh_eles(i)->set_transforms_inters_cubpts();
//      }
//    }

//    // Set metrics at volume cubpts
//    //if (FlowSol->rank==0) cout << "Deform: setting element transforms at volume cubature points ... " << endl;
//    for(int i=0;i<FlowSol->n_ele_types;i++) {
//      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
//        FlowSol->mesh_eles(i)->set_transforms_vol_cubpts();
//      }
//    }
  /// }
}

void mesh::write_mesh(double sim_time)
{

  if (run_input.mesh_format==0) {
    write_mesh_gambit(sim_time);
  }else if (run_input.mesh_format==1) {
    write_mesh_gmsh(sim_time);
  }else{
    cerr << "Mesh Output Type: " << run_input.mesh_format << endl;
    FatalError("ERROR: Trying to write unrecognized mesh format ... ");
  }
}

void mesh::write_mesh_gambit(double sim_time)
{
  cout << "Gambit mesh writer not yet implemented!" << endl;
}

void mesh::write_mesh_gmsh(double sim_time)
{

  string filename (run_input.mesh_file);
  ostringstream sstream;
  sstream << sim_time;
  string suffix = "_" + sstream.str();
  int find = suffix.find_first_of(".");
  if (find != suffix.npos) suffix.replace(find,1,"_");
  filename.insert(filename.size()-4,suffix);

  fstream file;
  file.open(filename.c_str(),ios::out);

  // write header
  file << "$MeshFormat" << endl << "2.2 0 8" << endl;
  file << "$EndMeshFormat" << endl;

  // write boundary info
  file << "$PhysicalNames" << endl << n_bnds << endl;
  for (int i=0; i<n_bnds; i++) {
    if (bc_list(i) == -1) {
      file << n_dims << " "; // volume cell
      file << i+1  << " " << "\"FLUID\"" << endl;
    }else{
      file << 1 << " ";  // edge
      file << i+1  << " " << "\"" << bc_string[bc_list(i)] << "\"" << endl;
    }

  }
  file << "$EndPhysicalNames" << endl;
  // write nodes
  file << "$Nodes" << endl << n_verts_global << endl;
  file << setprecision(12);
  for (int i=0; i<n_verts; i++) {
    file << i+1 << " " << xv(0)(i,0) << " " << xv(0)(i,1) << " ";
    if (n_dims==2) {
      file << 0;
    }else{
      file << xv(0)(i,2);
    }
    file << endl;
  }
  file << "$EndNodes" << endl;

  // write elements
  // note: n_cells_global not currently defined.  Fix!!  Needed for MPI.
  file << "$Elements" << endl << n_eles << endl;
  int gmsh_type, bcid;
  int ele_start = 0; // more setup needed for writing from parallel
  for (int i=ele_start; i<ele_start+n_eles; i++) {
    for (bcid=1; bcid<n_bnds+1; bcid++) {
      if (bc_list(bcid-1)==bctype_c(i)) break; // bc_list wrong size?
    }
    if (ctype(i)==0) {
      // triangle
      if (c2n_v(i)==3) {
        gmsh_type = 2;
        file << i+1  << " " << gmsh_type << " 2 " << bcid << " " << bcid;
        file << " " << iv2ivg(c2v(i,0))+1 << " " << iv2ivg(c2v(i,1))+1 << " " << iv2ivg(c2v(i,2))+1 << endl;
      }else if (c2n_v(i)==6) {
        gmsh_type = 9;
        file << i+1  << " " << gmsh_type << " 2 " << bcid << " " << bcid;
        file << " " << iv2ivg(c2v(i,0))+1 << " " << iv2ivg(c2v(i,1))+1 << " " << iv2ivg(c2v(i,2))+1;
        file << " " << iv2ivg(c2v(i,3))+1 << " " << iv2ivg(c2v(i,4))+1 << " " << iv2ivg(c2v(i,5))+1 << endl;
      }else if (c2n_v(i)==9) {
        gmsh_type = 21;
        FatalError("Cubic triangle not implemented");
      }
    }else if (ctype(i)==1) {
      // quad
      if (c2n_v(i)==4) {
        gmsh_type = 3;
        file << i+1 << " " << gmsh_type << " 2 " << bcid << " " << bcid;
        file << " " << iv2ivg(c2v(i,0))+1 << " " << iv2ivg(c2v(i,1))+1 << " " << iv2ivg(c2v(i,3))+1 << " " << iv2ivg(c2v(i,2))+1 << endl;
      }else if (c2n_v(i)==8) {
        gmsh_type = 16;
        file << i+1 << " " << gmsh_type << " 2 " << bcid << " " << bcid;
        file << " " << iv2ivg(c2v(i,0))+1 << " " << iv2ivg(c2v(i,1))+1 << " " << iv2ivg(c2v(i,2))+1 << " " << iv2ivg(c2v(i,3))+1;
        file << " " << iv2ivg(c2v(i,4))+1 << " " << iv2ivg(c2v(i,5))+1 << " " << iv2ivg(c2v(i,6))+1 << " " << iv2ivg(c2v(i,7))+1 << endl;
      }else if (c2n_v(i)==9) {
        gmsh_type = 10;
        file << i+1 << " " << gmsh_type << " 2 " << bcid << " " << bcid;
        file << " " << iv2ivg(c2v(i,0))+1 << " " << iv2ivg(c2v(i,2))+1 << " " << iv2ivg(c2v(i,8))+1 << " " << iv2ivg(c2v(i,6))+1 << " " << iv2ivg(c2v(i,1))+1;
        file << " " << iv2ivg(c2v(i,5))+1 << " " << iv2ivg(c2v(i,7))+1 << " " << iv2ivg(c2v(i,3))+1 << " " << iv2ivg(c2v(i,4))+1 << endl;
      }
    }else if (ctype(i)==4) {
      //hex
      if (c2n_v(i)==8) {
        gmsh_type = 5;
        file << i+1  << " " << gmsh_type << " 2 " << bcid << " " << bcid;
        file << " " << iv2ivg(c2v(i,1))+1 << " " << iv2ivg(c2v(i,1))+1 << " " << iv2ivg(c2v(i,3))+1 << " " << iv2ivg(c2v(i,2))+1;
        file << " " << iv2ivg(c2v(i,4))+1 << " " << iv2ivg(c2v(i,5))+1 << " " << iv2ivg(c2v(i,7))+1 << " " << iv2ivg(c2v(i,6))+1 << endl;
      }
    }
  }
  //cout << "SIZE(e2v): " << e2v.get_dim(0) << "," << e2v.get_dim(1) << endl;
  //cout << "N_FACES: " << n_faces << endl;
  /* write non-interior 'elements' (boundary faces) */
  /** ONLY FOR 2D CURRENTLY -- To fix, add array<array<int>> boundFaces to mesh class
      * (same as boundPts, but for faces) - since faces, not edges, needed for 3D */
  // also, only for linear edges currently [Gmsh: 1==linear edge, 8==quadtratic edge]
  /*int faceid = n_cells_global + 1;
    int nv = 0;
    for (int i=0; i<n_bnds; i++) {
        nv = boundPts(i).get_dim(0);
        set<int> edges;
        int iv;
        for (int j=0; j<nv; j++) {
            iv = boundPts(i)(j);
            for (int k=0; k<v2n_e(iv); k++) {
                edges.insert(v2e(j)(k));
                cout << "Edge #: " << v2e(j)(k) << endl;
                if (v2e(j)(k) > n_faces) {
                    cout << "** n_faces=" << n_faces << " but v2e(" << j << ")(" << k << ")=" << v2e(j)(k) << "!!" << endl;
                    cin.get();
                }
            }
        }
        set<int>::iterator it;
        for (it=edges.begin(); it!=edges.end(); it++) {
            file << faceid << " 1 2 " << i+1 << " " << i+1 << " " << e2v(*it,0)+1 << " " << e2v(*it,1)+1 << endl;
            cout << faceid << " 1 2 " << i+1 << " " << i+1 << " " << e2v(*it,0)+1 << " " << e2v(*it,1)+1 << endl;
            faceid++;
        }
    }*/
  file << "$EndElements" << endl;
  file.close();
}

double mesh::check_grid(solution* FlowSol) {
  unsigned short iDim;
  unsigned long iElem, ElemCounter = 0;
  double Area, Volume, MinArea = DBL_MAX, MinVolume = DBL_MAX;
  //double MaxArea = -1E22, MaxVolume = -1E22  // never used
  bool NegVol;

  /*--- Load up each triangle and tetrahedron to check for negative volumes. ---*/

  for (iElem = 0; iElem < n_eles; iElem++) {
    /*--- Triangles ---*/
    if (n_dims == 2) {

      double a[2], b[2];
      for (iDim = 0; iDim < n_dims; iDim++) {
        a[iDim] = xv(0)(c2v(iElem,0),iDim)-xv(0)(c2v(iElem,1),iDim);
        b[iDim] = xv(0)(c2v(iElem,1),iDim)-xv(0)(c2v(iElem,2),iDim);
      }

      Area = 0.5*fabs(a[0]*b[1]-a[1]*b[0]);

      //MaxArea = max(MaxArea, Area);
      MinArea = min(MinArea, Area);

      NegVol = (MinArea < 0);
    }

    /*--- Tetrahedra ---*/
    if (n_dims == 3) {
      double r1[3], r2[3], r3[3], CrossProduct[3];

      for (iDim = 0; iDim < n_dims; iDim++) {
        r1[iDim] = xv(0)(c2v(iElem,1),iDim) - xv(0)(c2v(iElem,0),iDim);
        r2[iDim] = xv(0)(c2v(iElem,2),iDim) - xv(0)(c2v(iElem,0),iDim);
        r3[iDim] = xv(0)(c2v(iElem,3),iDim) - xv(0)(c2v(iElem,0),iDim);
      }

      CrossProduct[0] = (r1[1]*r2[2] - r1[2]*r2[1])*r3[0];
      CrossProduct[1] = (r1[2]*r2[0] - r1[0]*r2[2])*r3[1];
      CrossProduct[2] = (r1[0]*r2[1] - r1[1]*r2[0])*r3[2];

      Volume = (CrossProduct[0] + CrossProduct[1] + CrossProduct[2])/6.0;

      //MaxVolume = max(MaxVolume, Volume);
      MinVolume = min(MinVolume, Volume);

      NegVol = (MinVolume < 0);
    }

    if (NegVol) ElemCounter++;
  }

#ifdef MPI
  unsigned long ElemCounter_Local = ElemCounter; ElemCounter = 0;
  double MaxVolume_Local = MaxVolume; MaxVolume = 0.0;
  double MinVolume_Local = MinVolume; MinVolume = 0.0;

  MPI::COMM_WORLD.Allreduce(&ElemCounter_Local, &ElemCounter, 1, MPI::UNSIGNED_LONG, MPI::SUM);
  //MPI::COMM_WORLD.Allreduce(&MaxVolume_Local, &MaxVolume, 1, MPI::DOUBLE, MPI::MAX);
  MPI::COMM_WORLD.Allreduce(&MinVolume_Local, &MinVolume, 1, MPI::DOUBLE, MPI::MIN);
#endif
  /*
    if ((ElemCounter != 0) && (FlowSol->rank == MASTER_NODE))
        cout <<"There are " << ElemCounter << " elements with negative volume.\n" << endl;
    */
  if (n_dims == 2) return MinArea;
  else return MinVolume;
}


void mesh::rigid_move(void) {
#ifdef _CPU
  if (rk_step==0) {
    for (int i=4; i>0; i--) {
      for (int j=0; j<xv(i).get_dim(0); j++) {
        for (int k=0; k<n_dims; k++) {
          xv(i)(j,k) = xv(i-1)(j,k);
        }
      }
    }
  }

  for (int i=0; i<n_verts; i++) {
    // Useful for simple cases / debugging
    for (int j=0; j<n_dims; j++) {
      xv(0)(i,j) = xv(0)(i,j) + run_input.bound_vel_simple(0)(2*j  )*run_input.bound_vel_simple(0)(6+j)*sin(run_input.bound_vel_simple(0)(6+j)*time);
      xv(0)(i,j)+=              run_input.bound_vel_simple(0)(2*j+1)*run_input.bound_vel_simple(0)(6+j)*cos(run_input.bound_vel_simple(0)(6+j)*time);
    }
  }

  update();
#endif

#ifdef _GPU
  for (int i=0;i<FlowSol->n_ele_types;i++) {
    FlowSol->mesh_eles(i)->rigid_move(rk_time);
    FlowSol->mesh_eles(i)->rigid_grid_velocity(rk_time);
    FlowSol->mesh_eles(i)->set_transforms_dynamic();
  }
#endif
}

void mesh::perturb(void)
{
#ifdef _CPU
  if (rk_step==0) {
    // Push back previous time-advance level
    for (int i=4; i>0; i--) {
      for (int j=0; j<xv(i).get_dim(0); j++) {
        for (int k=0; k<n_dims; k++) {
          xv(i)(j,k) = xv(i-1)(j,k);
        }
      }
    }
  }

  for (int i=0; i<n_verts; i++) {
    /// Taken from Kui, AIAA-2010-5031-661
    xv(0)(i,0) = xv_0(i,0) + 2*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/10)*sin(2*pi*rk_time/10);
    xv(0)(i,1) = xv_0(i,1) + 2*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/10)*sin(2*pi*rk_time/10);
  }

  update();
#endif

#ifdef _GPU
  for (int i=0;i<FlowSol->n_ele_types;i++) {
    FlowSol->mesh_eles(i)->perturb_shape(rk_time);
    //FlowSol->mesh_eles(i)->calc_grid_velocity();
    FlowSol->mesh_eles(i)->perturb_grid_velocity(rk_time);
    FlowSol->mesh_eles(i)->set_transforms_dynamic();
  }
#endif
}


void mesh::initialize_displacement()
{
  displacement.setup(n_verts,n_dims);
  displacement.initialize_to_zero();
}

void mesh::set_boundary_displacements_eles(void)
{
  int ib0, ib1, ivb, iv, icv, ic, spt, ctype, j;
  array<double> disp(n_dims);

  // Calculate displacement for boundaries at next time step
  for (ib0 = 0; ib0 < n_bnds; ib0++) {
    if (bound_flags(ib0) == MOTION_ENABLED) {
      for (ib1=0; ib1<n_moving_bnds; ib1++) {
        if (bc_list(ib0)==bc_num[run_input.boundary_flags(ib1)]) break;
      }
      for (ivb = 0; ivb < nBndPts(ib0); ivb++) {
        iv = boundPts(ib0)(ivb); // Global id of boundary vertex
        for (j = 0; j<n_dims; j++) {
          //displacement(iv,j) = xv(0)(iv,j) + motion_params(ib1)(2*j  )*motion_params(ib1)(6+j)*sin(motion_params(ib1)(6+j)*(time+run_input.dt));
          displacement(iv,j) = motion_params(ib1)(2*j  )*sin(motion_params(ib1)(6+j)*(time+run_input.dt));
          displacement(iv,j)+= motion_params(ib1)(2*j+1)*cos(motion_params(ib1)(6+j)*(time+run_input.dt));
        }
      }
    }
  }

  // Assign displacement values to eles
  for (iv=0; iv<n_verts; iv++) {
    for (j=0; j<n_dims; j++)
      disp(j) = displacement(iv,j);

    // Set displacement to be average of values from all surrounding cells
    for (icv=0; icv<v2n_c(iv); icv++) {
      ctype = v2ctype(iv,icv);
      ic = v2c(iv,icv);
      spt = v2spt(iv,icv);
      FlowSol->mesh_eles(ctype)->set_displacement_spt(spt,ic,disp);
    }
  }

  // update solution (elas_disu_upts) in all cells which have been modified above

}

/*! update displacements of mesh nodes using values computed from linear-elasticity solution in each ele */
void mesh::update_displacements(void)
{
  int iv, iiv, dim, spt, ic, ctype;
  array<double> disp(n_dims);

  for (iv=0; iv<n_verts; iv++) {
    for (dim=0; dim<n_dims; dim++)
      displacement(iv,dim) = 0.;

    // Set displacement to be average of values from all surrounding cells
    for (iiv=0; iiv<v2n_c(iv); iiv++) {
      ctype = v2ctype(iv,iiv);
      ic = v2c(iv,iiv);
      spt = v2spt(iv,iiv);
      FlowSol->mesh_eles(ctype)->get_displacement(spt,ic,disp);
      for (dim=0; dim<n_dims; dim++)
        displacement(iv,dim) += disp(dim)/v2n_c(iv);
//      if (disp(0)!=0 || disp(1)!=0) {
//        cout << "Hooray!" << endl << flush << cin.get();
//      }
    }
  }
}

/*! using the final displacements, update the list of vertex positions */
void mesh::update_mesh_nodes(void)
{
  int iv, dim;
  double new_coord;

  if (rk_step==0) {
    // Push back previous time-advance level w/o de-/re-allocating memory
    for (int i=4; i>0; i--) {
      for (int j=0; j<xv(i).get_dim(0); j++) {
        for (int k=0; k<n_dims; k++) {
          xv(i)(j,k) = xv(i-1)(j,k);
        }
      }
    }
  }

  for (iv = 0; iv < n_verts; iv++) {
    for (dim = 0; dim < n_dims; dim++) {
      new_coord = xv(0)(iv,dim) + displacement(iv,dim);
      if (fabs(new_coord) < eps*eps) new_coord = 0.0; // necessary?
      xv(0)(iv,dim) = new_coord;
    }
  }
}

/*! Apply the boundary displacements to the flux points on the boundaries */
void mesh::apply_boundary_displacements_fpts(void)
{
  int ib, ic, icg, loc_c, loc_f;

  for (ib=0; ib<n_bnds; ib++) {
    for (ic=0; ic<bc_ncells(ib); ic++) {
      icg = bccells(ib)(ic);
      loc_c = ic2loc_c(icg);
      loc_f = bcfaces(ib)(ic);
      FlowSol->mesh_eles(ctype(icg))->apply_boundary_displacement_fpts(loc_f,loc_c);
    }
  }
}

/* If using moving mesh, need to advance the Geometric Conservation Law
 * (GCL) first to get updated Jacobians. Necessary to preserve freestream
 * on arbitrarily deforming mesh. See Kui Ou's Ph.D. thesis for details. */
/* Residual for Geometric Conservation Law (GCL) */
/*
if (run_input.GCL) {
  CalcResidualGCL(&FlowSol);

  // Time integration for Geometric Conservation Law (GCL) using a RK scheme
  for(j=0; j<FlowSol.n_ele_types; j++) {

    // Time Advance
    FlowSol.mesh_eles(j)->AdvanceGCL(i, FlowSol.adv_type);

    // Extrapolate Jacobians to flux points
    FlowSol.mesh_eles(j)->extrapolate_GCL_solution(0);

    // Reset transforms using updated Jacobians
    FlowSol.mesh_eles(j)->correct_dynamic_transforms();
  }
}*/
