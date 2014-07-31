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

  bc_name["Sub_In_Simp"] = 1;
  bc_name["Sub_Out_Simp"] = 2;
  bc_name["Sub_In_Char"] = 3;
  bc_name["Sub_Out_Char"] = 4;
  bc_name["Sup_In"] = 5;
  bc_name["Sup_Out"] = 6;
  bc_name["Slip_Wall"] = 7;
  bc_name["Cyclic"] = 9;
  bc_name["Isotherm_Fix"] = 11;
  bc_name["Adiabat_Fix"] = 12;
  bc_name["Isotherm_Move"] = 13;
  bc_name["Adiabat_Move"] = 14;
  bc_name["Char"] = 15;
  bc_name["Slip_Wall_Dual"] = 16;
  bc_name["AD_Wall"] = 50;

  bc_flag[1] = "Sub_In_Simp";
  bc_flag[2] = "Sub_Out_Simp";
  bc_flag[3] = "Sub_In_Char";
  bc_flag[4] = "Sub_Out_Char";
  bc_flag[5] = "Sup_In";
  bc_flag[6]= "Sup_Out";
  bc_flag[7]= "Slip_Wall";
  bc_flag[9]= "Cyclic";
  bc_flag[11]= "Isotherm_Fix";
  bc_flag[12]= "Adiabat_Fix";
  bc_flag[13]= "Isotherm_Move";
  bc_flag[14]= "Adiabat_Move";
  bc_flag[15]= "Char";
  bc_flag[16]= "Slip_Wall_Dual";
  bc_flag[50]= "AD_Wall";
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

void mesh::move(int _iter, int in_rk_step, solution *FlowSol) {
  iter = _iter;
  rk_step = in_rk_step;
  time = FlowSol->time;
  rk_time = time; //+run_input.dt*RK_c(rk_step);
  run_input.rk_time = rk_time;

  if (run_input.motion == 1) {
    deform(FlowSol);
  }else if (run_input.motion == 2) {
    rigid_move(FlowSol);
  }else if (run_input.motion == 3) {
    perturb(FlowSol);
  }else{
    // Do Nothing
  }
}

void mesh::deform(struct solution* FlowSol) {
  array<double> stiff_mat_ele;
  int failedIts = 0;

  array<int> nodes; // NEW ADDITION 3/26/2014

  /// cout << endl << ">>>>>>>>> Beginning Mesh Deformation >>>>>>>>>" << endl;
  int pt_0,pt_1,pt_2,pt_3;
  bool check;

  min_vol = check_grid(FlowSol);
  set_min_length();
  /* cout << "n_verts: " << n_verts << ", ";
    cout << "n_dims: " << n_dims << ", ";
    cout << "min_vol = " << min_vol << endl; */

  // Setup stiffness matrices for each individual element,
  // combine all element-level matrices into global matrix
  //stiff_mat.setup(n_eles);
  LinSysSol.Initialize(n_verts,n_dims,0.0); /// should it be n_verts or n_verts_global?
  LinSysRes.Initialize(n_verts,n_dims,0.0);
  StiffnessMatrix.Initialize(n_verts,n_verts_global,n_dims,n_dims,v2e,v2n_e,e2v);

  /*--- Loop over the total number of grid deformation iterations. The surface
    deformation can be divided into increments to help with stability. In
    particular, the linear elasticity equations hold only for small deformations. ---*/
  for (int iGridDef_Iter = 0; iGridDef_Iter < run_input.n_deform_iters; iGridDef_Iter++) {
    //cout << ">>Iteration " << iGridDef_Iter+1 << " of " << run_input.n_deform_iters << endl;
    /*--- Initialize vector and sparse matrix ---*/

    LinSysSol.SetValZero();
    LinSysRes.SetValZero();
    StiffnessMatrix.SetValZero();

    /*--- Compute the stiffness matrix entries for all nodes/elements in the
        mesh. FEA uses a finite element method discretization of the linear
        elasticity equations (transfers element stiffnesses to point-to-point). ---*/

    for (int ic=0; ic<n_eles; ic++) {
//      //--- NEW ADDITION 3/26/14 ---//
//      nodes.setup(c2n_v(ic));
//      for (int iNode=0; iNode<c2n_v(ic); iNode++) {
//        nodes(iNode) = iv2ivg(c2v(ic,iNode)); // will the iv2ivg be needed for MPI?
//      }
//      if (n_dims == 2) {
//        set_stiffmat_ele_2d(stiff_mat_ele,ic,min_vol);
//      }else if (n_dims == 3) {
//        set_stiffmat_ele_3d(stiff_mat_ele,ic,min_vol);
//      }
//      add_FEA_stiffMat(stiff_mat_ele,nodes);
//      //----------------------------//

      switch(ctype(ic))
      {
        case TRI:
          pt_0 = iv2ivg(c2v(ic,0));
          pt_1 = iv2ivg(c2v(ic,1));
          pt_2 = iv2ivg(c2v(ic,2));
          check = set_2D_StiffMat_ele_tri(stiff_mat_ele,ic);
          add_StiffMat_EleTri(stiff_mat_ele,pt_0,pt_1,pt_2);
          break;
        case QUAD:
          pt_0 = iv2ivg(c2v(ic,0));
          pt_1 = iv2ivg(c2v(ic,1));
          pt_2 = iv2ivg(c2v(ic,2));
          pt_3 = iv2ivg(c2v(ic,3));
          set_2D_StiffMat_ele_quad(stiff_mat_ele,ic);
          add_StiffMat_EleQuad(stiff_mat_ele,pt_0,pt_1,pt_2,pt_3);
          break;
        default:
          FatalError("Element type not yet supported for mesh motion - supported types are tris and quads");
          break;
      }
      if (!check) {
        failedIts++;
        if (failedIts > 5) FatalError("ERROR: negative volumes encountered during mesh motion.");
      }else{
        failedIts=0;
      }
    }

    /*--- Compute the tolerance of the linear solver using MinLength ---*/
    solver_tolerance = min_length * 1E-2;

    /*--- Set the boundary displacements (as prescribed by the design variable
        perturbations controlling the surface shape) as a Dirichlet BC. ---*/
    set_boundary_displacements(FlowSol);

    /*--- Fix the location of any points in the domain, if requested. ---*/
    /*
        if (config->GetHold_GridFixed())
            SetDomainDisplacements(FlowSol);
        */

    /*--- Communicate any prescribed boundary displacements via MPI,
        so that all nodes have the same solution and r.h.s. entries
        across all paritions. ---*/
    /// HELP!!! Need Tom/Francisco to decipher what's being sent & how it's used
    //StiffMatrix.SendReceive_Solution(LinSysSol, FlowSol);
    //StiffMatrix.SendReceive_Solution(LinSysRes, FlowSol);

    /*--- Definition of the preconditioner matrix vector multiplication, and linear solver ---*/
    CMatrixVectorProduct* mat_vec = new CSysMatrixVectorProduct(StiffnessMatrix, FlowSol);
    CPreconditioner* precond      = new CLU_SGSPreconditioner(StiffnessMatrix, FlowSol);
    CSysSolve *system             = new CSysSolve();

    /*--- Solve the linear system ---*/
    bool display_statistics = false;
    LinSolIters = system->FGMRES(LinSysRes, LinSysSol, *mat_vec, *precond, solver_tolerance, 100, display_statistics, FlowSol);

    /*--- Deallocate memory needed by the Krylov linear solver ---*/
    delete system;
    delete mat_vec;
    delete precond;

    /*--- Update the grid coordinates and cell volumes using the solution
        of the linear system (usol contains the x, y, z displacements). ---*/
    update_grid_coords();

    /*--- Check for failed deformation (negative volumes). ---*/
    min_vol = check_grid(FlowSol);
    set_min_length();

    bool mesh_monitor = false;
    if (FlowSol->rank == 0 && mesh_monitor) {
      cout << "Non-linear iter.: " << iGridDef_Iter << "/" << run_input.n_deform_iters
           << ". Linear iter.: " << LinSolIters << ". Min vol.: " << min_vol
           << ". Error: " << solver_tolerance << "." <<endl;
    }
  }

  /*--- Update grid velocity & dynamic element transforms ---*/
  update(FlowSol);

  /*--- Now that deformation is complete & velocity is set, update the
      'official' vertex coordinates ---*/
  //xv = xv_new;

  /*--- Deallocate vectors for the linear system. ---*/
  LinSysSol.~CSysVector();
  LinSysRes.~CSysVector();
  StiffnessMatrix.~CSysMatrix();
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

/*! set individual-element stiffness matrix for a triangle */
bool mesh::set_2D_StiffMat_ele_tri(array<double> &stiffMat_ele, int ele_id)
{
  int iPoint;
  int n_spts = c2n_v(ele_id);

  array<double> pos_spts;
  pos_spts.setup(n_spts,n_dims);

  for (int i=0; i<n_spts; i++) {
    iPoint = c2v(ele_id,i);
    for (int j=0; j<n_dims; j++) {
      pos_spts(i,j) = xv(0)(iPoint,j);
    }
  }

  stiffMat_ele.setup(6,6);
  stiffMat_ele.initialize_to_zero();

  // ----------- Create single-element stiffness matrix ---------------
  // Copied from SU2
  unsigned short iDim, iVar, jVar, kVar;
  double B_Matrix[6][12], BT_Matrix[12][6], D_Matrix[6][6], Aux_Matrix[12][6];
  double a[3], b[3], c[3], Area, E, Mu, Lambda;

  for (iDim = 0; iDim < n_dims; iDim++) {
    a[iDim] = pos_spts(0,iDim)-pos_spts(2,iDim);
    b[iDim] = pos_spts(1,iDim)-pos_spts(2,iDim);
  }

  Area = 0.5*fabs(a[0]*b[1]-a[1]*b[0]);

  if (Area < 0.0) {

    /*--- The initial grid has degenerate elements ---*/
    return false;
  }else{

    /*--- Each element uses their own stiffness which is inversely
        proportional to the area/volume of the cell. Using Mu = E & Lambda = -E
        is a modification to help allow rigid rotation of elements (see
        "Robust Mesh Deformation using the Linear Elasticity Equations" by
        R. P. Dwight. ---*/

    E = 1.0 / Area * fabs(min_vol);
    Mu = E;
    Lambda = -E;

    a[0] = 0.5 * (pos_spts(1,0)*pos_spts(2,1)-pos_spts(2,0)*pos_spts(1,1)) / Area;
    a[1] = 0.5 * (pos_spts(2,0)*pos_spts(0,1)-pos_spts(0,0)*pos_spts(2,1)) / Area;
    a[2] = 0.5 * (pos_spts(0,0)*pos_spts(1,1)-pos_spts(1,0)*pos_spts(0,1)) / Area;

    b[0] = 0.5 * (pos_spts(1,1)-pos_spts(2,1)) / Area;
    b[1] = 0.5 * (pos_spts(2,1)-pos_spts(0,1)) / Area;
    b[2] = 0.5 * (pos_spts(0,1)-pos_spts(1,1)) / Area;

    c[0] = 0.5 * (pos_spts(2,0)-pos_spts(1,0)) / Area;
    c[1] = 0.5 * (pos_spts(0,0)-pos_spts(2,0)) / Area;
    c[2] = 0.5 * (pos_spts(1,0)-pos_spts(0,0)) / Area;

    /*--- Compute the B Matrix ---*/
    B_Matrix[0][0] = b[0];  B_Matrix[0][1] = 0.0;   B_Matrix[0][2] = b[1];  B_Matrix[0][3] = 0.0;   B_Matrix[0][4] = b[2];  B_Matrix[0][5] = 0.0;
    B_Matrix[1][0] = 0.0;   B_Matrix[1][1] = c[0];  B_Matrix[1][2] = 0.0;   B_Matrix[1][3] = c[1];  B_Matrix[1][4] = 0.0;   B_Matrix[1][5] = c[2];
    B_Matrix[2][0] = c[0];  B_Matrix[2][1] = b[0];  B_Matrix[2][2] = c[1];  B_Matrix[2][3] = b[1];  B_Matrix[2][4] = c[2];  B_Matrix[2][5] = b[2];

    for (iVar = 0; iVar < 3; iVar++)
      for (jVar = 0; jVar < 6; jVar++)
        BT_Matrix[jVar][iVar] = B_Matrix[iVar][jVar];

    /*--- Compute the D Matrix (for plane strain and 3-D)---*/
    D_Matrix[0][0] = Lambda + 2.0*Mu;		D_Matrix[0][1] = Lambda;            D_Matrix[0][2] = 0.0;
    D_Matrix[1][0] = Lambda;            D_Matrix[1][1] = Lambda + 2.0*Mu;   D_Matrix[1][2] = 0.0;
    D_Matrix[2][0] = 0.0;               D_Matrix[2][1] = 0.0;               D_Matrix[2][2] = Mu;

    /*--- Compute the BT.D Matrix ---*/
    /// IMPLEMENT BLAS?
    for (iVar = 0; iVar < 6; iVar++) {
      for (jVar = 0; jVar < 3; jVar++) {
        Aux_Matrix[iVar][jVar] = 0.0;
        for (kVar = 0; kVar < 3; kVar++)
          Aux_Matrix[iVar][jVar] += BT_Matrix[iVar][kVar]*D_Matrix[kVar][jVar];
      }
    }

    /*--- Compute the BT.D.B Matrix (stiffness matrix) ---*/
    /// IMPLEMENT BLAS?
    for (iVar = 0; iVar < 6; iVar++) {
      for (jVar = 0; jVar < 6; jVar++) {
        stiffMat_ele(iVar,jVar) = 0.0;
        for (kVar = 0; kVar < 3; kVar++)
          stiffMat_ele(iVar,jVar) += Area * Aux_Matrix[iVar][kVar]*B_Matrix[kVar][jVar];
      }
    }

    return true;
  }
}

/*! set individual-element stiffness matrix for a quadrilateral */
bool mesh::set_2D_StiffMat_ele_quad(array<double> &stiffMat_ele,int ele_id) {
  FatalError("ERROR: Sorry, mesh motion on quads not yet implemented.  :( ");
}


// ---- **NEW** Added 3/26/14 ---- //
void mesh::set_stiffmat_ele_2d(array<double> &stiffMat_ele, int ic, double scale)
{
  double B_Matrix[3][8], D_Matrix[3][3], Aux_Matrix[8][3];
  double Xi = 0.0, Eta = 0.0, Det, E, Lambda, Mu;
  unsigned short iNode, iVar, jVar, kVar, iGauss, nGauss;
  double DShapeFunction[8][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
  double Location[4][3], Weight[4], CoordCorners[8][3];
  unsigned short nVar = (unsigned short)n_dims;

  // First, get the ID's & coordinates of the nodes for this element
  int nNodes = c2n_v(ic);
  array<int> nodes(nNodes);
  array<double> coords(nNodes,2);

  for (int i=0; i<nNodes; i++) {
    nodes(i) = iv2ivg(c2v(ic,i));
    for (int j=0; j<n_dims; j++) {
      CoordCorners[i][j] = xv(0)(nodes(i),j);
      coords(i,j) = xv(0)(nodes(i),j);
    }
  }

  /*--- Each element uses their own stiffness which is inversely
   proportional to the area/volume of the cell. Using Mu = E & Lambda = -E
   is a modification to help allow rigid rotation of elements (see
   "Robust Mesh Deformation using the Linear Elasticity Equations" by
   R. P. Dwight. ---*/

  /*--- Integration formulae from "Shape functions and points of
   integration of the Résumé" by Josselin Delmas (2013) ---*/

  // Initialize the stiffness matrix for this element accordingly
  switch(ctype(ic))
  {
    case TRI:
      // note that this is for first-order integration only (higher-order [curved-edge] elements not currently supported)
      stiffMat_ele.setup(6,6);
      stiffMat_ele.initialize_to_zero();
      nGauss = 1;
      Location[0][0] = 0.333333333333333;  Location[0][1] = 0.333333333333333;  Weight[0] = 0.5;
      break;
    case QUAD:
      // note that this is for first-order integration only (higher-order [curved-edge] elements not currently supported)
      stiffMat_ele.setup(8,8);
      stiffMat_ele.initialize_to_zero();
      nGauss = 4;
      Location[0][0] = -0.577350269189626;  Location[0][1] = -0.577350269189626;  Weight[0] = 1.0;
      Location[1][0] = 0.577350269189626;   Location[1][1] = -0.577350269189626;  Weight[1] = 1.0;
      Location[2][0] = 0.577350269189626;   Location[2][1] = 0.577350269189626;   Weight[2] = 1.0;
      Location[3][0] = -0.577350269189626;  Location[3][1] = 0.577350269189626;   Weight[3] = 1.0;
      break;
  }

  for (iGauss = 0; iGauss < nGauss; iGauss++) {

    Xi = Location[iGauss][0]; Eta = Location[iGauss][1];

    if (nNodes == 3) Det = ShapeFunc_Triangle(Xi, Eta, CoordCorners, DShapeFunction);
    if (nNodes == 4) Det = ShapeFunc_Rectangle(Xi, Eta, CoordCorners, DShapeFunction);

    /*--- Compute the B Matrix ---*/

    for (iVar = 0; iVar < 3; iVar++)
      for (jVar = 0; jVar < nNodes*nVar; jVar++)
        B_Matrix[iVar][jVar] = 0.0;

    for (iNode = 0; iNode < nNodes; iNode++) {
      B_Matrix[0][0+iNode*nVar] = DShapeFunction[iNode][0];
      B_Matrix[1][1+iNode*nVar] = DShapeFunction[iNode][1];

      B_Matrix[2][0+iNode*nVar] = DShapeFunction[iNode][1];
      B_Matrix[2][1+iNode*nVar] = DShapeFunction[iNode][0];
    }

    /*--- Impose a type of stiffness for each element (proportional to inverse of volume) ---*/

    E = scale / (Weight[iGauss] * Det) ;
    Mu = E;
    Lambda = -E;

    /*--- Compute the D Matrix (for plane strain and 3-D)---*/

    D_Matrix[0][0] = Lambda + 2.0*Mu;		D_Matrix[0][1] = Lambda;            D_Matrix[0][2] = 0.0;
    D_Matrix[1][0] = Lambda;            D_Matrix[1][1] = Lambda + 2.0*Mu;   D_Matrix[1][2] = 0.0;
    D_Matrix[2][0] = 0.0;               D_Matrix[2][1] = 0.0;               D_Matrix[2][2] = Mu;

    /*--- Compute the BT.D Matrix ---*/

    for (iVar = 0; iVar < nNodes*nVar; iVar++) {
      for (jVar = 0; jVar < 3; jVar++) {
        Aux_Matrix[iVar][jVar] = 0.0;
        for (kVar = 0; kVar < 3; kVar++)
          Aux_Matrix[iVar][jVar] += B_Matrix[kVar][iVar]*D_Matrix[kVar][jVar];
      }
    }

    /*--- Compute the BT.D.B Matrix (stiffness matrix), and add to the original
     matrix using Gauss integration ---*/

    for (iVar = 0; iVar < nNodes*nVar; iVar++) {
      for (jVar = 0; jVar < nNodes*nVar; jVar++) {
        for (kVar = 0; kVar < 3; kVar++) {
          stiffMat_ele(iVar,jVar) += Weight[iGauss] * Aux_Matrix[iVar][kVar]*B_Matrix[kVar][jVar] * Det;
        }
      }
    }
  }
}


// ---- **NEW** Added 3/26/14 ---- //
void mesh::set_stiffmat_ele_3d(array<double> &stiffMat_ele, int ic, double scale)
{
  double B_Matrix[6][24], D_Matrix[6][6], Aux_Matrix[24][6];
  double Xi = 0.0, Eta = 0.0, Mu = 0.0, Det, E, Lambda, Nu, Avg_Wall_Dist;
  unsigned short iNode, jNode, iVar, jVar, kVar, iGauss, nGauss;
  double DShapeFunction[8][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
  double Location[8][3], Weight[8], CoordCorners[8][3];
  unsigned short nVar = (unsigned short)n_dims;

  // First, get the ID's & coordinates of the nodes for this element
  int nNodes = c2n_v(ic);
  array<int> nodes(nNodes);

  for (int i=0; i<nNodes; i++) {
    nodes(i) = iv2ivg(c2v(ic,i));
    for (int j=0; j<n_dims; j++) {
      CoordCorners[i][j] = xv(0)(nodes(i),j);
    }
  }

  /*--- Each element uses their own stiffness which is inversely
   proportional to the area/volume of the cell. Using Mu = E & Lambda = -E
   is a modification to help allow rigid rotation of elements (see
   "Robust Mesh Deformation using the Linear Elasticity Equations" by
   R. P. Dwight. ---*/

  /*--- Integration formulae from "Shape functions and points of
   integration of the Résumé" by Josselin Delmas (2013) ---*/

  // Initialize the stiffness matrix for this element accordingly
  switch(ctype(ic))
  {
    case TET:
      stiffMat_ele.setup(12,12);
      stiffMat_ele.initialize_to_zero();
      /*--- Tetrahedrons. Nodes of numerical integration at 1 point (order 1). ---*/
      nGauss = 1;
      Location[0][0] = 0.25;  Location[0][1] = 0.25;  Location[0][2] = 0.25;  Weight[0] = 0.166666666666666;
      break;
    case PYRAMID:
      stiffMat_ele.setup(15,15);
      stiffMat_ele.initialize_to_zero();
      /*--- Pyramids. Nodes numerical integration at 5 points. ---*/
      nGauss = 5;
      Location[0][0] = 0.5;   Location[0][1] = 0.0;   Location[0][2] = 0.1531754163448146;  Weight[0] = 0.133333333333333;
      Location[1][0] = 0.0;   Location[1][1] = 0.5;   Location[1][2] = 0.1531754163448146;  Weight[1] = 0.133333333333333;
      Location[2][0] = -0.5;  Location[2][1] = 0.0;   Location[2][2] = 0.1531754163448146;  Weight[2] = 0.133333333333333;
      Location[3][0] = 0.0;   Location[3][1] = -0.5;  Location[3][2] = 0.1531754163448146;  Weight[3] = 0.133333333333333;
      Location[4][0] = 0.0;   Location[4][1] = 0.0;   Location[4][2] = 0.6372983346207416;  Weight[4] = 0.133333333333333;
      break;
    case PRISM:
      stiffMat_ele.setup(18,18);
      stiffMat_ele.initialize_to_zero();
      /*--- Wedge. Nodes of numerical integration at 6 points (order 3 in Xi, order 2 in Eta and Mu ). ---*/
      nGauss = 6;
      Location[0][0] = 0.5;                 Location[0][1] = 0.5;                 Location[0][2] = -0.577350269189626;  Weight[0] = 0.166666666666666;
      Location[1][0] = -0.577350269189626;  Location[1][1] = 0.0;                 Location[1][2] = 0.5;                 Weight[1] = 0.166666666666666;
      Location[2][0] = 0.5;                 Location[2][1] = -0.577350269189626;  Location[2][2] = 0.0;                 Weight[2] = 0.166666666666666;
      Location[3][0] = 0.5;                 Location[3][1] = 0.5;                 Location[3][2] = 0.577350269189626;   Weight[3] = 0.166666666666666;
      Location[4][0] = 0.577350269189626;   Location[4][1] = 0.0;                 Location[4][2] = 0.5;                 Weight[4] = 0.166666666666666;
      Location[5][0] = 0.5;                 Location[5][1] = 0.577350269189626;   Location[5][2] = 0.0;                 Weight[5] = 0.166666666666666;
      break;
    case HEX:
      stiffMat_ele.setup(24,24);
      stiffMat_ele.initialize_to_zero();
      /*--- Hexahedrons. Nodes of numerical integration at 6 points (order 3). ---*/
      nGauss = 8;
      Location[0][0] = -0.577350269189626;  Location[0][1] = -0.577350269189626;  Location[0][2] = -0.577350269189626;  Weight[0] = 1.0;
      Location[1][0] = -0.577350269189626;  Location[1][1] = -0.577350269189626;  Location[1][2] = 0.577350269189626;   Weight[1] = 1.0;
      Location[2][0] = -0.577350269189626;  Location[2][1] = 0.577350269189626;   Location[2][2] = -0.577350269189626;  Weight[2] = 1.0;
      Location[3][0] = -0.577350269189626;  Location[3][1] = 0.577350269189626;   Location[3][2] = 0.577350269189626;   Weight[3] = 1.0;
      Location[4][0] = 0.577350269189626;   Location[4][1] = -0.577350269189626;  Location[4][2] = -0.577350269189626;  Weight[4] = 1.0;
      Location[5][0] = 0.577350269189626;   Location[5][1] = -0.577350269189626;  Location[5][2] = 0.577350269189626;   Weight[5] = 1.0;
      Location[6][0] = 0.577350269189626;   Location[6][1] = 0.577350269189626;   Location[6][2] = -0.577350269189626;  Weight[6] = 1.0;
      Location[7][0] = 0.577350269189626;   Location[7][1] = 0.577350269189626;   Location[7][2] = 0.577350269189626;   Weight[7] = 1.0;
      break;
  }

  for (iGauss = 0; iGauss < nGauss; iGauss++) {

    Xi = Location[iGauss][0]; Eta = Location[iGauss][1];  Mu = Location[iGauss][2];

    if (nNodes == 4) Det = ShapeFunc_Tetra(Xi, Eta, Mu, CoordCorners, DShapeFunction);
    if (nNodes == 5) Det = ShapeFunc_Pyram(Xi, Eta, Mu, CoordCorners, DShapeFunction);
    if (nNodes == 6) Det = ShapeFunc_Wedge(Xi, Eta, Mu, CoordCorners, DShapeFunction);
    if (nNodes == 8) Det = ShapeFunc_Hexa(Xi, Eta, Mu, CoordCorners, DShapeFunction);

    /*--- Compute the B Matrix ---*/

    for (iVar = 0; iVar < 6; iVar++)
      for (jVar = 0; jVar < nNodes*nVar; jVar++)
        B_Matrix[iVar][jVar] = 0.0;

    for (iNode = 0; iNode < nNodes; iNode++) {
      B_Matrix[0][0+iNode*nVar] = DShapeFunction[iNode][0];
      B_Matrix[1][1+iNode*nVar] = DShapeFunction[iNode][1];
      B_Matrix[2][2+iNode*nVar] = DShapeFunction[iNode][2];

      B_Matrix[3][0+iNode*nVar] = DShapeFunction[iNode][1];
      B_Matrix[3][1+iNode*nVar] = DShapeFunction[iNode][0];

      B_Matrix[4][1+iNode*nVar] = DShapeFunction[iNode][2];
      B_Matrix[4][2+iNode*nVar] = DShapeFunction[iNode][1];

      B_Matrix[5][0+iNode*nVar] = DShapeFunction[iNode][2];
      B_Matrix[5][2+iNode*nVar] = DShapeFunction[iNode][0];
    }

    /*--- Impose a type of stiffness for each element ---*/

    E = scale / (Weight[iGauss] * Det) ;
    Mu = E;
    Lambda = -E;


    /*--- Compute the D Matrix (for plane strain and 3-D)---*/

    D_Matrix[0][0] = Lambda + 2.0*Mu;	D_Matrix[0][1] = Lambda;					D_Matrix[0][2] = Lambda;					D_Matrix[0][3] = 0.0;	D_Matrix[0][4] = 0.0;	D_Matrix[0][5] = 0.0;
    D_Matrix[1][0] = Lambda;					D_Matrix[1][1] = Lambda + 2.0*Mu;	D_Matrix[1][2] = Lambda;					D_Matrix[1][3] = 0.0;	D_Matrix[1][4] = 0.0;	D_Matrix[1][5] = 0.0;
    D_Matrix[2][0] = Lambda;					D_Matrix[2][1] = Lambda;					D_Matrix[2][2] = Lambda + 2.0*Mu;	D_Matrix[2][3] = 0.0;	D_Matrix[2][4] = 0.0;	D_Matrix[2][5] = 0.0;
    D_Matrix[3][0] = 0.0;							D_Matrix[3][1] = 0.0;							D_Matrix[3][2] = 0.0;							D_Matrix[3][3] = Mu;	D_Matrix[3][4] = 0.0;	D_Matrix[3][5] = 0.0;
    D_Matrix[4][0] = 0.0;							D_Matrix[4][1] = 0.0;							D_Matrix[4][2] = 0.0;							D_Matrix[4][3] = 0.0;	D_Matrix[4][4] = Mu;	D_Matrix[4][5] = 0.0;
    D_Matrix[5][0] = 0.0;							D_Matrix[5][1] = 0.0;							D_Matrix[5][2] = 0.0;							D_Matrix[5][3] = 0.0;	D_Matrix[5][4] = 0.0;	D_Matrix[5][5] = Mu;


    /*--- Compute the BT.D Matrix ---*/

    for (iVar = 0; iVar < nNodes*nVar; iVar++) {
      for (jVar = 0; jVar < 6; jVar++) {
        Aux_Matrix[iVar][jVar] = 0.0;
        for (kVar = 0; kVar < 6; kVar++)
          Aux_Matrix[iVar][jVar] += B_Matrix[kVar][iVar]*D_Matrix[kVar][jVar];
      }
    }

    /*--- Compute the BT.D.B Matrix (stiffness matrix), and add to the original
     matrix using Gauss integration ---*/

    for (iVar = 0; iVar < nNodes*nVar; iVar++) {
      for (jVar = 0; jVar < nNodes*nVar; jVar++) {
        for (kVar = 0; kVar < 6; kVar++) {
          stiffMat_ele(iVar,jVar) += Weight[iGauss] * Aux_Matrix[iVar][kVar]*B_Matrix[kVar][jVar] * Det;
        }
      }
    }

  }
}

// ---- **NEW** 3/26/14
void mesh::add_FEA_stiffMat(array<double> &stiffMat_ele, array<int> &PointCorners) {
  unsigned short iVar, jVar, iDim, jDim;
  unsigned short nVar = (unsigned short)n_dims;
  unsigned short nNodes = (unsigned short)PointCorners.get_dim(0);

  double **StiffMatrix_Node;
  StiffMatrix_Node = new double* [nVar];
  for (iVar = 0; iVar < nVar; iVar++)
    StiffMatrix_Node[iVar] = new double [nVar];

  for (iVar = 0; iVar < nVar; iVar++)
    for (jVar = 0; jVar < nVar; jVar++)
      StiffMatrix_Node[iVar][jVar] = 0.0;

  /*--- Transform the stiffness matrix for the hexahedral element into the
   contributions for the individual nodes relative to each other. ---*/

  for (iVar = 0; iVar < nNodes; iVar++) {
    for (jVar = 0; jVar < nNodes; jVar++) {

      for (iDim = 0; iDim < nVar; iDim++) {
        for (jDim = 0; jDim < nVar; jDim++) {
          StiffMatrix_Node[iDim][jDim] = stiffMat_ele((iVar*nVar)+iDim,(jVar*nVar)+jDim);
        }
      }

      StiffnessMatrix.AddBlock(PointCorners(iVar), PointCorners(jVar), StiffMatrix_Node);

    }
  }

  /*--- Deallocate memory and exit ---*/

  for (iVar = 0; iVar < nVar; iVar++)
    delete StiffMatrix_Node[iVar];
  delete [] StiffMatrix_Node;

}

double mesh::ShapeFunc_Triangle(double Xi, double Eta, double CoordCorners[8][3], double DShapeFunction[8][4]) {

  int i, j, k;
  double c0, c1, xsj;
  double xs[3][3], ad[3][3];

  /*--- Shape functions ---*/

  DShapeFunction[0][3] = 1-Xi-Eta;
  DShapeFunction[1][3] = Xi;
  DShapeFunction[2][3] = Eta;

  /*--- dN/d xi, dN/d eta, dN/d mu ---*/

  DShapeFunction[0][0] = -1.0;  DShapeFunction[0][1] = -1.0;
  DShapeFunction[1][0] = 1;     DShapeFunction[1][1] = 0.0;
  DShapeFunction[2][0] = 0;     DShapeFunction[2][1] = 1;

  /*--- Jacobian transformation ---*/

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      xs[i][j] = 0.0;
      for (k = 0; k < 3; k++) {
        xs[i][j] = xs[i][j]+CoordCorners[k][j]*DShapeFunction[k][i];
      }
    }
  }

  /*--- Adjoint to jacobian ---*/

  ad[0][0] = xs[1][1];
  ad[0][1] = -xs[0][1];
  ad[1][0] = -xs[1][0];
  ad[1][1] = xs[0][0];

  /*--- Determinant of jacobian ---*/

  xsj = ad[0][0]*ad[1][1]-ad[0][1]*ad[1][0];

  /*--- Jacobian inverse ---*/

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      xs[i][j] = ad[i][j]/xsj;
    }
  }

  /*--- Derivatives with repect to global coordinates ---*/

  for (k = 0; k < 3; k++) {
    c0 = xs[0][0]*DShapeFunction[k][0]+xs[0][1]*DShapeFunction[k][1]; // dN/dx
    c1 = xs[1][0]*DShapeFunction[k][0]+xs[1][1]*DShapeFunction[k][1]; // dN/dy
    DShapeFunction[k][0] = c0; // store dN/dx instead of dN/d xi
    DShapeFunction[k][1] = c1; // store dN/dy instead of dN/d eta
  }

  return xsj;

}

double mesh::ShapeFunc_Rectangle(double Xi, double Eta, double CoordCorners[8][3], double DShapeFunction[8][4]) {

  int i, j, k;
  double c0, c1, xsj;
  double xs[3][3], ad[3][3];

  /*--- Shape functions ---*/

  DShapeFunction[0][3] = 0.25*(1.0-Xi)*(1.0-Eta);
  DShapeFunction[1][3] = 0.25*(1.0+Xi)*(1.0-Eta);
  DShapeFunction[2][3] = 0.25*(1.0+Xi)*(1.0+Eta);
  DShapeFunction[3][3] = 0.25*(1.0-Xi)*(1.0+Eta);

  /*--- dN/d xi, dN/d eta, dN/d mu ---*/

  DShapeFunction[0][0] = -0.25*(1.0-Eta); DShapeFunction[0][1] = -0.25*(1.0-Xi);
  DShapeFunction[1][0] =  0.25*(1.0-Eta); DShapeFunction[1][1] = -0.25*(1.0+Xi);
  DShapeFunction[2][0] =  0.25*(1.0+Eta); DShapeFunction[2][1] =  0.25*(1.0+Xi);
  DShapeFunction[3][0] = -0.25*(1.0+Eta); DShapeFunction[3][1] =  0.25*(1.0-Xi);

  /*--- Jacobian transformation ---*/

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      xs[i][j] = 0.0;
      for (k = 0; k < 4; k++) {
        xs[i][j] = xs[i][j]+CoordCorners[k][j]*DShapeFunction[k][i];
      }
    }
  }

  /*--- Adjoint to jacobian ---*/

  ad[0][0] = xs[1][1];
  ad[0][1] = -xs[0][1];
  ad[1][0] = -xs[1][0];
  ad[1][1] = xs[0][0];

  /*--- Determinant of jacobian ---*/

  xsj = ad[0][0]*ad[1][1]-ad[0][1]*ad[1][0];

  /*--- Jacobian inverse ---*/

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      xs[i][j] = ad[i][j]/xsj;
    }
  }

  /*--- Derivatives with repect to global coordinates ---*/

  for (k = 0; k < 4; k++) {
    c0 = xs[0][0]*DShapeFunction[k][0]+xs[0][1]*DShapeFunction[k][1]; // dN/dx
    c1 = xs[1][0]*DShapeFunction[k][0]+xs[1][1]*DShapeFunction[k][1]; // dN/dy
    DShapeFunction[k][0] = c0; // store dN/dx instead of dN/d xi
    DShapeFunction[k][1] = c1; // store dN/dy instead of dN/d eta
  }

  return xsj;

}

double mesh::ShapeFunc_Hexa(double Xi, double Eta, double Mu, double CoordCorners[8][3], double DShapeFunction[8][4]) {

  int i, j, k;
  double a0, a1, a2, c0, c1, c2, xsj;
  double ss[3], xs[3][3], ad[3][3];
  double s0[8] = {-0.5, 0.5, 0.5,-0.5,-0.5, 0.5,0.5,-0.5};
  double s1[8] = {-0.5,-0.5, 0.5, 0.5,-0.5,-0.5,0.5, 0.5};
  double s2[8] = {-0.5,-0.5,-0.5,-0.5, 0.5, 0.5,0.5, 0.5};

  ss[0] = Xi;
  ss[1] = Eta;
  ss[2] = Mu;

  /*--- Shape functions ---*/

  for (i = 0; i < 8; i++) {
    a0 = 0.5+s0[i]*ss[0]; // shape function in xi-direction
    a1 = 0.5+s1[i]*ss[1]; // shape function in eta-direction
    a2 = 0.5+s2[i]*ss[2]; // shape function in mu-direction
    DShapeFunction[i][0] = s0[i]*a1*a2; // dN/d xi
    DShapeFunction[i][1] = s1[i]*a0*a2; // dN/d eta
    DShapeFunction[i][2] = s2[i]*a0*a1; // dN/d mu
    DShapeFunction[i][3] = a0*a1*a2; // actual shape function N
  }

  /*--- Jacobian transformation ---*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = 0.0;
      for (k = 0; k < 8; k++) {
        xs[i][j] = xs[i][j]+CoordCorners[k][j]*DShapeFunction[k][i];
      }
    }
  }

  /*--- Adjoint to jacobian ---*/

  ad[0][0] = xs[1][1]*xs[2][2]-xs[1][2]*xs[2][1];
  ad[0][1] = xs[0][2]*xs[2][1]-xs[0][1]*xs[2][2];
  ad[0][2] = xs[0][1]*xs[1][2]-xs[0][2]*xs[1][1];
  ad[1][0] = xs[1][2]*xs[2][0]-xs[1][0]*xs[2][2];
  ad[1][1] = xs[0][0]*xs[2][2]-xs[0][2]*xs[2][0];
  ad[1][2] = xs[0][2]*xs[1][0]-xs[0][0]*xs[1][2];
  ad[2][0] = xs[1][0]*xs[2][1]-xs[1][1]*xs[2][0];
  ad[2][1] = xs[0][1]*xs[2][0]-xs[0][0]*xs[2][1];
  ad[2][2] = xs[0][0]*xs[1][1]-xs[0][1]*xs[1][0];

  /*--- Determinant of jacobian ---*/

  xsj = xs[0][0]*ad[0][0]+xs[0][1]*ad[1][0]+xs[0][2]*ad[2][0];

  /*--- Jacobian inverse ---*/
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = ad[i][j]/xsj;
    }
  }

  /*--- Derivatives with repect to global coordinates ---*/

  for (k = 0; k < 8; k++) {
    c0 = xs[0][0]*DShapeFunction[k][0]+xs[0][1]*DShapeFunction[k][1]+xs[0][2]*DShapeFunction[k][2]; // dN/dx
    c1 = xs[1][0]*DShapeFunction[k][0]+xs[1][1]*DShapeFunction[k][1]+xs[1][2]*DShapeFunction[k][2]; // dN/dy
    c2 = xs[2][0]*DShapeFunction[k][0]+xs[2][1]*DShapeFunction[k][1]+xs[2][2]*DShapeFunction[k][2]; // dN/dz
    DShapeFunction[k][0] = c0; // store dN/dx instead of dN/d xi
    DShapeFunction[k][1] = c1; // store dN/dy instead of dN/d eta
    DShapeFunction[k][2] = c2; // store dN/dz instead of dN/d mu
  }

  return xsj;

}

double mesh::ShapeFunc_Tetra(double Xi, double Eta, double Mu, double CoordCorners[8][3], double DShapeFunction[8][4]) {

  int i, j, k;
  double c0, c1, c2, xsj;
  double xs[3][3], ad[3][3];

  /*--- Shape functions ---*/

  DShapeFunction[0][3] = Xi;
  DShapeFunction[1][3] = Eta;
  DShapeFunction[2][3] = Mu;
  DShapeFunction[3][3] = 1.0 - Xi - Eta - Mu;

  /*--- dN/d xi, dN/d eta, dN/d mu ---*/

  DShapeFunction[0][0] = 1.0;   DShapeFunction[0][1] = 0.0;   DShapeFunction[0][2] = 0.0;
  DShapeFunction[1][0] = 0.0;   DShapeFunction[1][1] = 1.0;   DShapeFunction[1][2] = 0.0;
  DShapeFunction[2][0] = 0.0;   DShapeFunction[2][1] = 0.0;   DShapeFunction[2][2] = 1.0;
  DShapeFunction[3][0] = -1.0;  DShapeFunction[3][1] = -1.0;  DShapeFunction[3][2] = -1.0;

  /*--- Jacobian transformation ---*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = 0.0;
      for (k = 0; k < 4; k++) {
        xs[i][j] = xs[i][j]+CoordCorners[k][j]*DShapeFunction[k][i];
      }
    }
  }

  /*--- Adjoint to jacobian ---*/

  ad[0][0] = xs[1][1]*xs[2][2]-xs[1][2]*xs[2][1];
  ad[0][1] = xs[0][2]*xs[2][1]-xs[0][1]*xs[2][2];
  ad[0][2] = xs[0][1]*xs[1][2]-xs[0][2]*xs[1][1];
  ad[1][0] = xs[1][2]*xs[2][0]-xs[1][0]*xs[2][2];
  ad[1][1] = xs[0][0]*xs[2][2]-xs[0][2]*xs[2][0];
  ad[1][2] = xs[0][2]*xs[1][0]-xs[0][0]*xs[1][2];
  ad[2][0] = xs[1][0]*xs[2][1]-xs[1][1]*xs[2][0];
  ad[2][1] = xs[0][1]*xs[2][0]-xs[0][0]*xs[2][1];
  ad[2][2] = xs[0][0]*xs[1][1]-xs[0][1]*xs[1][0];

  /*--- Determinant of jacobian ---*/

  xsj = xs[0][0]*ad[0][0]+xs[0][1]*ad[1][0]+xs[0][2]*ad[2][0];

  /*--- Jacobian inverse ---*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = ad[i][j]/xsj;
    }
  }

  /*--- Derivatives with repect to global coordinates ---*/

  for (k = 0; k < 4; k++) {
    c0 = xs[0][0]*DShapeFunction[k][0]+xs[0][1]*DShapeFunction[k][1]+xs[0][2]*DShapeFunction[k][2]; // dN/dx
    c1 = xs[1][0]*DShapeFunction[k][0]+xs[1][1]*DShapeFunction[k][1]+xs[1][2]*DShapeFunction[k][2]; // dN/dy
    c2 = xs[2][0]*DShapeFunction[k][0]+xs[2][1]*DShapeFunction[k][1]+xs[2][2]*DShapeFunction[k][2]; // dN/dz
    DShapeFunction[k][0] = c0; // store dN/dx instead of dN/d xi
    DShapeFunction[k][1] = c1; // store dN/dy instead of dN/d eta
    DShapeFunction[k][2] = c2; // store dN/dz instead of dN/d mu
  }

  return xsj;

}

double mesh::ShapeFunc_Pyram(double Xi, double Eta, double Mu, double CoordCorners[8][3], double DShapeFunction[8][4]) {

  int i, j, k;
  double c0, c1, c2, xsj;
  double xs[3][3], ad[3][3];

  /*--- Shape functions ---*/
  double Den = 4.0*(1.0 - Mu);

  DShapeFunction[0][3] = (-Xi+Eta+Mu-1.0)*(-Xi-Eta+Mu-1.0)/Den;
  DShapeFunction[1][3] = (-Xi-Eta+Mu-1.0)*(Xi-Eta+Mu-1.0)/Den;
  DShapeFunction[2][3] = (Xi+Eta+Mu-1.0)*(Xi-Eta+Mu-1.0)/Den;
  DShapeFunction[3][3] = (Xi+Eta+Mu-1.0)*(-Xi+Eta+Mu-1.0)/Den;
  DShapeFunction[4][3] = Mu;

  /*--- dN/d xi, dN/d eta, dN/d mu ---*/

  DShapeFunction[0][0] = 0.5 + (0.5*Xi)/(1.0 - Mu);
  DShapeFunction[0][1] = (0.5*Eta)/(-1.0 + Mu);
  DShapeFunction[0][2] = (-0.25 - 0.25*Eta*Eta + (0.5 - 0.25*Mu)*Mu + 0.25*Xi*Xi)/((-1.0 + Mu)*(-1.0 + Mu));

  DShapeFunction[1][0] = (0.5*Xi)/(-1.0 + Mu);
  DShapeFunction[1][1] = (-0.5 - 0.5*Eta + 0.5*Mu)/(-1.0 + Mu);
  DShapeFunction[1][2] = (-0.25 + 0.25*Eta*Eta + (0.5 - 0.25*Mu)*Mu - 0.25*Xi*Xi)/((-1.0 + Mu)*(-1.0 + Mu));

  DShapeFunction[2][0] = -0.5 + (0.5*Xi)/(1.0 - 1.0*Mu);
  DShapeFunction[2][1] = (0.5*Eta)/(-1.0 + Mu);
  DShapeFunction[2][2] = (-0.25 - 0.25*Eta*Eta + (0.5 - 0.25*Mu)*Mu + 0.25*Xi*Xi)/((-1.0 + Mu)*(-1.0 + Mu));

  DShapeFunction[3][0] = (0.5*Xi)/(-1.0 + Mu);
  DShapeFunction[3][1] = (0.5 - 0.5*Eta - 0.5*Mu)/(-1.0 + Mu);
  DShapeFunction[3][2] = (-0.25 + 0.25*Eta*Eta + (0.5 - 0.25*Mu)*Mu - 0.25*Xi*Xi)/((-1.0 + Mu)*(-1.0 + Mu));

  DShapeFunction[4][0] = 0.0;
  DShapeFunction[4][1] = 0.0;
  DShapeFunction[4][2] = 1.0;

  /*--- Jacobian transformation ---*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = 0.0;
      for (k = 0; k < 5; k++) {
        xs[i][j] = xs[i][j]+CoordCorners[k][j]*DShapeFunction[k][i];
      }
    }
  }

  /*--- Adjoint to jacobian ---*/

  ad[0][0] = xs[1][1]*xs[2][2]-xs[1][2]*xs[2][1];
  ad[0][1] = xs[0][2]*xs[2][1]-xs[0][1]*xs[2][2];
  ad[0][2] = xs[0][1]*xs[1][2]-xs[0][2]*xs[1][1];
  ad[1][0] = xs[1][2]*xs[2][0]-xs[1][0]*xs[2][2];
  ad[1][1] = xs[0][0]*xs[2][2]-xs[0][2]*xs[2][0];
  ad[1][2] = xs[0][2]*xs[1][0]-xs[0][0]*xs[1][2];
  ad[2][0] = xs[1][0]*xs[2][1]-xs[1][1]*xs[2][0];
  ad[2][1] = xs[0][1]*xs[2][0]-xs[0][0]*xs[2][1];
  ad[2][2] = xs[0][0]*xs[1][1]-xs[0][1]*xs[1][0];

  /*--- Determinant of jacobian ---*/

  xsj = xs[0][0]*ad[0][0]+xs[0][1]*ad[1][0]+xs[0][2]*ad[2][0];

  /*--- Jacobian inverse ---*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = ad[i][j]/xsj;
    }
  }

  /*--- Derivatives with repect to global coordinates ---*/

  for (k = 0; k < 5; k++) {
    c0 = xs[0][0]*DShapeFunction[k][0]+xs[0][1]*DShapeFunction[k][1]+xs[0][2]*DShapeFunction[k][2]; // dN/dx
    c1 = xs[1][0]*DShapeFunction[k][0]+xs[1][1]*DShapeFunction[k][1]+xs[1][2]*DShapeFunction[k][2]; // dN/dy
    c2 = xs[2][0]*DShapeFunction[k][0]+xs[2][1]*DShapeFunction[k][1]+xs[2][2]*DShapeFunction[k][2]; // dN/dz
    DShapeFunction[k][0] = c0; // store dN/dx instead of dN/d xi
    DShapeFunction[k][1] = c1; // store dN/dy instead of dN/d eta
    DShapeFunction[k][2] = c2; // store dN/dz instead of dN/d mu
  }

  return xsj;

}

double mesh::ShapeFunc_Wedge(double Xi, double Eta, double Mu, double CoordCorners[8][3], double DShapeFunction[8][4]) {

  int i, j, k;
  double c0, c1, c2, xsj;
  double xs[3][3], ad[3][3];

  /*--- Shape functions ---*/

  DShapeFunction[0][3] = 0.5*Eta*(1.0-Xi);
  DShapeFunction[1][3] = 0.5*Mu*(1.0-Xi);;
  DShapeFunction[2][3] = 0.5*(1.0-Eta-Mu)*(1.0-Xi);
  DShapeFunction[3][3] = 0.5*Eta*(Xi+1.0);
  DShapeFunction[4][3] = 0.5*Mu*(Xi+1.0);
  DShapeFunction[5][3] = 0.5*(1.0-Eta-Mu)*(Xi+1.0);

  /*--- dN/d Xi, dN/d Eta, dN/d Mu ---*/

  DShapeFunction[0][0] = -0.5*Eta;            DShapeFunction[0][1] = 0.5*(1.0-Xi);      DShapeFunction[0][2] = 0.0;
  DShapeFunction[1][0] = -0.5*Mu;             DShapeFunction[1][1] = 0.0;               DShapeFunction[1][2] = 0.5*(1.0-Xi);
  DShapeFunction[2][0] = -0.5*(1.0-Eta-Mu);   DShapeFunction[2][1] = -0.5*(1.0-Xi);     DShapeFunction[2][2] = -0.5*(1.0-Xi);
  DShapeFunction[3][0] = 0.5*Eta;             DShapeFunction[3][1] = 0.5*(Xi+1.0);      DShapeFunction[3][2] = 0.0;
  DShapeFunction[4][0] = 0.5*Mu;              DShapeFunction[4][1] = 0.0;               DShapeFunction[4][2] = 0.5*(Xi+1.0);
  DShapeFunction[5][0] = 0.5*(1.0-Eta-Mu);    DShapeFunction[5][1] = -0.5*(Xi+1.0);     DShapeFunction[5][2] = -0.5*(Xi+1.0);

  /*--- Jacobian transformation ---*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = 0.0;
      for (k = 0; k < 6; k++) {
        xs[i][j] = xs[i][j]+CoordCorners[k][j]*DShapeFunction[k][i];
      }
    }
  }

  /*--- Adjoint to jacobian ---*/

  ad[0][0] = xs[1][1]*xs[2][2]-xs[1][2]*xs[2][1];
  ad[0][1] = xs[0][2]*xs[2][1]-xs[0][1]*xs[2][2];
  ad[0][2] = xs[0][1]*xs[1][2]-xs[0][2]*xs[1][1];
  ad[1][0] = xs[1][2]*xs[2][0]-xs[1][0]*xs[2][2];
  ad[1][1] = xs[0][0]*xs[2][2]-xs[0][2]*xs[2][0];
  ad[1][2] = xs[0][2]*xs[1][0]-xs[0][0]*xs[1][2];
  ad[2][0] = xs[1][0]*xs[2][1]-xs[1][1]*xs[2][0];
  ad[2][1] = xs[0][1]*xs[2][0]-xs[0][0]*xs[2][1];
  ad[2][2] = xs[0][0]*xs[1][1]-xs[0][1]*xs[1][0];

  /*--- Determinant of jacobian ---*/

  xsj = xs[0][0]*ad[0][0]+xs[0][1]*ad[1][0]+xs[0][2]*ad[2][0];

  /*--- Jacobian inverse ---*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      xs[i][j] = ad[i][j]/xsj;
    }
  }

  /*--- Derivatives with repect to global coordinates ---*/

  for (k = 0; k < 6; k++) {
    c0 = xs[0][0]*DShapeFunction[k][0]+xs[0][1]*DShapeFunction[k][1]+xs[0][2]*DShapeFunction[k][2]; // dN/dx
    c1 = xs[1][0]*DShapeFunction[k][0]+xs[1][1]*DShapeFunction[k][1]+xs[1][2]*DShapeFunction[k][2]; // dN/dy
    c2 = xs[2][0]*DShapeFunction[k][0]+xs[2][1]*DShapeFunction[k][1]+xs[2][2]*DShapeFunction[k][2]; // dN/dz
    DShapeFunction[k][0] = c0; // store dN/dx instead of dN/d xi
    DShapeFunction[k][1] = c1; // store dN/dy instead of dN/d eta
    DShapeFunction[k][2] = c2; // store dN/dz instead of dN/d mu
  }

  return xsj;

}


/*!
 * Transform element-defined stiffness matrix into node-base stiffness matrix for inclusion
 * into global stiffness matrix 'StiffMatrix'
 */
void mesh::add_StiffMat_EleTri(array<double> StiffMatrix_Elem, int id_pt_0,
                               int id_pt_1, int id_pt_2) {
  unsigned short nVar = n_dims;

  // Transform local -> global node ID
  id_pt_0 = iv2ivg(id_pt_0);
  id_pt_1 = iv2ivg(id_pt_1);
  id_pt_2 = iv2ivg(id_pt_1);

  array<double> StiffMatrix_Node;
  StiffMatrix_Node.setup(nVar,nVar);
  StiffMatrix_Node.initialize_to_zero();

  /*--- Transform the stiffness matrix for the triangular element into the
   contributions for the individual nodes relative to each other. ---*/
  StiffMatrix_Node(0,0) = StiffMatrix_Elem(0,0);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(0,1);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(1,0);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(1,1);
  StiffnessMatrix.AddBlock(id_pt_0, id_pt_0, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(0,2);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(0,3);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(1,2);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(1,3);
  StiffnessMatrix.AddBlock(id_pt_0, id_pt_1, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(0,4);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(0,5);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(1,4);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(1,5);
  StiffnessMatrix.AddBlock(id_pt_0, id_pt_2, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(2,0);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(2,1);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(3,0);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(3,1);
  StiffnessMatrix.AddBlock(id_pt_1, id_pt_0, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(2,2);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(2,3);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(3,2);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(3,3);
  StiffnessMatrix.AddBlock(id_pt_1, id_pt_1, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(2,4);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(2,5);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(3,4);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(3,5);
  StiffnessMatrix.AddBlock(id_pt_1, id_pt_2, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(4,0);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(4,1);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(5,0);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(5,1);
  StiffnessMatrix.AddBlock(id_pt_2, id_pt_0, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(4,2);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(4,3);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(5,2);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(5,3);
  StiffnessMatrix.AddBlock(id_pt_2, id_pt_1, StiffMatrix_Node);

  StiffMatrix_Node(0,0) = StiffMatrix_Elem(4,4);	StiffMatrix_Node(0,1) = StiffMatrix_Elem(4,5);
  StiffMatrix_Node(1,0) = StiffMatrix_Elem(5,4);	StiffMatrix_Node(1,1) = StiffMatrix_Elem(5,5);
  StiffnessMatrix.AddBlock(id_pt_2, id_pt_2, StiffMatrix_Node);
}

void mesh::add_StiffMat_EleQuad(array<double> StiffMatrix_Elem, int id_pt_0,
                                int id_pt_1, int id_pt_2, int id_pt_3)
{
  FatalError("ERROR: Mesh motion not setup on quads yet  :( ");
}


void mesh::update(solution* FlowSol)
{
  // Update grid velocity & transfer to upts, fpts
  //if (FlowSol->rank==0) cout << "Deform: updating grid velocity" << endl;

  set_grid_velocity(FlowSol,run_input.dt);

  // Update element shape points
  //if (FlowSol->rank==0) cout << "Deform: updating element shape points" << endl;

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
  //if (FlowSol->rank==0) cout << "Deform: updating element transforms ... " << endl;
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

void mesh::write_mesh(int mesh_type,double sim_time)
{
  if (mesh_type==0) {
    write_mesh_gambit(sim_time);
  }else if (mesh_type==1) {
    write_mesh_gmsh(sim_time);
  }else{
    cerr << "Mesh Output Type: " << mesh_type << endl;
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
      file << i+1  << " " << "\"" << bc_flag[bc_list(i)] << "\"" << endl;
    }

  }
  file << "$EndPhysicalNames" << endl;
  // write nodes
  file << "$Nodes" << endl << n_verts_global << endl;
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
      * (same as boundPts, but for faces) - since since faces, not edges, needed for 3D */
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
  //cout << "$EndElements" << endl;
  file.close();
}

void mesh::update_grid_coords(void)
{
  unsigned short iDim;
  unsigned long iPoint, total_index;
  double new_coord;

  /*--- Update the grid coordinates using the solution of the linear system
   after grid deformation (LinSysSol contains the x, y, z displacements). ---*/

  for (int i=4; i>0; i--) {
    xv(i) = xv(i-1);
  }

  for (iPoint = 0; iPoint < n_verts; iPoint++) {
    for (iDim = 0; iDim < n_dims; iDim++) {
      total_index = iPoint*n_dims + iDim;
      new_coord = xv(0)(iPoint,iDim) + LinSysSol[total_index];
      if (fabs(new_coord) < eps*eps) new_coord = 0.0;
      xv(0)(iPoint,iDim) = new_coord;
    }
  }
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

void mesh::set_boundary_displacements(solution *FlowSol)
{
  unsigned short iDim, nDim = FlowSol->n_dims, iBound, axis = 0;
  unsigned long iPoint, total_index, iVertex;
  //double MeanCoord[3];
  double VarIncrement = 1.0;

  /*--- If requested (no by default) impose the surface deflections in
    increments and solve the grid deformation equations iteratively with
    successive small deformations. ---*/

  VarIncrement = 1.0/((double)run_input.n_deform_iters);

  /*--- As initialization, set to zero displacements of all the surfaces except the symmetry
     plane and the receive boundaries. ---*/

  for (iBound = 0; iBound < n_bnds; iBound++) {
    //        my version: if ((bound_flag(ibound) != SYMMETRY_PLANE) && bound_flag(iBound) != MPI_BOUND)) {
    for (iVertex = 0; iVertex < nBndPts(iBound); iVertex++) {
      /// is iv2ivg needed for this?
      iPoint = iv2ivg(boundPts(iBound)(iVertex));
      for (iDim = 0; iDim < n_dims; iDim++) {
        total_index = iPoint*n_dims + iDim;
        LinSysRes[total_index] = 0.0;
        LinSysSol[total_index] = 0.0;
        StiffnessMatrix.DeleteValsRowi(total_index);
      }
    }
    //        }
  }

  /*--- Set to zero displacements of the normal component for the symmetry plane condition ---*/
  /*for (iBound = 0; iBound < config->GetnMarker_All(); iBound++) {
        if ((config->GetMarker_All_Boundary(iBound) == SYMMETRY_PLANE) && (nDim == 3)) {

            for (iDim = 0; iDim < nDim; iDim++) MeanCoord[iDim] = 0.0;
            for (iVertex = 0; iVertex < geometry->nVertex[iBound]; iVertex++) {
                iPoint = geometry->vertex[iBound][iVertex]->GetNode();
                VarCoord = geometry->node[iPoint]->GetCoord();
                for (iDim = 0; iDim < nDim; iDim++)
                    MeanCoord[iDim] += VarCoord[iDim]*VarCoord[iDim];
            }
            for (iDim = 0; iDim < nDim; iDim++) MeanCoord[iDim] = sqrt(MeanCoord[iDim]);

            if ((MeanCoord[0] <= MeanCoord[1]) && (MeanCoord[0] <= MeanCoord[2])) axis = 0;
            if ((MeanCoord[1] <= MeanCoord[0]) && (MeanCoord[1] <= MeanCoord[2])) axis = 1;
            if ((MeanCoord[2] <= MeanCoord[0]) && (MeanCoord[2] <= MeanCoord[1])) axis = 2;

            for (iVertex = 0; iVertex < geometry->nVertex[iBound]; iVertex++) {
                iPoint = geometry->vertex[iBound][iVertex]->GetNode();
                total_index = iPoint*nDim + axis;
                LinSysRes[total_index] = 0.0;
                LinSysSol[total_index] = 0.0;
                StiffnessMatrix.DeleteValsRowi(total_index);
            }
        }
    }*/

  array<double> VarCoord(n_dims);
  /*VarCoord(0) = run_input.bound_vel_simple(0)(0)*run_input.dt;
  VarCoord(1) = run_input.bound_vel_simple(0)(1)*run_input.dt;*/
  VarCoord(0) = 0;
  VarCoord(1) = run_input.bound_vel_simple(0)(0)*cos(2*run_input.bound_vel_simple(0)(1)*pi*rk_time)*run_input.dt;
  /// cout << "number of boundaries: " << n_bnds << endl;
  /*--- Set the known displacements, note that some points of the moving surfaces
    could be on on the symmetry plane, we should specify DeleteValsRowi again (just in case) ---*/
  for (iBound = 0; iBound < n_bnds; iBound++) {
    if (bound_flags(iBound) == MOTION_ENABLED) {
      for (iVertex = 0; iVertex < nBndPts(iBound); iVertex++) {
        iPoint = boundPts(iBound)(iVertex);
        // get amount which each point is supposed to move at this time step
        // **for now, set to a constant (setup data structure(s) later)**
        //VarCoord = geometry->vertex[iBound][iVertex]->GetVarCoord();
        for (iDim = 0; iDim < nDim; iDim++) {
          total_index = iPoint*nDim + iDim;
          LinSysRes[total_index] = VarCoord(iDim) * VarIncrement;
          LinSysSol[total_index] = VarCoord(iDim) * VarIncrement;
          StiffnessMatrix.DeleteValsRowi(total_index);
        }
      }
    }
  }
}

void mesh::rigid_move(solution* FlowSol) {
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

  update(FlowSol);
#endif

#ifdef _GPU
  for (int i=0;i<FlowSol->n_ele_types;i++) {
    FlowSol->mesh_eles(i)->rigid_move(rk_time);
    FlowSol->mesh_eles(i)->rigid_grid_velocity(rk_time);
    FlowSol->mesh_eles(i)->set_transforms_dynamic();
  }
#endif
}

void mesh::perturb(solution* FlowSol)
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

  update(FlowSol);
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
