/*!
 * \file mesh_deform.cpp
 * \brief  - Perform mesh deformation using linear elasticity method
 * \author - Current development: Aerospace Computing Laboratory (ACL) directed
 *                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
 * \version 1.0.0
 *
 * HiFiLES (High Fidelity Large Eddy Simulation).
 * Copyright (C) 2013 Aerospace Computing Laboratory.
 */

#include "../include/mesh.h"
#include "../include/geometry.h"
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

void mesh::move(int _iter, int time_level, solution *FlowSol) {
  iter = _iter;

  if (run_input.motion == 1) {
    deform(FlowSol);
  }else if (run_input.motion == 2) {
    rigid_move(FlowSol);
  }else if (run_input.motion == 3) {
    perturb(FlowSol);
  }
}

void mesh::deform(struct solution* FlowSol) {
  array<double> stiff_mat_ele;
  int failedIts = 0;
  /*
    if (FlowSol->n_dims == 2) {
        stiff_mat_ele.setup(6,6);
    }else{
        FatalError("3D Mesh motion not implemented yet!");
    }
    */
  /// cout << endl << ">>>>>>>>> Beginning Mesh Deformation >>>>>>>>>" << endl;
  int pt_0,pt_1,pt_2,pt_3;
  bool check;

  min_vol = check_grid(FlowSol);
  set_min_length();
  /** cout << "n_verts: " << n_verts << ", ";
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
  xv = xv_new;

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
    length2 = pow((xv(e2v(i,0),0)-xv(e2v(i,1),0)),2) + pow((xv(e2v(i,0),1)-xv(e2v(i,1),1)),2);
    min_length2 = fmin(min_length2,length2);
  }

  min_length = sqrt(min_length2);
}

void mesh::set_grid_velocity(solution* FlowSol, double dt)
{
  time = iter*dt;
  // calculate velocity using simple backward-Euler
  for (int i=0; i<n_verts; i++) {
    for (int j=0; j<n_dims; j++) {
      /// --- IMPLEMENT RK45 TIMESTEPPING ---
      if (run_input.adv_type == 0) {
        vel_new(i,j) = (xv_new(i,j) - xv(i,j))/dt;
      }else if (run_input.adv_type == 3) {
        cout << "Terribly sorry, but RK45 timestepping for mesh velocity has not been implemented yet! ";
        cout << " Using Forward Euler instead." << endl;
        vel_new(i,j) = (xv_new(i,j) - xv(i,j))/dt;
      }

      /// Analytic solution for perturb test-case
      //vel_new(i,0) = 4*pi/10*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/8)*cos(2*pi*time/10); // from Kui
      //vel_new(i,1) = 6*pi/10*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/8)*cos(4*pi*time/10);
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
    FlowSol->mesh_eles(i)->set_grid_vel_fpts();
    FlowSol->mesh_eles(i)->set_grid_vel_upts();
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
      pos_spts(i,j) = xv(iPoint,j);
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
    for (iVar = 0; iVar < 6; iVar++) {
      for (jVar = 0; jVar < 3; jVar++) {
        Aux_Matrix[iVar][jVar] = 0.0;
        for (kVar = 0; kVar < 3; kVar++)
          Aux_Matrix[iVar][jVar] += BT_Matrix[iVar][kVar]*D_Matrix[kVar][jVar];
      }
    }

    /*--- Compute the BT.D.B Matrix (stiffness matrix) ---*/
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
        pos(k) = xv_new(c2v(ic,iv),k);
      }
      FlowSol->mesh_eles(ele_type)->set_dynamic_shape_node(iv,local_id,pos);
    }
  }

  // Update element transforms
  //if (FlowSol->rank==0) cout << "Deform: updating element transforms ... " << endl;
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      FlowSol->mesh_eles(i)->set_transforms(run_input.run_type);
    }
  }

  // Set metrics at interface cubpts
  //if (FlowSol->rank==0) cout << "Deform: setting element transforms at interface cubature points ... " << endl;
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      FlowSol->mesh_eles(i)->set_transforms_inters_cubpts();
    }
  }

  // Set metrics at volume cubpts
  //if (FlowSol->rank==0) cout << "Deform: setting element transforms at volume cubature points ... " << endl;
  for(int i=0;i<FlowSol->n_ele_types;i++) {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
      FlowSol->mesh_eles(i)->set_transforms_vol_cubpts();
    }
  }
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
    file << i+1 << " " << xv(i,0) << " " << xv(i,1) << " ";
    if (n_dims==2) {
      file << 0;
    }else{
      file << xv(i,2);
    }
    file << endl;
  }
  file << "$EndNodes" << endl;

  // write elements
  file << "$Elements" << endl << n_cells_global << endl;
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

  for (iPoint = 0; iPoint < n_verts; iPoint++) {
    for (iDim = 0; iDim < n_dims; iDim++) {
      total_index = iPoint*n_dims + iDim;
      new_coord = xv(iPoint,iDim) + LinSysSol[total_index];
      if (fabs(new_coord) < eps*eps) new_coord = 0.0;
      xv_new(iPoint,iDim) = new_coord;
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
        a[iDim] = xv(c2v(iElem,0),iDim)-xv(c2v(iElem,1),iDim);
        b[iDim] = xv(c2v(iElem,1),iDim)-xv(c2v(iElem,2),iDim);
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
        r1[iDim] = xv(c2v(iElem,1),iDim) - xv(c2v(iElem,0),iDim);
        r2[iDim] = xv(c2v(iElem,2),iDim) - xv(c2v(iElem,0),iDim);
        r3[iDim] = xv(c2v(iElem,3),iDim) - xv(c2v(iElem,0),iDim);
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
  VarCoord(0) = run_input.bound_vel_simple(0)*run_input.dt;
  VarCoord(1) = run_input.bound_vel_simple(1)*run_input.dt;
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
  time = iter*run_input.dt;
  for (int i=0; i<n_verts; i++) {
    // Useful for simple cases / debugging
    xv_new(i,0) = xv_0(i,0) + 0.0*cos(2*pi*time/5)*run_input.dt;
    xv_new(i,1) = xv_0(i,1) + 0.5*cos(2*pi*time/5)*run_input.dt;

    //xv_new(i,0) = xv(i,0) + run_input.bound_vel_simple(0)*run_input.dt;
    //xv_new(i,1) = xv(i,1) + run_input.bound_vel_simple(1)*run_input.dt;
  }

  update(FlowSol);

  xv = xv_new;
  iter++;
}

void mesh::perturb(solution* FlowSol) {
  time = iter*run_input.dt;
  for (int i=0; i<n_verts; i++) {
    /// Taken from Kui, AIAA-2010-5031-661
    xv_new(i,0) = xv_0(i,0) + 1*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/10)*sin(2*pi*time/100);
    xv_new(i,1) = xv_0(i,1) + .75*sin(pi*xv_0(i,0)/10)*sin(pi*xv_0(i,1)/10)*sin(4*pi*time/100);
  }

  update(FlowSol);

  xv = xv_new;
  iter++;
}
