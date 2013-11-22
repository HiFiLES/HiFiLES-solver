/*!
 * \file eles.h
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

#pragma once

#include "array.h"
#include "input.h"

#if defined _GPU
#include "cuda_runtime_api.h"
#include "cusparse_v2.h"
#endif

class eles
{	
	public:
	
	// #### constructors ####
	
	// default constructor
	
	eles();

  // default destructor
 
  ~eles(); 

	// #### methods ####

	/*! setup */
	void setup(int in_n_eles, int in_max_s_spts_per_ele, int in_run_type);

  /*! returns the number of ppts for all eles */
  int calc_num_ppts();

  array<double> calc_pos_pnode_vert(int in_ele, int in_vert);

  array<double> calc_pos_pnode_edge(int in_ele, int in_edge, int in_edge_ppt);

  array<double> calc_pos_pnode_face(int in_ele, int in_face, int in_face_ppt);

  array<double> calc_pos_pnode_interior(int in_ele, int in_interior_ppt);

  void set_pnode_vert(int in_ele, int in_vert, int in_pnode);

  void set_pnode_edge(int in_ele, int in_edge, int in_edge_ppt, int in_pnode);

  void set_pnode_face(int in_ele, int in_face, int in_face_ppt,  int in_pnode);

  void set_pnode_interior(int in_ele, int in_interior_ppt, int in_pnode);

  int get_n_interior_ppts();

  int get_n_ppts_per_face(int in_face);

  int get_max_n_ppts_per_face();

  virtual void create_map_ppt(void)=0;

	/*! setup initial conditions */
	void set_ics(double& time);

  /*! read data from restart file */
  void read_restart_data(ifstream& restart_file);

  /*! write data to restart file */
  void write_restart_data(ofstream& restart_file);

	/*! move all to from cpu to gpu */
	void mv_all_cpu_gpu(void);

	/*! copy transformed discontinuous solution at solution points to cpu */
	void cp_disu_upts_gpu_cpu(void);

	/*! copy transformed discontinuous solution at solution points to gpu */
  void cp_disu_upts_cpu_gpu(void);

  void cp_grad_disu_upts_gpu_cpu(void);

	/*! copy determinant of jacobian at solution points to cpu */
	void cp_detjac_upts_gpu_cpu(void);

  /*! copy divergence at solution points to cpu */
  void cp_div_tconf_upts_gpu_cpu(void);

	/*! remove transformed discontinuous solution at solution points from cpu */
	void rm_disu_upts_cpu(void);

	/*! remove determinant of jacobian at solution points from cpu */
	void rm_detjac_upts_cpu(void);

  /*! calculate the discontinuous solution at the flux points */
	void calc_disu_fpts(int in_disu_upts_from);
	
	/*! Calculate filtered solution globally */
	void calc_disuf_upts(int in_disu_upts_from);

	/*! calculate transformed discontinuous inviscid flux at solution points */
  void calc_tdisinvf_upts(int in_disu_upts_from);
  
  /*! calculate divergence of transformed discontinuous flux at solution points */
  void calc_div_tdisf_upts(int in_div_tconf_upts_to);
  
  /*! calculate normal transformed discontinuous flux at flux points */
  void calc_norm_tdisf_fpts(void);
  
  /*! calculate divergence of transformed continuous flux at solution points */
  void calc_div_tconf_upts(int in_div_tconf_upts_to);
  
	/*! calculate uncorrected transformed gradient of the discontinuous solution at the solution points */
  void calc_uncor_tgrad_disu_upts(int in_disu_upts_from);
	
  /*! calculate corrected gradient of the discontinuous solution at solution points */
	void calc_cor_grad_disu_upts(void);
		
  /*! calculate corrected gradient of the discontinuous solution at flux points */
	void calc_cor_grad_disu_fpts(void);

	/*! calculate corrected gradient of solution at flux points */
	//void calc_cor_grad_disu_fpts(void);
	
	/*! calculate transformed discontinuous viscous flux at solution points */
  void calc_tdisvisf_upts(int in_disu_upts_from);
    
  /*! calculate divergence of transformed discontinuous viscous flux at solution points */
  //void calc_div_tdisvisf_upts(int in_div_tconinvf_upts_to);
    
  /*! calculate normal transformed discontinuous viscous flux at flux points */
  //void calc_norm_tdisvisf_fpts(void);
    
  /*! calculate divergence of transformed continuous viscous flux at solution points */
  //void calc_div_tconvisf_upts(int in_div_tconinvf_upts_to);
  
  /*! advance with rk11 (forwards euler) */
  void advance_rk11(void);
  
  /*! advance with rk33 (three-stage third-order runge-kutta) */
  void advance_rk33(int in_step);
  
  /*! advance with rk44 (four-stage forth-order runge-kutta) */
  void advance_rk44(int in_step);
  
  /*! advance with rk45 (five-stage forth-order low-storage runge-kutta) */
  void advance_rk45(int in_step);

  /*! get number of elements */
  int get_n_eles(void);
       
  // get number of ppts_per_ele
  int get_n_ppts_per_ele(void);

  // get number of peles_per_ele 
  int get_n_peles_per_ele(void);

  // get number of verts_per_ele 
  int get_n_verts_per_ele(void);

  /*! get number of solution points per element */
  int get_n_upts_per_ele(void);
  
  /*! get element type */
  int get_ele_type(void);
      
  /*! get number of dimensions */
  int get_n_dims(void);

  /*! get number of fields */
  int get_n_fields(void);
  
  /*! set shape */
  void set_shape(int in_max_n_spts_per_ele);

  /*! set shape node */
  void set_shape_node(int in_spt, int in_ele, array<double>& in_pos);

  /*! set bc type */
  void set_bctype(int in_ele, int in_inter, int in_bctype);

  /*! set bc type */
  void set_bdy_ele2ele(void);

  /*! set number of shape points */
  void set_n_spts(int in_ele, int in_n_spts);

  /*!  set global element number */
  void set_ele2global_ele(int in_ele, int in_global_ele);

  /*! get a pointer to the transformed discontinuous solution at a flux point */
	double* get_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);
  
  /*! get a pointer to the normal transformed continuous flux at a flux point */
  double* get_norm_tconf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);
       
  /*! get a pointer to the determinant of the jacobian at a flux point */
  double* get_detjac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);
       
  /*! get a pointer to the magntiude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at flux points */
	double* get_mag_tnorm_dot_inv_detjac_mul_jac_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_ele);
        
  /*! get a pointer to the normal at a flux point */
  double* get_norm_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);
        
  /*! get a pointer to the coordinates at a flux point */
  double* get_loc_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_ele);
        
  /*! get a pointer to delta of the transformed discontinuous solution at a flux point */
	double* get_delta_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);

	/*! get a pointer to gradient of discontinuous solution at a flux point */
	double* get_grad_disu_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_dim, int in_field, int in_ele);

	/*! get a pointer to the normal transformed continuous viscous flux at a flux point */
	//double* get_norm_tconvisf_fpts_ptr(int in_inter_local_fpt, int in_ele_local_inter, int in_field, int in_ele);
  
  /*! set opp_0 */
  void set_opp_0(int in_sparse);
  
  /*! set opp_1 */
	void set_opp_1(int in_sparse);

	/*! set opp_2 */	
	void set_opp_2(int in_sparse);
	
	/*! set opp_3 */
	void set_opp_3(int in_sparse);
	
	/*! set opp_4 */
	void set_opp_4(int in_sparse);
	
	/*! set opp_5 */
	void set_opp_5(int in_sparse);
	
	/*! set opp_6 */
	void set_opp_6(int in_sparse);
	
	/*! set opp_p */
	void set_opp_p(void);

	/*! set opp_p */
	void set_opp_inters_cubpts(void);

	/*! set opp_p */
	void set_opp_volume_cubpts(void);

	/*! set opp_r */
	void set_opp_r(void);
	
  /*! calculate position of the plot points */
  void calc_pos_ppts(int in_ele, array<double>& out_pos_ppts);
      
  /*! get position of the plot points inside one cell*/
  void get_pos_ppts(int in_ele, array<double>& out_pos_ppts);

  /*! get position of a single plot point */
  array<double> get_pos_ppt(int in_ele, int in_ppt);

  /*! set position of the plot points for all eles*/
  void set_pos_ppts();

  void set_rank(int in_rank);

  virtual void set_connectivity_plot()=0;

  void set_disu_upts_to_zero_other_levels(void);

  int* get_connectivity_plot_ptr();

  array<int> get_connectivity_plot();

  void get_plotq_ppts(int in_ele, array<double> &out_plotq_ppts, array<double>& plotq_pnodes);

  /*! return the list of pnodes on face loc_f of cell ic_l */
  void get_face_pnode_list(array<int>& out_inter_pnodes, int ic_l, int loc_f, int& out_n_inter_pnodes);

  /*! calculate solution at the plot points */
  void calc_disu_ppts(int in_ele, array<double>& out_disu_ppts);
       
  /*! calculate solution at the plot points and store it*/
  void add_contribution_to_pnodes(array<double> &plotq_pnodes);

  /*! calculate position of a solution point */
  void calc_pos_upt(int in_upt, int in_ele, array<double>& out_pos);
       
  /*! returns position of a solution point */
  double get_loc_upt(int in_upt, int in_dim);

  /*! set transforms */
  void set_transforms(int in_run_type);
       
  /*! set transforms at the interface cubature points*/
  void set_transforms_inters_cubpts(void);

  /*! set transforms at the volume cubature points*/
  void set_transforms_vol_cubpts(void);

  /*! calculate position */
  void calc_pos(array<double> in_loc, int in_ele, array<double>& out_pos);

  /*! calculate derivative of position */
  void calc_d_pos(array<double> in_loc, int in_ele, array<double>& out_d_pos);
  
  /*! calculate second derivative of position */
  void calc_dd_pos(array<double> in_loc, int in_ele, array<double>& out_dd_pos);
  
  // #### virtual methods ####

  virtual void setup_ele_type_specific(int in_run_type)=0;

  virtual int read_restart_info(ifstream& restart_file)=0;  

  virtual void write_restart_info(ofstream& restart_file)=0;  

  /*! Compute interface jacobian determinant on face */
  virtual double compute_inter_detjac_inters_cubpts(int in_inter, array<double> d_pos)=0;

  /*! evaluate nodal basis */
	virtual double eval_nodal_basis(int in_index, array<double> in_loc)=0;

  /*! evaluate nodal basis for restart file*/
	virtual double eval_nodal_basis_restart(int in_index, array<double> in_loc)=0;

	/*! evaluate derivative of nodal basis */
	virtual double eval_d_nodal_basis(int in_index, int in_cpnt, array<double> in_loc)=0;

  virtual void fill_opp_3(array<double>& opp_3)=0;
	
	/*! evaluate divergence of vcjh basis */
	//virtual double eval_div_vcjh_basis(int in_index, array<double>& loc)=0;

	/*! evaluate nodal shape basis */
	virtual double eval_nodal_s_basis(int in_index, array<double> in_loc, int in_n_spts)=0;
	
	/*! evaluate derivative of nodal shape basis */
  virtual void eval_d_nodal_s_basis(array<double> &d_nodal_s_basis, array<double> in_loc, int in_n_spts)=0;
	
	/*! evaluate second derivative of nodal shape basis */
  virtual void eval_dd_nodal_s_basis(array<double> &dd_nodal_s_basis, array<double> in_loc, int in_n_spts)=0;

	/*! Calculate SGS flux */
	void calc_sgsf_upts(array<double>& temp_u, array<double>& temp_grad_u, double& detjac, int upt, array<double>& temp_sgsf);

	/*! Filter state variables and calculate Leonard tensor in an element */
	void calc_disuf_upts_ele(array<double>& in_u, array<double>& out_u);

	/*! Calculate element volume */
	virtual double calc_ele_vol(double& detjac)=0;

  double compute_res_upts(int in_norm_type, int in_field);
   
	/*! calculate body forcing at solution points */
  void calc_body_force_upts(array <double>& vis_force, array <double>& body_force);

	/*! add body forcing at solution points */
	void add_body_force_upts(array <double>& body_force);

	/*! Compute volume integral of diagnostic quantities */
	void CalcDiagnostics(int n_diagnostics, array <double>& diagnostic_array);

  void compute_wall_forces(array<double>& inv_force, array<double>& vis_force,ofstream& cp_file );

  array<double> compute_error(int in_norm_type, double& time);
  
  array<double> get_pointwise_error(array<double>& sol, array<double>& grad_sol, array<double>& loc, double& time, int in_norm_type);


	protected:
		
	// #### members ####
	
	/*! viscous flag */
	int viscous;
	
	/*! number of elements */
	int n_eles;

	/*! number of elements that have a boundary face*/
	int n_bdy_eles;

	/*!  number of dimensions */
	int n_dims;
	
	/*!  number of fields */
	int n_fields;
	
	/*! order of solution polynomials */
	int order; 

  /*! order of interface cubature rule */
  int inters_cub_order;

  /*! order of interface cubature rule */
  int volume_cub_order;

	/*! order of solution polynomials in restart file*/
	int order_rest;
	
	/*! number of solution points per element */
	int n_upts_per_ele;

	/*! number of solution points per element */
	int n_upts_per_ele_rest;
	
	/*! number of flux points per element */
	int n_fpts_per_ele;

  int n_verts_per_ele;
  int n_edges_per_ele;

  array<int> vert_to_ppt;
  array<int> edge_ppt_to_ppt;
  array< array<int> > face_ppt_to_ppt;
  array< array<int> > face2_ppt_to_ppt;
  array<int> interior_ppt_to_ppt;

  array<int> n_ppts_per_face;
  array<int> n_ppts_per_face2;

  int max_n_ppts_per_face;
  int n_ppts_per_edge;
  int n_interior_ppts;

  array<int> connectivity_plot;

	/*! plotting resolution */
	int p_res;
	
	/*! solution point type */
	int upts_type;
	
	/*! flux point type */
	int fpts_type;
	
	/*! number of plot points per element */
	int n_ppts_per_ele;

	/*! number of plot elements per element */
	int n_peles_per_ele;

  /*! Global cell number of element */
  array<int> ele2global_ele;	

  /*! Global cell number of element */
  array<int> bdy_ele2ele;	
 
  /*! Boundary condition type of faces */ 
  array<int> bctype;

	/*! number of shape points per element */
	array<int> n_spts_per_ele;
	
	/*! transformed normal at flux points */
	array<double> tnorm_fpts;

	/*! transformed normal at flux points */
	array< array<double> > tnorm_inters_cubpts;

	/*! location of solution points in standard element */
	array<double> loc_upts;

	/*! location of solution points in standard element */
	array<double> loc_upts_rest;

	/*! location of flux points in standard element */
	array<double> tloc_fpts;

	/*! location of interface cubature points in standard element */
	array< array<double> > loc_inters_cubpts;

	/*! weight of interface cubature points in standard element */
	array< array<double> > weight_inters_cubpts;

	/*! location of volume cubature points in standard element */
	array<double> loc_volume_cubpts;

	/*! weight of cubature points in standard element */
	array<double> weight_volume_cubpts;

  /*! transformed normal at cubature points */
	array< array<double> > tnorm_cubpts;

	/*! location of plot points in standard element */
	array<double> loc_ppts;
	
	/*! location of shape points in standard element (simplex elements only)*/
	array<double> loc_spts;
	
	/*! number of interfaces per element */
	int n_inters_per_ele;
	
	/*! number of flux points per interface */
	array<int> n_fpts_per_inter; 

	/*! number of cubature points per interface */
	array<int> n_cubpts_per_inter; 

	/*! number of cubature points per interface */
	int n_cubpts_per_ele; 

	/*! element type (0=>quad,1=>tri,2=>tet,3=>pri,4=>hex) */
	int ele_type; 
	
	/*! order of polynomials defining shapes */
	int s_order;
	
	/*! shape */
	array<double> shape;
	
	/*! temporary solution storage at a single solution point */
	array<double> temp_u;

	/*! temporary solution gradient storage */
	array<double> temp_grad_u;

	/*! Matrix of filter weights at solution points */
	array<double> filter_upts;

	/* Leonard tensors for WSM model */
	array<double> Lm, Hm;

	/*! temporary flux storage */
	array<double> temp_f;
	array<double> temp_sgsf;
	
	/*! number of storage levels for time-integration scheme */
	int n_adv_levels;
	
	/*! determinant of jacobian at solution points */
	array<double> detjac_upts;
	
	/*! determinant of jacobian at flux points */
	array<double> detjac_fpts;

	/*! determinant of volume jacobian at flux points */
	array< array<double> > vol_detjac_inters_cubpts;

	/*! determinant of volume jacobian at cubature points */
	array< array<double> > vol_detjac_vol_cubpts;

	/*! inverse of (determinant of jacobian multiplied by jacobian) at solution points */
	array<double> inv_detjac_mul_jac_upts;
	
	array<double> inv_detjac_mul_jac_fpts;
	
	/*! magntiude of normal dot inverse of (determinant of jacobian multiplied by jacobian) at flux points */
	array<double> mag_tnorm_dot_inv_detjac_mul_jac_fpts;

	/*! determinant of interface jacobian at flux points */
	array< array<double> > inter_detjac_inters_cubpts;

	/*! normal at flux points*/
	array<double> norm_fpts;
	
  /*! physical coordinates at flux points*/
	array<double> loc_fpts;

	/*! normal at interface cubature points*/
	array< array<double> > norm_inters_cubpts;

	/*!
	description: transformed discontinuous solution at the solution points
	indexing: \n
	matrix mapping:
	*/
	array< array<double> > disu_upts;

	/*!
	plot data at plot points
	*/
	 array<int> ppt_to_pnode;

  /*! position at the plot points */
  array< array<double> > pos_ppts;

	/*!
	description: transformed discontinuous solution at the flux points \n
	indexing: (in_fpt, in_field, in_ele) \n
	matrix mapping: (in_fpt || in_field, in_ele)
	*/
	array<double> disu_fpts;

	/*!
	description: transformed discontinuous flux at the solution points \n
	indexing: (in_upt, in_dim, in_field, in_ele) \n
	matrix mapping: (in_upt, in_dim || in_field, in_ele)
	*/
	array<double> tdisf_upts;
	
	/*!
	normal transformed discontinuous flux at the flux points
	indexing: \n
	matrix mapping:
	*/
	array<double> norm_tdisf_fpts;
	
	/*!
	normal transformed continuous flux at the flux points
	indexing: \n
	matrix mapping:
	*/
	array<double> norm_tconf_fpts;
	
	/*!
	divergence of transformed continuous flux at the solution points
	indexing: \n
	matrix mapping:
	*/
	array< array<double> > div_tconf_upts;
	
	/*! delta of the transformed discontinuous solution at the flux points   */
	array<double> delta_disu_fpts;

	/*! gradient of discontinuous solution at solution points */
	array<double> grad_disu_upts;
	
	/*! gradient of discontinuous solution at flux points */
	array<double> grad_disu_fpts;

	/*! transformed discontinuous viscous flux at the solution points */
	//array<double> tdisvisf_upts;
	
	/*! normal transformed discontinuous viscous flux at the flux points */
	//array<double> norm_tdisvisf_fpts;
	
	/*! normal transformed continuous viscous flux at the flux points */
	//array<double> norm_tconvisf_fpts;
		
	/*! transformed gradient of determinant of jacobian at solution points */
	array<double> tgrad_detjac_upts;
	
	/*! transformed gradient of determinant of jacobian at flux points */
	array<double> tgrad_detjac_fpts;

  array<double> d_nodal_s_basis;
  array<double> dd_nodal_s_basis;
	// TODO: change naming (comments) to reflect reuse

  #ifdef _GPU
  cusparseHandle_t handle;
  #endif 

	/*! operator to go from transformed discontinuous solution at the solution points to transformed discontinuous solution at the flux points */
	array<double> opp_0;
	array<double> opp_0_data;
	array<int> opp_0_cols;
	array<int> opp_0_b;
	array<int> opp_0_e;
	int opp_0_sparse;

  #ifdef _GPU
  array<double> opp_0_ell_data;
  array<int> opp_0_ell_indices;
  int opp_0_nnz_per_row;
  #endif

	/*! operator to go from transformed discontinuous inviscid flux at the solution points to divergence of transformed discontinuous inviscid flux at the solution points */
	array< array<double> > opp_1;
	array< array<double> > opp_1_data;
	array< array<int> > opp_1_cols;
	array< array<int> > opp_1_b;
	array< array<int> > opp_1_e;
	int opp_1_sparse;
  #ifdef _GPU
  array< array<double> > opp_1_ell_data;
  array< array<int> > opp_1_ell_indices;
  array<int> opp_1_nnz_per_row;
  #endif
	
	/*! operator to go from transformed discontinuous inviscid flux at the solution points to normal transformed discontinuous inviscid flux at the flux points */
	array< array<double> > opp_2;
	array< array<double> > opp_2_data;
	array< array<int> > opp_2_cols;
	array< array<int> > opp_2_b;
	array< array<int> > opp_2_e;
	int opp_2_sparse;
  #ifdef _GPU
  array< array<double> > opp_2_ell_data;
  array< array<int> > opp_2_ell_indices;
  array<int> opp_2_nnz_per_row;
  #endif
	
	/*! operator to go from normal correction inviscid flux at the flux points to divergence of correction inviscid flux at the solution points*/
	array<double> opp_3;
	array<double> opp_3_data;
	array<int> opp_3_cols;
	array<int> opp_3_b;
	array<int> opp_3_e;
	int opp_3_sparse;
  #ifdef _GPU
  array<double> opp_3_ell_data;
  array<int> opp_3_ell_indices;
  int opp_3_nnz_per_row;
  #endif
	
	/*! operator to go from transformed solution at solution points to transformed gradient of transformed solution at solution points */
	array< array<double> >  opp_4;
	array< array<double> >  opp_4_data;
	array< array<int> > opp_4_cols;
	array< array<int> > opp_4_b;
	array< array<int> > opp_4_e;
	int opp_4_sparse;
  #ifdef _GPU
  array< array<double> > opp_4_ell_data;
  array< array<int> > opp_4_ell_indices;
  array< int > opp_4_nnz_per_row;
  #endif
	
	/*! operator to go from transformed solution at flux points to transformed gradient of transformed solution at solution points */
	array< array<double> > opp_5;
	array< array<double> > opp_5_data;
	array< array<int> > opp_5_cols;
	array< array<int> > opp_5_b;
	array< array<int> > opp_5_e;
	int opp_5_sparse;
  #ifdef _GPU
  array< array<double> > opp_5_ell_data;
  array< array<int> > opp_5_ell_indices;
  array<int> opp_5_nnz_per_row;
  #endif
	
	/*! operator to go from transformed solution at solution points to transformed gradient of transformed solution at flux points */
	array<double> opp_6;
	array<double> opp_6_data;
	array<int> opp_6_cols;
	array<int> opp_6_b;
	array<int> opp_6_e;
	int opp_6_sparse;
  #ifdef _GPU
  array<double> opp_6_ell_data;
  array<int> opp_6_ell_indices;
  int opp_6_nnz_per_row;
  #endif
	
	/*! operator to go from discontinuous solution at the solution points to discontinuous solution at the plot points */
	array<double> opp_p;

  array< array<double> > opp_inters_cubpts;
  array<double> opp_volume_cubpts;

	/*! operator to go from discontinuous solution at the restart points to discontinuous solution at the solutoin points */
	array<double> opp_r;

	/*! dimensions for blas calls */
	int Arows, Acols;
	int Brows, Bcols;
	int Astride, Bstride, Cstride;
	
	/*! general settings for mkl sparse blas */
	char matdescra[6];
	
	/*! transpose setting for mkl sparse blas */
	char transa;
	
	/*! zero for mkl sparse blas */
	double zero;
	
	/*! one for mkl sparse blas */
	double one;
	
	/*! number of fields multiplied by number of elements */
	int n_fields_mul_n_eles;
	
	/*! number of dimensions multiplied by number of solution points per element */
	int n_dims_mul_n_upts_per_ele;

  int rank;

};
