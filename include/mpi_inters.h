/*!
 * \file mpi_inters.h
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

#include "inters.h"
#include "mpi_inters.h"
#include "array.h"
#include "solution.h"

#ifdef _MPI
#include "mpi.h"
#endif

struct solution; // forwards declaration

class mpi_inters: public inters
{
public:

  // #### constructors ####

  // default constructor

  mpi_inters();

  // default destructor

  ~mpi_inters();

  // #### methods ####

  /*! setup mpi_inters */
  void setup(int in_n_inters, int in_inter_type, int in_run_type);

  void set_nproc(int in_nproc, int in_rank);

  void set_nout_proc(int in_nout,int in_p);

  void set_mpi_requests(int in_number_of_request);

  void send_disu_fpts();

  void receive_disu_fpts();

  void send_epsilon_fpts();

  void receive_epsilon_fpts();

  void send_cor_grad_disu_fpts();

  void receive_cor_grad_disu_fpts();

  void set_mpi(int in_inter, int in_ele_type_l, int in_ele_l, int in_local_inter_l, int rot_tag, int in_run_type, struct solution* FlowSol);

  void calc_norm_tconinvf_fpts_mpi(void);
  void calc_norm_tconvisf_fpts_mpi(void);

  /*! move all from cpu to gpu */
  void mv_all_cpu_gpu(void);

protected:

  // #### members ####

  array<double*> disu_fpts_r;
  array<double*> grad_disu_fpts_r;
  array<double*> epsilon_fpts_r;

  int nproc;
  int rank;
  int Nmess;

  array<double> out_buffer_disu, in_buffer_disu;
  array<int> Nout_proc;

  // Viscous
  array<double> out_buffer_grad_disu, in_buffer_grad_disu;
  array<double> out_buffer_epsilon_fpts, in_buffer_epsilon_fpts;

#ifdef _MPI
  MPI_Request *mpi_out_requests;
  MPI_Request *mpi_in_requests;
  MPI_Request *mpi_out_requests_grad;
  MPI_Request *mpi_in_requests_grad;

  MPI_Status *mpi_instatus;
  MPI_Status *mpi_outstatus;
#endif

};
