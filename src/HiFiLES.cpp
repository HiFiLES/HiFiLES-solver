/*!
 * \file HiFiLES.cpp
 * \brief Main subrotuine of HiFiLES.
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL) directed
 *                                by Prof. Jameson. (Aero/Astro Dept. Stanford University).
 * \version 1.0.0
 *
 * HiFiLES (High Fidelity Large Eddy Simulation).
 * Copyright (C) 2013 Aerospace Computing Laboratory.
 */

#include <iostream>
#include <fstream>

#include "../include/global.h"
#include "../include/array.h"
#include "../include/funcs.h"
#include "../include/input.h"
#include "../include/flux.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/solution.h"

#ifdef _MPI
#include "mpi.h"
#endif

using namespace std;

int main(int argc, char *argv[]) {
  
	int rank = 0, error_state = 0;
  int i, j;                           /*!< Loop iterators */
  int i_steps = 0;                    /*!< Iteration index */
  ifstream run_input_file;            /*!< Config input file */
  clock_t init, final;                /*!< To control the time */
  struct solution FlowSol;            /*!< Main structure with the flow solution and geometry */
  ofstream write_force, write_stats;  /*!< Output files (forces and statistics) */

  /*! Check the command line input. */
  if (argc < 2) { cout << "ERROR: No input file specified ... " << endl; return(0); }
  
  /*! Initialize MPI. */
#ifdef _MPI
  MPI_Init(&argc, &argv);
  int nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#endif
  
  if (rank == 0) {
    cout << " _    _  _  ______  _  _       ______   _____ " << endl;
    cout << "| |  | |(_)|  ____|(_)| |     |  ____| / ____|" << endl;
    cout << "| |__| | _ | |__    _ | |     | |__   | (___  " << endl;
    cout << "|  __  || ||  __|  | || |     |  __|   \\___ \\ " << endl;
    cout << "| |  | || || |     | || |____ | |____  ____) |" << endl;
    cout << "|_|  |_||_||_|     |_||______||______||_____/ " << endl;
    cout << "                                              " << endl;
    cout << "Aerospace Computing Lab (Stanford University) " << endl;
  }

  /////////////////////////////////////////////////
  /// Read config file and mesh
  /////////////////////////////////////////////////
  
  /*! Read the config file and store the information in run_input. */
  run_input_file.open(argv[1], ifstream::in);
  if (!run_input_file) FatalError("Unable to open input file");
  run_input.setup(run_input_file, rank);
  
  /*! Set the input values in the FlowSol structure. */
  SetInput(&FlowSol);

  /*! Read the mesh file from a file. */
  GeoPreprocess(run_input.run_type, &FlowSol);
  
  InitSolution(&FlowSol);
  
  init = clock();
  
  /*! Just output. */
  if (run_input.run_type == 1) {
		plot_continuous(&FlowSol);
	  /*! Finalize MPI. */
#ifdef _MPI
	  MPI_Finalize();
#endif
		/*! Exit. */
		return(0);
	}
  
  /////////////////////////////////////////////////
  /// Pre-processing
  /////////////////////////////////////////////////
  
  /*! Variable initialization. */
  error_state = 0;
  FlowSol.ene_hist = 1000.;
  FlowSol.grad_ene_hist = 1000.;
  
  /*! Warning about body forcing term for periodic channel. */
  if (run_input.equation == 0 and run_input.run_type == 0 and run_input.forcing == 1) {
    if(run_input.monitor_force_freq>100)
      cout<<"WARNING: when running the periodic channel, it is necessary to add a body forcing term to prevent the flow decaying to zero. Make sure monitor_force_freq is set to a relatively small number, e.g. 100"<<endl;
    FlowSol.body_force.setup(5);
    for (i=0; i<5; i++) FlowSol.body_force(i)=0.0;
  }
  
  /*! Compute forces in the initial solution. */
  if (FlowSol.rank == 0) {
    write_force.open("force000.dat", ios::app);
    write_force << "new run" << endl;
    write_force.close();
    
    write_stats.open("statfile.dat");
    write_stats << "time ";
    for(j=0; j<run_input.n_diagnostics; ++j) { write_stats << run_input.diagnostics(j) << " "; }
    write_stats << endl;
    write_stats.close();
  }
  
  /*! Dump initial Paraview or tecplot file. */
  if (FlowSol.write_type == 0) write_vtu(FlowSol.ini_iter+i_steps, &FlowSol);
  else if (FlowSol.write_type == 1) write_tec(FlowSol.ini_iter+i_steps, &FlowSol);
  else FatalError("ERROR: Trying to write unrecognized file format ... ");
  
  /*! Compute diagnostics at t=0. */
  if(run_input.diagnostics_freq) {
    CalcDiagnostics(FlowSol.ini_iter+i_steps, FlowSol.time, &FlowSol);
  }
  
  if (FlowSol.rank == 0) cout << endl;
  
  /*! Main solver loop (outer loop). */
  while(i_steps < FlowSol.n_steps) {
    
    /////////////////////////////////////////////////
    /// Flow solver
    /////////////////////////////////////////////////
    
    /*! Advance the solution one time-step using a forward Euler method. */
    if(FlowSol.adv_type == 0) {
      CalcResidual(&FlowSol);
      for(j=0; j<FlowSol.n_ele_types; j++) FlowSol.mesh_eles(j)->advance_rk11();
    }
    
    /*! Advance the solution one time-step using a RK45 method. */
    else if(FlowSol.adv_type==3) {
      for(i=0; i<5; i++) {
        CalcResidual(&FlowSol);
        for(j=0; j<FlowSol.n_ele_types; j++) FlowSol.mesh_eles(j)->advance_rk45(i);
      }
    }
    
    /*! Time integration not implemented. */
    else { cout << "ERROR: Time integration type not recognised ... " << endl; }
    
    /*! Update total time. */
    FlowSol.time += run_input.dt;
    
    /*! Increase the iteration index. */
    i_steps++;
    
    /////////////////////////////////////////////////
    /// Post-processing (visualization)
    /////////////////////////////////////////////////
    
    /*! Dump residual and error. */
    if(i_steps%run_input.monitor_res_freq==0 ) {
      
      error_state = monitor_residual(FlowSol.ini_iter+i_steps, &FlowSol);
      
      if (error_state) cout << "error_state=" << error_state << "rank=" << FlowSol.rank << endl;
      
      if (run_input.test_case != 0) {
#ifdef _MPI
        /*! Check state of other processors. */
        MPI_Alltoall(&error_state, 1, MPI_INT, FlowSol.error_states.get_ptr_cpu(), 1, MPI_INT,MPI_COMM_WORLD);
        for (j=0; j<nproc; j++) { if (FlowSol.error_states(j) == 1) error_state=1; }
#endif
        if (error_state == 1) return 1;
        compute_error(FlowSol.ini_iter+i_steps, &FlowSol);
      }
      
    }
    
    /*! Dump forces. */
    if(i_steps%run_input.monitor_force_freq == 0 && run_input.equation == 0) {
      compute_forces(FlowSol.ini_iter+i_steps, FlowSol.time, &FlowSol);
    }
    
    if (i_steps%run_input.monitor_res_freq == 0 || i_steps%run_input.monitor_force_freq == 0)
      if (FlowSol.rank == 0) cout << endl;
    
    /*! Dump diagnostics. */
    if (i_steps%run_input.diagnostics_freq == 0) {
      CalcDiagnostics(FlowSol.ini_iter+i_steps, FlowSol.time, &FlowSol);
    }
    
    if (i_steps%run_input.diagnostics_freq == 0)
      if (FlowSol.rank == 0) cout << endl;
    
    /*! Dump Paraview or Tecplot file. */
    if(i_steps%FlowSol.plot_freq == 0) {
      if(FlowSol.write_type == 0) write_vtu(FlowSol.ini_iter+i_steps, &FlowSol);
      else if(FlowSol.write_type == 1) write_tec(FlowSol.ini_iter+i_steps, &FlowSol);
      else FatalError("ERROR: Trying to write unrecognized file format ... ");
    }
    
    /*! Dump restart file. */
    if(i_steps%FlowSol.restart_dump_freq==0) {
      write_restart(FlowSol.ini_iter+i_steps, &FlowSol);
    }
    
  }
  
  /////////////////////////////////////////////////
  /// End simulation
  /////////////////////////////////////////////////
  
  /*! Compute execution time. */
  final = clock()-init;
  printf("Execution time= %f s\n", (double) final/((double) CLOCKS_PER_SEC));
  
  /*! Finalize MPI. */
#ifdef _MPI
  MPI_Finalize();
#endif
  
}
