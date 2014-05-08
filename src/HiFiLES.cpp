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

#ifdef _GPU
#include "util.h"
#endif

using namespace std;

int main(int argc, char *argv[]) {
  
  int rank = 0, error_state = 0;
  int i, j;                           /*!< Loop iterators */
  int i_steps = 0;                    /*!< Iteration index */
  int RKSteps;                        /*!< Number of RK steps */
  ifstream run_input_file;            /*!< Config input file */
  clock_t init_time, final_time;                /*!< To control the time */
  struct solution FlowSol;            /*!< Main structure with the flow solution and geometry */
  ofstream write_hist;                /*!< Output files (forces, statistics, and history) */
  
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
    cout << " __    __   __   _______  __          __       _______     _______." << endl;
    cout << "|  |  |  | |  | |   ____||  |        |  |     |   ____|   /       |" << endl;
    cout << "|  |__|  | |  | |  |__   |  |  _____ |  |     |  |__     |   (----`" << endl;
    cout << "|   __   | |  | |   __|  |  | |____| |  |     |   __|     \\   \\" << endl;
    cout << "|  |  |  | |  | |  |     |  |        |  `----.|  |____.----)   |" << endl;
    cout << "|__|  |__| |__| |__|     |__|        |_______||_______|_______/" << endl;
    cout << "                                              " << endl;
    cout << "Aerospace Computing Laboratory (Stanford University) " << endl;
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
  
  GeoPreprocess(&FlowSol);
  
  InitSolution(&FlowSol);
  
  init_time = clock();
  
  /////////////////////////////////////////////////
  /// Pre-processing
  /////////////////////////////////////////////////
  
  /*! Variable initialization. */
  
  error_state = 0;
  FlowSol.ene_hist = 1000.;
  FlowSol.grad_ene_hist = 1000.;
  
  /*! Warning about body forcing term for periodic channel. */

  if (run_input.equation == 0 && run_input.forcing == 1) {
    if(run_input.monitor_force_freq > 100)
      cout<<"WARNING: when running the periodic channel, it is necessary to add a body forcing"<<endl;
      cout<<"term to prevent the flow decaying to zero. Make sure monitor_force_freq is set to a"<<endl;
      cout<<"relatively small number, e.g. 100"<<endl;
    FlowSol.body_force.setup(5);
    for (i=0; i<5; i++) FlowSol.body_force(i)=0.0;
  }
  
  /*! Initialize forces, integral quantities, and residuals. */

  if (FlowSol.rank == 0) {
    
    FlowSol.inv_force.setup(5);
    FlowSol.vis_force.setup(5);
    FlowSol.norm_residual.setup(5);
    FlowSol.integral_quantities.setup(run_input.n_integral_quantities);
    
    for (i=0; i<5; i++) {
      FlowSol.inv_force(i)=0.0;
      FlowSol.vis_force(i)=0.0;
      FlowSol.norm_residual(i)=0.0;
    }
    for (i=0; i<run_input.n_integral_quantities; i++)
      FlowSol.integral_quantities(i)=0.0;

  }
  
  /*! Copy solution and gradients from GPU to CPU, ready for the following routines */
#ifdef _GPU

  CopyGPUCPU(&FlowSol);

#endif

  /*! Dump initial Paraview or tecplot file. */
  
  if (FlowSol.write_type == 0) write_vtu(FlowSol.ini_iter+i_steps, &FlowSol);
  else if (FlowSol.write_type == 1) write_tec(FlowSol.ini_iter+i_steps, &FlowSol);
  else FatalError("ERROR: Trying to write unrecognized file format ... ");
  
  if (FlowSol.rank == 0) cout << endl;
  
  /////////////////////////////////////////////////
  /// Flow solver
  /////////////////////////////////////////////////
  
  /*! Main solver loop (outer loop). */
  
  while(i_steps < FlowSol.n_steps) {
    
    if (FlowSol.adv_type == 0) RKSteps = 1;
    if (FlowSol.adv_type == 3) RKSteps = 5;
    
    for(i=0; i < RKSteps; i++) {
      
      /*! Spatial integration. */

      CalcResidual(&FlowSol);
      
      /*! Time integration usign a RK scheme */
      
      for(j=0; j<FlowSol.n_ele_types; j++) {
        
        FlowSol.mesh_eles(j)->AdvanceSolution(i, FlowSol.adv_type);
        
      }
      
    }

    /*! Update total time, and increase the iteration index. */
    
    FlowSol.time += run_input.dt;
    i_steps++;
    
    /*! Copy solution and gradients from GPU to CPU, ready for the following routines */
#ifdef _GPU

    if(i_steps%FlowSol.plot_freq == 0 || i_steps%run_input.monitor_res_freq == 0 ||
       i_steps%FlowSol.restart_dump_freq==0 || i_steps%run_input.monitor_force_freq==0) {

      CopyGPUCPU(&FlowSol);

    }

#endif

    /*! Force, integral quantities, and residual computation and output. */

    if(i_steps%run_input.monitor_res_freq == 0 ) {

      /*! Compute the value of the forces. */
      
      CalcForces(FlowSol.ini_iter+i_steps, &FlowSol);
      
      /*! Compute integral quantities. */
      
      CalcIntegralQuantities(FlowSol.ini_iter+i_steps, &FlowSol);
      
      /*! Compute the norm of the residual. */
      
      CalcNormResidual(&FlowSol);
      
      /*! Output the history file. */
      
      HistoryOutput(FlowSol.ini_iter+i_steps, init_time, &write_hist, &FlowSol);

      /*! Output error if run has exact solution */
      if (run_input.test_case != 0){
          compute_error(FlowSol.ini_iter + i_steps, &FlowSol);
        }
      
    }
    
    if (i_steps%run_input.monitor_res_freq == 0 || i_steps%run_input.monitor_force_freq == 0)
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
  
  /*! Close convergence history file. */
  
  if (rank == 0)
    write_hist.close();
  
  /*! Compute execution time. */
  
  final_time = clock()-init_time;
  printf("Execution time= %f s\n", (double) final_time/((double) CLOCKS_PER_SEC));
  
  /*! Finalize MPI. */
  
#ifdef _MPI
  MPI_Finalize();
#endif
  
}
