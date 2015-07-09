/*!
 * \file HiFiLES.cpp
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
  mesh Mesh;                          /*!< Store mesh details & perform mesh motion */
  
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

  run_input.setup(argv[1], rank);
  
  /*! Set the input values in the FlowSol structure. */
  
  SetInput(&FlowSol);
  
  /*! Read the mesh file from a file. */
  
  GeoPreprocess(&FlowSol, Mesh);
  
  InitSolution(&FlowSol);
  
  init_time = clock();
  
  /////////////////////////////////////////////////
  /// Pre-processing
  /////////////////////////////////////////////////
  
  /*! Variable initialization. */
  
  error_state = 0;
  FlowSol.ene_hist = 1000.;
  FlowSol.grad_ene_hist = 1000.;
    
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

      /* If using moving mesh, need to advance the Geometric Conservation Law
       * (GCL) first to get updated Jacobians. Necessary to preserve freestream
       * on arbitrarily deforming mesh. See Kui Ou's Ph.D. thesis for details. */
      if (run_input.motion > 0) {

        /* Update the mesh */
        Mesh.move(FlowSol.ini_iter+i_steps,i,&FlowSol);

      }

      /*! Spatial integration. */

      CalcResidual(FlowSol.ini_iter+i_steps, i, &FlowSol);
      
      /*! Time integration usign a RK scheme */
      
      for(j=0; j<FlowSol.n_ele_types; j++) {
        
        FlowSol.mesh_eles(j)->AdvanceSolution(i, FlowSol.adv_type);
        
      }
      
    }

    /*! Update total time, and increase the iteration index. */
    
    FlowSol.time += run_input.dt;
    run_input.time = FlowSol.time;
    i_steps++;
    
    /*! Copy solution and gradients from GPU to CPU, ready for the following routines */
#ifdef _GPU

    if(i_steps == 1 || i_steps%FlowSol.plot_freq == 0 ||
       i_steps%run_input.monitor_res_freq == 0 || i_steps%FlowSol.restart_dump_freq==0) {

      CopyGPUCPU(&FlowSol);

    }

#endif

    /*! Force, integral quantities, and residual computation and output. */

    if( i_steps == 1 || i_steps%run_input.monitor_res_freq == 0 ) {

      /*! Compute the value of the forces. */
      
      CalcForces(FlowSol.ini_iter+i_steps, &FlowSol);
      
      /*! Compute integral quantities. */
      
      CalcIntegralQuantities(FlowSol.ini_iter+i_steps, &FlowSol);
      
      /*! Compute time-averaged quantities. */
      
      CalcTimeAverageQuantities(&FlowSol);

      /*! Compute the norm of the residual. */
      
      CalcNormResidual(&FlowSol);
      
      /*! Output the history file. */
      
      HistoryOutput(FlowSol.ini_iter+i_steps, init_time, &write_hist, &FlowSol);
      
      if (FlowSol.rank == 0) cout << endl;
    }
    
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
  
  if (rank == 0) {
    write_hist.close();
  
  /*! Compute execution time. */
  
  final_time = clock()-init_time;
  printf("Execution time= %f s\n", (double) final_time/((double) CLOCKS_PER_SEC));
    }
  /*! Finalize MPI. */
  
#ifdef _MPI
  MPI_Finalize();
#endif
  
}
