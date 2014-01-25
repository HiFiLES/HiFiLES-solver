/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * metis.c
 *
 * This file contains the driving routine for multilevel method
 *
 * Started 8/28/94
 * George
 *
 * $Id: metis.c 5993 2009-01-07 02:09:57Z karypis $
 *
 */

#include "metisbin.h"



/*************************************************************************
* Let the game begin
**************************************************************************/
int main(int argc, char *argv[])
{
  idx_t i, nparts, OpType, options[10], nbytes;
  idx_t *part, *perm, *iperm, *sizes;
  graph_t graph;
  char filename[256];
  idx_t numflag = 0, wgtflag = 0, edgecut;


  if (argc != 11) {
    printf("Usage: %s <GraphFile> <Nparts> <Mtype> <Rtype> <IPtype> <Oflags> <Pfactor> <Nseps> <OPtype> <Options> \n",argv[0]);
    exit(0);
  }
    
  strcpy(filename, argv[1]);
  nparts = atoi(argv[2]);
  options[METIS_OPTION_CTYPE] = atoi(argv[3]);
  options[METIS_OPTION_RTYPE] = atoi(argv[4]);
  options[METIS_OPTION_ITYPE] = atoi(argv[5]);
  options[METIS_OPTION_OFLAGS] = atoi(argv[6]);
  options[METIS_OPTION_PFACTOR] = atoi(argv[7]);
  options[METIS_OPTION_NSEPS] = atoi(argv[8]);
  OpType = atoi(argv[9]); 
  options[METIS_OPTION_DBGLVL] = atoi(argv[10]);


  ReadGraph(&graph, filename, &wgtflag);
  if (graph.nvtxs <= 0) {
    printf("Empty graph. Nothing to do.\n");
    exit(0);
  }
  printf("Partitioning a graph with %"PRIDX" vertices and %"PRIDX" edges\n", graph.nvtxs, graph.nedges/2);

  METIS_EstimateMemory(&graph.nvtxs, graph.xadj, graph.adjncy, &numflag, &OpType, &nbytes);
  printf("Metis will need %"PRIDX" Mbytes\n", nbytes/(1024*1024));

  part = perm = iperm = NULL;

  options[0] = 1;
  switch (OpType) {
    case OP_PMETIS:
      printf("Recursive Partitioning... ------------------------------------------\n");
      part = imalloc(graph.nvtxs, "main: part");

      METIS_PartGraphRecursive(&graph.nvtxs, graph.xadj, graph.adjncy, graph.vwgt, graph.adjwgt, 
                               &wgtflag, &numflag, &nparts, options, &edgecut, part);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, part, graph.nvtxs, nparts)); 

      printf("  %"PRIDX"-way Edge-Cut: %7"PRIDX"\n", nparts, edgecut);
      ComputePartitionInfo(&graph, nparts, part);

      gk_free((void **)&part, LTERM);
      break;
    case OP_KMETIS:
      printf("K-way Partitioning... -----------------------------------------------\n");
      part = imalloc(graph.nvtxs, "main: part");

      METIS_PartGraphKway(&graph.nvtxs, graph.xadj, graph.adjncy, graph.vwgt, graph.adjwgt, 
                          &wgtflag, &numflag, &nparts, options, &edgecut, part);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, part, graph.nvtxs, nparts)); 

      printf("  %"PRIDX"-way Edge-Cut: %7"PRIDX"\n", nparts, edgecut);
      ComputePartitionInfo(&graph, nparts, part);

      gk_free((void **)&part, LTERM);
      break;
    case OP_OEMETIS:
      gk_free((void **)&graph.vwgt, &graph.adjwgt, LTERM);

      printf("Edge-based Nested Dissection Ordering... ----------------------------\n");
      perm = imalloc(graph.nvtxs, "main: perm");
      iperm = imalloc(graph.nvtxs, "main: iperm");

      METIS_EdgeND(&graph.nvtxs, graph.xadj, graph.adjncy, &numflag, options, perm, iperm);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, iperm, graph.nvtxs, 0)); 

      ComputeFillIn(&graph, iperm);

      gk_free((void **)&perm, &iperm, LTERM);
      break;
    case OP_ONMETIS:
      gk_free((void **)&graph.vwgt, &graph.adjwgt, LTERM);

      printf("Node-based Nested Dissection Ordering... ----------------------------\n");
      perm = imalloc(graph.nvtxs, "main: perm");
      iperm = imalloc(graph.nvtxs, "main: iperm");

      METIS_NodeND(&graph.nvtxs, graph.xadj, graph.adjncy, &numflag, options, perm, iperm);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, iperm, graph.nvtxs, 0)); 

      ComputeFillIn(&graph, iperm);

      gk_free((void **)&perm, &iperm, LTERM);
      break;
    case OP_ONWMETIS:
      gk_free((void **)&graph.adjwgt, LTERM);

      printf("WNode-based Nested Dissection Ordering... ---------------------------\n");
      perm = imalloc(graph.nvtxs, "main: perm");
      iperm = imalloc(graph.nvtxs, "main: iperm");

      METIS_NodeWND(&graph.nvtxs, graph.xadj, graph.adjncy, graph.vwgt, &numflag, options, perm, iperm);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, iperm, graph.nvtxs, 0)); 

      ComputeFillIn(&graph, iperm);

      gk_free((void **)&perm, &iperm, LTERM);
      break;
    case 6:
      gk_free((void **)&graph.vwgt, &graph.adjwgt, LTERM);

      printf("Node-based Nested Dissection Ordering... ----------------------------\n");
      perm = imalloc(graph.nvtxs, "main: perm");
      iperm = imalloc(graph.nvtxs, "main: iperm");
      sizes = imalloc(2*nparts, "main: sizes");

      METIS_NodeNDP(graph.nvtxs, graph.xadj, graph.adjncy, nparts, options, perm, iperm, sizes);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, iperm, graph.nvtxs, 0)); 

      ComputeFillIn(&graph, iperm);

      for (i=0; i<2*nparts-1; i++)
        printf("%"PRIDX" ", sizes[i]);
      printf("\n");

      gk_free((void **)&perm, &iperm, &sizes, LTERM);
      break;
    case 7:
      printf("K-way Vol Partitioning... -------------------------------------------\n");
      part = imalloc(graph.nvtxs, "main: part");

      METIS_PartGraphVKway(&graph.nvtxs, graph.xadj, graph.adjncy, graph.vwgt, NULL, 
            &wgtflag, &numflag, &nparts, options, &edgecut, part);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, part, graph.nvtxs, nparts)); 

      printf("  %"PRIDX"-way Volume: %7"PRIDX"\n", nparts, edgecut);
      ComputePartitionInfo(&graph, nparts, part);

      gk_free((void **)&part, LTERM);
      break;
    case 9:
      printf("K-way Partitioning (with vwgts)... ----------------------------------\n");
      part = imalloc(graph.nvtxs, "main: part");
      graph.vwgt = imalloc(graph.nvtxs, "main: graph.vwgt");
      for (i=0; i<graph.nvtxs; i++)
        graph.vwgt[i] = graph.xadj[i+1]-graph.xadj[i]+1;
      wgtflag = 2;

      METIS_PartGraphKway(&graph.nvtxs, graph.xadj, graph.adjncy, graph.vwgt, graph.adjwgt, 
                          &wgtflag, &numflag, &nparts, options, &edgecut, part);

      IFSET(options[METIS_OPTION_DBGLVL], DBG_OUTPUT, WritePartition(filename, part, graph.nvtxs, nparts)); 

      printf("  %"PRIDX"-way Edge-Cut: %7"PRIDX"\n", nparts, edgecut);
      ComputePartitionInfo(&graph, nparts, part);

      gk_free((void **)&part, LTERM);
      break;
    case 10:
      break;
    default:
      errexit("Unknown");
  }

  gk_free((void **)&graph.xadj, &graph.adjncy, &graph.vwgt, &graph.adjwgt, LTERM);
}  


