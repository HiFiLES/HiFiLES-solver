/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * pometis.c
 *
 * This is the entry point of the ILUT
 *
 * Started 10/19/95
 * George
 *
 * $Id: parmetis.c,v 1.5 2003/07/30 21:18:54 karypis Exp $
 *
 */

#include <parmetisbin.h>

/*************************************************************************
* Let the game begin
**************************************************************************/
int main(int argc, char *argv[])
{
  idx_t i, j, npes, mype, optype, nparts, adptf, options[10];
  idx_t *order, *sizes;
  graph_t graph;
  MPI_Comm comm;
  idx_t numflag=0, wgtflag=0, ndims=3, edgecut;
  idx_t mtype, rtype, p_nseps, s_nseps, seed, dbglvl;
  real_t ubfrac;
  size_t opc;

  MPI_Init(&argc, &argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  if (argc != 10) {
    if (mype == 0) {
      printf("Usage: %s <graph-file> <op-type> <seed> <dbglvl> <mtype> <rtype> <p_nseps> <s_nseps> <ubfrac>\n", argv[0]);
      printf("  op-type: 1=ParNodeND_V3, 2=ParNodeND_V32, 3=SerNodeND\n");
      printf("  mtype: %"PRIDX"=LOCAL, %"PRIDX"=GLOBAL\n", 
          (idx_t)PARMETIS_MTYPE_LOCAL, (idx_t)PARMETIS_MTYPE_GLOBAL);
      printf("  rtype: %"PRIDX"=GREEDY, %"PRIDX"=2PHASE\n", 
          (idx_t)PARMETIS_SRTYPE_GREEDY, (idx_t)PARMETIS_SRTYPE_2PHASE);
    }

    MPI_Finalize();
    exit(0);
  }

  optype = atoi(argv[2]);

  if (mype == 0) 
    printf("reading file: %s\n", argv[1]);
  ParallelReadGraph(&graph, argv[1], comm);
  if (mype == 0) 
    printf("done\n");

  order = ismalloc(graph.nvtxs, mype, "main: order");
  sizes = imalloc(2*npes, "main: sizes");

  switch (optype) {
    case 1: 
      options[0] = 1;
      options[PMV3_OPTION_SEED]   = atoi(argv[3]);
      options[PMV3_OPTION_DBGLVL] = atoi(argv[4]);
      ParMETIS_V3_NodeND(graph.vtxdist, graph.xadj, graph.adjncy, &numflag, 
          options, order, sizes, &comm);
      break;
    case 2: 
      seed    = atoi(argv[3]);
      dbglvl  = atoi(argv[4]);
      mtype   = atoi(argv[5]);
      rtype   = atoi(argv[6]);
      p_nseps = atoi(argv[7]);
      s_nseps = atoi(argv[8]);
      ubfrac  = atof(argv[9]);
      ParMETIS_V32_NodeND(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, 
          &numflag, &mtype, &rtype, &p_nseps, &s_nseps, &ubfrac, &seed, &dbglvl, 
          order, sizes, &comm);
      break;
    case 3: 
      options[0] = 0;
      ParMETIS_SerialNodeND(graph.vtxdist, graph.xadj, graph.adjncy, &numflag, 
          options, order, sizes, &comm);
      break;
    default:
      if (mype == 0) 
        printf("Uknown optype of %"PRIDX"\n", optype);
      MPI_Finalize();
      exit(0);
  }

  WriteOVector(argv[1], graph.vtxdist, order, comm);

  /* print the partition sizes and the separators */
  if (mype == 0) {
    opc = 0;
    nparts = 1<<log2Int(npes);
    for (i=0; i<2*nparts-1; i++) {
      printf(" %6"PRIDX"", sizes[i]);
      if (i >= nparts)
        opc += sizes[i]*(sizes[i]+1)/2;
    }
    printf("\nTopSep OPC: %zu\n", opc);
  }


  gk_free((void **)&order, &sizes, &graph.vtxdist, &graph.xadj, &graph.adjncy, 
         &graph.vwgt, &graph.adjwgt, LTERM);

  MPI_Comm_free(&comm);

  MPI_Finalize();

  return 0;
}

