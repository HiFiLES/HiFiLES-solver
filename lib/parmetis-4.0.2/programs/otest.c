/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * main.c
 * 
 * This file contains code for testing teh adaptive partitioning routines
 *
 * Started 5/19/97
 * George
 *
 * $Id: ptest.c,v 1.3 2003/07/22 21:47:20 karypis Exp $
 *
 */

#include <parmetisbin.h>


/*************************************************************************/
/*! Entry point of the testing routine */
/*************************************************************************/
idx_t main(idx_t argc, char *argv[])
{
  idx_t mype, npes;
  MPI_Comm comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  if (argc != 2) {
    if (mype == 0)
      printf("Usage: %s <graph-file>\n", argv[0]);

    MPI_Finalize();
    exit(0);
  }

  TestParMetis_GPart(argv[1], comm); 

  MPI_Comm_free(&comm);

  MPI_Finalize();

  return 0;
}



/***********************************************************************************/
/*! This function tests the various graph partitioning and ordering routines */
/***********************************************************************************/
void TestParMetis(char *filename, MPI_Comm comm)
{
  idx_t ncon, nparts, npes, mype, opt2, realcut;
  graph_t graph, mgraph;
  idx_t *part, *mpart, *savepart, *order, *sizes;
  idx_t numflag=0, wgtflag=0, options[10], edgecut, ndims;
  real_t ipc2redist, *xyz, *tpwgts = NULL, ubvec[MAXNCON];

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  ndims = 2;

  ParallelReadGraph(&graph, filename, comm);
  xyz = ReadTestCoordinates(&graph, filename, 2, comm);
  MPI_Barrier(comm);

  part = imalloc(graph.nvtxs, "TestParMetis_V3: part");
  tpwgts = rmalloc(MAXNCON*npes*2, "TestParMetis_V3: tpwgts");
  rset(MAXNCON, 1.05, ubvec);
  graph.vwgt = ismalloc(graph.nvtxs*5, 1, "TestParMetis_V3: vwgt");


  /*======================================================================
  / ParMETIS_V3_PartKway
  /=======================================================================*/
  options[0] = 1;
  options[1] = 3;
  options[2] = 1;
  wgtflag = 2;
  numflag = 0;
  edgecut = 0;
  for (nparts=2*npes; nparts>=npes/2 && nparts > 0; nparts = nparts/2) {
    for (ncon=1; ncon<=5; ncon+=2) {

      if (ncon > 1 && nparts > 1)
        Mc_AdaptGraph(&graph, part, ncon, nparts, comm);
      else
        iset(graph.nvtxs, 1, graph.vwgt);

      for (opt2=1; opt2<=2; opt2++) {
        options[2] = opt2;

        rset(nparts*ncon, 1.0/(real_t)nparts, tpwgts);
        if (mype == 0)
          printf("\nTesting ParMETIS_V3_PartKway with options[1-2] = {%"PRIDX" %"PRIDX"}, Ncon: %"PRIDX", Nparts: %"PRIDX"\n", options[1], options[2], ncon, nparts);

        ParMETIS_V3_PartKway(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, NULL, &wgtflag,
        &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);

        if (mype == 0) {
          printf("ParMETIS_V3_PartKway reported a cut of %"PRIDX"\n", edgecut);
        }
      }
    }
  }


  /*======================================================================
  / ParMETIS_V3_PartGeomKway 
  /=======================================================================*/
  options[0] = 1;
  options[1] = 3;
  wgtflag = 2;
  numflag = 0;
  for (nparts=2*npes; nparts>=npes/2 && nparts > 0; nparts = nparts/2) {
    for (ncon=1; ncon<=5; ncon+=2) {

      if (ncon > 1)
        Mc_AdaptGraph(&graph, part, ncon, nparts, comm);
      else
        iset(graph.nvtxs, 1, graph.vwgt);

      for (opt2=1; opt2<=2; opt2++) {
        options[2] = opt2;

        rset(nparts*ncon, 1.0/(real_t)nparts, tpwgts);
        if (mype == 0)
          printf("\nTesting ParMETIS_V3_PartGeomKway with options[1-2] = {%"PRIDX" %"PRIDX"}, Ncon: %"PRIDX", Nparts: %"PRIDX"\n", options[1], options[2], ncon, nparts);

        ParMETIS_V3_PartGeomKway(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, NULL, &wgtflag,
          &numflag, &ndims, xyz, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);

        if (mype == 0) {
          printf("ParMETIS_V3_PartGeomKway reported a cut of %"PRIDX"\n", edgecut);
        }
      }
    }
  }



  /*======================================================================
  / ParMETIS_V3_PartGeom 
  /=======================================================================*/
  wgtflag = 0;
  numflag = 0;
  if (mype == 0)
    printf("\nTesting ParMETIS_V3_PartGeom\n");

/*  ParMETIS_V3_PartGeom(graph.vtxdist, &ndims, xyz, part, &comm); */

  if (mype == 0) 
    printf("ParMETIS_V3_PartGeom partition complete\n");
/*
  realcut = ComputeRealCut(graph.vtxdist, part, filename, comm);
  if (mype == 0) 
    printf("ParMETIS_V3_PartGeom reported a cut of %"PRIDX"\n", realcut);
*/

  /*======================================================================
  / ParMETIS_V3_RefineKway 
  /=======================================================================*/
  options[0] = 1;
  options[1] = 3;
  options[2] = 1;
  options[3] = COUPLED;
  nparts = npes;
  wgtflag = 0;
  numflag = 0;
  ncon = 1;
  rset(nparts*ncon, 1.0/(real_t)nparts, tpwgts);

  if (mype == 0)
    printf("\nTesting ParMETIS_V3_RefineKway with default options (before move)\n");

  ParMETIS_V3_RefineKway(graph.vtxdist, graph.xadj, graph.adjncy, NULL, NULL, &wgtflag,
    &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);

  MALLOC_CHECK(NULL);

  if (mype == 0) {
    printf("ParMETIS_V3_RefineKway reported a cut of %"PRIDX"\n", edgecut);
  }


  MALLOC_CHECK(NULL);

  /* Compute a good partition and move the graph. Do so quietly! */
  options[0] = 0;
  nparts = npes;
  wgtflag = 0;
  numflag = 0;
  ncon = 1;
  rset(nparts*ncon, 1.0/(real_t)nparts, tpwgts);
  ParMETIS_V3_PartKway(graph.vtxdist, graph.xadj, graph.adjncy, NULL, NULL, &wgtflag,
    &numflag, &ncon, &npes, tpwgts, ubvec, options, &edgecut, part, &comm);
  TestMoveGraph(&graph, &mgraph, part, comm);
  gk_free((void **)&(graph.vwgt), LTERM);
  mpart = ismalloc(mgraph.nvtxs, mype, "TestParMetis_V3: mpart");
  savepart = imalloc(mgraph.nvtxs, "TestParMetis_V3: savepart");

  MALLOC_CHECK(NULL);

  /*======================================================================
  / ParMETIS_V3_RefineKway 
  /=======================================================================*/
  options[0] = 1;
  options[1] = 3;
  options[3] = COUPLED;
  nparts = npes;
  wgtflag = 0;
  numflag = 0;

  for (ncon=1; ncon<=5; ncon+=2) {
    for (opt2=1; opt2<=2; opt2++) {
      options[2] = opt2;

      rset(nparts*ncon, 1.0/(real_t)nparts, tpwgts);
      if (mype == 0)
        printf("\nTesting ParMETIS_V3_RefineKway with options[1-3] = {%"PRIDX" %"PRIDX" %"PRIDX"}, Ncon: %"PRIDX", Nparts: %"PRIDX"\n", options[1], options[2], options[3], ncon, nparts);
      ParMETIS_V3_RefineKway(mgraph.vtxdist, mgraph.xadj, mgraph.adjncy, NULL, NULL, &wgtflag,
        &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, mpart, &comm);

      if (mype == 0) {
        printf("ParMETIS_V3_RefineKway reported a cut of %"PRIDX"\n", edgecut);
      }
    }
  }


  /*======================================================================
  / ParMETIS_V3_AdaptiveRepart
  /=======================================================================*/
  mgraph.vwgt = ismalloc(mgraph.nvtxs*5, 1, "TestParMetis_V3: mgraph.vwgt");
  mgraph.vsize = ismalloc(mgraph.nvtxs, 1, "TestParMetis_V3: mgraph.vsize");
  AdaptGraph(&mgraph, 4, comm); 
  options[0] = 1;
  options[1] = 7;
  options[3] = COUPLED;
  wgtflag = 2;
  numflag = 0;

  for (nparts=2*npes; nparts>=npes/2; nparts = nparts/2) {

    ncon = 1;
    wgtflag = 0;
    options[0] = 0;
    rset(nparts*ncon, 1.0/(real_t)nparts, tpwgts);
    ParMETIS_V3_PartKway(mgraph.vtxdist, mgraph.xadj, mgraph.adjncy, NULL, NULL,
    &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, savepart, &comm);
    options[0] = 1;
    wgtflag = 2;

    for (ncon=1; ncon<=3; ncon+=2) {
      rset(nparts*ncon, 1.0/(real_t)nparts, tpwgts);

      if (ncon > 1)
        Mc_AdaptGraph(&mgraph, savepart, ncon, nparts, comm);
      else
        AdaptGraph(&mgraph, 4, comm); 
/*        iset(mgraph.nvtxs, 1, mgraph.vwgt); */

      for (ipc2redist=1000.0; ipc2redist>=0.001; ipc2redist/=1000.0) {
        for (opt2=1; opt2<=2; opt2++) {
          icopy(mgraph.nvtxs, savepart, mpart);
          options[2] = opt2;

          if (mype == 0)
            printf("\nTesting ParMETIS_V3_AdaptiveRepart with options[1-3] = {%"PRIDX" %"PRIDX" %"PRIDX"}, ipc2redist: %.3"PRREAL", Ncon: %"PRIDX", Nparts: %"PRIDX"\n", options[1], options[2], options[3], ipc2redist, ncon, nparts);

          ParMETIS_V3_AdaptiveRepart(mgraph.vtxdist, mgraph.xadj, mgraph.adjncy, mgraph.vwgt,
            mgraph.vsize, NULL, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, &ipc2redist,
            options, &edgecut, mpart, &comm);

          if (mype == 0) {
            printf("ParMETIS_V3_AdaptiveRepart reported a cut of %"PRIDX"\n", edgecut);
          }
        }
      }
    }
  }

  free(mgraph.vwgt);
  free(mgraph.vsize);



  /*======================================================================
  / ParMETIS_V3_NodeND 
  /=======================================================================*/
  sizes = imalloc(2*npes, "TestParMetis_V3: sizes");
  order = imalloc(graph.nvtxs, "TestParMetis_V3: sizes");

  options[0] = 1;
  options[PMV3_OPTION_DBGLVL] = 3;
  options[PMV3_OPTION_SEED] = 1;
  numflag = 0;

  for (opt2=1; opt2<=2; opt2++) {
    options[PMV3_OPTION_IPART] = opt2;

    if (mype == 0)
      printf("\nTesting ParMETIS_V3_NodeND with options[1-3] = {%"PRIDX" %"PRIDX" %"PRIDX"}\n", options[1], options[2], options[3]);

    ParMETIS_V3_NodeND(graph.vtxdist, graph.xadj, graph.adjncy, &numflag, options,
      order, sizes, &comm);
  }


  gk_free((void **)&tpwgts, &part, &mpart, &savepart, &order, &sizes, LTERM);

}



/******************************************************************************
* This function takes a partition vector that is distributed and reads in
* the original graph and computes the edgecut
*******************************************************************************/
idx_t ComputeRealCut(idx_t *vtxdist, idx_t *part, char *filename, MPI_Comm comm)
{
  idx_t i, j, nvtxs, mype, npes, cut;
  idx_t *xadj, *adjncy, *gpart;
  MPI_Status status;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  if (mype != 0) {
    MPI_Send((void *)part, vtxdist[mype+1]-vtxdist[mype], IDX_T, 0, 1, comm);
  }
  else {  /* Processor 0 does all the rest */
    gpart = imalloc(vtxdist[npes], "ComputeRealCut: gpart");
    icopy(vtxdist[1], part, gpart);

    for (i=1; i<npes; i++) 
      MPI_Recv((void *)(gpart+vtxdist[i]), vtxdist[i+1]-vtxdist[i], IDX_T, i, 1, comm, &status);

    ReadMetisGraph(filename, &nvtxs, &xadj, &adjncy);

    /* OK, now compute the cut */
    for (cut=0, i=0; i<nvtxs; i++) {
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (gpart[i] != gpart[adjncy[j]])
          cut++;
      }
    }
    cut = cut/2;

    gk_free((void **)&gpart, &xadj, &adjncy, LTERM);

    return cut;
  }
  return 0;
}


/******************************************************************************
* This function takes a partition vector that is distributed and reads in
* the original graph and computes the edgecut
*******************************************************************************/
idx_t ComputeRealCut2(idx_t *vtxdist, idx_t *mvtxdist, idx_t *part, idx_t *mpart, char *filename, MPI_Comm comm)
{
  idx_t i, j, nvtxs, mype, npes, cut;
  idx_t *xadj, *adjncy, *gpart, *gmpart, *perm, *sizes;
  MPI_Status status;


  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  if (mype != 0) {
    MPI_Send((void *)part, vtxdist[mype+1]-vtxdist[mype], IDX_T, 0, 1, comm);
    MPI_Send((void *)mpart, mvtxdist[mype+1]-mvtxdist[mype], IDX_T, 0, 1, comm);
  }
  else {  /* Processor 0 does all the rest */
    gpart = imalloc(vtxdist[npes], "ComputeRealCut: gpart");
    icopy(vtxdist[1], part, gpart);
    gmpart = imalloc(mvtxdist[npes], "ComputeRealCut: gmpart");
    icopy(mvtxdist[1], mpart, gmpart);

    for (i=1; i<npes; i++) {
      MPI_Recv((void *)(gpart+vtxdist[i]), vtxdist[i+1]-vtxdist[i], IDX_T, i, 1, comm, &status);
      MPI_Recv((void *)(gmpart+mvtxdist[i]), mvtxdist[i+1]-mvtxdist[i], IDX_T, i, 1, comm, &status);
    }

    /* OK, now go and reconstruct the permutation to go from the graph to mgraph */
    perm = imalloc(vtxdist[npes], "ComputeRealCut: perm");
    sizes = ismalloc(npes+1, 0, "ComputeRealCut: sizes");

    for (i=0; i<vtxdist[npes]; i++)
      sizes[gpart[i]]++;
    MAKECSR(i, npes, sizes);
    for (i=0; i<vtxdist[npes]; i++)
      perm[i] = sizes[gpart[i]]++;

    /* Ok, now read the graph from the file */
    ReadMetisGraph(filename, &nvtxs, &xadj, &adjncy);

    /* OK, now compute the cut */
    for (cut=0, i=0; i<nvtxs; i++) {
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (gmpart[perm[i]] != gmpart[perm[adjncy[j]]])
          cut++;
      }
    }
    cut = cut/2;

    gk_free((void **)&gpart, &gmpart, &perm, &sizes, &xadj, &adjncy, LTERM);

    return cut;
  }

  return 0;
}



/******************************************************************************
* This function takes a graph and its partition vector and creates a new
* graph corresponding to the one after the movement
*******************************************************************************/
void TestMoveGraph(graph_t *ograph, graph_t *omgraph, idx_t *part, MPI_Comm comm)
{
  idx_t npes, mype;
  ctrl_t ctrl;
  wspace_t wspace;
  graph_t *graph, *mgraph;
  idx_t options[5] = {0, 0, 1, 0, 0};

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  SetUpCtrl(&ctrl, npes, 0, comm); 
  ctrl.CoarsenTo = 1;  /* Needed by SetUpGraph, otherwise we can FP errors */
  graph = SetUpGraph(&ctrl, ograph->vtxdist, ograph->xadj, NULL, ograph->adjncy, NULL, 0);
  AllocateWSpace(&ctrl, graph, &wspace);

  SetUp(&ctrl, graph, &wspace);
  graph->where = part;
  graph->ncon = 1;
  mgraph = MoveGraph(&ctrl, graph, &wspace);

  omgraph->gnvtxs = mgraph->gnvtxs;
  omgraph->nvtxs = mgraph->nvtxs;
  omgraph->nedges = mgraph->nedges;
  omgraph->vtxdist = mgraph->vtxdist;
  omgraph->xadj = mgraph->xadj;
  omgraph->adjncy = mgraph->adjncy;
  mgraph->vtxdist = NULL;
  mgraph->xadj = NULL;
  mgraph->adjncy = NULL;
  FreeGraph(mgraph);

  graph->where = NULL;
  FreeInitialGraphAndRemap(graph, 0, 1);
  FreeWSpace(&wspace);
}  


/*****************************************************************************
*  This function sets up a graph data structure for partitioning
*****************************************************************************/
graph_t *SetUpGraph(ctrl_t *ctrl, idx_t *vtxdist, idx_t *xadj,
   idx_t *vwgt, idx_t *adjncy, idx_t *adjwgt, idx_t wgtflag)
{
  idx_t mywgtflag;

  mywgtflag = wgtflag;
  return Mc_SetUpGraph(ctrl, 1, vtxdist, xadj, vwgt, adjncy, adjwgt, &mywgtflag);
}


