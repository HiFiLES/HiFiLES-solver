/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * proto.h
 *
 * This file contains header files
 *
 * Started 10/19/95
 * George
 *
 * $Id: proto.h 10076 2011-06-03 15:36:39Z karypis $
 *
 */


/* pio.c */
void ParallelReadGraph(graph_t *, char *, MPI_Comm);
void Mc_ParallelWriteGraph(ctrl_t *, graph_t *, char *, idx_t, idx_t);
void ReadTestGraph(graph_t *, char *, MPI_Comm);
real_t *ReadTestCoordinates(graph_t *, char *, idx_t *, MPI_Comm);
void ReadMetisGraph(char *, idx_t *, idx_t **, idx_t **);
void Mc_SerialReadGraph(graph_t *, char *, idx_t *, MPI_Comm);
void Mc_SerialReadMetisGraph(char *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t **, idx_t **, idx_t **, idx_t **, idx_t *);
void WritePVector(char *gname, idx_t *vtxdist, idx_t *part, MPI_Comm comm);
void WriteOVector(char *gname, idx_t *vtxdist, idx_t *order, MPI_Comm comm);



/* adaptgraph */
void AdaptGraph(graph_t *, idx_t, MPI_Comm);
void AdaptGraph2(graph_t *, idx_t, MPI_Comm);
void Mc_AdaptGraph(graph_t *, idx_t *, idx_t, idx_t, MPI_Comm);


/* ptest.c */
void TestParMetis_GPart(char *filename, char *xyzfile, MPI_Comm comm);
idx_t ComputeRealCut(idx_t *, idx_t *, char *, MPI_Comm);
idx_t ComputeRealCutFromMoved(idx_t *, idx_t *, idx_t *, idx_t *, char *, MPI_Comm);
void TestMoveGraph(graph_t *, graph_t *, idx_t *, MPI_Comm);
graph_t *TestSetUpGraph(ctrl_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t);

/* mienio.c */
void mienIO(mesh_t *, char *, idx_t, idx_t, MPI_Comm);

/* meshio.c */
void ParallelReadMesh(mesh_t *, char *, MPI_Comm);


