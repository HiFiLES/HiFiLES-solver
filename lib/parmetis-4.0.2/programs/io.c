/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * pio.c
 *
 * This file contains routines related to I/O
 *
 * Started 10/19/94
 * George
 *
 * $Id: io.c,v 1.1 2003/07/22 21:47:18 karypis Exp $
 *
 */

#include <parmetisbin.h>
#define	MAXLINE	64*1024*1024

/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
void ParallelReadGraph(graph_t *graph, char *filename, MPI_Comm comm)
{
  idx_t i, k, l, pe;
  idx_t npes, mype, ier;
  idx_t gnvtxs, nvtxs, your_nvtxs, your_nedges, gnedges;
  idx_t maxnvtxs = -1, maxnedges = -1;
  idx_t readew = -1, readvw = -1, dummy, edge;
  idx_t *vtxdist, *xadj, *adjncy, *vwgt, *adjwgt;
  idx_t *your_xadj, *your_adjncy, *your_vwgt, *your_adjwgt, graphinfo[4];
  idx_t fmt, ncon, nobj;
  MPI_Status stat;
  char *line = NULL, *oldstr, *newstr;
  FILE *fpin = NULL;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist = ismalloc(npes+1, 0, "ReadGraph: vtxdist");

  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL) {
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    line = gk_cmalloc(MAXLINE+1, "line");

    while (fgets(line, MAXLINE, fpin) && line[0] == '%');

    fmt = ncon = nobj = 0;
    sscanf(line, "%"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX"", 
        &gnvtxs, &gnedges, &fmt, &ncon, &nobj);
    gnedges *=2;
    readew = (fmt%10 > 0);
    readvw = ((fmt/10)%10 > 0);
    graph->ncon = ncon = (ncon == 0 ? 1 : ncon);
    graph->nobj = nobj = (nobj == 0 ? 1 : nobj);

    /*printf("Nvtxs: %"PRIDX", Nedges: %"PRIDX", Ncon: %"PRIDX"\n", gnvtxs, gnedges, ncon); */

    graphinfo[0] = ncon;
    graphinfo[1] = nobj;
    graphinfo[2] = readvw;
    graphinfo[3] = readew;
    gkMPI_Bcast((void *)graphinfo, 4, IDX_T, npes-1, comm);

    /* Construct vtxdist and send it to all the processors */
    vtxdist[0] = 0;
    for (i=0,k=gnvtxs; i<npes; i++) {
      l = k/(npes-i);
      vtxdist[i+1] = vtxdist[i]+l;
      k -= l;
    }

    gkMPI_Bcast((void *)vtxdist, npes+1, IDX_T, npes-1, comm);
  }
  else {
    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    gkMPI_Bcast((void *)graphinfo, 4, IDX_T, npes-1, comm);
    graph->ncon = ncon = graphinfo[0];
    graph->nobj = nobj = graphinfo[1];
    readvw = graphinfo[2];
    readew = graphinfo[3];

    gkMPI_Bcast((void *)vtxdist, npes+1, IDX_T, npes-1, comm);
  }

  if ((ncon > 1 && !readvw) || (nobj > 1 && !readew)) {
    printf("fmt and ncon/nobj are inconsistant.  Exiting...\n");
    gkMPI_Finalize();
    exit(-1);
  }


  graph->gnvtxs = vtxdist[npes];
  nvtxs = graph->nvtxs = vtxdist[mype+1]-vtxdist[mype];
  xadj  = graph->xadj  = imalloc(graph->nvtxs+1, "ParallelReadGraph: xadj");
  vwgt  = graph->vwgt  = imalloc(graph->nvtxs*ncon, "ParallelReadGraph: vwgt");

  /*******************************************/
  /* Go through first time and generate xadj */
  /*******************************************/
  if (mype == npes-1) {
    maxnvtxs = vtxdist[1];
    for (i=1; i<npes; i++) 
      maxnvtxs = (maxnvtxs < vtxdist[i+1]-vtxdist[i] ? vtxdist[i+1]-vtxdist[i] : maxnvtxs);

    your_xadj = imalloc(maxnvtxs+1, "your_xadj");
    your_vwgt = ismalloc(maxnvtxs*ncon, 1, "your_vwgt");

    maxnedges = 0;
    for (pe=0; pe<npes; pe++) {
      your_nvtxs = vtxdist[pe+1]-vtxdist[pe];

      for (i=0; i<your_nvtxs; i++) {
        your_nedges = 0;

        while (fgets(line, MAXLINE, fpin) && line[0] == '%'); /* skip lines with '#' */
        oldstr = line;
        newstr = NULL;

        if (readvw) {
          for (l=0; l<ncon; l++) {
            your_vwgt[i*ncon+l] = strtoidx(oldstr, &newstr, 10);
            oldstr = newstr;
          }
        }

        for (;;) {
          edge = strtoidx(oldstr, &newstr, 10) -1;
          oldstr = newstr;

          if (edge < 0)
            break;

          if (readew) {
            for (l=0; l<nobj; l++) {
              dummy  = strtoidx(oldstr, &newstr, 10);
              oldstr = newstr;
            }
          }
          your_nedges++;
        }
        your_xadj[i] = your_nedges;
      }

      MAKECSR(i, your_nvtxs, your_xadj);
      maxnedges = (maxnedges < your_xadj[your_nvtxs] ? your_xadj[your_nvtxs] : maxnedges);

      if (pe < npes-1) {
        gkMPI_Send((void *)your_xadj, your_nvtxs+1, IDX_T, pe, 0, comm);
        gkMPI_Send((void *)your_vwgt, your_nvtxs*ncon, IDX_T, pe, 1, comm);
      }
      else {
        for (i=0; i<your_nvtxs+1; i++)
          xadj[i] = your_xadj[i];
        for (i=0; i<your_nvtxs*ncon; i++)
          vwgt[i] = your_vwgt[i];
      }
    }
    fclose(fpin);
    gk_free((void **)&your_xadj, &your_vwgt, LTERM);
  }
  else {
    gkMPI_Recv((void *)xadj, nvtxs+1, IDX_T, npes-1, 0, comm, &stat);
    gkMPI_Recv((void *)vwgt, nvtxs*ncon, IDX_T, npes-1, 1, comm, &stat);
  }

  graph->nedges = xadj[nvtxs];
  adjncy = graph->adjncy = imalloc(xadj[nvtxs], "ParallelReadGraph: adjncy");
  adjwgt = graph->adjwgt = imalloc(xadj[nvtxs]*nobj, "ParallelReadGraph: adjwgt");

  /***********************************************/
  /* Now go through again and record adjncy data */
  /***********************************************/
  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL){
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      gkMPI_Finalize();
      exit(0);
    }

    /* get first line again */
    while (fgets(line, MAXLINE, fpin) && line[0] == '%');

    your_adjncy = imalloc(maxnedges, "your_adjncy");
    your_adjwgt = ismalloc(maxnedges*nobj, 1, "your_adjwgt");

    for (pe=0; pe<npes; pe++) {
      your_nvtxs  = vtxdist[pe+1]-vtxdist[pe];
      your_nedges = 0;

      for (i=0; i<your_nvtxs; i++) {
        while (fgets(line, MAXLINE, fpin) && line[0] == '%');
        oldstr = line;
        newstr = NULL;

        if (readvw) {
          for (l=0; l<ncon; l++) {
            dummy  = strtoidx(oldstr, &newstr, 10);
            oldstr = newstr;
          }
        }

        for (;;) {
          edge   = strtoidx(oldstr, &newstr, 10) -1;
          oldstr = newstr;

          if (edge < 0)
            break;

          your_adjncy[your_nedges] = edge;
          if (readew) {
            for (l=0; l<nobj; l++) {
              your_adjwgt[your_nedges*nobj+l] = strtoidx(oldstr, &newstr, 10);
              oldstr = newstr;
            }
          }
          your_nedges++;
        }
      }
      if (pe < npes-1) {
        gkMPI_Send((void *)your_adjncy, your_nedges, IDX_T, pe, 0, comm);
        gkMPI_Send((void *)your_adjwgt, your_nedges*nobj, IDX_T, pe, 1, comm);
      }
      else {
        for (i=0; i<your_nedges; i++)
          adjncy[i] = your_adjncy[i];
        for (i=0; i<your_nedges*nobj; i++)
          adjwgt[i] = your_adjwgt[i];
      }
    }
    fclose(fpin);
    gk_free((void **)&your_adjncy, &your_adjwgt, &line, LTERM);
  }
  else {
    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      gkMPI_Finalize();
      exit(0);
    }

    gkMPI_Recv((void *)adjncy, xadj[nvtxs], IDX_T, npes-1, 0, comm, &stat);
    gkMPI_Recv((void *)adjwgt, xadj[nvtxs]*nobj, IDX_T, npes-1, 1, comm, &stat);
  }

}



/*************************************************************************
* This function writes a distributed graph to file
**************************************************************************/
void Mc_ParallelWriteGraph(ctrl_t *ctrl, graph_t *graph, char *filename,
     idx_t nparts, idx_t testset)
{
  idx_t h, i, j;
  idx_t npes, mype, penum, gnedges;
  char partfile[256];
  FILE *fpin;
  MPI_Comm comm;

  comm = ctrl->comm;
  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  gnedges = GlobalSESum(ctrl, graph->nedges);
  sprintf(partfile, "%s.%"PRIDX".%"PRIDX".%"PRIDX"", filename, testset, graph->ncon, nparts);

  if (mype == 0) {
    if ((fpin = fopen(partfile, "w")) == NULL)
      errexit("Failed to open file %s", partfile);

    fprintf(fpin, "%"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX"\n", 
        graph->gnvtxs, gnedges/2, (idx_t)11, graph->ncon, (idx_t)1);
    fclose(fpin);
  }

  gkMPI_Barrier(comm);
  for (penum=0; penum<npes; penum++) {
    if (mype == penum) {

      if ((fpin = fopen(partfile, "a")) == NULL)
        errexit("Failed to open file %s", partfile);

      for (i=0; i<graph->nvtxs; i++) {
        for (h=0; h<graph->ncon; h++)
          fprintf(fpin, "%"PRIDX" ", graph->vwgt[i*graph->ncon+h]);

        for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++) {
          fprintf(fpin, "%"PRIDX" ", graph->adjncy[j]+1);
          fprintf(fpin, "%"PRIDX" ", graph->adjwgt[j]);
        }
      fprintf(fpin, "\n");
      }
      fclose(fpin);
    }
    gkMPI_Barrier(comm);
  }

  return;
}


/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
void ReadTestGraph(graph_t *graph, char *filename, MPI_Comm comm)
{
  idx_t i, k, l, npes, mype;
  idx_t nvtxs, penum, snvtxs;
  idx_t *gxadj, *gadjncy;  
  idx_t *vtxdist, *sxadj, *ssize = NULL;
  MPI_Status status;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist = ismalloc(npes+1, 0, "ReadGraph: vtxdist");

  if (mype == 0) {
    ssize = ismalloc(npes, 0, "ReadGraph: ssize");

    ReadMetisGraph(filename, &nvtxs, &gxadj, &gadjncy);

    printf("Nvtxs: %"PRIDX", Nedges: %"PRIDX"\n", nvtxs, gxadj[nvtxs]);

    /* Construct vtxdist and send it to all the processors */
    vtxdist[0] = 0;
    for (i=0,k=nvtxs; i<npes; i++) {
      l = k/(npes-i);
      vtxdist[i+1] = vtxdist[i]+l;
      k -= l;
    }
  }

  gkMPI_Bcast((void *)vtxdist, npes+1, IDX_T, 0, comm);

  graph->gnvtxs = vtxdist[npes];
  graph->nvtxs = vtxdist[mype+1]-vtxdist[mype];
  graph->xadj = imalloc(graph->nvtxs+1, "ReadGraph: xadj");

  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      snvtxs = vtxdist[penum+1]-vtxdist[penum];
      sxadj = imalloc(snvtxs+1, "ReadGraph: sxadj");

      icopy(snvtxs+1, gxadj+vtxdist[penum], sxadj);
      for (i=snvtxs; i>=0; i--)
        sxadj[i] -= sxadj[0];

      ssize[penum] = gxadj[vtxdist[penum+1]] - gxadj[vtxdist[penum]];

      if (penum == mype) 
        icopy(snvtxs+1, sxadj, graph->xadj);
      else
        gkMPI_Send((void *)sxadj, snvtxs+1, IDX_T, penum, 1, comm); 

      gk_free((void **)&sxadj, LTERM);
    }
  }
  else 
    gkMPI_Recv((void *)graph->xadj, graph->nvtxs+1, IDX_T, 0, 1, comm, &status);


  graph->nedges = graph->xadj[graph->nvtxs];
  graph->adjncy = imalloc(graph->nedges, "ReadGraph: graph->adjncy");

  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      if (penum == mype) 
        icopy(ssize[penum], gadjncy+gxadj[vtxdist[penum]], graph->adjncy);
      else
        gkMPI_Send((void *)(gadjncy+gxadj[vtxdist[penum]]), ssize[penum], IDX_T, penum, 1, comm); 
    }

    gk_free((void **)&ssize, LTERM);
  }
  else 
    gkMPI_Recv((void *)graph->adjncy, graph->nedges, IDX_T, 0, 1, comm, &status);

  graph->vwgt = NULL;
  graph->adjwgt = NULL;

  if (mype == 0) 
    gk_free((void **)&gxadj, &gadjncy, LTERM);

  MALLOC_CHECK(NULL);
}



/*************************************************************************/
/*! Reads the coordinates associated with the vertices of a graph */
/*************************************************************************/
real_t *ReadTestCoordinates(graph_t *graph, char *filename, idx_t *r_ndims, 
            MPI_Comm comm)
{
  idx_t i, j, k, npes, mype, penum, ndims;
  real_t *xyz, *txyz;
  float ftmp;
  char line[8192];
  FILE *fpin=NULL;
  idx_t *vtxdist;
  MPI_Status status;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist;

  if (mype == 0) {
    if ((fpin = fopen(filename, "r")) == NULL) 
      errexit("Failed to open file %s\n", filename);

    /* determine the number of dimensions */
    if (fgets(line, 8191, fpin) == NULL)
      errexit("Failed to read from file %s\n", filename);
    ndims = sscanf(line, "%e %e %e", &ftmp, &ftmp, &ftmp);
    fclose(fpin);
    if ((fpin = fopen(filename, "r")) == NULL) 
      errexit("Failed to open file %s\n", filename);
  }
  gkMPI_Bcast((void *)&ndims, 1, IDX_T, 0, comm);
  *r_ndims = ndims;

  xyz = rmalloc(graph->nvtxs*ndims, "ReadTestCoordinates");
  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      txyz = rmalloc((vtxdist[penum+1]-vtxdist[penum])*ndims, "ReadTestCoordinates");
      for (k=0, i=vtxdist[penum]; i<vtxdist[penum+1]; i++, k++) {
        for (j=0; j<ndims; j++)
          if (fscanf(fpin, "%"SCREAL" ", txyz+k*ndims+j) != 1)
            errexit("Failed to read coordinate for node\n");
      }

      if (penum == mype) 
        memcpy((void *)xyz, (void *)txyz, sizeof(real_t)*ndims*k);
      else 
        gkMPI_Send((void *)txyz, ndims*k, REAL_T, penum, 1, comm); 
      gk_free((void **)&txyz, LTERM);
    }
    fclose(fpin);
  }
  else 
    gkMPI_Recv((void *)xyz, ndims*graph->nvtxs, REAL_T, 0, 1, comm, &status);

  return xyz;
}



/*************************************************************************
* This function reads the spd matrix
**************************************************************************/
void ReadMetisGraph(char *filename, idx_t *r_nvtxs, idx_t **r_xadj, idx_t **r_adjncy)
{
  idx_t i, k, edge, nvtxs, nedges;
  idx_t *xadj, *adjncy;
  char *line, *oldstr, *newstr;
  FILE *fpin;

  line = gk_cmalloc(MAXLINE+1, "ReadMetisGraph: line");

  if ((fpin = fopen(filename, "r")) == NULL) {
    printf("Failed to open file %s\n", filename);
    exit(0);
  }

  oldstr = fgets(line, MAXLINE, fpin);
  sscanf(line, "%"PRIDX" %"PRIDX"", &nvtxs, &nedges);
  nedges *=2;

  xadj = imalloc(nvtxs+1, "ReadGraph: xadj");
  adjncy = imalloc(nedges, "ReadGraph: adjncy");

  /* Start reading the graph file */
  for (xadj[0]=0, k=0, i=0; i<nvtxs; i++) {
    oldstr = fgets(line, MAXLINE, fpin);
    oldstr = line;
    newstr = NULL;

    for (;;) {
      edge = strtoidx(oldstr, &newstr, 10) -1;
      oldstr = newstr;

      if (edge < 0)
        break;

      adjncy[k++] = edge;
    } 
    xadj[i+1] = k;
  }

  fclose(fpin);

  gk_free((void **)&line, LTERM);

  *r_nvtxs = nvtxs;
  *r_xadj = xadj;
  *r_adjncy = adjncy;
}


/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
void Mc_SerialReadGraph(graph_t *graph, char *filename, idx_t *wgtflag, MPI_Comm comm)
{
  idx_t i, k, l, npes, mype;
  idx_t nvtxs, ncon, nobj, fmt;
  idx_t penum, snvtxs;
  idx_t *gxadj, *gadjncy, *gvwgt, *gadjwgt;  
  idx_t *vtxdist, *sxadj, *ssize = NULL;
  MPI_Status status;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist = ismalloc(npes+1, 0, "ReadGraph: vtxdist");

  if (mype == 0) {
    ssize = ismalloc(npes, 0, "ReadGraph: ssize");

    Mc_SerialReadMetisGraph(filename, &nvtxs, &ncon, &nobj, &fmt, &gxadj, &gvwgt,
	&gadjncy, &gadjwgt, wgtflag);

    printf("Nvtxs: %"PRIDX", Nedges: %"PRIDX"\n", nvtxs, gxadj[nvtxs]);

    /* Construct vtxdist and send it to all the processors */
    vtxdist[0] = 0;
    for (i=0,k=nvtxs; i<npes; i++) {
      l = k/(npes-i);
      vtxdist[i+1] = vtxdist[i]+l;
      k -= l;
    }
  }

  gkMPI_Bcast((void *)(&fmt), 1, IDX_T, 0, comm);
  gkMPI_Bcast((void *)(&ncon), 1, IDX_T, 0, comm);
  gkMPI_Bcast((void *)(&nobj), 1, IDX_T, 0, comm);
  gkMPI_Bcast((void *)(wgtflag), 1, IDX_T, 0, comm);
  gkMPI_Bcast((void *)vtxdist, npes+1, IDX_T, 0, comm);

  graph->gnvtxs = vtxdist[npes];
  graph->nvtxs = vtxdist[mype+1]-vtxdist[mype];
  graph->ncon = ncon;
  graph->xadj = imalloc(graph->nvtxs+1, "ReadGraph: xadj");
  /*************************************************/
  /* distribute xadj array */
  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      snvtxs = vtxdist[penum+1]-vtxdist[penum];
      sxadj = imalloc(snvtxs+1, "ReadGraph: sxadj");

      icopy(snvtxs+1, gxadj+vtxdist[penum], sxadj);
      for (i=snvtxs; i>=0; i--)
        sxadj[i] -= sxadj[0];

      ssize[penum] = gxadj[vtxdist[penum+1]] - gxadj[vtxdist[penum]];

      if (penum == mype) 
        icopy(snvtxs+1, sxadj, graph->xadj);
      else
        MPI_Send((void *)sxadj, snvtxs+1, IDX_T, penum, 1, comm); 

      gk_free((void **)&sxadj, LTERM);
    }
  }
  else 
    gkMPI_Recv((void *)graph->xadj, graph->nvtxs+1, IDX_T, 0, 1, comm,
		&status);



  graph->nedges = graph->xadj[graph->nvtxs];
  graph->adjncy = imalloc(graph->nedges, "ReadGraph: graph->adjncy");
  /*************************************************/
  /* distribute adjncy array */
  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      if (penum == mype) 
        icopy(ssize[penum], gadjncy+gxadj[vtxdist[penum]], graph->adjncy);
      else
        gkMPI_Send((void *)(gadjncy+gxadj[vtxdist[penum]]), ssize[penum],
		IDX_T, penum, 1, comm); 
    }

  }
  else 
    gkMPI_Recv((void *)graph->adjncy, graph->nedges, IDX_T, 0, 1, comm,
		&status);


  graph->adjwgt = imalloc(graph->nedges*nobj, "ReadGraph: graph->adjwgt");
  if (fmt%10 > 0) {
    /*************************************************/
    /* distribute adjwgt array */
    if (mype == 0) {
      for (penum=0; penum<npes; penum++) {
        ssize[penum] *= nobj;
        if (penum == mype)
          icopy(ssize[penum], gadjwgt+(gxadj[vtxdist[penum]]*nobj), graph->adjwgt);
        else
          gkMPI_Send((void *)(gadjwgt+(gxadj[vtxdist[penum]]*nobj)), ssize[penum],
                IDX_T, penum, 1, comm);
      }

    }
    else
      gkMPI_Recv((void *)graph->adjwgt, graph->nedges*nobj, IDX_T, 0, 1,
		comm, &status);

  }
  else {
    for (i=0; i<graph->nedges*nobj; i++)
      graph->adjwgt[i] = 1;
  }

  graph->vwgt = imalloc(graph->nvtxs*ncon, "ReadGraph: graph->vwgt");
  if ((fmt/10)%10 > 0) {
    /*************************************************/
    /* distribute vwgt array */

    if (mype == 0) {
      for (penum=0; penum<npes; penum++) {
        ssize[penum] = (vtxdist[penum+1]-vtxdist[penum])*ncon;

        if (penum == mype) 
          icopy(ssize[penum], gvwgt+(vtxdist[penum]*ncon), graph->vwgt);
        else
          gkMPI_Send((void *)(gvwgt+(vtxdist[penum]*ncon)), ssize[penum],
		IDX_T, penum, 1, comm);
      }

      gk_free((void **)&ssize, LTERM);
    }
    else
      gkMPI_Recv((void *)graph->vwgt, graph->nvtxs*ncon, IDX_T, 0, 1,
		comm, &status);

  }
  else {
    for (i=0; i<graph->nvtxs*ncon; i++)
      graph->vwgt[i] = 1;
  }

  if (mype == 0) 
    gk_free((void **)&gxadj, &gadjncy, &gvwgt, &gadjwgt, LTERM);

  MALLOC_CHECK(NULL);
}



/*************************************************************************
* This function reads the spd matrix
**************************************************************************/
void Mc_SerialReadMetisGraph(char *filename, idx_t *r_nvtxs, idx_t *r_ncon, 
        idx_t *r_nobj, idx_t *r_fmt, idx_t **r_xadj, idx_t **r_vwgt, 
        idx_t **r_adjncy, idx_t **r_adjwgt, idx_t *wgtflag)
{
  idx_t i, k, l;
  idx_t ncon, nobj, edge, nvtxs, nedges;
  idx_t *xadj, *adjncy, *vwgt, *adjwgt;
  char *line, *oldstr, *newstr;
  idx_t fmt, readew, readvw;
  idx_t ewgt[1024];
  FILE *fpin;

  line = gk_cmalloc(MAXLINE+1, "line");

  if ((fpin = fopen(filename, "r")) == NULL) {
    printf("Failed to open file %s\n", filename);
    exit(-1);
  }

  oldstr = fgets(line, MAXLINE, fpin);
  fmt = ncon = nobj = 0;
  sscanf(line, "%"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX"", &nvtxs, &nedges, &fmt, &ncon, &nobj);
  readew = (fmt%10 > 0);
  readvw = ((fmt/10)%10 > 0);

  *wgtflag = 0;
  if (readew)
    *wgtflag += 1;
  if (readvw)
    *wgtflag += 2;

  if ((ncon > 0 && !readvw) || (nobj > 0 && !readew)) {
    printf("fmt and ncon/nobj are inconsistant.\n");
    exit(-1);
  }

  nedges *=2;
  ncon = (ncon == 0 ? 1 : ncon);
  nobj = (nobj == 0 ? 1 : nobj);

  xadj = imalloc(nvtxs+1, "ReadGraph: xadj");
  adjncy = imalloc(nedges, "Mc_ReadGraph: adjncy");
  vwgt = (readvw ? imalloc(ncon*nvtxs, "RG: vwgt") : NULL);
  adjwgt = (readew ? imalloc(nobj*nedges, "RG: adjwgt") : NULL);

  /* Start reading the graph file */
  for (xadj[0]=0, k=0, i=0; i<nvtxs; i++) {
    while (fgets(line, MAXLINE, fpin) && line[0] == '%');
    oldstr = line;
    newstr = NULL;

    if (readvw) {
      for (l=0; l<ncon; l++) {
        vwgt[i*ncon+l] = strtoidx(oldstr, &newstr, 10);
        oldstr = newstr;
      }
    }

    for (;;) {
      edge   = strtoidx(oldstr, &newstr, 10) -1;
      oldstr = newstr;

      if (readew) {
        for (l=0; l<nobj; l++) {
          ewgt[l] = strtoreal(oldstr, &newstr);
          oldstr = newstr;
        }
      }

      if (edge < 0)
        break;

      adjncy[k] = edge;
      if (readew)
        for (l=0; l<nobj; l++)
          adjwgt[k*nobj+l] = ewgt[l];
      k++;
    }
    xadj[i+1] = k;
  }

  fclose(fpin);

  gk_free((void **)&line, LTERM);

  *r_nvtxs = nvtxs;
  *r_ncon = ncon;
  *r_nobj = nobj;
  *r_fmt = fmt;
  *r_xadj = xadj;
  *r_vwgt = vwgt;
  *r_adjncy = adjncy;
  *r_adjwgt = adjwgt;
}



/*************************************************************************
* This function writes out a partition vector
**************************************************************************/
void WritePVector(char *gname, idx_t *vtxdist, idx_t *part, MPI_Comm comm)
{
  idx_t i, j, k, l, rnvtxs, npes, mype, penum;
  FILE *fpin;
  idx_t *rpart;
  char partfile[256];
  MPI_Status status;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  if (mype == 0) {
    sprintf(partfile, "%s.part", gname);
    if ((fpin = fopen(partfile, "w")) == NULL) 
      errexit("Failed to open file %s", partfile);

    for (i=0; i<vtxdist[1]; i++)
      fprintf(fpin, "%"PRIDX"\n", part[i]);

    for (penum=1; penum<npes; penum++) {
      rnvtxs = vtxdist[penum+1]-vtxdist[penum];
      rpart = imalloc(rnvtxs, "rpart");
      MPI_Recv((void *)rpart, rnvtxs, IDX_T, penum, 1, comm, &status);

      for (i=0; i<rnvtxs; i++)
        fprintf(fpin, "%"PRIDX"\n", rpart[i]);

      gk_free((void **)&rpart, LTERM);
    }
    fclose(fpin);
  }
  else
    MPI_Send((void *)part, vtxdist[mype+1]-vtxdist[mype], IDX_T, 0, 1, comm); 

}


/*************************************************************************
* This function writes out the ordering vector
**************************************************************************/
void WriteOVector(char *gname, idx_t *vtxdist, idx_t *order, MPI_Comm comm)
{
  idx_t i, j, k, l, rnvtxs, npes, mype, penum;
  FILE *fpout;
  idx_t *rorder, *gorder;
  char orderfile[256];
  MPI_Status status;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  if (mype == 0) {
    gorder = ismalloc(vtxdist[npes], 0, "WriteOVector: gorder");

    sprintf(orderfile, "%s.order.%"PRIDX"", gname, npes);
    if ((fpout = fopen(orderfile, "w")) == NULL) 
      errexit("Failed to open file %s", orderfile);

    for (i=0; i<vtxdist[1]; i++) {
      gorder[order[i]]++;
      fprintf(fpout, "%"PRIDX"\n", order[i]);
    }

    for (penum=1; penum<npes; penum++) {
      rnvtxs = vtxdist[penum+1]-vtxdist[penum];
      rorder = imalloc(rnvtxs, "rorder");
      MPI_Recv((void *)rorder, rnvtxs, IDX_T, penum, 1, comm, &status);

      for (i=0; i<rnvtxs; i++) {
        gorder[rorder[i]]++;
        fprintf(fpout, "%"PRIDX"\n", rorder[i]);
      }

      gk_free((void **)&rorder, LTERM);
    }
    fclose(fpout);

    /* Check the global ordering */
    for (i=0; i<vtxdist[npes]; i++) {
      if (gorder[i] != 1)
        printf("Global ordering problems with index: %"PRIDX" [%"PRIDX"]\n", i, gorder[i]);
    }
    gk_free((void **)&gorder, LTERM);
  }
  else
    MPI_Send((void *)order, vtxdist[mype+1]-vtxdist[mype], IDX_T, 0, 1, comm); 

}


/*************************************************************************
* This function reads a mesh from a file
**************************************************************************/
void ParallelReadMesh(mesh_t *mesh, char *filename, MPI_Comm comm)
{
  idx_t i, j, k, pe;
  idx_t npes, mype, ier;
  idx_t gnelms, nelms, your_nelms, etype, maxnelms;
  idx_t maxnode, gmaxnode, minnode, gminnode;
  idx_t *elmdist, *elements;
  idx_t *your_elements;
  MPI_Status stat;
  char *line = NULL, *oldstr, *newstr;
  FILE *fpin = NULL;
  idx_t esize, esizes[5] = {-1, 3, 4, 8, 4};
  idx_t mgcnum, mgcnums[5] = {-1, 2, 3, 4, 2};

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  elmdist = mesh->elmdist = ismalloc(npes+1, 0, "ReadGraph: elmdist");

  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL){
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    MPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      fclose(fpin);
      MPI_Finalize();
      exit(0);
    }

    line = gk_cmalloc(MAXLINE+1, "line");

    while (fgets(line, MAXLINE, fpin) && line[0] == '%');
    sscanf(line, "%"PRIDX" %"PRIDX"", &gnelms, &etype);

    /* Construct elmdist and send it to all the processors */
    elmdist[0] = 0;
    for (i=0,j=gnelms; i<npes; i++) {
      k = j/(npes-i);
      elmdist[i+1] = elmdist[i]+k;
      j -= k;
    }

    MPI_Bcast((void *)elmdist, npes+1, IDX_T, npes-1, comm);
  }
  else {
    MPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    MPI_Bcast((void *)elmdist, npes+1, IDX_T, npes-1, comm);
  }

  MPI_Bcast((void *)(&etype), 1, IDX_T, npes-1, comm);

  gnelms = mesh->gnelms = elmdist[npes];
  nelms = mesh->nelms = elmdist[mype+1]-elmdist[mype];
  mesh->etype = etype;
  esize = esizes[etype];
  mgcnum = mgcnums[etype];

  elements = mesh->elements = imalloc(nelms*esize, "ParallelReadMesh: elements");

  if (mype == npes-1) {
    maxnelms = 0;
    for (i=0; i<npes; i++) {
      maxnelms = (maxnelms > elmdist[i+1]-elmdist[i]) ?
      maxnelms : elmdist[i+1]-elmdist[i];
    }

    your_elements = imalloc(maxnelms*esize, "your_elements");

    for (pe=0; pe<npes; pe++) {
      your_nelms = elmdist[pe+1]-elmdist[pe];
      for (i=0; i<your_nelms; i++) {

        oldstr = fgets(line, MAXLINE, fpin);
        oldstr = line;
        newstr = NULL;

        /*************************************/
        /* could get element weigts here too */
        /*************************************/

        for (j=0; j<esize; j++) {
          your_elements[i*esize+j] = strtoidx(oldstr, &newstr, 10);
          oldstr = newstr;
        }
      }

      if (pe < npes-1) {
        MPI_Send((void *)your_elements, your_nelms*esize, IDX_T, pe, 0, comm);
      }
      else {
        for (i=0; i<your_nelms*esize; i++)
          elements[i] = your_elements[i];
      }
    }
    fclose(fpin);
    gk_free((void **)&your_elements, LTERM);
  }
  else {
    MPI_Recv((void *)elements, nelms*esize, IDX_T, npes-1, 0, comm, &stat);
  }

  /*********************************/
  /* now check for number of nodes */
  /*********************************/
  minnode = imin(nelms*esize, elements);
  MPI_Allreduce((void *)&minnode, (void *)&gminnode, 1, IDX_T, MPI_MIN, comm);
  for (i=0; i<nelms*esize; i++)
    elements[i] -= gminnode;

  maxnode = imax(nelms*esize, elements);
  MPI_Allreduce((void *)&maxnode, (void *)&gmaxnode, 1, IDX_T, MPI_MAX, comm);
  mesh->gnns = gmaxnode+1;

  if (mype==0) 
    printf("Nelements: %"PRIDX", Nnodes: %"PRIDX", EType: %"PRIDX"\n", 
        gnelms, mesh->gnns, etype);
}


