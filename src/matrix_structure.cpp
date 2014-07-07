/*!
 * \file matrix_structure.cpp
 * \brief Main subroutines for doing the sparse structures.
 * \author - Original Author: Aerospace Design Laboratory (Stanford University) <http://su2.stanford.edu>.
           - Current development: Aerospace Computing Laboratory (ACL)
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

#include "../include/matrix_structure.hpp"

CSysMatrix::CSysMatrix(void) {
  
  /*--- Array initialization ---*/
	matrix            = NULL;
	row_ptr           = NULL;
	col_ind           = NULL;
	block             = NULL;
	prod_block_vector = NULL;
    prod_row_vector   = NULL;
    aux_vector        = NULL;
    invM              = NULL;
    LineletBool       = NULL;
    LineletPoint      = NULL;
  
}

CSysMatrix::~CSysMatrix(void) {

    if (matrix != NULL)             delete [] matrix;
    if (row_ptr != NULL)            delete [] row_ptr;
    if (col_ind != NULL)            delete [] col_ind;
    if (block != NULL)              delete [] block;
    if (prod_block_vector != NULL)  delete [] prod_block_vector;
    if (prod_row_vector != NULL)    delete [] prod_row_vector;
    if (aux_vector != NULL)         delete [] aux_vector;
    if (invM != NULL)               delete [] invM;
    if (LineletBool != NULL)        delete [] LineletBool;
    if (LineletPoint != NULL)       delete [] LineletPoint;

    /*--- Need to set to NULL to avoid double-delete[] ---*/
    matrix            = NULL;
    row_ptr           = NULL;
    col_ind           = NULL;
    block             = NULL;
    prod_block_vector = NULL;
    prod_row_vector   = NULL;
    aux_vector        = NULL;
    invM              = NULL;
    LineletBool       = NULL;
    LineletPoint      = NULL;
}

void CSysMatrix::Initialize(int n_verts, int n_verts_global, int n_var, int n_eqns, array<array<int> > &v2e, array<int> &v2n_e, array<int> &e2v) {
	unsigned long iPoint, *row_ptr, *col_ind, *vneighs, index, nnz;
    unsigned short iNeigh, nNeigh, Max_nNeigh, iEdge;

    nPoint = n_verts;              // Assign number of points in the mesh (on processor)

	/*--- Don't delete *row_ptr, *col_ind because they are asigned to the Jacobian structure. ---*/
	row_ptr = new unsigned long [nPoint+1];
	row_ptr[0] = 0;
	for (iPoint = 0; iPoint < nPoint; iPoint++)
        row_ptr[iPoint+1] = row_ptr[iPoint]+(v2n_e(iPoint)+1); // +1 -> to include diagonal element
	nnz = row_ptr[nPoint];
  
	col_ind = new unsigned long [nnz];
  
    Max_nNeigh = 0;
    for (iPoint = 0; iPoint < nPoint; iPoint++) {
        nNeigh = v2n_e(iPoint);
        if (nNeigh > Max_nNeigh) Max_nNeigh = nNeigh;
    }
	vneighs = new unsigned long [Max_nNeigh+1]; // +1 -> to include diagonal
  
	for (iPoint = 0; iPoint < nPoint; iPoint++) {
        nNeigh = v2n_e(iPoint);
        for (iNeigh = 0; iNeigh < nNeigh; iNeigh++) {
            iEdge = v2e(iPoint)(iNeigh);
            if (e2v(iEdge,0) == iPoint) {
                vneighs[iNeigh] = e2v(iEdge,1);
            }else{
                vneighs[iNeigh] = e2v(iEdge,0);
            }
        }
		vneighs[nNeigh] = iPoint;
		sort(vneighs,vneighs+nNeigh+1);
		index = row_ptr[iPoint];
		for (iNeigh = 0; iNeigh <= nNeigh; iNeigh++) {
			col_ind[index] = vneighs[iNeigh];
			index++;
		}
	}
  
    /*--- Set the indices in the in the sparce matrix structure ---*/
    SetIndexes(n_verts, n_verts_global, n_var, n_eqns, row_ptr, col_ind, nnz);
  
    /*--- Initialization to zero ---*/
    SetValZero();
  
	delete[] vneighs;
}

void CSysMatrix::SetIndexes(int n_verts, int n_verts_global, int n_var, int n_eqns, unsigned long* val_row_ptr, unsigned long* val_col_ind, unsigned long val_nnz) {
  
    nPoint = n_verts;              // Assign number of points in the mesh (on processor)
    nPointDomain = n_verts_global;  // Assign number of points in the mesh (across all procs)
    nVar = n_var;                  // Assign number of vars in each block system
    nEqn = n_eqns;                   // Assign number of eqns in each block system
	nnz = val_nnz;                    // Assign number of possible non zero blocks
	row_ptr = val_row_ptr;
	col_ind = val_col_ind;
	
    matrix            = new double [nnz*nVar*nEqn];	// Reserve memory for the values of the matrix
    block             = new double [nVar*nEqn];
    prod_block_vector = new double [nEqn];
    prod_row_vector   = new double [nVar];
    aux_vector        = new double [nVar];
    invM              = new double [nPoint*nVar*nEqn];	// Reserve memory for the values of the inverse of the preconditioner

    /*--- Memory initialization ---*/
    unsigned long iVar;
    for (iVar = 0; iVar < nnz*nVar*nEqn; iVar++)    matrix[iVar] = 0.0;
    for (iVar = 0; iVar < nVar*nEqn; iVar++)        block[iVar] = 0.0;
    for (iVar = 0; iVar < nEqn; iVar++)             prod_block_vector[iVar] = 0.0;
    for (iVar = 0; iVar < nVar; iVar++)             prod_row_vector[iVar] = 0.0;
    for (iVar = 0; iVar < nVar; iVar++)             aux_vector[iVar] = 0.0;
    for (iVar = 0; iVar < nPoint*nVar*nEqn; iVar++) invM[iVar] = 0.0;
  
}

void CSysMatrix::GetBlock(unsigned long block_i, unsigned long block_j) {
	unsigned long step = 0, index, iVar;
	
	for (index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
        //step++;
		if (col_ind[index] == block_j) {
			for (iVar = 0; iVar < nVar*nEqn; iVar++)
                //block[iVar] = matrix[(row_ptr[block_i]+step-1)*nVar*nEqn+iVar];
                block[iVar] = matrix[(index)*nVar*nEqn+iVar];
			break;
		}
	}
}

void CSysMatrix::DisplayBlock(void) {
	unsigned short iVar, jVar;
	
	for (iVar = 0; iVar < nVar; iVar++) {
		for (jVar = 0; jVar < nEqn; jVar++)
			cout << block[iVar*nEqn+jVar] << "  ";
		cout << endl;
	}
}

void CSysMatrix::ReturnBlock(double **val_block) {
	unsigned short iVar, jVar;
	for (iVar = 0; iVar < nVar; iVar++)
		for (jVar = 0; jVar < nEqn; jVar++)
			val_block[iVar][jVar] = block[iVar*nEqn+jVar];
}

void CSysMatrix::SetBlock(unsigned long block_i, unsigned long block_j, double **val_block) {
	unsigned long iVar, jVar, index, step = 0;
	
	for (index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		step++;
		if (col_ind[index] == block_j) {
			for (iVar = 0; iVar < nVar; iVar++)
				for (jVar = 0; jVar < nEqn; jVar++)
					matrix[(row_ptr[block_i]+step-1)*nVar*nEqn+iVar*nEqn+jVar] = val_block[iVar][jVar];
			break;
		}
	}
}

void CSysMatrix::AddBlock(unsigned long block_i, unsigned long block_j, double **val_block) {
	unsigned long iVar, jVar, index, step = 0;
	
	for (index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		step++;
		if (col_ind[index] == block_j) {
			for (iVar = 0; iVar < nVar; iVar++)
				for (jVar = 0; jVar < nEqn; jVar++)
					matrix[(row_ptr[block_i]+step-1)*nVar*nEqn+iVar*nEqn+jVar] += val_block[iVar][jVar];
			break;
		}
	}
}

void CSysMatrix::AddBlock(unsigned long block_i, unsigned long block_j, array<double> val_block) {
	unsigned long iVar, jVar, index, step = 0;
	
	for (index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		step++;
		if (col_ind[index] == block_j) {
			for (iVar = 0; iVar < nVar; iVar++)
				for (jVar = 0; jVar < nEqn; jVar++)
					matrix[(row_ptr[block_i]+step-1)*nVar*nEqn+iVar*nEqn+jVar] += val_block(iVar,jVar);
			break;
		}
	}
}

void CSysMatrix::SubtractBlock(unsigned long block_i, unsigned long block_j, double **val_block) {
	unsigned long iVar, jVar, index, step = 0;
	
	for (index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		step++;
		if (col_ind[index] == block_j) {
			for (iVar = 0; iVar < nVar; iVar++)
				for (jVar = 0; jVar < nEqn; jVar++)
					matrix[(row_ptr[block_i]+step-1)*nVar*nEqn+iVar*nEqn+jVar] -= val_block[iVar][jVar];
			break;
		}
	}
}

void CSysMatrix::AddVal2Diag(unsigned long block_i, double val_matrix) {
	unsigned long step = 0, iVar, index;
	
	for (index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		step++;
		if (col_ind[index] == block_i) {	// Only elements on the diagonal
			for (iVar = 0; iVar < nVar; iVar++)
				matrix[(row_ptr[block_i]+step-1)*nVar*nVar+iVar*nVar+iVar] += val_matrix;
			break;
		}
	}
}

void CSysMatrix::AddVal2Diag(unsigned long block_i,  double* val_matrix, unsigned short num_dim) {
	unsigned long step = 0, iVar, iSpecies;
	
	for (unsigned long index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		step++;
		if (col_ind[index] == block_i) {	// Only elements on the diagonal
			for (iVar = 0; iVar < nVar; iVar++) {
				iSpecies = iVar/(num_dim + 2);
				matrix[(row_ptr[block_i]+step-1)*nVar*nVar+iVar*nVar+iVar] += val_matrix[iSpecies];
			}
			break;
		}
	}
}

void CSysMatrix::AddVal2Diag(unsigned long block_i,  double* val_matrix, unsigned short val_nDim, unsigned short val_nDiatomics) {
	unsigned long step = 0, iVar, iSpecies;
	
	for (unsigned long index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		step++;
		if (col_ind[index] == block_i) {	// Only elements on the diagonal
			for (iVar = 0; iVar < nVar; iVar++) {
        if (iVar < (val_nDim+3)*val_nDiatomics) iSpecies = iVar / (val_nDim+3);
        else iSpecies = (iVar - (val_nDim+3)*val_nDiatomics) / (val_nDim+2) + val_nDiatomics;
				matrix[(row_ptr[block_i]+step-1)*nVar*nVar+iVar*nVar+iVar] += val_matrix[iSpecies];
			}
			break;
		}
	}
}


void CSysMatrix::DeleteValsRowi(unsigned long i) {
	unsigned long block_i = i/nVar;
	unsigned long row = i - block_i*nVar;
	unsigned long index, iVar;
  
	for (index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++) {
		for (iVar = 0; iVar < nVar; iVar++)
			matrix[index*nVar*nVar+row*nVar+iVar] = 0.0; // Delete row values in the block
		if (col_ind[index] == block_i)
			matrix[index*nVar*nVar+row*nVar+row] = 1.0; // Set 1 to the diagonal element
	}
}

double CSysMatrix::SumAbsRowi(unsigned long i) {
	unsigned long block_i = i/nVar;
	unsigned long row = i - block_i*nVar;
  
	double sum = 0;
	for (unsigned long index = row_ptr[block_i]; index < row_ptr[block_i+1]; index++)
		for (unsigned long iVar = 0; iVar < nVar; iVar ++)
			sum += fabs(matrix[index*nVar*nVar+row*nVar+iVar]);
  
	return sum;
}

void CSysMatrix::Gauss_Elimination(unsigned long block_i, double* rhs) {
	unsigned short jVar, kVar;
	short iVar;
	double weight, aux;
  
	GetBlock(block_i, block_i);
  
	if (nVar == 1) {
    if (fabs(block[0]) < eps) cout <<"Gauss' elimination error, value:" << abs(block[0]) << "." << endl;
		rhs[0] /= block[0];
    }
	else {

    //cout << "Performing Gauss Elimination to get UT matrix" << endl;
    /*cout << "Block (" << block_i << "," << block_i << "):" << endl;
    DisplayBlock();*/
    /*--- Transform system in Upper Matrix ---*/
    for (iVar = 1; iVar < (short)nVar; iVar++) {
      for (jVar = 0; jVar < iVar; jVar++) {
        if (fabs(block[jVar*nVar+jVar]) < eps) cout <<"Gauss' elimination error, value:" << fabs(block[jVar*nVar+jVar]) << "." << endl;
        weight = block[iVar*nVar+jVar] / block[jVar*nVar+jVar];
        for (kVar = jVar; kVar < nVar; kVar++)
          block[iVar*nVar+kVar] -= weight*block[jVar*nVar+kVar];
        rhs[iVar] -= weight*rhs[jVar];
      }
    }
    
    /*--- Backwards substitution ---*/
    if (fabs(block[nVar*nVar-1]) < eps) cout <<"Gauss' elimination error, value:" << fabs(block[nVar*nVar-1]) << "." << endl;
    rhs[nVar-1] = rhs[nVar-1] / block[nVar*nVar-1];
    for (iVar = nVar-2; iVar >= 0; iVar--) {
      aux = 0.0;
      for (jVar = iVar+1; jVar < nVar; jVar++)
        aux += block[iVar*nVar+jVar]*rhs[jVar];
      if (fabs(block[iVar*nVar+iVar]) < eps) cout <<"Gauss' elimination error, value:" << fabs(block[iVar*nVar+iVar]) << "." << endl;
      rhs[iVar] = (rhs[iVar]-aux) / block[iVar*nVar+iVar];
      if (iVar == 0) break;
    }
	}
}

void CSysMatrix::Gauss_Elimination(double* Block, double* rhs) {
	unsigned short jVar, kVar;
	short iVar;
	double weight;
  double aux;
  
	/*--- Copy block matrix, note that the original matrix
	 is modified by the algorithm---*/
	for (kVar = 0; kVar < nVar; kVar++)
		for (jVar = 0; jVar < nVar; jVar++)
			block[kVar*nVar+jVar] = Block[kVar*nVar+jVar];
  
  
	if (nVar == 1) {
    if (fabs(block[0]) < eps) cout <<"Gauss' elimination error." << endl;
		rhs[0] /= block[0];
  }
	else {
		/*--- Transform system in Upper Matrix ---*/
		for (iVar = 1; iVar < (short)nVar; iVar++) {
			for (jVar = 0; jVar < iVar; jVar++) {
        if (fabs(block[jVar*nVar+jVar]) < eps) cout <<"Gauss' elimination error." << endl;
				weight = block[iVar*nVar+jVar] / block[jVar*nVar+jVar];
				for (kVar = jVar; kVar < nVar; kVar++)
					block[iVar*nVar+kVar] -= weight*block[jVar*nVar+kVar];
				rhs[iVar] -= weight*rhs[jVar];
			}
		}
		
		/*--- Backwards substitution ---*/
    if (fabs(block[nVar*nVar-1]) < eps) cout <<"Gauss' elimination error." << endl;
		rhs[nVar-1] = rhs[nVar-1] / block[nVar*nVar-1];
		for (iVar = nVar-2; iVar >= 0; iVar--) {
			aux = 0.0;
			for (jVar = iVar+1; jVar < nVar; jVar++)
				aux += block[iVar*nVar+jVar]*rhs[jVar];
      if (fabs(block[iVar*nVar+iVar]) < eps) cout <<"Gauss' elimination error." << endl;
			rhs[iVar] = (rhs[iVar]-aux) / block[iVar*nVar+iVar];
			if (iVar == 0) break;
		}
	}
	
}

void CSysMatrix::ProdBlockVector(unsigned long block_i, unsigned long block_j, const CSysVector & vec) {
	unsigned long j = block_j*nVar;
	unsigned short iVar, jVar;
  
	GetBlock(block_i, block_j);
  
	for (iVar = 0; iVar < nVar; iVar++) {
		prod_block_vector[iVar] = 0;
		for (jVar = 0; jVar < nVar; jVar++)
			prod_block_vector[iVar] += block[iVar*nVar+jVar]*vec[j+jVar];
	}
}

void CSysMatrix::UpperProduct(CSysVector & vec, unsigned long row_i) {
	unsigned long iVar, index;
  
	for (iVar = 0; iVar < nVar; iVar++)
		prod_row_vector[iVar] = 0;
  
	for (index = row_ptr[row_i]; index < row_ptr[row_i+1]; index++) {
		if (col_ind[index] > row_i) {
			ProdBlockVector(row_i, col_ind[index], vec);
			for (iVar = 0; iVar < nVar; iVar++)
				prod_row_vector[iVar] += prod_block_vector[iVar];
		}
	}
}

void CSysMatrix::LowerProduct(CSysVector & vec, unsigned long row_i) {
	unsigned long iVar, index;
  
	for (iVar = 0; iVar < nVar; iVar++)
		prod_row_vector[iVar] = 0;
  
	for (index = row_ptr[row_i]; index < row_ptr[row_i+1]; index++) {
		if (col_ind[index] < row_i) {
			ProdBlockVector(row_i, col_ind[index], vec);
			for (iVar = 0; iVar < nVar; iVar++)
				prod_row_vector[iVar] += prod_block_vector[iVar];
		}
	}
  
}

void CSysMatrix::DiagonalProduct(CSysVector & vec, unsigned long row_i) {
	unsigned long iVar, index;
  
	for (iVar = 0; iVar < nVar; iVar++)
		prod_row_vector[iVar] = 0;
  
	for (index = row_ptr[row_i]; index < row_ptr[row_i+1]; index++) {
		if (col_ind[index] == row_i) {
			ProdBlockVector(row_i,col_ind[index],vec);
			for (iVar = 0; iVar < nVar; iVar++)
				prod_row_vector[iVar] += prod_block_vector[iVar];
		}
	}
}

#ifdef foo
void CSysMatrix::SendReceive_Solution(CSysVector & x, solution *FlowSol) {
  unsigned short iVar, iMarker, MarkerS, MarkerR;
    unsigned long iVertex, iPoint, nVertexS, nVertexR, nBufferS_Vector, nBufferR_Vector;
    double *Buffer_Receive = NULL, *Buffer_Send = NULL;
    int send_to, receive_from;

#ifdef _MPI
  MPI_Status status;
  MPI_Request send_request, recv_request;
#endif
  
    for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
        // only used in serial if perodic boundaries
        if ((config->GetMarker_All_Boundary(iMarker) == SEND_RECEIVE) &&
        (config->GetMarker_All_SendRecv(iMarker) > 0)) {
			
            MarkerS = iMarker;  MarkerR = iMarker+1;
      
      send_to = config->GetMarker_All_SendRecv(MarkerS)-1;
            receive_from = abs(config->GetMarker_All_SendRecv(MarkerR))-1;
			
            nVertexS = FlowSol->nVertex[MarkerS];  nVertexR = FlowSol->nVertex[MarkerR];
            nBufferS_Vector = nVertexS*nVar;        nBufferR_Vector = nVertexR*nVar;
      
      /*--- Allocate Receive and send buffers  ---*/
      Buffer_Receive = new double[nBufferR_Vector];
      Buffer_Send = new double[nBufferS_Vector];
      
      /*--- Copy the solution that should be sent ---*/
      for (iVertex = 0; iVertex < nVertexS; iVertex++) {
        iPoint = FlowSol->vertex[MarkerS][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          Buffer_Send[iVertex*nVar+iVar] = x[iPoint*nVar+iVar];
      }
      
#ifdef _MPI     
      /*--- Send/Receive information using Sendrecv ---*/
      MPI_Sendrecv(Buffer_Send, nBufferS_Vector, MPI_DOUBLE, send_to, 0,
                   Buffer_Receive, nBufferR_Vector, MPI_DOUBLE, receive_from, 0);
#else
      
      /*--- Receive information without MPI ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        iPoint = FlowSol->vertex[MarkerR][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          Buffer_Receive[iVar*nVertexR+iVertex] = Buffer_Send[iVar*nVertexR+iVertex];
      }
      
#endif
      
      /*--- Deallocate send buffer ---*/
      delete [] Buffer_Send;
      
      /*--- Do the coordinate transformation ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        
        /*--- Find point and its type of transformation ---*/
        iPoint = FlowSol->vertex[MarkerR][iVertex]->GetNode();
        
        /*--- Copy transformed conserved variables back into buffer. ---*/
        for (iVar = 0; iVar < nVar; iVar++)
          x[iPoint*nVar+iVar] = Buffer_Receive[iVertex*nVar+iVar];
        
      }
      
      /*--- Deallocate receive buffer ---*/
      delete [] Buffer_Receive;
      
    }
    
    }
}
#endif

void CSysMatrix::RowProduct(const CSysVector & vec, unsigned long row_i) {
	unsigned long iVar, index;
  
	for (iVar = 0; iVar < nVar; iVar++)
		prod_row_vector[iVar] = 0;
  
	for (index = row_ptr[row_i]; index < row_ptr[row_i+1]; index++) {
		ProdBlockVector(row_i, col_ind[index], vec);
		for (iVar = 0; iVar < nVar; iVar++)
			prod_row_vector[iVar] += prod_block_vector[iVar];
	}
}

void CSysMatrix::MatrixVectorProduct(const CSysVector & vec, CSysVector & prod) {
	unsigned long prod_begin, vec_begin, mat_begin, index, iVar, jVar, row_i;

#ifdef MPI
  MPI_Status status;
  MPI_Request send_request, recv_request;
#endif
  
	/*--- Some checks for consistency between CSysMatrix and the CSysVectors ---*/
	if ( (nVar != vec.GetNVar()) || (nVar != prod.GetNVar()) ) {
		cerr << "CSysMatrix::MatrixVectorProduct(const CSysVector&, CSysVector): "
    << "nVar values incompatible." << endl;
		throw(-1);
	}
	if ( (nPoint != vec.GetNBlk()) || (nPoint != prod.GetNBlk()) ) {
		cerr << "CSysMatrix::MatrixVectorProduct(const CSysVector&, CSysVector): "
    << "nPoint and nBlk values incompatible." << endl;
		throw(-1);
	}
  
	prod = 0.0; // set all entries of prod to zero
	for (row_i = 0; row_i < nPointDomain; row_i++) {
		prod_begin = row_i*nVar; // offset to beginning of block row_i
		for (index = row_ptr[row_i]; index < row_ptr[row_i+1]; index++) {
			vec_begin = col_ind[index]*nVar; // offset to beginning of block col_ind[index]
			mat_begin = (index*nVar*nVar); // offset to beginning of matrix block[row_i][col_ind[indx]]
			for (iVar = 0; iVar < nVar; iVar++) {
				for (jVar = 0; jVar < nVar; jVar++) {
					prod[(const unsigned int)(prod_begin+iVar)] += matrix[(const unsigned int)(mat_begin+iVar*nVar+jVar)]*vec[(const unsigned int)(vec_begin+jVar)];
				}
			}
		}
	}
  
  /*--- MPI Parallelization ---*/
  //SendReceive_Solution(prod, geometry, config);
}

void CSysMatrix::GetMultBlockBlock(double *c, double *a, double *b) {
	unsigned long iVar, jVar, kVar;
	
	for(iVar = 0; iVar < nVar; iVar++)
		for(jVar = 0; jVar < nVar; jVar++) {
			c[iVar*nVar+jVar] = 0.0;
			for(kVar = 0; kVar < nVar; kVar++)
				c[iVar*nVar+jVar] += a[iVar*nVar+kVar] * b[kVar*nVar+jVar];
		}
}

void CSysMatrix::GetMultBlockVector(double *c, double *a, double *b) {
	unsigned long iVar, jVar;
	
	for(iVar = 0; iVar < nVar; iVar++) {
		c[iVar] =  0.0;
		for(jVar = 0; jVar < nVar; jVar++)
			c[iVar] += a[iVar*nVar+jVar] * b[jVar];
	}
}

void CSysMatrix::GetSubsBlock(double *c, double *a, double *b) {
	unsigned long iVar, jVar;
	
	for(iVar = 0; iVar < nVar; iVar++)
		for(jVar = 0; jVar < nVar; jVar++)
			c[iVar*nVar+jVar] = a[iVar*nVar+jVar] - b[iVar*nVar+jVar];
}

void CSysMatrix::GetSubsVector(double *c, double *a, double *b) {
	unsigned long iVar;
	
	for(iVar = 0; iVar < nVar; iVar++)
		c[iVar] = a[iVar] - b[iVar];
}

void CSysMatrix::InverseBlock(double *Block, double *invBlock) {
	unsigned long iVar, jVar;
  
	for (iVar = 0; iVar < nVar; iVar++) {
		for (jVar = 0; jVar < nVar; jVar++)
			aux_vector[jVar] = 0.0;
		aux_vector[iVar] = 1.0;
		
		/*--- Compute the i-th column of the inverse matrix ---*/
		Gauss_Elimination(Block, aux_vector);
		
		for (jVar = 0; jVar < nVar; jVar++)
			invBlock[jVar*nVar+iVar] = aux_vector[jVar];
	}
	
}

void CSysMatrix::InverseDiagonalBlock(unsigned long block_i, double **invBlock) {
	unsigned long iVar, jVar;
  
	for (iVar = 0; iVar < nVar; iVar++) {
		for (jVar = 0; jVar < nVar; jVar++)
			aux_vector[jVar] = 0.0;
		aux_vector[iVar] = 1.0;
    
		/*--- Compute the i-th column of the inverse matrix ---*/
		Gauss_Elimination(block_i, aux_vector);
		for (jVar = 0; jVar < nVar; jVar++)
			invBlock[jVar][iVar] = aux_vector[jVar];
	}
  
}

void CSysMatrix::ComputeLU_SGSPreconditioner(const CSysVector & vec, CSysVector & prod) {
  unsigned long iPoint, iVar;

  /*--- There are two approaches to the parallelization (AIAA-2000-0927):
   1. Use a special scheduling algorithm which enables data parallelism by regrouping edges. This method has the
      advantage of producing exactly the same result as the single processor case, but it suffers from severe overhead
      penalties for parallel loop initiation, heavy interprocessor communications and poor load balance.
   2. Split the computational domain into several nonoverlapping regions according to the number of processors, and apply
      the SGS method inside of each region with (or without) some special interprocessor boundary treatment. This approach
      may suffer from convergence degradation but takes advantage of minimal parallelization overhead and good load balance. ---*/

    /*--- First part of the symmetric iteration: (D+L).x* = b ---*/
    /*cout <<"--ComputLU_SGSPreconditioner--" << endl;
    cout << "n_dims: " << nVar << ", nPointDomain: " << nPointDomain << endl;
    cout << "--paused--" << endl; cin.get();*/
    for (iPoint = 0; iPoint < nPointDomain; iPoint++) {
        LowerProduct(prod, iPoint);                                        // Compute L.x*
        for (iVar = 0; iVar < nVar; iVar++)
            aux_vector[iVar] = vec[iPoint*nVar+iVar] - prod_row_vector[iVar]; // Compute aux_vector = b - L.x*
        Gauss_Elimination(iPoint, aux_vector);                            // Solve D.x* = aux_vector
        for (iVar = 0; iVar < nVar; iVar++)
            prod[iPoint*nVar+iVar] = aux_vector[iVar];                       // Assesing x* = solution
        /*if(iPoint%500==0) {
            cout << "** pause **" << endl;
            cin.get();
        }*/
    }

    /*--- Inner send-receive operation the solution vector ---*/
    //SendReceive_Solution(prod, geometry, config);

    /*--- Second part of the symmetric iteration: (D+U).x_(1) = D.x* ---*/
    for (iPoint = nPointDomain-1; (int)iPoint >= 0; iPoint--) {
        DiagonalProduct(prod, iPoint);                 // Compute D.x*
        for (iVar = 0; iVar < nVar; iVar++)
            aux_vector[iVar] = prod_row_vector[iVar];   // Compute aux_vector = D.x*
        UpperProduct(prod, iPoint);                    // Compute U.x_(n+1)
        for (iVar = 0; iVar < nVar; iVar++)
            aux_vector[iVar] -= prod_row_vector[iVar];  // Compute aux_vector = D.x*-U.x_(n+1)
        Gauss_Elimination(iPoint, aux_vector);        // Solve D.x* = aux_vector
        for (iVar = 0; iVar < nVar; iVar++)
            prod[iPoint*nVar + iVar] = aux_vector[iVar]; // Assesing x_(1) = solution
    }

  /*--- Final send-receive operation the solution vector (redundant in CFD simulations) ---*/
    //SendReceive_Solution(prod, geometry, config);

}
