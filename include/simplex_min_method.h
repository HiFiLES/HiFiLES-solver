#ifndef SIMPLEX_MIN_METHOD_H
#define SIMPLEX_MIN_METHOD_H

/*!
 This file is an implementation of the downhill simplex optimization algorithm using C++.
*/

/*
 *  Copyright (C) 2010 Botao Jia

 * Obtained from: http://www.codeguru.com/cpp/article.php/c17505/Simplex-Optimization-Algorithm-and-Implemetation-in-C-Programming.htm
*/

//simplex.h
#include <vector>
#include <limits>
#include <algorithm>
#include <functional>
#include <iostream>
#include "Array.h"

template<class D, class OP>
std::vector<D> simplex_min_method(OP f,                   //target function
                       std::vector<D> init,    //initial guess of the parameters
                       D tol=1E8*std::numeric_limits<D>::epsilon(), //termination criteria
                       std::vector<std::vector<D> > x =  std::vector<std::vector<D> >(),//x: The Simplex
                       int max_iterations = 1E2){    //iteration step number

  int N=init.size();                         //space dimension
  const double a=1.0, b=1.0, g=0.5, h=0.5;   //coefficients
  //a: reflection  -> xr
  //b: expansion   -> xe
  //g: contraction -> xc
  //h: full contraction to x1
  std::vector<D> xcentroid_old(N,0);   //simplex center * (N+1)
  std::vector<D> xcentroid_new(N,0);   //simplex center * (N+1)
  std::vector<D> vf(N+1,0);            //f evaluated at simplex vertexes
  int x1=0, xn=0, xnp1=0;         //x1:   f(x1) = min { f(x1), f(x2)...f(x_{n+1} }
  //xnp1: f(xnp1) = max { f(x1), f(x2)...f(x_{n+1} }
  //xn:   f(xn)<f(xnp1) && f(xn)> all other f(x_i)
  int cnt=0; //iteration step number

  if(x.size()==0) //if no initial simplex is specified
    { //construct the trial simplex
      //based upon the initial guess parameters
      std::vector<D> del( init );
      std::transform(del.begin(), del.end(), del.begin(),
                     std::bind2nd( std::divides<D>() , 20) );//'20' is picked
      //assuming initial trail close to true

      for(int i=0; i<N; ++i){
          std::vector<D> tmp( init );
          tmp[i] +=  del[i];
          x.push_back( tmp );
        }
      x.push_back(init);//x.size()=N+1, x[i].size()=N

      //xcentriod
      std::transform(init.begin(), init.end(),
                     xcentroid_old.begin(), std::bind2nd(std::multiplies<D>(), N+1) );
    }//constructing the simplex finished

  //optimization begins
  for(cnt=0; cnt<max_iterations; ++cnt){

      for(int i=0;i<N+1;++i){
          vf[i]= f(x[i]);
        }

      x1=0; xn=0; xnp1=0;//find index of max, second max, min of vf.

      for(int i=0;i<vf.size();++i){
          if(vf[i]<vf[x1]){
              x1=i;
            }
          if(vf[i]>vf[xnp1]){
              xnp1=i;
            }
        }

      xn=x1;

      for(int i=0; i<vf.size();++i){
          if( vf[i]<vf[xnp1] && vf[i]>vf[xn] )
            xn=i;
        }
      //x1, xn, xnp1 are found

      std::vector<D> xg(N, 0);//xg: centroid of the N best vertexes
      for(int i=0; i<x.size(); ++i){
          if(i!=xnp1)
            std::transform(xg.begin(), xg.end(), x[i].begin(), xg.begin(), std::plus<D>() );
        }
      std::transform(xg.begin(), xg.end(),
                     x[xnp1].begin(), xcentroid_new.begin(), std::plus<D>());
      std::transform(xg.begin(), xg.end(), xg.begin(),
                     std::bind2nd(std::divides<D>(), N) );
      //xg found, xcentroid_new updated

      //termination condition
      D diff=0;          //calculate the difference of the simplex centers
      //see if the difference is less than the termination criteria
      for(int i=0; i<N; ++i)
        diff += fabs(xcentroid_old[i]-xcentroid_new[i]);

      if (diff/N < tol) break;              //terminate the optimizer
      else xcentroid_old.swap(xcentroid_new); //update simplex center

      //reflection:
      std::vector<D> xr(N,0);
      for( int i=0; i<N; ++i)
        xr[i]=xg[i]+a*(xg[i]-x[xnp1][i]);
      //reflection, xr found

      D fxr=f(xr);//record function at xr

      if(vf[x1]<=fxr && fxr<=vf[xn])
        std::copy(xr.begin(), xr.end(), x[xnp1].begin() );

      //expansion:
      else if(fxr<vf[x1]){
          std::vector<D> xe(N,0);
          for( int i=0; i<N; ++i)
            xe[i]=xr[i]+b*(xr[i]-xg[i]);
          if( f(xe) < fxr )
            std::copy(xe.begin(), xe.end(), x[xnp1].begin() );
          else
            std::copy(xr.begin(), xr.end(), x[xnp1].begin() );
        }//expansion finished,  xe is not used outside the scope

      //contraction:
      else if( fxr > vf[xn] ){
          std::vector<D> xc(N,0);
          for( int i=0; i<N; ++i)
            xc[i]=xg[i]+g*(x[xnp1][i]-xg[i]);
          if( f(xc) < vf[xnp1] )
            std::copy(xc.begin(), xc.end(), x[xnp1].begin() );
          else{

              for( int i=0; i<x.size(); ++i ){
                  if( i!=x1 ){
                      for(int j=0; j<N; ++j)
                        x[i][j] = x[x1][j] + h * ( x[i][j]-x[x1][j] );
                    }
                }

            }
        }//contraction finished, xc is not used outside the scope

    }//optimization is finished

  if(cnt == max_iterations){//max number of iterations reached before tol is satisfied
      //printf("Iteration limit reached, result may not be optimal at %s:%d\n",__FILE__,__LINE__);
    }
  return x[x1];
}

#endif // SIMPLEX_MIN_METHOD_H

