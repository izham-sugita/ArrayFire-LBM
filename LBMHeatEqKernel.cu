#include<cuda.h>
#include "blocksize.h"

//play with the block size to find your best performance
//hint: blockDim_x should be bigger than blockDim_y
//hint: blockDim_x*blockDim_y should not exceed 1024 (gpu dependent)
// #define  blockDim_x       32
// #define  blockDim_y       32


/*fused advection-collision*/
__global__ void fusedAdvColl
(
int nx,
int ny,
float omega,
float* f1,
float* f2,
float* f3,
float* f4,
float* feq,
float* fn1,
float* fn2,
float* fn3,
float* fn4
)
{
   int    j,    jx,   jy;
   jy = blockDim.y*blockIdx.y + threadIdx.y;
   jx = blockDim.x*blockIdx.x + threadIdx.x;

if(jx>0 && jx<nx-1){
if(jy>0 && jy<ny-1){   

/*Slow memory access*/
/* 
j = ny*jx + jy;
fn1[j+ny] = (1.0 - omega)*f1[j] + omega*feq[j]; 
fn2[j+1] = (1.0 - omega)*f2[j] + omega*feq[j]; 
fn3[j-ny] = (1.0 - omega)*f3[j] + omega*feq[j]; 
fn4[j-1] = (1.0 - omega)*f4[j] + omega*feq[j];
*/

/*
j = nx*jy + jx;
fn1[j+1] = (1.0 - omega)*f1[j] + omega*feq[j]; 
fn2[j+nx] = (1.0 - omega)*f2[j] + omega*feq[j]; 
fn3[j-1] = (1.0 - omega)*f3[j] + omega*feq[j]; 
fn4[j-nx] = (1.0 - omega)*f4[j] + omega*feq[j];
*/

j = nx*jy + jx;
fn1[j] = (1.0 - omega)*f1[j-1] + omega*feq[j-1];
fn2[j] = (1.0 - omega)*f2[j-nx] + omega*feq[j-nx];
fn3[j] = (1.0 - omega)*f3[j+1] + omega*feq[j+1];
fn4[j] = (1.0 - omega)*f4[j+nx] + omega*feq[j+nx];

  }
 }


}

__global__ void macroscopic2
(
int nx,
int ny,
float* fn1,
float* fn2,
float* fn3,
float* fn4,
float* f1,
float* f2,
float* f3,
float* f4,
float* Tempn,
float* feq
)
{
   int    j,    jx,   jy;

   jy = blockDim.y*blockIdx.y + threadIdx.y;
   jx = blockDim.x*blockIdx.x + threadIdx.x;

   //j = ny*jx + jy; //cause slow memory access
   j = nx*jy + jx;

	Tempn[j] = fn1[j] + fn2[j] + fn3[j] + fn4[j];
	feq[j] = 0.25*Tempn[j];
	f1[j] = fn1[j];
	f2[j] = fn2[j];
	f3[j] = fn3[j];
	f4[j] = fn4[j];

}

__global__ void fusedAll
(
int nx,
int ny,
float omega,
float* f1,
float* f2,
float* f3,
float* f4,
float* feq,
float* fn1,
float* fn2,
float* fn3,
float* fn4,
float* Tempn
)
{

   int    j,    jx,   jy;
   jy = blockDim.y*blockIdx.y + threadIdx.y;
   jx = blockDim.x*blockIdx.x + threadIdx.x;

if(jx>0 && jx<nx-1){
if(jy>0 && jy<ny-1){   

j = nx*jy + jx;
fn1[j] = (1.0 - omega)*f1[j-1] + omega*feq[j-1]; 
fn2[j] = (1.0 - omega)*f2[j-nx] + omega*feq[j-nx]; 
fn3[j] = (1.0 - omega)*f3[j+1] + omega*feq[j+1]; 
fn4[j] = (1.0 - omega)*f4[j+nx] + omega*feq[j+nx];
Tempn[j] = fn1[j] + fn2[j] + fn3[j] + fn4[j];

}}

__syncthreads();

/* Updating */
if(jx>0 && jx<nx-1){
if(jy>0 && jy<ny-1){
feq[j] = 0.25*Tempn[j];
f1[j] = fn1[j];
f2[j] = fn2[j];
f3[j] = fn3[j];
f4[j] = fn4[j];

}
}

}


//kernel wrapper
float  LBMdiffusion2d
// ====================================================================
//
// purpose    :  2-dimensional diffusion equation solved by LBM
//
// date       :  July 9, 2018
// programmer :  Muhammad Izham aka Sugita
// place      :  Universiti Malaysia Perlis
//
(
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   float    omega,      /* relaxation parameter                      */
float* f1,
float* f2,
float* f3, 
float* f4,
float* fn1,
float* fn2,
float* fn3,
float* fn4,
float* feq,
float* Tempn
)
{

dim3  grid(nx/blockDim_x,ny/blockDim_y,1);
dim3  threads(blockDim_x,blockDim_y,1);

/*
fusedAdvColl<<<grid,threads>>>(nx,ny,omega,f1,f2,f3,f4,feq,fn1,fn2,fn3,fn4);
macroscopic2<<<grid,threads>>>(nx,ny,fn1,fn2,fn3,fn4,f1,f2,f3,f4,Tempn,feq);
*/

/*Fixed!*/
fusedAll<<<grid,threads>>>(nx,ny,omega,f1,f2,f3,f4,feq,fn1,fn2,fn3,fn4,Tempn);

return (float)(nx*ny)*7.0;

}


