/*
program     : Calling CUDA-kernel using ArrayFire v3.4 library. ArrayFire v3.4 handles
	      the boilerplate of CUDA-API.

date        : 11th Sept 2017 (11/09/2017)

coder       : Muhammad Izham a.k.a Sugita

institution : Universiti Malaysia Perlis (UniMAP)

contact     : sugita5019@gmail.com

*/

/*
How to compile:
$ nvcc -ccbin=g++ -std=c++11 -o filename filename.cu -lcuda -lcudart -lafcuda
*/

#include<arrayfire.h>

#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<iostream>

#include "sys/time.h"
#include "time.h"


#include "blocksize.h"

#define  pi          4.0*atan(1.0)

#define  MIN2(x, y)  ((x) < (y) ? (x) : (y))

using namespace af;
using namespace std;

extern float LBMdiffusion2d(int, int, float, float*, float*, float*, float*, 
float*, float*, float*, float*, 
float*, float* );

void timestamp(FILE *);

int main()
{

struct timeval start,finish;
double duration;
float  flops=0.0;

 //int imax,jmax;
 //cout<<"Enter imax and jmax "<<endl;
 //cin>>imax>>jmax;
 
int imax = 256;
int jmax = 256;
std::cout<<"Enter imax, jmax? default is 256,256"<<std::endl;
cin>>imax;
cin>>jmax;

int nodes = imax*jmax;
  
float dx  = 1.0/((float)imax - 1);
float dy  = 1.0/((float)jmax - 1);
float dt  = dx;

  //float dx = 1.0;
  //float dy = 1.0;
  //float dt = 1.0;
  float kappa = 1.0;

cout<<"Adjust kappa? Default is 1.0 "<<endl;
cin>>kappa;


  /*adjusted kappa*/
  kappa =  kappa*( 1.0/float(imax-1));
  
  float omega = 1.0 /( 2.0*kappa + 0.5);
  

/*Initiate distribution function from host*/
 float* h_f1;
 float* h_f2;
 float* h_f3;
 float* h_f4;
 float* h_feq;
 
 float* h_fn1;
 float* h_fn2;
 float* h_fn3;
 float* h_fn4;
  
 h_f1 = new float [nodes];
 h_f2 = new float [nodes];
 h_f3 = new float [nodes];
 h_f4 = new float [nodes];
 h_feq = new float [nodes];

 h_fn1 = new float [nodes];
 h_fn2 = new float [nodes];
 h_fn3 = new float [nodes];
 h_fn4 = new float [nodes];

/*Initiate from host*/
float* h_Temp;
h_Temp = new float [nodes];

 
 /*Initiating temperature distribution*/
for(int i=0; i<imax; ++i){
 for(int j=0; j<jmax; ++j){
   int id = i*jmax + j; 
 h_Temp[id] = sin((float)i * dx * pi )*sin( (float)j * dy * pi);
 h_feq[id] = 0.25*h_Temp[id];
 
  }
}

int index = (imax/2)*jmax + (jmax/2);
printf("Temp[%d] = %f\n", index, h_Temp[index]);

//random number for initial equilibrium
srand(time(NULL));
for(int i=0; i<imax; ++i){
for(int j=0; j<jmax; ++j){
int id = i*jmax + j;
h_f1[id] = h_feq[id] + 0.001*float(rand())/float(RAND_MAX);
h_f2[id] = h_feq[id] + 0.001*float(rand())/float(RAND_MAX);
h_f3[id] = h_feq[id] + 0.001*float(rand())/float(RAND_MAX);
h_f4[id] = h_feq[id] + 0.001*float(rand())/float(RAND_MAX);

h_fn1[id] = h_feq[id]; 
h_fn2[id] = h_feq[id]; 
h_fn3[id] = h_feq[id]; 
h_fn4[id] = h_feq[id]; 

}
}

int itermax = 20000;

//Initiate ArrayFire class on GPU 

/*
array Temp0(nodes, h_Temp);
array Temp1(nodes, h_Temp); 
float *d_Temp0 = Temp0.device<float>();
float *d_Temp1 = Temp1.device<float>();
*/

array f1(nodes, h_f1);
array f2(nodes, h_f2);
array f3(nodes, h_f3);
array f4(nodes, h_f4);

float* d_f1 = f1.device<float>();
float* d_f2 = f2.device<float>();
float* d_f3 = f3.device<float>();
float* d_f4 = f4.device<float>();

array fn1(nodes, h_fn1);
array fn2(nodes, h_fn2);
array fn3(nodes, h_fn3);
array fn4(nodes, h_fn4);

float* d_fn1 = fn1.device<float>();
float* d_fn2 = fn2.device<float>();
float* d_fn3 = fn3.device<float>();
float* d_fn4 = fn4.device<float>();

//array Temp(nodes, h_Temp);
array feq(nodes, h_feq);

//float* d_Temp = Temp.device<float>();
float* d_feq = feq.device<float>();

/*Initial array object for update*/
array Tempn(nodes, h_Temp); //equals to initial temperature

float* d_Tempn = Tempn.device<float>();

sync();

float* h_copy;
h_copy = new float [nodes];

Tempn.host(h_copy);
int id = (imax/2)*jmax + (jmax/2);
printf("Position = %d, peak value: %f\n", id, h_copy[id]);
printf("Omega = %f\n", omega);


gettimeofday(&start, NULL);

// time loop
for(int iter =0; iter < itermax; ++iter){


flops += LBMdiffusion2d(imax,jmax,omega,
 d_f1, d_f2, d_f3, d_f4, d_fn1, d_fn2, d_fn3, d_fn4, d_feq, d_Tempn);


// Use unlock to return back to ArrayFire stream. Otherwise just use the
// d_Temp0 and d_Temp1 pointer.
// Temp0.unlock();
// Temp1.unlock();

if(iter > 0 && iter % 1000 == 0){

//output to file
int id = (imax/2)*jmax + (jmax/2);
Tempn.host(h_copy);
printf("time(%d) = %f, peak value for Tempn: %f\n", iter, (float)iter*dt, h_copy[id]);
}

//Update pointer
//d_Temp0 = d_Temp1;

/*Update pointer for distribution function*/
/*d_f1 = d_fn1;
d_f2 = d_fn2;
d_f3 = d_fn3;
d_f4 = d_fn4;
d_feq = d_feqn;
*/
}
// end time loop

 
gettimeofday(&finish, NULL);
duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
printf("Total operations : %f\n", flops);
flops = flops/(duration*1.0e09); 

printf("Elapsed time:%lf secs\n", duration);
printf("Time per loop: %lf secs\n", duration/(double)itermax);
printf("Performance : %.2f GFlops\n", flops);

FILE* fparam;
fparam = fopen("Parameter.txt","a"); 
fprintf(fparam,"%d\t %d\t %dx%d\t  %.2f\t", blockDim_x, blockDim_y, imax, jmax, flops);
timestamp(fparam);
fclose(fparam);
 

/*Transfering calculation data to host */
float* h_Temp0 = new float[nodes];
Tempn.host(h_Temp0);

FILE* fp;
fp = fopen("cuda_LBMheateq.csv","w");
fprintf(fp,"x, y, z, temp\n");
for(int i=0; i<imax; ++i){
 for(int j=0; j<jmax; ++j){
 int   id = i*jmax + j;
 float xg = (float)i*dx;
 float yg = (float)j*dy;
 fprintf(fp,"%f, %f, %f, %f\n", xg, yg, h_Temp0[id], h_Temp0[id]);
 }
}
fclose(fp);

 
 delete [] h_Temp0;

delete [] h_f1, h_f2, h_f3, h_f4;
delete [] h_fn1, h_fn2, h_fn3, h_fn4;
delete [] h_Temp, h_feq;
 

}
