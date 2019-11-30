#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 

typedef struct {
  int width;				
  int height;				
  int stride;
  float* elements;
} Matrix;
 
float rand(float a,float b)
{

	return(b - a) * ((float)rand() / RAND_MAX) + a;
}


__device__ float GetElement(const Matrix A, int row, int col)
{
  return A.elements[row * A.stride + col];
}
 

__device__ void SetElement(Matrix A, int row, int col, float value)
{
  A.elements[row * A.stride + col] = value;
}
 

#define BLOCK_SIZE 1
 

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
			      + BLOCK_SIZE * col];
  return Asub;
}
 

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
 

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
 
  Matrix d_A;
  d_A.width = d_A.stride = A.width; d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc((void**)&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size,
       cudaMemcpyHostToDevice);
  Matrix d_B;
  d_B.width = d_B.stride = B.width; d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc((void**)&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size,
       cudaMemcpyHostToDevice);
  
  Matrix d_C;
  d_C.width = d_C.stride = C.width; d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  cudaMalloc((void**)&d_C.elements, size);
  
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  
  cudaMemcpy(C.elements, d_C.elements, size,
       cudaMemcpyDeviceToHost);
 
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}
 

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
  
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
  
  float Cvalue = 0;
  
  int row = threadIdx.y;
  int col = threadIdx.x;

  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    
    Matrix Asub = GetSubMatrix(A, blockRow, m);
   
    Matrix Bsub = GetSubMatrix(B, m, blockCol);
    
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
   
   
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);
   
    __syncthreads();
    
    for (int e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += As[row][e] * Bs[e][col];
    
    __syncthreads();
  }
 

  SetElement(Csub, row, col, Cvalue);
  
  __syncthreads();
  __syncthreads();			
}
 
static Matrix
cons_Matrix (int height_, int width_)
{
  Matrix A;
  A.height = height_;
  A.width = width_;
  A.stride = width_;
  A.elements = (float*) malloc(sizeof(*A.elements) * width_ * height_);
  for (int row = 0; row < height_; row++)
    for (int col = 0; col < width_; col++)
      A.elements[row * width_ + col] = rand(0.0,50.0);
  return A;
}
 
static void
print_Matrix (Matrix A, char *name)
{
 
  for (int row = 0; row < A.height; row++){
	  
    for (int col = 0; col < A.width; col++)
      printf ("%5.1f ",  A.elements[row * A.stride + col]);
  printf("\n");}
}

 
int main(int argc, char **argv)
{
	time_t czas;
	
	srand( (unsigned int)time(&czas));
		
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int p = atoi(argv[3]);
  
  Matrix A = cons_Matrix(m, n);
  Matrix B = cons_Matrix(n, p);
  Matrix C = cons_Matrix(m, p);
  
  MatMul(A, B, C);

  printf("\n");
  print_Matrix(A, "A");
  printf("\n");
  print_Matrix(B, "B");
  printf("\n");
  print_Matrix(C, "C");
  printf("\n");
  return 0;
}