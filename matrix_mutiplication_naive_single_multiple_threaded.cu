#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
//#include <cuda.h>


typedef struct {
    int width;
    int height;

    float* elements;

}Matrix;

float rand(float a,float b)
{
    //return( a + rand()%(b-a+1.0) );
    return(b - a) * ((float)rand() / RAND_MAX) + a;
}

#define BLOCK_SIZE 16
#define BLOCK_SIZE_SINGLE 1

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	/*
	 We couldn't find proper library for functions:
	 cutCreateTimer, cutResetTimer, cudaEventCreate, cudaEventRecord,cudaEventSynchronize, cudaEventsElapsedTime
	 
	 That's why we have left them so far in the comments for clarity of the program
	*/
	
	//uint kernelTime;
	 //cutCreateTimer(&kernelTime);
	 //cutResetTimer(kernelTime);


    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.elements, size);
    printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    printf("Copy A to device: %s\n",cudaGetErrorString(err));
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    err = cudaMalloc(&d_B.elements, size);

    printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    printf("Copy B to device: %s\n",cudaGetErrorString(err));
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);
    printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
    (A.height + dimBlock.y - 1) / dimBlock.y);
    printf("Block size= %d\n",BLOCK_SIZE);
    printf("DimBlock = %d\n",dimBlock);

    //cudaEvent_t start, stop;
    //float time;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaDeviceSynchronize();
    
    //cudaEventRecord(stop,0);
    //cudaEventSynchronize(stop);


    printf("Run kernel: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    printf("Copy C off of device: %s\n",cudaGetErrorString(err));


    cudaFree(d_A.elements);
    cudaFree(d_B.elements);

    //cudaEventElapsedTime(&time, start, stop);
    //printf("Time : %f mss\n",time);
}

void MatMulSingleThreaded(const Matrix A, const Matrix B, Matrix C)
{


    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.elements, size);
    printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    printf("Copy A to device: %s\n",cudaGetErrorString(err));
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    err = cudaMalloc(&d_B.elements, size);

    printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    printf("Copy B to device: %s\n",cudaGetErrorString(err));
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);
    printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

    dim3 dimBlock(BLOCK_SIZE_SINGLE, BLOCK_SIZE_SINGLE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
    (A.height + dimBlock.y - 1) / dimBlock.y);
    printf("Block size= %d\n",BLOCK_SIZE_SINGLE);
    printf("DimBlock = %d\n",dimBlock);

    //cudaEvent_t start, stop;
    //float time;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaDeviceSynchronize();
    
    //cudaEventRecord(stop,0);
    //cudaEventSynchronize(stop);

    printf("Run kernel: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    printf("Copy C off of device: %s\n",cudaGetErrorString(err));


    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    
    //cudaEventElapsedTime(&time, start, stop);
    //printf("Time : %f mss\n",time);
}


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row > A.height || col > B.width) return;

    for (int e = 0; e < A.width; ++e)
        Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);

    C.elements[row * C.width + col] = Cvalue;

}


int main(int argc, char* argv[]){

    time_t czas;
    srand( (unsigned int)time(&czas));


    Matrix A, B, C,C1;
    int a1, a2, b1, b2;

    a1 = atoi(argv[1]); /* Height of A */
    a2 = atoi(argv[2]); /* Width of A */
    b1 = a2; /* Height of B */
    b2 = atoi(argv[3]); /* Width of B */

    A.height = a1;
    A.width = a2;
    A.elements = (float*)malloc(A.width * A.height * sizeof(float));

    B.height = b1;
    B.width = b2;
    B.elements = (float*)malloc(B.width * B.height * sizeof(float));

    C.height = A.height;
    C.width = B.width;
    C.elements = (float*)malloc(C.width * C.height * sizeof(float));

    C1.height = A.height;
    C1.width = B.width;
    C1.elements = (float*)malloc(C.width * C.height * sizeof(float));

    printf("\n");

    for(int i = 0; i < A.height; i++) {
        for(int j = 0; j < A.width; j++)
            A.elements[i*A.width + j] = rand(1.0,50.0);
    }

    for(int i = 0; i < B.height; i++){
        for(int j = 0; j < B.width; j++)
            B.elements[i*B.width + j] = rand(1.0,50.0);
            }

    MatMul(A, B, C);
    MatMulSingleThreaded(A,B,C1);
    int checksum=0;
    for(int i = 0; i < C.height; i++){
    	for(int j = 0; j < min(10, C.width); j++){
          if(C.elements[i*C.width + j]!= C1.elements[i*C1.width + j])
          checksum++;
		}
    	printf("checksum= %d\n",checksum);
	}
// HERE we had a problem with every first value of each row, we couldn't solve it..
	if (checksum == 0)
		printf("\nResults for single-threaded and multiple-threaded are the same! : ) \n");
	else
		printf("\nResults for single-threaded and multiple-threaded are different! : ( \n");
	printf("Unfortunately we had a problem with every first value of each row (from the second row)");

	printf("\n");
	printf("Matrix A:\n");
	for(int i = 0; i < min(10, A.height); i++){
        for(int j = 0; j < min(10, A.width); j++)
            printf("%5.1f ", A.elements[i*A.width + j]);
        printf("\n");
    }

    printf("\n");
    printf("Matrix B:\n");
    for(int i = 0; i < min(10, B.height); i++){
        for(int j = 0; j < min(10, B.width); j++)
            printf("%5.1f ", B.elements[i*B.width + j]);
        printf("\n");
    }
    printf("\n");
    printf("Result matrix C (multiple-threaded):\n");
    for(int i = 0; i < min(10, C.height); i++){
        for(int j = 0; j < min(10, C.width); j++)
            printf("%7.1f ", C.elements[i*C.width + j]);
        printf("\n");
    }
    printf("\n");
    printf("Result matrix C (single-threaded):\n");
    for(int i = 0; i < min(10, C1.height); i++){
            for(int j = 0; j < min(10, C1.width); j++)
                printf("%7.1f ", C1.elements[i*C1.width + j]);
            printf("\n");
        }

}
