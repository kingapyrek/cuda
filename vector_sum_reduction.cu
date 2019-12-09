#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024
#define NUM_OF_ELEMS 1024

#define funcCheck(stmt) {                                            \
    cudaError_t err = stmt;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
        printf( "Failed to run stmt %d ", __LINE__);                 \
        printf( "Got CUDA error ...  %s ", cudaGetErrorString(err)); \
        return -1;                                                   \
    }                                                                \
}

__global__  void total(float * input, float * output, int len)
{

    __shared__ float partialSum[2*BLOCK_SIZE];
    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    if ((start + t) < len)
    {
        partialSum[t] = input[start + t];
    }
    else
    {
        partialSum[t] = 0.0;
    }
    if ((start + blockDim.x + t) < len)
    {
        partialSum[blockDim.x + t] = input[start + blockDim.x + t];
    }
    else
    {
        partialSum[blockDim.x + t] = 0.0;
    }

    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
      __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    __syncthreads();

    if (t == 0 && (globalThreadId*2) < len)
    {
        output[blockIdx.x] = partialSum[t];
    }
}

int main(int argc, char ** argv)
{

    float * hostInput;
    float * hostOutput;
    float * deviceInput;
    float * deviceOutput;

    int numInputElements = NUM_OF_ELEMS;
    int numOutputElements;
    hostInput = (float *) malloc(sizeof(float) * numInputElements);

    for (int i=0; i < NUM_OF_ELEMS; i++)
    {
        hostInput[i] = 1.0;
    }

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1))
    {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));


    funcCheck(cudaMalloc((void **)&deviceInput, numInputElements * sizeof(float)));
    funcCheck(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));

    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 DimGrid( numOutputElements, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);


    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 1; i < numOutputElements; i++)
    {
        hostOutput[0] += hostOutput[i];
    }

    printf("Reduced Sum = %f\n", hostOutput[0]);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    free(hostInput);
    free(hostOutput);

    return 0;
}
