#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

float rand(float a,float b)
{
	//return( a + rand()%(b-a+1.0) );
	return(b - a) * ((float)rand() / RAND_MAX) + a;
}


int ci(int row, int column, int nColumns) {
  return row*nColumns+column;
}

int main(int argc, char* argv[])
{
	time_t czas;
	srand( (unsigned int)time(&czas));

  int rowD = atoi(argv[1]);;
  int colD = atoi(argv[2]);; 
  int rowE = colD; 
  int colE = atoi(argv[3]);; 
  int rowF = rowD;
  int colF = colE;

  // initialize data
  thrust::device_vector<float> D(rowD * colD);
  thrust::device_vector<float> E(rowE * colE);
  thrust::device_vector<float> F(rowF * colF);

  std::cout << "\n";

  for (size_t i = 0; i < rowD; i++){
    for (size_t j = 0; j < colD; j++){
      D[ci(i,j,colD)]=rand(1.0,50.0);
      std::cout <<std::setprecision(1) << D[ci(i,j,colD)] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  for (size_t i = 0; i < rowE; i++){
    for (size_t j = 0; j < colE; j++){
      E[ci(i,j,colE)]=rand(1.0,50.0);
      std::cout <<std::setprecision(1) << E[ci(i,j,colE)] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  for (size_t i = 0; i < rowF; i++){
    for (size_t j = 0; j < colF; j++)
      F[ci(i,j,colF)]=0;
  }

  cublasHandle_t handle;

 
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "CUBLAS initialization error\n";
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      colE, rowD, colD,
                                      &alpha, thrust::raw_pointer_cast(&E[0]), colE,
                                              thrust::raw_pointer_cast(&D[0]), colD,
                                      &beta,  thrust::raw_pointer_cast(&F[0]), colE);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "kernel execution error.\n";
  }


  for (size_t i = 0; i < rowF; i++){
    for (size_t j = 0; j < colF; j++){
      std::cout <<std::setprecision(1) << F[ci(i,j,colF)] << " ";
   }
    std::cout << "\n";
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "shutdown error (A)\n";
  }


  return 0;
}
