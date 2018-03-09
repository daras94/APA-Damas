#include "UtilGPU.cuh"

/*
	Transfiere el contenoido de la mmemoria de la GPU a la memoria, Los Argumentos 
	son los siguientes y retorna el codigo de error en caso de error
 		- c    =  puntero a entero en el que devolvemos.
		- dev  =  puntero a enteros usado para tranferir a la GPU.
		- size =  devuelve el tamaño del valor a transferir.
*/
cudaError_t setCudaMemcpy(int *c,  int *dev, unsigned int size) {
	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus = cudaMemcpy(c, dev, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		ERROR_MSS("Fallo el la operacion cudaMemcpy !!");
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// Saltamos a erro y liberamos la memoria.
	Error:
		cudaFree(dev);
		return cudaStatus;
}

/*
	Asignamos memoria a la variable pasada por parametro para pasarlas a la GPU, , Los Argumentos 
	son los siguientes y retorna el codigo de error en caso de error
		- c    =  puntero a entero en el que devolvemos.
		- dev  =  puntero a enteros usado para tranferir a la GPU.
		- size =  devuelve el tamaño del valor a transferir.
*/
cudaError_t setCudaMalloc(int *dev, unsigned int size) {
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaError_t cudaStatus = cudaMalloc((void**)&dev, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		ERROR_MSS("Fallo el la operacion cudaMalloc !!");
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Saltamos a erro y liberamos la memoria.
Error:
	cudaFree(dev);
	return cudaStatus;
}