#include "UtilGPU.cuh"

/*
	Transfiere el contenoido de la mmemoria de la GPU a la memoria, Los Argumentos 
	son los siguientes y retorna el codigo de error en caso de error
 		- c    =  puntero a entero en el que devolvemos.
		- dev  =  puntero a enteros usado para tranferir a la GPU.
		- size =  devuelve el tamaño del valor a transferir.
*/
void setCudaMemcpyToHost(long*& c, long*& dev, int size) {
	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus = cudaMemcpy(c, dev, size * sizeof(long), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		ERROR_MSS("Fallo el la operacion cudaMemcpy Device to Host!!");
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Saltamos a erroe y liberamos la memoria.
	return;
	Error:
		cudaFree(dev);
		cout << endl;
		system("pause");
}

void setCudaMemcpyToDevice(long*& c, long*& dev, int size) {
	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus = cudaMemcpy(c, dev, size * sizeof(long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		ERROR_MSS("Fallo el la operacion cudaMemcpy Host to Device!!");
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Saltamos a erroe y liberamos la memoria.
	return;
Error:
	cudaFree(dev);
	cout << endl;
	system("pause");
}

/*
	Asignamos memoria a la variable pasada por parametro para pasarlas a la GPU, , Los Argumentos 
	son los siguientes y retorna el codigo de error en caso de error
		- c    =  puntero a entero en el que devolvemos.
		- dev  =  puntero a enteros usado para tranferir a la GPU.
		- size =  devuelve el tamaño del valor a transferir.
*/
void setCudaMalloc(long*& dev, int size) {
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaError_t cudaStatus = cudaMalloc((void**)&dev, size * sizeof(long));
	if (cudaStatus != cudaSuccess) {
		ERROR_MSS("Fallo el la operacion cudaMalloc !!");
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Saltamos a error y liberamos la memoria.
	return;
Error:
	cudaFree(&dev);
	cout << endl;
	system("pause");
}