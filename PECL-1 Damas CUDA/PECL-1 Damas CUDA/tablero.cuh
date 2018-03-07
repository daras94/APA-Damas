#pragma once
#include "UtilGPU.cuh"
#include "GPUCarasteristic.h"

// Declaracion de constantes.
#define TAM_TESELA 16		// tesela mas optima.
#define NUM_DIMENSION 3     // Numero de dimesiones posibles en la GPU.

// Declaracion de strut para alamacenar info gpu para su configuracion
typedef struct InfoGPU {
	double numThreadMaxPerSM;
	double numThreadMasPerBlock;
	double numRegPerBlock;
	size_t sharedMemPerBlock;
	double maxDimThreadBlock[NUM_DIMENSION];
	double maxDimGridSize[NUM_DIMENSION];
} info_gpu;

// Declaracion de funciones y metodos.
void getCofigPlay(int devian, cudaDeviceProp *deviceProp, info_gpu *myConfGpu);
void setGpuForPlay(cudaDeviceProp *devProp, info_gpu *myConfGpu);