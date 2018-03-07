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
} *info_gpu;

// Declaracion de varibales globales.

// Declaracion de funciones y metodos.
info_gpu getCofigPlay(int devian, cudaDeviceProp *deviceProp);
void setGpuForPlay(cudaDeviceProp *devProp);