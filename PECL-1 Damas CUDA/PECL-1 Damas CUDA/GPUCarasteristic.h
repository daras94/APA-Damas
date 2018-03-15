#pragma once
#include "UtilGPU.cuh"
#include <stdio.h>

// Declaracion de variables Globales.

// definicion de funciones y metodos.
void getCarrasteristicForGPU(cudaDeviceProp *devProp);
void echoCarGPUs(int devianCurrent, cudaDeviceProp *devProp);
void selectGpuCurrent(cudaDeviceProp *devProp, int *devianCurrent);
void fotterCarGPU(cudaDeviceProp *devProp, int deviceCurren);
bool getDevCuda(int *deviceCont);