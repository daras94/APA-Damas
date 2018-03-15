#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "UtilGPU.cuh"
#include "tablero.cuh"
#include "cpu_bitmap.h"


#define TAM_TESELA 16  // tesela mas optima.

// Funciones y metodos del host.
void iniciarInterfaz(double numThreads, long *tablero);

// Funcion y metodos GPU
__global__ void kernelGUI(unsigned char *imagen, double numThread);