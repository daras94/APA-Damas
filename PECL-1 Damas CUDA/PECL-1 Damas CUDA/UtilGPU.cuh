#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

// Declaracion de feficiones genricas.


// Definiciones de colores.
#define ANSI_COLOR_RED     "\x1B[1;31m" 
#define ANSI_COLOR_GREEN   "\x1B[1;32m"
#define ANSI_COLOR_YELLOW  "\x1B[1;33m"
#define ANSI_COLOR_BLUE    "\x1B[1;34m"
#define ANSI_COLOR_MAGENTA "\x1B[1;35m"
#define ANSI_COLOR_CYAN    "\x1B[1;36m"
#define ANSI_COLOR_RESET   "\x1B[1;0m"	//Restablece el color del pront.

// Definiciones de colores.

// Definicion de macro para errores.
#define ERROR_MSS(A)	cout << ANSI_COLOR_RED " - ERROR:" ANSI_COLOR_RESET << A << endl;

// Declaracion de funciones y metodos.
cudaError_t setCudaMalloc(int *dev, unsigned int size);
cudaError_t setCudaMemcpy(int *c, int *dev, unsigned int size);
