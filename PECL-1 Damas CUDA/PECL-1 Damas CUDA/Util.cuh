#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

using namespace std;

//	Definiciones de colores.
#define ANSI_COLOR_RED     "\x1b[31m" 
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"	//Restablece el color del pront.

// Definicion de macro para errores.
#define ERROR_MSS(A) cout << ANSI_COLOR_RED " - ERROR:" ANSI_COLOR_RESET << A << endl;


//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t setCudaMalloc(int *dev, unsigned int size);
cudaError_t getCudaMalloc(int *c, int *dev, unsigned int size);