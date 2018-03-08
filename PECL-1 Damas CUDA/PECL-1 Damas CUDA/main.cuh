#include "UtilGPU.cuh"
#include <windows.h>
#include <tchar.h>
#include <conio.h>
#include <strsafe.h>
#include "UtilGPU.cuh"
#include "tablero.cuh"

// Declaracion de define.
#define DEFAULT_GPU 0

// Declaracion de varaibles globales.
int selectGPU = DEFAULT_GPU;
cudaDeviceProp devProp;				// struct de carrasteristicas de la GPU.
info_gpu infoMyGPU = { NULL };

// Declarcion de metodos de la clase de C para CUDA del main.
int main();

