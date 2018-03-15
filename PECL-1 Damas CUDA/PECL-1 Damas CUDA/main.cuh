#include "UtilGPU.cuh"
#include <windows.h>
#include <tchar.h>
#include <conio.h>
#include <strsafe.h>
#include "UtilGPU.cuh"
#include "tablero.cuh"
#include "tableroGUI.cuh"

// Declaracion de define.
#define DEFAULT_GPU 0

// Declaracion de variables globales.
int selectGPU = DEFAULT_GPU;
cudaDeviceProp devProp;				// struct de carasteristicas de la GPU.
info_gpu infoMyGPU = { NULL };

// Declaracion de metodos de la clase de C para CUDA del main.
int main();

