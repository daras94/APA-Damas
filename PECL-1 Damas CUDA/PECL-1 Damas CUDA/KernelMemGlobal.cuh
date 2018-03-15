#include "UtilGPU.cuh"
#ifdef __INTELLISENSE__ 
/*
	Inteligence y sus incompativilidades con cuda esto solucion
	el error de vs2015 de que no lo reconoce.
*/
#define __syncthreads(); 
#endif

// Definicion de constantes
#define TAM_BLOCK 1				// tesela mas optima.
#define TAM_TESELA 16			// tesela mas optima.
#define POS_TAB_JUEGO_EMPTY 10	// Posicion del tablero vacia si niguna ficha.

// Definicion de variables globales.
__device__ bool isBomtrasposeGlobalMem = false;

// Finciones y Metodos del Host.
bool launchKernelMemGlobal(double numThread, long* tablero, int* jugada);

// Funcion y metonos GPU
__global__ void DamasBomPlayGlobalMem(long *tablero, int width, int row, int col, int direcion);
__device__ bool isCamaradaGlobalMem(int pos, int movV, int movH, long *Tab, int width);
__device__ void purpleBomGlobalMem(long *Tab, int y, int x, int width);
__device__ void yellowBomGlobalMem(long *Tab, int x, int y, int width);