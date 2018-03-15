#include "UtilGPU.cuh"
#ifdef __INTELLISENSE__ 
/*
	Inteligence y sus incompativilidades con cuda esto solucion
	el error de vs2015 de que no lo reconoce.
*/
#define __syncthreads(); 
#endif

// Definicion de constantes
#define TAM_TESELA 16			// tesela mas optima.
#define POS_TAB_JUEGO_EMPTY 10	// Posicion del tablero vacia si niguna ficha.

// Definicion de variables globales.
__device__ bool isBomtrasposeSharedMem = false;

// Finciones y Metodos del Host.
bool launchKernelMemShared(double numThread, long* tablero, int* jugada);

// Funcion y metonos GPU
__global__ void DamasBomPlayMemShared(long *tablero, int numthread, int row, int col, int direcion);
__device__ bool isCamaradaSharedMem(int pos, int movV, int movH, long Tabs[TAM_TESELA][TAM_TESELA + 1]);
__device__ void purpleBomSharedMem(long Tabs[TAM_TESELA][TAM_TESELA + 1], int y, int x);
__device__ void yellowBomSharedMem(long *Tab, long Tabs[TAM_TESELA][TAM_TESELA + 1], int x, int y, int width);