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


// Finciones y Metodos del Host.
bool launchKernelMemShared(double numThread, long* tablero, int* jugada);

// Funcion y metonos GPU
__global__ void DamasBomPlay(long *tablero, int numthread, bool *error_play, int row, int col, int direcion);
__device__ bool isCamarada(int movV, int movH, long Tabs[TAM_TESELA][TAM_TESELA + 1]);