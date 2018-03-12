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

/*
	Declaracion de macros para calcular la tesela para no liarla cuando la matrix
	esta x debajo del tamaño de la tesela.
*/
#define TAM_TESELA_C(numThread) (TAM_TESELA/(((numThread/TAM_TESELA)>= TAM_TESELA)? 1 : 2));

// Finciones y Metodos del Host.
void launchKernelMemShared(double numThread, long* tablero, int* jugada, bool error_play);

// Funcion y metonos GPU
__global__ void DamasBomPlay(long *tablero, int numthread, int row, int col, int direcion);