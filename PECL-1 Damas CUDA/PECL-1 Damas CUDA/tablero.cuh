#pragma once
#include "UtilGPU.cuh"
#include <conio.h>
#include <Windows.h>
#include "GPUCarasteristic.h"

// Declaracion de constantes.
#define TAM_TESELA 16		// tesela mas optima.
#define NUM_DIMENSION 3     // Numero de dimesiones posibles en la GPU.
#define NIVEL_DIFICULTAD 5  // Numeros de niveles de dificultad
#define NUM_FICHAS	9

// Declaracion de variables globales.


// Declaracion de strut para alamacenar info gpu para su configuracion
typedef struct InfoGPU {
	double numThreadMaxPerSM;
	double numThreadMasPerBlock;
	double numRegPerBlock;
	size_t sharedMemPerBlock;
	double maxDimThreadBlock[NUM_DIMENSION];
	double maxDimGridSize[NUM_DIMENSION];
} info_gpu;

// Declaracion de funciones y metodos.
void getCofigPlay(int devian, cudaDeviceProp *deviceProp, info_gpu *myConfGpu);
double setGpuForPlayAuto(cudaDeviceProp *devProp, info_gpu *myConfGpu, int deviceCurrent);
int setDificultad();

// Funciones y metodos CPU
void generarTablero(int *tablero, double numThread, int dificultad);
void imprimirTablero(int *tablero, double numThread);
void imprimirColumnas(double numThread);
void playDamas(double numThread, int *tablero, info_gpu *myConfGpu, int dificultad);

// Funcion y metonos GPU
__global__ void ToyBlastManual(int *tablero, int filas, int columnas, int fila, int columna, int bomba);
__device__ void compruebaPiezas(int * tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaArriba(int *tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaAbajo(int *tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaDerecha(int *tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaIzquierda(int *tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaArribaDerecha(int *tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaAbajoDerecha(int *tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaArribaIzquierda(int *tablero, int columna, int fila, int filas, int columnas, int anterior);
__device__ void compruebaAbajoIzquierda(int *tablero, int columna, int fila, int filas, int columnas, int anterior);