#pragma once
#include "UtilGPU.cuh"
#include "KernelMemShared.cuh"
#include "GPUCarasteristic.h"
#include <conio.h>
#include <Windows.h>
#include <regex>
#include <math.h>

// Declaracion de constantes.
#define TAM_TESELA 16			// tesela mas optima.
#define NUM_DIMENSION 3			// Numero de dimesiones posibles en la GPU.
#define NUM_DIMENSION_TAB 3		// Numero de dimesiones del tablero de juego + la direcion de la jugada de la diagoonal.
#define NIVEL_DIFICULTAD 5		// Numeros de niveles de dificultad
#define POS_TAB_JUEGO_EMPTY 10	// Posicion del tablero vacia si niguna ficha.
#define NUM_FICHAS	9


// Declaracion de strut para alamacenar info gpu para su configuracion
typedef struct InfoGPU {
	double numThreadMaxPerSM;
	double numThreadMasPerBlock;
	double numRegPerBlock;
	size_t sharedMemPerBlock;
	double maxDimThreadBlock[NUM_DIMENSION];
	double maxDimGridSize[NUM_DIMENSION];
} info_gpu;

// Declaracion de funciones y metodos CPU.
void getCofigPlay(int devian, cudaDeviceProp *deviceProp, info_gpu *myConfGpu);
double setGpuForPlayAuto(cudaDeviceProp *devProp, info_gpu *myConfGpu, int deviceCurrent);
double setGpuForPlayManual(cudaDeviceProp *devProp, info_gpu *myConfGpu, int deviceCurrent);
int setDificultad();
long* generarTablero(double numThread, int dificultad);
void imprimirTablero(long *tablero, double numThread);
void imprimirColumnas(double numThread);
void playDamas(int typeKernel, double numThread, info_gpu *myConfGpu, int dificultad);
int *getRowAndColumn(string jug, double numThread);
bool launchKernel(int typeKernel, double numThread, long* tablero, int* jugada);