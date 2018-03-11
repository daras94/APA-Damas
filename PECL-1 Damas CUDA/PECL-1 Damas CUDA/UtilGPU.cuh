#pragma once
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>
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
#define COLOR_NEGRO 0
#define COLOR_AZUL 1
#define COLOR_VERDE 2
#define COLOR_AGUAMARINA 3
#define COLOR_ROJO 4
#define COLOR_PURPURA 5
#define COLOR_AMARILLO 6
#define COLOR_BLANCO 7
#define COLOR_GRIS 8
#define COLOR_AZUL_LIGHT 9
#define COLOR_VERDE_LIGHT 10
#define COLOR_AGUAMARINA_LIGHT 11
#define COLOR_RED_LIGHT 12
#define COLOR_PURPURA_LIGHT 13
#define COLOR_AMARILLO_LIGHT 14
#define COLOR_BLACO_LIGHT 15

// Definicion de macro para errores.
#define ERROR_MSS(A)	cout << ANSI_COLOR_RED " - ERROR:" ANSI_COLOR_RESET << A << endl;

// Definicion de Macros para Calulo de cores de letra con Color de fondo A es el Color de le fondo y B del Testo.
#define COLOR_TABLERO(B, F) B*16+F;

// Declaracion de funciones y metodos.
void setCudaMalloc(long*& dev, int size);
void setCudaMemcpyToHost(long*& c, long*& dev, int size);
void setCudaMemcpyToDevice(long*& c, long*& dev, int size);
