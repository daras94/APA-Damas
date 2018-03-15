#pragma once
#include <string.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>

using namespace std;

// Declaracion de funciones y metodos.
void guardarPartida(long *tablero, int nFilas, int nCols);
long *cargarPartida();
