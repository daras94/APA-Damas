#include "tablero.cuh"

/*
	Recupera las carrasteristicas nesesarias para realizar la configuracion del tablero.

	- devian	 = recibe un entero con la el id de la GPU que se va a usar para realizar la configuracion.
	- deviceProp = recibe un struct para almacenar las carrasteristicas de la GPU.
	- myConfGpu  = truckt pasado por referencia para almacenar las informacion de la gpu ge nos interesa.
*/
void getCofigPlay(int devian, cudaDeviceProp *deviceProp, info_gpu *myConfGpu) {
	cudaSetDevice(devian);														// Establecemos con que GPU queremos realizar la configuracion en funcion de disponer.
	cudaGetDeviceProperties(deviceProp, devian);
	myConfGpu->numThreadMaxPerSM = deviceProp->maxThreadsPerMultiProcessor;
	myConfGpu->numThreadMasPerBlock = deviceProp->maxThreadsPerBlock;
	myConfGpu->numRegPerBlock = deviceProp->regsPerBlock;
	myConfGpu->sharedMemPerBlock = deviceProp->sharedMemPerBlock;
	for (int i = 0; i < NUM_DIMENSION; i++) {
		myConfGpu->maxDimGridSize[i] = deviceProp->maxGridSize[i];
		myConfGpu->maxDimThreadBlock[i] = deviceProp->maxThreadsDim[i];
	}
}

double setGpuForPlayAuto(cudaDeviceProp *devProp,  info_gpu *myConfGpu, int deviceCurrent) {
	double *dimTamblero;
	int  gpuOpc;
	do {
		system("cls");
		cout << "/***************************************************************************************/" << endl;
		cout << "/*  +--> " << ANSI_COLOR_CYAN "Menu de configuracion de partida:" ANSI_COLOR_RESET << setw(47) << "*/" << endl;
		cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
			string modelGPU = devProp->name;
			cout << "/*  " ANSI_COLOR_MAGENTA "GPU " << deviceCurrent << ANSI_COLOR_RESET ") - " << modelGPU << setw(76 - modelGPU.length()) << "*/" << endl;
			cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
			double numThread = myConfGpu->maxDimThreadBlock[1];
			dimTamblero = new double[NUM_DIMENSION] { 2, 4, 8 };
			for (int i = 0; i < 3; i++) {
				dimTamblero[i] = numThread / dimTamblero[i];
				cout << "/*\t" ANSI_COLOR_MAGENTA << (i + 1) << ANSI_COLOR_RESET ") - Disponible tablero de juego de " ANSI_COLOR_GREEN << (dimTamblero[i] / TAM_TESELA) << "x" << (dimTamblero[i] / TAM_TESELA) << ANSI_COLOR_RESET << " " << dimTamblero[i] << " Threads" << setw(40) << "*/" << endl;
			}
			cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
			cout << "/*  - " << ANSI_COLOR_RED "AVISO: " ANSI_COLOR_RESET "Selecione el id de una GPU para jugar." << setw(40) << "*/" << endl;
		cout << "/***************************************************************************************/" << endl;
		cout << " - Selecione una opcion para juegar (" ANSI_COLOR_GREEN "Pulse 0 para salir de la configuracion" ANSI_COLOR_RESET "): ";
		cin >> gpuOpc;		// Entrada de texto por teclado.
		if (gpuOpc != 0 && (gpuOpc < 0 || gpuOpc > 3)) {
			ERROR_MSS("Error opcion de juego introducida no es valida.");
		}
	} while (gpuOpc != 0 && (gpuOpc < 0 || gpuOpc > 3));
	return dimTamblero[gpuOpc - 1];
}

int setDificultad() {
	int  dificultad;
	do {
		system("cls");
		cout << "/***************************************************************************************/" << endl;
		cout << "/*  +--> " << ANSI_COLOR_CYAN "Dificulta de la partida:" ANSI_COLOR_RESET << setw(47) << "*/" << endl;
		cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		string niveles[NIVEL_DIFICULTAD] = {"Muy Facil", "Facil", "Normal", "Avanzado", "Experto"};
		for (int  i = 0; i < NIVEL_DIFICULTAD; i++) {
			cout << "/*\t" ANSI_COLOR_MAGENTA << (i + 1) << ANSI_COLOR_RESET ") - "  << ANSI_COLOR_RESET << niveles[i] << setw(40) << "*/" << endl;
		}
		cout << "/*" << setw(87) << "*/" << endl;
		cout << "/***************************************************************************************/" << endl;
		cout << " - Selecione dificultad del juego (" ANSI_COLOR_GREEN "0 para salir de la partida" ANSI_COLOR_RESET "): ";
		cin >> dificultad;		// Entrada de texto por teclado.
		if (dificultad != 0 && (dificultad < 0 && dificultad > NIVEL_DIFICULTAD)) {
			ERROR_MSS("Error opcion de juego introducida no es valida.");
		}
	} while (dificultad != 0 && (dificultad < 0 && dificultad > NIVEL_DIFICULTAD));
	return dificultad;
}

//Generamos el tablero con números aleatorios en función de la dificultad
void generarTablero(int *tablero, double numThread, int dificultad) {
	srand(time(NULL));
	for (int i = 0; i < numThread; i++) {
		tablero[i] = rand() % dificultad + 1;
	}
}


//Rellenar tablero cuando hemos explotado bloques
void rellenarTablero(int *tablero, double numThread, int dificultad) {
	srand(time(NULL));
	for (int i = 0; i < (numThread / TAM_TESELA); i++) {
		if (tablero[i] == 0) {
			tablero[i] = rand() % 4 + dificultad + 1;
		}
	}
}

//Función que imprime el número de columnas que va a tener el tablero para que sea más facil elegir piezas
void imprimirColumnas(double numThread) {
	for (int i = 0; i < (numThread / TAM_TESELA); i++) {
		cout << ((i == 0)? setw(8) : (i < 9)? setw(5)  : setw(4)) << i + 1;
	}
	cout << "" << endl;
	for (int i = 0; i < (numThread / TAM_TESELA); i++) {
		cout << ((i == 0) ? setw(11) : setw(6)) << "|" << i + 1;
	}
	cout << "\n";
}
//Imprimimos el tablero
void imprimirTablero(int *tablero, double numThread) {
	imprimirColumnas(numThread);
	int color[NUM_FICHAS] = {7, 4, 11, 2, 5, 6, 3 };
	for (int i = 0; i < numThread / TAM_TESELA; i++) {
		cout << ((i < 9) ? setw(5) : setw(4)) << "-" << i + 1;
		for (int k = 0; k < numThread/TAM_TESELA; k++) {
			//Damos color en función del número imprimir
			int bloque = tablero[i * ((int)numThread / TAM_TESELA) + k];
			if (bloque < NUM_FICHAS) {
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color[bloque]);
			} else {
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7);
			}
			cout << "| " << bloque << " |";
		}
		cout << "\n";
	}
}

__device__ void compruebaPiezas(int * tablero, int columna, int fila, int filas, int columnas, int anterior)
{
	//compruebaPiezas(tablero, columnaHilo, filaHilo, filas, columnas, anterior);
	//Aquí vamos a indicarle hacia donde tiene que buscar en función de la posición del tablero en la cual nos encontremos
	//Primero comprobamos que no sea un tipo de bomba, en nuestro caso, las bombas van a ser 7 8 y 9 
	//7 elimina la fila, 8 la columna y 9 es el TNT
	if (tablero[(fila * columnas) + columna] != 7 && tablero[(fila * columnas) + columna] != 8 && tablero[(fila * columnas) + columna] != 9) {
		//EMPEZAMOS CON LAS PIEZAS DE LAS CUATRO ESQUINAS
		//SI ESTAMOS EN LA SUPERIOR IZQUIERDA SOLO PODEMOS COMPROBAR HACIA ABAJO, DERECHA Y DIAGONAL DERECHA
		if (fila == 0 && columna == 0) {
			compruebaDerecha(tablero, columna, fila, filas, columnas, anterior);
			compruebaAbajo(tablero, columna, fila, filas, columnas, anterior);
		}
		//SI ESTAMOS EN LA SUPERIOR DERECHA SOLO PODEMOS COMPROBAR HACIA ABAJO, IZQUIERDA Y DIAGONAL IZQUIERDA
		if (fila == 0 && columna == (columnas - 1)) {
			compruebaIzquierda(tablero, columna, fila, filas, columnas, anterior);
			compruebaAbajo(tablero, columna, fila, filas, columnas, anterior);
		}
		//SI ESTAMOS EN LA INFERIOR IZQUIERDA SOLO PODEMOS COMPROBAR HACIA ARRIBA, DERECHA Y DIAGONAL DERECHA
		if (fila == (filas - 1) && columna == 0) {
			compruebaDerecha(tablero, columna, fila, filas, columnas, anterior);
			compruebaArriba(tablero, columna, fila, filas, columnas, anterior);
		}
		//SI ESTAMOS EN LA INFERIOR DERECHA SOLO PODEMOS COMPROBAR HACIA ARRIBA, IZQUIERDA Y DIAGONAL IZQUIERDA
		if (fila == (filas - 1) && columna == (columnas - 1)) {
			compruebaArriba(tablero, columna, fila, filas, columnas, anterior);
			compruebaIzquierda(tablero, columna, fila, filas, columnas, anterior);
		}
		//UNA VEZ COMPROBADAS LAS ESQUINAS, AUN TENEMOS OTROS CUATRO CASOS ESPECIALES, ESTAR EN LA FILA DE ARRIBA, FILA DE ABAJO, COLUMNA DE LA IZQ Y COLUMNA DE LA DERECHA
		//SI ESTAMOS EN LA FILA DE ARRIBA SOLO PODEMOS IR HACIA IZQ, DERECHA, DIAGONAL DERECHA, DIAGONAL IZQUIERDA Y HACIA ABAJO
		if (fila == 0) {
			compruebaIzquierda(tablero, columna, fila, filas, columnas, anterior);
			compruebaDerecha(tablero, columna, fila, filas, columnas, anterior);
			compruebaAbajo(tablero, columna, fila, filas, columnas, anterior);
		}
		//SI ESTAMOS EN LA FILA DE ABAJO SOLO PODEMOS IR HACIA IZQ, DERECHA, DIAGONAL DERECHA, DIAGONAL IZQUIERDA Y ARRIBA
		if (fila == (filas - 1)) {
			compruebaIzquierda(tablero, columna, fila, filas, columnas, anterior);
			compruebaDerecha(tablero, columna, fila, filas, columnas, anterior);
			compruebaArriba(tablero, columna, fila, filas, columnas, anterior);
		}
		//SI ESTAMOS EN LA COLUMNA IZQUIERDA SOLO SE COMPRUEBA HACIA DERECHA, ARRIBA, ABAJO, DIAGONAL DERECHA Y DIAGONAL IZQ
		if (columna == 0) {
			compruebaDerecha(tablero, columna, fila, filas, columnas, anterior);
			compruebaArriba(tablero, columna, fila, filas, columnas, anterior);
			compruebaAbajo(tablero, columna, fila, filas, columnas, anterior);

		}
		//SI ESTAMOS EN LA COLUMNA DERECHA SOLO SE COMPRUEBA HACIA IZQUIERDA, ARRIBA, ABAJO, DIAGONAL DERECHA Y DIAGONAL IZQ
		if (columna == (columnas - 1)) {
			compruebaIzquierda(tablero, columna, fila, filas, columnas, anterior);
			compruebaArriba(tablero, columna, fila, filas, columnas, anterior);
			compruebaAbajo(tablero, columna, fila, filas, columnas, anterior);

		}
		//CUALQUIER OTRO CASO
		else {
			compruebaArriba(tablero, columna, fila, filas, columnas, anterior);
			compruebaAbajo(tablero, columna, fila, filas, columnas, anterior);
			compruebaDerecha(tablero, columna, fila, filas, columnas, anterior);
			compruebaIzquierda(tablero, columna, fila, filas, columnas, anterior);
		}
	}
	else { //BOMBAS
		   //7 elimina la fila, 8 la columna y 9 es el TNT
		if (tablero[(fila * columnas) + columna] == 7) {
			compruebaDerecha(tablero, columna, fila, filas, columnas, 7);
			compruebaIzquierda(tablero, columna, fila, filas, columnas, 7);
		}
		else if (tablero[(fila * columnas) + columna] == 8) {
			compruebaAbajo(tablero, columna, fila, filas, columnas, 8);
			compruebaArriba(tablero, columna, fila, filas, columnas, 8);
		}
		else if (tablero[(fila * columnas) + columna] == 9) {
			compruebaAbajo(tablero, columna, fila, filas, columnas, 9);
			compruebaArriba(tablero, columna, fila, filas, columnas, 9);
			compruebaDerecha(tablero, columna, fila, filas, columnas, 9);
			compruebaIzquierda(tablero, columna, fila, filas, columnas, 9);
			compruebaAbajoDerecha(tablero, columna, fila, filas, columnas, 9);
			compruebaAbajoIzquierda(tablero, columna, fila, filas, columnas, 9);
			compruebaArribaDerecha(tablero, columna, fila, filas, columnas, 9);
			compruebaArribaIzquierda(tablero, columna, fila, filas, columnas, 9);
		}
	}
}


__device__ void compruebaArriba(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	if (anterior == 8) {
		for (int i = 0; (fila - i) >= 0; i++) {
			tablero[((fila - i) * columnas) + columna] = 0;
		}
	}
	else if (anterior == 9) {
		tablero[(fila * columnas) + columna] = 0;
		if (fila != 0) {
			tablero[((fila - 1) * columnas) + columna] = 0;
		}
	}
	else {
		if (tablero[((fila - 1) * columnas) + columna] == anterior) {
			tablero[((fila - 1) * columnas) + columna] = 0;
			tablero[(fila * columnas) + columna] = 0;
			compruebaPiezas(tablero, columna, fila - 1, filas, columnas, anterior);
		}
	}
}

__device__ void compruebaAbajo(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	if (anterior == 8) {
		for (int i = 0; (fila + i) < filas; i++) {
			tablero[((fila + i) * columnas) + columna] = 0;
		}
	}
	else if (anterior == 9) {
		tablero[(fila * columnas) + columna] = 0;
		if (fila != (filas - 1)) {
			tablero[((fila + 1) * columnas) + columna] = 0;
		}
	}
	else {
		if (tablero[((fila + 1) * columnas) + columna] == anterior) {
			tablero[((fila + 1) * columnas) + columna] = 0;
			tablero[(fila * columnas) + columna] = 0;
			compruebaPiezas(tablero, columna, fila + 1, filas, columnas, anterior);
		}
	}
}

__device__ void compruebaDerecha(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	if (anterior == 7) {
		for (int i = 0; (fila + i) < columnas; i++) {
			tablero[(fila  * columnas) + i] = 0;
		}
	}
	else if (anterior == 9) {
		tablero[(fila * columnas) + columna] = 0;
		if (columna != (columnas - 1)) {
			tablero[(fila * columnas) + columna + 1] = 0;
		}
	}
	else {
		if (tablero[(fila * columnas) + (columna + 1)] == anterior) {
			tablero[(fila * columnas) + (columna + 1)] = 0;
			tablero[(fila * columnas) + columna] = 0;
			compruebaPiezas(tablero, columna + 1, fila, filas, columnas, anterior);
		}
	}
}

__device__ void compruebaIzquierda(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	if (anterior == 7) {
		for (int i = 0; (fila - i) >= 0; i++) {
			tablero[(fila  * columnas) - i] = 0;
		}
	}
	else if (anterior == 9) {
		tablero[(fila * columnas) + columna] = 0;
		if (columna != 0) {
			tablero[(fila * columnas) + columna - 1] = 0;
		}
	}
	else {
		if (tablero[(fila * columnas) + (columna - 1)] == anterior) {
			tablero[(fila * columnas) + (columna - 1)] = 0;
			tablero[(fila * columnas) + columna] = 0;
			compruebaPiezas(tablero, columna - 1, fila, filas, columnas, anterior);
		}
	}
}

__device__ void compruebaArribaDerecha(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	//en columna = columnas -1 y en fila = 0
	if (columna != columnas - 1 && fila != 0) {
		tablero[((fila - 1) * columnas) + columna + 1] = 0;
	}
}
__device__ void compruebaAbajoDerecha(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	//en columna = columnas - 1 y en fila = filas - 1
	if (columna != columnas - 1 && fila != filas - 1) {
		tablero[((fila + 1) * columnas) + columna + 1] = 0;
	}
}

__device__ void compruebaArribaIzquierda(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	//en columna = 0 y en fila = 0
	if (columna != 0 && fila != 0) {
		tablero[((fila - 1) * columnas) + columna - 1] = 0;
	}
}
__device__ void compruebaAbajoIzquierda(int *tablero, int columna, int fila, int filas, int columnas, int anterior) {
	//en columna = 0 y en fila = filas -1
	if (columna != 0 && fila != filas - 1) {
		tablero[((fila + 1) * columnas) + columna - 1] = 0;
	}
}


__global__ void ToyBlastManual(int *tablero, int filas, int columnas, int columna, int fila, int bomba) {
	//Recogemos la fila y la columna del hilo
	int columnaHilo = threadIdx.x;
	int filaHilo = threadIdx.y;

	//Si la fila y columna del hilo coincide con la que hemos pasado por teclado, llamamos a la funcion comprueba piezas para que vaya eliminando las que son iguales
	if (columnaHilo == columna && filaHilo == fila) {
		int anterior = tablero[(filaHilo * columnas) + columnaHilo];
		compruebaPiezas(tablero, columnaHilo, filaHilo, filas, columnas, anterior);
		int contador = 0;
		//Contamos ceros y generamos la bomba en función del número de bloques que explotamos
		for (int i = 0; i < filas * columnas; i++) {
			if (tablero[i] == 0) {
				contador++;
			}
		}

		if (contador >= 6 && anterior != 9 && anterior != 7 && anterior != 8) {
			tablero[(fila * columnas) + columna] = 9;
		}
		if (contador == 5) {
			tablero[(fila * columnas) + columna] = bomba; //Tengo que pasarle la bomba ya generada porque con curand me descuadraba todas las comprobaciones
		}
	}
	__syncthreads();
	//Sube los ceros que hemos colocado al comprobar la posicion pedida por teclado bajando hacia abajo los bloques
	for (int i = 0; i <= filas; i++) {
		if (columnaHilo > 0) {
			if (tablero[columnaHilo*columnas + filaHilo] == 0 && !tablero[(columnaHilo - 1)*columnas + filaHilo] == 0) {
				tablero[columnaHilo*columnas + filaHilo] = tablero[(columnaHilo - 1)*columnas + filaHilo];
				tablero[(columnaHilo - 1)*columnas + filaHilo] = 0;
			}
		}
		__syncthreads();
	}
}


// 
void playDamas(double numThread, int *tablero, info_gpu *myConfGpu, int dificultad) {
	int fila = 1, columna = 1;
	generarTablero(tablero, numThread, dificultad);
	while (fila != 0 || columna != 0) {
		imprimirTablero(tablero, numThread);
		cout << "Introduce la fila en la que esta la ficha que deseas eliminar (0 para salir): \n";
		cin >> fila;
		while (fila < 0 && fila > numThread / TAM_TESELA) {
			cout << "Numero de fila no valido, introduzca uno en rango 1 - " << numThread / TAM_TESELA << ":\n";
			cin >> fila;
		}
		cout << "Introduce la columna en la que esta la ficha que deseas eliminar (0 para salir): \n";
		cin >> columna;
		while (columna < 0 && columna > numThread / TAM_TESELA) {
			cout << "Numero de columna no valido, introduzca uno en rango 1 - " << numThread / TAM_TESELA << ":\n";
			cin >> columna;
		}
		int *tablero_gpu;
		//Reservamos memoria y copiamos el tablero en la GPU
		cudaMalloc((void**)&tablero_gpu, numThread * sizeof(double));
		cudaMemcpy(tablero_gpu, tablero, numThread * sizeof(double), cudaMemcpyHostToDevice);
		dim3 DimGrid(((myConfGpu -> numThreadMasPerBlock) / numThread), ((myConfGpu->numThreadMasPerBlock) / numThread));		
		dim3 DimBlock(numThread / TAM_TESELA, numThread / TAM_TESELA);
		ToyBlastManual <<<DimGrid, DimBlock, myConfGpu ->sharedMemPerBlock >>> (tablero_gpu, numThread / TAM_TESELA, numThread / TAM_TESELA, columna - 1, fila - 1, dificultad); //Aqui empieza la fiesta con CUDA. 
		cudaMemcpy(tablero, tablero_gpu, sizeof(double) * numThread, cudaMemcpyDeviceToHost);
		system("cls");
		rellenarTablero(tablero, numThread, dificultad);
		cudaFree(tablero_gpu);
	}
}

