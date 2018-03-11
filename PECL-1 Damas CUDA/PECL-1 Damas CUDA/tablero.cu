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
	double *dimTamblero, numThread;
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
			numThread = myConfGpu->maxDimThreadBlock[1];
			dimTamblero = new double[NUM_DIMENSION] { 2, 4, 8 };
			for (int i = 0; i < 3; i++) {
				dimTamblero[i] = numThread / dimTamblero[i];
				cout << "/*\t" << right << ANSI_COLOR_MAGENTA << (i + 1) << ANSI_COLOR_RESET ") - Disponible tablero de juego de " << ANSI_COLOR_GREEN << (dimTamblero[i] / TAM_TESELA) << "x" << (dimTamblero[i] / TAM_TESELA) << ANSI_COLOR_RESET " " << dimTamblero[i] << " Threads" << setw(40) << "*/" << endl;
			}
			cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
			cout << "/*  - " << ANSI_COLOR_RED "AVISO: " ANSI_COLOR_RESET "Selecione un tablero para la fiesta de CUDA." << setw(35) << "*/" << endl;
		cout << "/***************************************************************************************/" << endl;
		cout << " - Selecione una opcion para juegar (" ANSI_COLOR_GREEN "0 para salir de la configuracion" ANSI_COLOR_RESET "): ";
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
		cout << "/*  +--> " << ANSI_COLOR_CYAN "Nivel de dificulta de partida:" ANSI_COLOR_RESET << setw(52) << "*/" << endl;
		cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		string niveles[NIVEL_DIFICULTAD] = {"Muy Facil", "Facil", "Normal", "Avanzado", "Experto"};
		for (int  i = 0; i < NIVEL_DIFICULTAD; i++) {
			cout << "/*\t" ANSI_COLOR_MAGENTA << (i + 1) << ANSI_COLOR_RESET ") - "  << ANSI_COLOR_RESET << niveles[i] << setw(60 - niveles[i].length()) << "*/" << endl;
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
long *generarTablero(double numThread, int dificultad) {
	long row = 0, col = 0, *tablero = new long[(int)numThread];
	int numRowFicha = log2(numThread / TAM_TESELA);			// El numero de fichas para cada jugador en funcion de las dimensiones del tablero.
	srand(time(NULL));
	for (int i = 0; i < numThread; i++) { 
		row = i / ((int)numThread / TAM_TESELA);			// Calculamos la columna 
		col = ((row % 2) == 0)? 1 : 0;						// Calculamos el desplazamiento de la fichas en la colocacion.
		int bonba = rand() % dificultad;				    // Gennera Bombas en funcion de las dificultad selecionada.
		tablero[i] = (((col + i) % 2) == 0)? (row < numRowFicha)? 11 + bonba : POS_TAB_JUEGO_EMPTY : (row >= (numRowFicha * (numRowFicha - 1)))? 22 + bonba : POS_TAB_JUEGO_EMPTY;
	}
	return tablero;
}



//Función que imprime el número de columnas que va a tener el tablero para que sea más facil elegir piezas
void imprimirColumnas(double numThread) {
	for (int i = 0; i < (numThread / TAM_TESELA); i++) {
		cout << ((i == 0) ? setw(12) : (i < 9) ? setw(3) : setw(3.5)) << i + 1;
	}
	cout << "" << endl;
	for (int i = 0; i < (numThread / TAM_TESELA); i++) {
		cout << ((i == 0)? setw(12) : setw(3)) << "|";
	}
	cout << "" << endl;
}
//Imprimimos el tablero
void imprimirTablero(long *tablero, double numThread) {
	imprimirColumnas(numThread);
	for (int i = 0; i < numThread / TAM_TESELA; i++) {
		cout << setw(4) << i+1 << setw(3) << "-" << setw(3) << "";
		for (int k = 0; k < numThread/TAM_TESELA; k++) {								// Damos color en función del número imprimir
			int background = ((i + k) % 2 == 0) ? COLOR_BLANCO : COLOR_NEGRO;			// Color que contrulle el tablero.
			long bloque = tablero[i * ((int)numThread / TAM_TESELA) + k];
			//if (bloque < NUM_FICHAS) {												// Calculamos el color de la casilla.
				int color = COLOR_TABLERO(background, (new int[NUM_FICHAS] {background, COLOR_ROJO, COLOR_AZUL, COLOR_VERDE, COLOR_PURPURA, COLOR_AMARILLO, COLOR_AGUAMARINA, COLOR_PURPURA_LIGHT})[bloque % 10]); 
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
			//} 
				cout << " " << (((bloque - (bloque % 10)) > POS_TAB_JUEGO_EMPTY)? "#" : "O") << " ";
		}
		cout << ANSI_COLOR_RESET "" << endl;
	}
}

//compruebaPiezas(tablero, columnaHilo, filaHilo, filas, columnas, anterior);
//Aquí vamos a indicarle hacia donde tiene que buscar en función de la posición del tablero en la cual nos encontremos
//Primero comprobamos que no sea un tipo de bomba, en nuestro caso, las bombas van a ser 7 8 y 9 
//7 elimina la fila, 8 la columna y 9 es el TNT
__device__ void compruebaPiezas(long *tablero, int columna, int fila, int direcion) {
	int ficha = tablero[(fila * TAM_TESELA) + columna];
	switch (ficha % 10) {
		case 1:
		case 2:
			switch (direcion) {
				case 10:					// Movimiento superior-izquierda.
					//compruebaArribaIzquierda(tablero, columna, fila, ficha);
					break;
				case 11:					// Movimiento superior-derecha.
					compruebaArribaDerecha(tablero, columna, fila, ficha);
					break;					
				case 20:					// Movimiento inferior-izquierda.
					//compruebaAbajoIzquierda(tablero, columna, fila, ficha);
					break;					
				case 21:					// Movimiento inferior-derecha.
					//compruebaAbajoDerecha(tablero, columna, fila, ficha);
					break;
			}
			break;
		case 3:
		case 4:
		case 5:
		case 6:
		case 7:
			//compruebaAbajoDerecha(tablero, columna, fila, ficha);
			//compruebaAbajoIzquierda(tablero, columna, fila, ficha);
			compruebaArribaDerecha(tablero, columna, fila, ficha);
			//compruebaArribaIzquierda(tablero, columna, fila, ficha);
			break;
	}
}

/*
	Comprueba diagonal superior derecha.
*/
__device__ void compruebaArribaDerecha(long *tablero, int columna, int fila, int ficha) {
	//en columna = columnas -1 y en fila = 0
	if (columna != TAM_TESELA - 1 /*&& fila != 0*/) {
		tablero[fila - 1][&columna + 1] == ficha;
		//tablero[((fila - 1) * TAM_TESELA) + columna + 1] = ficha;
	}
}

/*
	Comprueba diagonal inferior derecha.
*/
__device__ void compruebaAbajoDerecha(long *tablero, int columna, int fila, int ficha) {
	//en columna = columnas - 1 y en fila = filas - 1
	if (columna != TAM_TESELA - 1 && fila != TAM_TESELA - 1) {
		tablero[((fila + 1) * TAM_TESELA) + columna + 1] = ficha;
	}
}

/*
	Comprueba diagonal inferior izquierda.
*/
__device__ void compruebaArribaIzquierda(long *tablero, int columna, int fila, int ficha) {
	//en columna = 0 y en fila = 0
	//if (columna != 0 /*&& fila != 0*/) {
		tablero[((fila - 1) * TAM_TESELA) + columna - 1] = ficha;
	//}
}

/*
	Comprueba diagonal supeiror izquierda.
*/
__device__ void compruebaAbajoIzquierda(long *tablero, int columna, int fila, int ficha) {
	//en columna = 0 y en fila = filas -1
	if (columna != 0 && fila != TAM_TESELA - 1) {
		tablero[((fila + 1) * TAM_TESELA) + columna - 1] = ficha;
	}
}


__global__ void DamasBomPlay(long *Tab, int numThread, int row, int col, int direcion) {
	__shared__ long Tabs[TAM_TESELA][TAM_TESELA];	// Memoria compratida para las seselas segmentado la matriz y usando la memoria compratida.
	int XxY = numThread / TAM_TESELA;				// Tamaño para las filas y las columnas.
	int tx = threadIdx.x, ty = threadIdx.y;			// Identicadores de filas y columnas de acuerdo con los hilos.
	int bx = blockIdx.x,  by = blockIdx.y;			// Identificadores de bloques  en el eje x e y.
	int Row = by * TAM_TESELA + ty;					// Calculamos la fila de la matriz teselada.
	int Col = bx * TAM_TESELA + tx;					// Calculamos la columna de la matriz teselada.
	
	/* 
		Si la fila y columna del hilo coincide con la pasada x a
	*/
	Tabs[ty][tx] = Tab[(Row * numThread) + Col];
	__syncthreads();
	if ((ty == col) && (tx == row) ) {
		printf("%d - %d | %d - %d \n", Col, Row, col, row);
		compruebaPiezas(*Tabs, Col, Row, direcion);
		Tabs[tx][ty] = POS_TAB_JUEGO_EMPTY;		// Marcmos como vacia la casilla horiginal en la que se encontraba la ficha.
	}
	__syncthreads();
	Tab[(Row * numThread)+ Col] = Tabs[ty][tx];
			
	/*if (Col == col && Row == row) {
		int anterior = tablero[(ty * columnas) + tx];
		compruebaPiezas(tablero, tx, ty, , columnas, anterior);
		int contador = 0;
		//Contamos ceros y generamos la bomba en función del número de bloques que explotamos
		for (int i = 0; i < filas * columnas; i++) {
			if (tablero[i] == 0) {
				contador++;
			}
		}

		if (contador >= 6 && anterior != 9 && anterior != 7 && anterior != 8) {
			tablero[(row * columnas) + col] = 9;
		}
		if (contador == 5) {
			tablero[(row * columnas) + col] = bomba; //Tengo que pasarle la bomba ya generada porque con curand me descuadraba todas las comprobaciones
		}
	}
	__syncthreads();
	//Sube los ceros que hemos colocado al comprobar la posicion pedida por teclado bajando hacia abajo los bloques
	for (int i = 0; i <= filas; i++) {
		if (tx > 0) {
			if (tablero[tx*columnas + ty] == 0 && !tablero[(tx - 1)*columnas + ty] == 0) {
				tablero[tx*columnas + ty] = tablero[(tx - 1)*columnas + ty];
				tablero[(tx - 1)*columnas + ty] = 0;
			}
		}
	}
	__syncthreads();*/
}


// 
void playDamas(double numThread, info_gpu *myConfGpu, int dificultad) {
	long *tablero = generarTablero(numThread, dificultad);
	long cont = 0;
	string input = { NULL };
	do {
		system("cls");
		cout << "/***************************************************************************************/" << endl;
		cout << "/*  +--> " << ANSI_COLOR_CYAN "Tablero de juego, Turno de juego de ficha: " ANSI_COLOR_RESET << (new string[2]{"#","O"})[(cont % 2 == 0)? 0 : 1] << setw(36) << "*/" << endl;
		cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
		cout << "  " << setw(87) << "  " << endl;
		imprimirTablero(tablero, numThread);
		cout << "  " << setw(87) << "  " << endl;
		cout << "/*  -----------------------------------------------------------------------------------  */" << endl;
		cout << "/*  - " << ANSI_COLOR_RED "AVISO: " ANSI_COLOR_RESET "Jugada con el formato X:Y:D (X = column, Y = row, " << setw(28) << " */" << endl;
		cout << "/*    " << setw(72) << " D = (10 = sup-izq, 20 = inf-izq, 11 = sup-dech, 21 = inf-dech)). "			   << setw(13) << " */" << endl;
		cout << "/*****************************************************************************************/" << endl;
		teclado:
		cout << " - Realice su jugada (" ANSI_COLOR_GREEN "0 para salir de la partida s para guardar la partida." ANSI_COLOR_RESET "): ";
		cin >> input;															// Entrada de texto por teclado.
		smatch match;
		regex  reg_expre{R"(\d{1,2}:\d{1,2}:(1|2){1}(0|1){1})"};				// Epresion regular para las filas y columnas.
		bool found = regex_match(input, match, reg_expre);						// Coparacion que busca un expresion de tipo fila:columna
		if (found) {
			int *jugada = getRowAndColumn(input, numThread);
			if (sizeof(jugada) < NUM_DIMENSION_TAB) {
				ERROR_MSS("Error en la columna o fila introducida.");
				goto teclado;
			} else {
				long *tablero_cuda;
				setCudaMalloc(tablero_cuda, numThread);							// Reservamos espacio de memoria para el tablero en la GPU.
				setCudaMemcpyToDevice(tablero_cuda, tablero, numThread);		// Tranferimos el tablero a la GPU.
				dim3 DimGrid(numThread / TAM_TESELA, numThread / TAM_TESELA);
				dim3 DimBlock(TAM_TESELA, TAM_TESELA);
				//size_t sharedMemByte = myConfGpu->sharedMemPerBlock;
				DamasBomPlay << <DimGrid, DimBlock>> > (tablero_cuda, ((int)numThread), jugada[1] - 1, jugada[0] - 1, jugada[2]); //Aqui empieza la fiesta con CUDA. 
				setCudaMemcpyToHost(tablero, tablero_cuda,  numThread);			// Trasferimos el tablero del GPU al HOST.
				cudaFree(tablero_cuda);
				cont++;
				char car = getchar();
				system("pause");
			}
		} else {
			switch ((char)&input) {
				case 's':									// Para la persistencia desde la partida.
					
					break;
				default:									// Carraterees no validos
					if (input != "0") {
						ERROR_MSS("Error carrater o movimiento introducido no valido no valida.");
						goto teclado;
					}
					break;
			}
		}
		
	} while (input != "0");
}

int *getRowAndColumn(string jug, double numThread) {
	string delimiter = ":", aux = jug + ":";
	int pos = 0, cont = 0, *rowCol = new int[NUM_DIMENSION_TAB];
	bool isNotErrorColRow = true;
	while ((pos = aux.find(delimiter)) != string::npos && isNotErrorColRow) {
		int token = stoi(aux.substr(0, pos));
		if (isNotErrorColRow = (token > 0 && token <= (numThread / TAM_TESELA))) {
			rowCol[cont] = token;
		}
		cout << token << endl;
		aux.erase(0, pos + delimiter.length());
		cont++;
	}

	return rowCol;
}

