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

/*
	Medodo que genera 3 posbles configuraciones de la dimeciones de la trageta grafica
	en funcion de la carateristicas de la GPU que posea el usuario.
		
		- devProp       = puntero a struc el cual contiene la informacion de la GPU es 
					      de la arquitectura de CUDA.
		- myConfGpu     = struck declarado en la cabecera de esta clase el cul usamos 
		                  para almacenar informacion concreta de la GPU.
		- deviceCurrent = Id de la GPU que actualmente tenemos selecionada como principal.
*/
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

/*
	Medodo que genera 3 posbles configuraciones de la dimeciones de la trageta grafica
	en funcion de la carateristicas de la GPU que posea el usuario.

		- devProp       = puntero a struc el cual contiene la informacion de la GPU es
						  de la arquitectura de CUDA.
		- myConfGpu     = struck declarado en la cabecera de esta clase el cul usamos
						  para almacenar informacion concreta de la GPU.
		- deviceCurrent = Id de la GPU que actualmente tenemos selecionada como principal.
*/
double setGpuForPlayManual(cudaDeviceProp *devProp, info_gpu *myConfGpu, int deviceCurrent) {
	int imput, cont = 0; double dim = 1;
	system("cls");
	cout << "/***************************************************************************************/" << endl;
	cout << "/*  +--> " << ANSI_COLOR_CYAN "Menu de configuracion de partida:" ANSI_COLOR_RESET << setw(47) << "*/" << endl;
	cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
	cout << "/*" << setw(87) << "*/" << endl;
	string modelGPU = devProp->name;
	cout << "/*  " ANSI_COLOR_MAGENTA "GPU " << deviceCurrent << ANSI_COLOR_RESET ") - " << modelGPU << setw(76 - modelGPU.length()) << "*/" << endl;
	cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
	cout << "/*  - " << ANSI_COLOR_RED "AVISO: " ANSI_COLOR_RESET "Cuidado con la configuracion quien rompe CUDA lo sufre." << setw(35) << "*/" << endl;
	cout << "/***************************************************************************************/" << endl;
	teclado:
	cout << " - Introduca el numero de " << ((cont == 0)? "filas" : "Columnas") << " (" ANSI_COLOR_GREEN "0 para salir" ANSI_COLOR_RESET "): ";
	cin >> imput;		// Entrada de texto por teclado.
	if (cont == 0 && imput != 0) {
		dim = dim * imput;
		cont++;
		goto teclado;
	}
	return dim;
}

/*
	Establece la dificultade la partida.
*/
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

/* 
	Generamos el tablero con un números de bonbas aleatorios en función de la dificultad.
*/
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

/* 
	Función que imprime el número de columnas que va a tener el tablero
	para que sea más facil elegir piezas.
*/
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
			int background = ((i + k) % 2 == 0) ? COLOR_GRIS : COLOR_NEGRO;			// Color que contrulle el tablero.
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

/*
	Medodo que se encarga de gestionar la partida, salvar la partida con persistencia
	y lanzar el kernel adecuado que el jugador aya selecionado.
*/ 
void playDamas(int typeKernel, double numThread, info_gpu *myConfGpu, int dificultad) {
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
		bool found = regex_match(input, match, reg_expre);						// Coparacion que busca un expresion de tipo fila:columna:direccion
		if (found) {
			int *jugada = getRowAndColumn(input, numThread);
			if (sizeof(jugada) < NUM_DIMENSION_TAB) {
				ERROR_MSS("Error en la columna o fila introducida.");
				goto teclado;
			} else {															// Inbocamos al metodo de lanzamiento de los kernels
				bool error_play = launchKernel(typeKernel, numThread, tablero, jugada);
				if (error_play) {
					ERROR_MSS("El movimento realizado no es valido.");
					goto teclado;
				}
				cont++;
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

/*
	Metodo Que inboca el kernel segun el tipo de kernel que se quiera ejecutar.

		- typeKernel = Entero que indica tipo de kernel a lanzar x Block, Mem_Shared or Bloques y Mem_Global.
		- mumThread  = recibe el numeroi de thread para realizar la configuracion de juego.
		- tablero	 = Recibe el tablero de juego generado por el host
		- jugada	 = Recibe la jugada realizada por algun jugador.
*/
bool launchKernel(int typeKernel, double numThread, long* tablero, int* jugada) {
	bool isErrorJugada = false;
	switch (typeKernel) {
		case 1:		// Memoria Compartida Con Colesencia y Teselada.
			launchKernelMemShared(numThread, tablero, jugada, isErrorJugada);
			break;
		case 2:		// Por Bloques.

			break;
		case 3:		// Por Bloques Con Memoria Compartida.

			break;
	}
	return isErrorJugada;
}

/*
	Realiza el Separado de los valores de la jugada pasada en el formato C:F:D 
	(C = Columna, F = Fila, D = Direcion) y debuelve la jugada como un array 
	de enteros. 
*/
int *getRowAndColumn(string jug, double numThread) {
	string delimiter = ":", aux = jug + ":";
	int pos = 0, cont = 0, *rowCol = new int[NUM_DIMENSION_TAB];
	bool isNotErrorColRow = true;
	while ((pos = aux.find(delimiter)) != string::npos && isNotErrorColRow) {
		int token = stoi(aux.substr(0, pos));
		if (isNotErrorColRow = (token > 0 && token <= (numThread / TAM_TESELA))) {
			rowCol[cont] = token;
		}
		aux.erase(0, pos + delimiter.length());
		cont++;
	}

	return rowCol;
}

