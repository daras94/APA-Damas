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
	string imput = { NULL }; int  cont = 0; double dim = 0;
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
	cout << " - Introduca el numero las filas y columnas con el formato Ej: 16x16 (" ANSI_COLOR_GREEN "0 para salir" ANSI_COLOR_RESET "): ";
	cin >> imput;		// Entrada de texto por teclado.
	smatch match;
	regex  reg_expre{R"(\d{1,2}x\d{1,2})"};								// Epresion regular para las filas y columnas.
	bool found = regex_match(imput, match, reg_expre);					// Coparacion que busca un expresion de tipo fila:columna:direccion
	if (found) {
		int *jugada = getRowAndColumn(imput, myConfGpu -> numThreadMasPerBlock, "x", 2);
		if (sizeof(jugada) < NUM_DIMENSION_TAB) {
			ERROR_MSS("Error en la columna o fila introducida.");
			goto teclado;
		} else {
			dim = (jugada[0] * jugada[1]);
		}
	} else {
		if (imput != "0") {
			ERROR_MSS("Carrates no reconocido pruebe de nuevo o rango escedido.");
			goto teclado;
		}
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
	int cont = 0, tam = (((int)numThread) / TAM_TESELA);
	long row = 0, col = 0, *tablero = new long[(tam * tam)];
	
	/*
		El numero de fichas para cada jugador en funcion de las dimensiones del tablero 
		y lo multiplicamos por el doble para cuando nos salimos de las dimensiones con-
		becionales de un tablero de damas.
	*/
	int numRowFicha = log2(tam) + ((tam > 8)? 2 : 0);			
	srand(time(NULL));
	for (size_t i = 0; i < tam; i++) {
		for (size_t j = 0; j < tam; j++) {
			int bom = rand() % (dificultad + 2);				    // Gennera Bombas en funcion de las dificultad selecionada.
			tablero[(i * tam + j)] = (((i + j + ((i > numRowFicha) ? 0 : 1))% 2 == 0)? ((i < numRowFicha)? 31 + bom : POS_TAB_JUEGO_EMPTY) : ((i >= (tam - numRowFicha))? 22 + bom : POS_TAB_JUEGO_EMPTY));
		}	
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
	for (size_t i = 0; i < numThread / TAM_TESELA; i++) {
		cout << setw(4) << i+1 << setw(3) << "-" << setw(3) << "";
		for (size_t k = 0; k < numThread/TAM_TESELA; k++) {							// Damos color en función del número imprimir
			int background = ((i + k) % 2 == 0) ? COLOR_BLANCO : COLOR_NEGRO;		// Color que contrulle el tablero.
			int bloque = tablero[(i * (((int)numThread) / TAM_TESELA) + k)];
			//if (bloque < NUM_FICHAS) {											// Calculamos el color de la casilla.
				int color = COLOR_TABLERO(background, (new int[NUM_FICHAS] {background, COLOR_ROJO, COLOR_AZUL_LIGHT, COLOR_VERDE, COLOR_PURPURA, COLOR_AMARILLO, COLOR_AGUAMARINA, COLOR_PURPURA_LIGHT})[bloque % 10]); 
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
			//} 
			cout << " " << (((bloque - (bloque % 10)) > POS_TAB_JUEGO_EMPTY * 2)? "#" : "O") << " ";
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
			int *jugada = getRowAndColumn(input, numThread, ":", NUM_DIMENSION);
			if (sizeof(jugada) < NUM_DIMENSION_TAB) {
				ERROR_MSS("Error en la columna o fila introducida.");
				goto teclado;
			} else {															// Inbocamos al metodo de lanzamiento de los kernels
				bool error_play = launchKernel(typeKernel, numThread, tablero, jugada);
				if (error_play) {
					ERROR_MSS("El movimento realizado no es valido.");
					goto teclado;
				}
				system("pause");
				cont++;
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
			isErrorJugada = launchKernelMemShared(numThread, tablero, jugada);
			break;
		case 2:		// Por Bloques.
			isErrorJugada = launchKernelMultyBlock(numThread, tablero, jugada);
			break;
		case 3:		// Por Bloques Memoria global.
			isErrorJugada = launchKernelMemGlobal(numThread, tablero, jugada);
			break;
	}
	return isErrorJugada;
}

/*
	Realiza el Separado de los valores de la jugada pasada en el formato C:F:D 
	(C = Columna, F = Fila, D = Direcion) o la configuracion de filas columnas
	y debuelve la jugada como un array de enteros. 
*/
int *getRowAndColumn(string jug, double numThread, string delimiter, int num_parametres) {
	string aux = jug + delimiter;
	size_t pos = 0, cont = 0;
	int *rowCol = new int[num_parametres];
	bool isNotErrorColRow = true;
	while ((pos = aux.find(delimiter)) != string::npos && isNotErrorColRow) {
		int token = stoi(aux.substr(0, pos));
		isNotErrorColRow = (token > 0) && ((token <= (numThread / TAM_TESELA)) || (cont == 2));
		if (isNotErrorColRow) {
			rowCol[cont] = token;
			cont++;
		}
		aux.erase(0, pos + delimiter.length());
	}
	return rowCol;
}

