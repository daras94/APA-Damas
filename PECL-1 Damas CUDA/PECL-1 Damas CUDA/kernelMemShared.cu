#include "KernelMemShared.cuh"

/*
	Kernel de ejecucion con memoria compartida teselada y coalescencia.

		- Tab		= Un copia del tablero dejuego pasada al device.
		- numThread = Numero de hilos del tablero.
		- row		= fila de ficha que va a realizar la jugada.
		- col		= columna en la que se encuentra la ficha que va a reaalizar la jugada.
		- dirrecion = direcion de la jugada pudiendo ser:
			|-> 10 = sup-izq.
			|-> 20 = inf-izq.
			|-> 11 = sup-dech.
			|-> 21 = inf-dech.
*/
__global__ void DamasBomPlayMemShared(long *Tab, int numThread, int row, int col, int direcion) {
	__shared__ long Tabs[TAM_TESELA][TAM_TESELA + 1];	// Matriz teselada en memoria compartida.
	int Row = blockIdx.y * gridDim.y + threadIdx.y;		// Calculamos la fila de la matriz teselada.
	int Col = blockIdx.x * gridDim.x + threadIdx.x;		// Calculamos la columna de la matriz teselada.
	int width = numThread / TAM_TESELA;					// Calculamos el tamaño en funcion del ancho.
	/*
		Para evitar las violaciones de acceso esas que la version 5.3 de nising daban como
		errores de grid pero en realida son violaciones de memoria dpor accesos de hilos.
	*/
	if (Row < gridDim.y && Col < gridDim.x) {
		/*
			Caragamos la matriz en la matriz teselada con coalalesesian y sin comflitos en
			bancos de memoria para las GPU 2.x y 3.x (Es un precarga de datos).
		*/
		Tabs[threadIdx.y][threadIdx.x] = Tab[(Row* width + Col)];
		__syncthreads();
		int tx = blockIdx.x * gridDim.x + row;			// Calculamos el indice x de la jugada en la matriz teselda.
		int ty = blockIdx.y * gridDim.y + col;			// Calculamos el indice y de la jugada en la matriz teselda.
		/*
			Cuando encontramos el hilo que coincide con la jugada ejecutamos la jugada.
		*/
		if ((tx == Row) && (ty == Col)) {
			int movV = (new int[2]{ -1, 1 })[(direcion % 10)];								// Determinamos el movimiento vertical en funcion de la direcion recibida
			int movH = (new int[2]{ -1, 1 })[((direcion - (direcion % 10)) / 10) - 1];		// Determinamos el movimiento Horizontal en funcion de la direcion recibida
			int type_bom = Tabs[threadIdx.y][threadIdx.x] % 10;								// Determinamos el tipo de bomaba de la que se trata.
			bool isPacMan = false;															// la ficha se convierte en pacma cunado se encuentra una ficha contraria y se la come.
			for (size_t i = 1; i <= ((type_bom > 2)? type_bom : 1); i++) {
				isBomtrasposeSharedMem = false;	// Desactivamos las bombas de trasposicion para que su efecto solo dure una jugada.
				/*
					Determinamos si es error de jugada y se lo comunicamos al host o finaliza el 
					recorido de una bomba las bombas abazan tantas casillas en diagonal como va-
					lor de su tipo o asta que se convierta en pacman coman o se encuentre los li-
					mites del tablero os ata encontrar una ficha amiga.
				*/
				if (isCamaradaSharedMem(i, movV, movH, Tabs) && !isPacMan) {
					isPacMan = (Tabs[threadIdx.y + (i * movH)][threadIdx.x + (i * movV)] != POS_TAB_JUEGO_EMPTY); // Determinamos si somos PacMAn
					Tabs[threadIdx.y + (i * movH)][threadIdx.x + (i * movV)] = Tab[tx  * width + ty];			  // Insertamos la ficha en la nueva posicion.
					Tabs[threadIdx.y + ((i - 1) * movH)][threadIdx.x + ((i - 1) * movV)] = POS_TAB_JUEGO_EMPTY;	  // Ponemos en blanco la poscion previa de mi ficha.
				} else {
					int posMovH = threadIdx.y + (i * movH);
					int posMovV = threadIdx.x + (i * movV);
					/*
						Haber Tenemos 5 bombas que cuando la ficha se conbiertan en pacman o llequen
						a los limites del tablero esplota (pudiendo crear alteraciones espaciales en
						el tablero). COMENCEMOS!!!
					*/
					if (isPacMan /*|| ((posMovH < 0) || (posMovV < 0)) || ((posMovH > gridDim.y) || (posMovV > gridDim.x))*/) {
						switch (type_bom) {
							case 4:			// BOM Purpura!!, La bomba de Radial elimina todo openete en el radio de una casilla.
								purpleBomSharedMem(Tabs, posMovH, posMovV);
								printf(ANSI_COLOR_GREEN " - BOM Purple, Radial BOM!!" ANSI_COLOR_RESET "\n");
								Tabs[threadIdx.y + ((i - 1) * movH)][threadIdx.x + ((i - 1) * movV)] = POS_TAB_JUEGO_EMPTY;	
								break;
							case 7:			// BOM Rosa!!, La Bomba de transposcicion no mata pero si altera las dimensiones.
								isBomtrasposeSharedMem = true;
								printf(ANSI_COLOR_GREEN " - BOM Rose, traspose BOM!!" ANSI_COLOR_RESET "\n");
								Tabs[threadIdx.y + ((i - 1) * movH)][threadIdx.x + ((i - 1) * movV)] = POS_TAB_JUEGO_EMPTY;	
								break;		
						}
						break; // Me parece un buena forma de optimizar el kerne para salir del bucle cuando el resto de ciclos no son nesesarios.
					}
				}
			}			
		}
		__syncthreads();
		/*
			Cargamos el contenido de las matrizes teselada (pre cargada)en nuestra matriz 
			resultante, ademas puede generar la bomba de trasposicion si esta es activada.
		*/
		Tab[(Row* width + Col)] = Tabs[((isBomtrasposeSharedMem) ? threadIdx.x : threadIdx.y)][((isBomtrasposeSharedMem) ? threadIdx.y : threadIdx.x)]; 
	}
}

__device__ void yellowBomSharedMem(long *Tab, long Tabs[TAM_TESELA][TAM_TESELA + 1], int x, int y, int width) { // Si me da tiempo ago que rompa mas cosas ademas de acer la traspuesta
	if (isBomtrasposeSharedMem) {
		// por si da tempo a añadir mas cosas a esta bomaba.
	}
}

__device__ void purpleBomSharedMem(long Tabs[TAM_TESELA][TAM_TESELA + 1], int x, int y) {
	long fichaInMov = Tabs[x][y];
	for (size_t i = 0; i < ((x > gridDim.x)? 3 : 4); i++) {
		int row = (y > gridDim.y) ? 0 : 1;
		for (size_t j = 0; j < 2; j++) {
			int victimas = Tabs[(y + row) + i][(x + (new int[2]{ 1, -1 })[j])];
			if ((victimas - (victimas % 10)) != (fichaInMov - (fichaInMov % 10))) {
				Tabs[(y + row) + i][(x + (new int[2]{ 1, -1 })[j])] = POS_TAB_JUEGO_EMPTY;
			}
		}
	}
	//Tabs[(y + row) + i][(x + (new int[2]{ 1, -1 })[i])] = POS_TAB_JUEGO_EMPTY;
}


/*
	Determina si el movimiento cuando encuentra una ficha en su camino la puede comer si
	no es ficha amiga si es ficha amiga movimiento no valido;
*/
__device__ bool isCamaradaSharedMem(int pos, int movV, int movH, long Tabs[TAM_TESELA][TAM_TESELA + 1]) {
	bool isFriend = ((threadIdx.y + (pos * movH)) != -1) && ((threadIdx.x + (pos * movV)) != -1) &&
				    ((threadIdx.y + (pos * movH)) < gridDim.y) && ((threadIdx.x + (pos * movV)) < gridDim.x);
	if (isFriend) {
		long fichaInMov = Tabs[threadIdx.y + (pos * movH)][threadIdx.x + (pos * movV)];
		long fichaVictima = Tabs[threadIdx.y + ((pos - 1) * movH)][threadIdx.x + ((pos - 1) * movV)];
		isFriend = isFriend && (fichaVictima - ((fichaVictima % 10)) != (fichaInMov - (fichaInMov % 10)));
	}
	return isFriend;
}

/*
	Realiza la inbocacion al kernel de memoria compartida con coalecencia y teselada.

		- numThread  = Numreo de hilos de nuestra matriz.
		- tablero	 = Puntero al tablero de juego generado en el host.
		- jugada	 = Array de enteros el cual contiene la jugada realizada.
		- error_play = Bolean pasado por referencia para notificar errores de jugada realizados. 
*/
bool launchKernelMemShared(double numThread, long* tablero, int* jugada) {
	bool  error_play = true;
	long *tablero_cuda;
	setCudaMalloc(tablero_cuda, ((int)numThread));							// Reservamos espacio de memoria para el tablero en la GPU.
	setCudaMemcpyToDevice(tablero_cuda, tablero, ((int)numThread));		// Tranferimos el tablero a la GPU.
	dim3 dimGrid_c((int)numThread / TAM_TESELA, (int)numThread / TAM_TESELA);
	dim3 dimBlock_c(TAM_TESELA, TAM_TESELA);
	/*
		Aqui empieza la fiesta con CUDA, y mi total y asoluta faltas de hora de sueño SORPRISE.
	*/
	DamasBomPlayMemShared <<<dimGrid_c, dimBlock_c >> > (tablero_cuda, ((int)numThread), jugada[1] - 1, jugada[0] - 1, jugada[2]);
	setCudaMemcpyToHost(tablero, tablero_cuda, (int)numThread);			// Trasferimos el tablero del GPU al HOST.
	cudaFree(tablero_cuda);											// Liberamos memoria llamamos al recolector de basura.
	return false/*error_play*/;
}