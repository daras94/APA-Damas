#include "KernelMemGlobal.cuh"
/*
Kernel de ejecucion con memoria cOMPRATIDA TESELaDA Y COALECENCIA.

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
__global__ void DamasBomPlayGlobalMem(long *Tab, int numthread, int row, int col, int direcion) {
	int ty = threadIdx.y;		// Identificador de hilo del eje Y.
	int tx = threadIdx.x;		// Identificador de hilo del eje X.
	int width = numthread / TAM_TESELA;
	if ((ty < width) && (tx < width)) {
		/*
			Cuando encontramos el hilo que coincide con la jugada ejecutamos la jugada.
		*/
		if ((tx == col) && (ty == row)) {
			int movV = (new int[2]{ -1, 1 })[(direcion % 10)];								// Determinamos el movimiento vertical en funcion de la direcion recibida
			int movH = (new int[2]{ -1, 1 })[((direcion - (direcion % 10)) / 10) - 1];		// Determinamos el movimiento Horizontal en funcion de la direcion recibida
			int type_bom = Tab[ty * width + tx] % 10;										// Determinamos el tipo de bomaba de la que se trata.
			bool isPacMan = false;															// la ficha se convierte en pacma cunado se encuentra una ficha contraria y se la come.
			for (size_t i = 1; i <= ((type_bom > 2) ? type_bom : 1); i++) {
				isBomtrasposeGlobalMem = false;	// Desactivamos las bombas de trasposicion para que su efecto solo dure una jugada.
				/*
					Determinamos si es error de jugada y se lo comunicamos al host o finaliza el
					recorido de una bomba las bombas abazan tantas casillas en diagonal como va-
					lor de su tipo o asta que se convierta en pacman coman o se encuentre los li-
					mites del tablero os ata encontrar una ficha amiga.
				*/
				if (isCamaradaGlobal(i, movV, movH, Tab, width) && !isPacMan) {
					isPacMan = (Tab[(ty + (i * movH))* width + (tx + (i * movV))] != POS_TAB_JUEGO_EMPTY);								  // Determinamos si somos PacMAn
					Tab[(ty + (i * movH))* width + (tx + (i * movV))] = Tab[(row + ((i - 1) * movH)) * width + (col +((i - 1) * movV))];  // Insertamos la ficha en la nueva posicion.
					Tab[(ty + ((i - 1) * movH))* width + (tx + ((i - 1) * movV))] = POS_TAB_JUEGO_EMPTY;								  // Ponemos en blanco la poscion previa de mi ficha.
				} else {
					int posMov = (threadIdx.y + (i * movH)) * width + (threadIdx.x + (i * movV));
					/*
						Haber Tenemos 5 bombas que cuando la ficha se conbiertan en pacman o llequen
						a los limites del tablero esplota (pudiendo crear alteraciones espaciales en
						el tablero). COMENCEMOS!!!
					*/
					if (isPacMan || ((posMov < 0) || (posMov > (width * width)))) {
						switch (type_bom) {
							case 4:			// BOM Purpura!!, La bomba de Radial elimina todo openete en el radio de una casilla.
								purpleBom(Tab, movH, movV, width);
								printf(ANSI_COLOR_GREEN " BOM Purple, Radial BOM!!" ANSI_COLOR_RESET "\n");
								Tab[(ty + ((i - 1) * movH)) * width + (tx + ((i - 1) * movV))] = POS_TAB_JUEGO_EMPTY;
								break;
							case 7:			// BOM Rosita!!, La Bomba de transposcicion no mata pero si altera las dimensiones.
								isBomtrasposeGlobalMem = true;
								printf(ANSI_COLOR_GREEN " BOM Rose, traspose BOM!!" ANSI_COLOR_RESET "\n");
								Tab[(ty + ((i - 1) * movH)) * width + (tx + ((i - 1) * movV))] = POS_TAB_JUEGO_EMPTY;
								break;
						}
					}
				}
			}
			/*
				Cargamos el contenido de las matrizes teselada (pre cargada)en nuestra matriz
				resultante, ademas puede generar la bomba de trasposicion si esta es activada.
			*/
			Tab[(ty* width + tx)] = Tab[((isBomtrasposeGlobalMem) ? threadIdx.x : threadIdx.y) * width + ((isBomtrasposeGlobalMem) ? threadIdx.y : threadIdx.x)];
		}
	}
}

__device__ void yellowBom(long *Tab, int x, int y, int width) { // Si me da tiempo ago que rompa mas cosas ademas de acer la traspuesta
	if (isBomtrasposeGlobalMem) {

	}
}

__device__ void purpleBom(long *Tab, int x, int y, int width) {
	long fichaInMov = Tab[x * width * y];
	for (size_t i = 0; i < ((x > gridDim.x) ? 2 : 3); i++) {
		int row = (y > gridDim.y) ? 0 : 1;
		for (size_t j = 0; j < 2; j++) {
			int victimas = Tab[((y + row) + i) * width + (x + (new int[2]{ 1, -1 })[j])];
			if ((victimas - (victimas % 10)) != (fichaInMov - (fichaInMov % 10))) {
				Tab[((y + row) + i) * width + (x + (new int[2]{ 1, -1 })[j])] = POS_TAB_JUEGO_EMPTY;
			}
		}
	}
	//Tabs[(y + row) + i][(x + (new int[2]{ 1, -1 })[i])] = POS_TAB_JUEGO_EMPTY;
}


/*
	Determina si el movimiento cuando encuentra una ficha en su camino la puede comer si
	no es ficha amiga si es ficha amiga movimiento no valido;
*/
__device__ bool isCamaradaGlobal(int pos, int movV, int movH, long *Tab, int width) {
	bool isFriend = ((threadIdx.y + (pos * movH))* width + (threadIdx.x + (pos * movV)) != -1) &&
					((threadIdx.y + (pos * movH))* width + (threadIdx.x + (pos * movV))  < (width * width));
	if (isFriend) {
		long fichaInMov = Tab[(threadIdx.y + (pos * movH)) * width + (threadIdx.x + (pos * movV))];
		long fichaVictima = Tab[(threadIdx.y + ((pos - 1) * movH)) * width + (threadIdx.x + ((pos - 1) * movV))];
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
bool launchKernelMemGlobal(double numThread, long* tablero, int* jugada) {
	bool  error_play = true;
	long *tablero_cuda;
	setCudaMalloc(tablero_cuda, ((int)numThread));							// Reservamos espacio de memoria para el tablero en la GPU.
	setCudaMemcpyToDevice(tablero_cuda, tablero, ((int)numThread));			// Tranferimos el tablero a la GPU.
	dim3 dimGrid_c(TAM_BLOCK, TAM_BLOCK);
	dim3 dimBlock_c(numThread / TAM_TESELA, numThread / TAM_TESELA);
	/*
		Aqui empieza la fiesta con CUDA, y mi total y asoluta faltas de hora de sueño SORPRISE.
		Heee en memoria GLOBAL.
	*/
	DamasBomPlayGlobalMem <<<dimGrid_c, dimBlock_c >>> (tablero_cuda, numThread, jugada[1] - 1, jugada[0] - 1, jugada[2]);
	setCudaMemcpyToHost(tablero, tablero_cuda, (int)numThread);			// Trasferimos el tablero del GPU al HOST.
	cudaFree(tablero_cuda);												// Liberamos memoria llamamos al recolector de basura.
	return false; // Pertenece a algo que no dio tiempo y es mostrar errores de jugada desde el kernel.
}