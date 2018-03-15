#include "KernelMultiBlock.cuh"
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
__global__ void DamasBomPlayMultiBlock(long *Tab, int numthread, int row, int col, int direcion) {
	int Col = blockIdx.y * TAM_TESELA + threadIdx.y;	// Identificador de hilo del eje Y, lo usamos para calular la columna.
	int Row = blockIdx.x * TAM_TESELA + threadIdx.x;	// Identificador de hilo del eje X, lo usamos para calular la fila.
	int width = numthread / TAM_TESELA;
	if ((Col < width) && (Row < width)) {
		__syncthreads();
		/*
			Cuando encontramos el hilo que coincide con la jugada ejecutamos la jugada.
		*/
		if ((Row == col) && (Col == row)) {
			int movV = (new int[2]{ -1, 1 })[(direcion % 10)];								// Determinamos el movimiento vertical en funcion de la direcion recibida
			int movH = (new int[2]{ -1, 1 })[((direcion - (direcion % 10)) / 10) - 1];		// Determinamos el movimiento Horizontal en funcion de la direcion recibida
			int type_bom = Tab[Col * width + Row] % 10;										// Determinamos el tipo de bomaba de la que se trata.
			bool isPacMan = false;															// la ficha se convierte en pacma cunado se encuentra una ficha contraria y se la come.
			for (size_t i = 1; i <= ((type_bom > 2) ? type_bom : 1); i++) {
				isBomtrasposeMultiBlock = false;											// Desactivamos las bombas de trasposicion para que su efecto solo dure una jugada.
				/*
					Determinamos si es error de jugada y se lo comunicamos al host o finaliza el
					recorido de una bomba las bombas abazan tantas casillas en diagonal como va-
					lor de su tipo o asta que se convierta en pacman coman o se encuentre los li-
					mites del tablero os ata encontrar una ficha amiga.
				*/
				if (isCamaradaMultyBlock(Col,  Row, i, movV, movH, Tab, width) && !isPacMan) {
					isPacMan = (Tab[(Col + (i * movH))* width + (Row + (i * movV))] != POS_TAB_JUEGO_EMPTY);								 // Determinamos si somos PacMan y por lo tanto podemos comer fantasmas es decir damas.
					Tab[(Col + (i * movH))* width + (Row + (i * movV))] = Tab[(row + ((i - 1) * movV)) * width + (col + ((i - 1) * movH))];  // Insertamos la ficha en la nueva posicion.
					Tab[(Col + ((i - 1) * movH))* width + (Row + ((i - 1) * movV))] = POS_TAB_JUEGO_EMPTY;									 // Ponemos en blanco la poscion previa de mi ficha.
				} else {
					/*
						Haber Tenemos 5 bombas que cuando la ficha se conbiertan en pacman o llequen
						a los limites del tablero esplota (pudiendo crear alteraciones espaciales en
						el tablero). COMENCEMOS!!!
					*/
					if (isPacMan && ((Col + ((i - 1) * movH) > -1) && (Row + ((i - 1) * movV) < width))) {
						switch (type_bom) {
							case 4:			// BOM Purpura!!, La bomba de Radial elimina todo openete en el radio de una casilla.
								purpleBomMultyBlock((Col + ((i - 1) * movH)), (Row + ((i - 1) * movV)), Tab, width);
								printf(ANSI_COLOR_GREEN " BOM Purple, Radial BOM!!" ANSI_COLOR_RESET "\n");
								Tab[(Col + ((i - 1) * movH)) * width + (Row + ((i - 1) * movV))] = POS_TAB_JUEGO_EMPTY;
								break;
							case 7:			// BOM Rosita!!, La Bomba de transposcicion no mata pero si altera las dimensiones.
								isBomtrasposeMultiBlock = true;
								roseBomMultyBlock(Tab, (Col + ((i - 1) * movH)), (Row + ((i - 1) * movV)), width);
								printf(ANSI_COLOR_GREEN " BOM Rose, traspose BOM!!" ANSI_COLOR_RESET "\n");
								Tab[(Col + ((i - 1) * movH)) * width + (Row + ((i - 1) * movV))] = POS_TAB_JUEGO_EMPTY;
								break;
						}
						break; // Me parece un buena forma de optimizar el kerne para salir del bucle cuando el resto de ciclos no son nesesarios.
					}
				}
			}
		}
	}
}

__device__ void roseBomMultyBlock(long *Tab, int x, int y, int width) { // Si me da tiempo ago que rompa mas cosas ademas de acer la traspuesta
	if (isBomtrasposeMultiBlock) {
		long fichaInMov = Tab[x * width + y];
		for (size_t i = 1; i < width; i++) {
			int victimas = Tab[(x + (i * (-1)) * width + y)];
			if ((victimas - (victimas % 10)) != (fichaInMov - (fichaInMov % 10))) {
				Tab[(x * width + (y + (i * (-1))))] = POS_TAB_JUEGO_EMPTY;
			}
		}
	}
}

__device__ void purpleBomMultyBlock(int Col, int Row, long *Tab, int width) {
	long fichaInMov = Tab[Col * width + Row];
	for (size_t i = 1; i < width; i++) {
		for (size_t j = 0; j < 2; j++) {
			int victimas = Tab[((Col + (i * (new int[2]{ 1, -1 })[j])) * width + Row)];
			if ((victimas - (victimas % 10)) != (fichaInMov - (fichaInMov % 10))) {
				Tab[((Col + (i * (new int[2]{ 1, -1 })[j])) * width + (Row + j))] = POS_TAB_JUEGO_EMPTY;
			}
		}
	}
}


/*
	Determina si el movimiento cuando encuentra una ficha en su camino la puede comer si
	no es ficha amiga si es ficha amiga movimiento no valido;
*/
__device__ bool isCamaradaMultyBlock(int col, int row, int pos, int movV, int movH, long *Tab, int width) {
	bool isFriend = ((col + (pos * movH) > -1) && (col + (pos * movH) < width)) &&
					((row + (pos * movV) > -1) && (row + (pos * movV) < width));
	if (isFriend) {
		long fichaInMov = Tab[(col + (pos * movH)) * width + (row + (pos * movV))];
		long fichaVictima = Tab[(col + ((pos - 1) * movH)) * width + (row + ((pos - 1) * movV))];
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
bool launchKernelMultyBlock(double numThread, long* tablero, int* jugada) {
	bool  error_play = true;
	long *tablero_cuda;
	setCudaMalloc(tablero_cuda, ((int)numThread));							// Reservamos espacio de memoria para el tablero en la GPU.
	setCudaMemcpyToDevice(tablero_cuda, tablero, ((int)numThread));			// Tranferimos el tablero a la GPU.
	dim3 dimGrid_c(numThread / TAM_TESELA, numThread / TAM_TESELA);
	dim3 dimBlock_c(TAM_TESELA, TAM_TESELA);								// OJO para mi es TAM_TESELA respecto a teoria es TITLE_WIDTH.
	/*
		Aqui empieza la fiesta con CUDA, y mi total y asoluta faltas de hora de sueño SORPRISE.
		Heee en ahora en Multi Bloque.
	*/
	DamasBomPlayMultiBlock << <dimGrid_c, dimBlock_c >> > (tablero_cuda, numThread, jugada[1] - 1, jugada[0] - 1, jugada[2]);
	setCudaMemcpyToHost(tablero, tablero_cuda, (int)numThread);			// Trasferimos el tablero del GPU al HOST.
	cudaFree(tablero_cuda);												// Liberamos memoria llamamos al recolector de basura.
	return false; // Pertenece a algo que no dio tiempo y es mostrar errores de jugada desde el kernel.
}