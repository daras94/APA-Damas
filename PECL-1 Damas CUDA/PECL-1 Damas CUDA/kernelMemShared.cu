#include "KernelMemShared.cuh"

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
__global__ void DamasBomPlay(long *Tab, int numThread, bool *error_play, int row, int col, int direcion) {
	__shared__ long Tabs[TAM_TESELA][TAM_TESELA + 1];   // Matriz teselada en memoria compartida.
	int Row = blockIdx.y * TAM_TESELA + threadIdx.y;	// Calculamos la fila de la matriz teselada.
	int Col = blockIdx.x * TAM_TESELA + threadIdx.x;	// Calculamos la columna de la matriz teselada.
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
				/*
					Determinamos si es error de jugada y se lo comunicamos al host o finaliza el 
					recorido de una bomba las bombas abazan tantas casillas en diagonal como va-
					lor de su tipo o asta que se convierta en pacman coman o se encuentre los li-
					mites del tablero os ata encontrar una ficha amiga.
				*/
				if ((*error_play = isCamarada(i, movV, movH, Tabs)) && !isPacMan) {
					isPacMan = (Tabs[threadIdx.y + (i * movH)][threadIdx.x + (i * movV)] != POS_TAB_JUEGO_EMPTY); // Determinamos si somos PacMAn
					Tabs[threadIdx.y + (i * movH)][threadIdx.x + (i * movV)] = Tab[tx* width + ty];				  // Insertamos la ficha en la nueva posicion.
					Tabs[threadIdx.y + ((i - 1) * movH)][threadIdx.x + ((i - 1) * movV)] = POS_TAB_JUEGO_EMPTY;	  // Ponemos en blanco la poscion previa de mi ficha.
				} else {
					int posMovH = threadIdx.y + ((i - 1) * movH);
					int posMovV = threadIdx.x + ((i - 1) * movV);
					/*
						Haber Tenemos 5 bombas que cuando la ficha se conbiertan en pacman o llequen
						a los limites del tablero esplota (pudiendo crear alteraciones espaciales en
						el tablero). COMENCEMOS!!!
					*/
					if (isPacMan || (posMovH == -1) || (posMovV == -1) || (posMovH > gridDim.y) || (posMovV > gridDim.x)) {
						switch (type_bom) {
							case 3:			// BOM Verde!!, 

								break;
							case 4:			// BOM Purpura!!, La bomba de Radial elimina todo openete en el radio de una casilla.
								purpleBom(Tabs, posMovH, posMovV);
								break;
							case 7:			// BOM Amarillo!!, La Bomba de transposcicion no mata pero si altera las dimensiones.
								isBomtraspose = true;
								break;		
						}
						Tabs[threadIdx.y + ((i - 1) * movH)][threadIdx.x + ((i - 1) * movV)] = POS_TAB_JUEGO_EMPTY;	  // Ponemos en blanco la poscion previa de mi ficha.
						break;
					}
				}
			}			
		}
		__syncthreads();
		yellowBom(Tab, Tabs, Row, Col, width); // Ejecutamos la bomba de trasposicion si es el caso.
		/*
			Cargamos el contenido de las matrizes teselada (pre cargada)en nuestra matriz 
			resultante.
		*/
		Tab[(Row* width + Col)] = Tabs[threadIdx.y][threadIdx.x];
	}
}

__device__ void yellowBom(long *Tab, long Tabs[TAM_TESELA][TAM_TESELA + 1], int x, int y, int width) {
	if (isBomtraspose) {
		for (size_t j = 0; j < TAM_TESELA; j += gridDim.x) {
			Tabs[threadIdx.x][threadIdx.y + j] = Tab[((x + j)* width + y)];
		}
	}
}

__device__ void purpleBom(long Tabs[TAM_TESELA][TAM_TESELA + 1], int x, int y) {
	if (x > gridDim.x) {
		
	} else if (y > gridDim.y){

	}
	for (size_t i = 0; i < ((x > gridDim.x)? 2 : 3); i++) {
		int row = (y > gridDim.y) ? 0 : 1;
		Tabs[(y + 1) + i][(x + 1)] = POS_TAB_JUEGO_EMPTY;
		Tabs[(y + 1) + i][(x - 1)] = POS_TAB_JUEGO_EMPTY;
	}
}


/*
	Determina si el movimiento cuando encuentra una ficha en su camino la puede comer si
	no es ficha amiga si es ficha amiga movimiento no valido;
*/
__device__ bool isCamarada(int pos, int movV, int movH, long Tabs[TAM_TESELA][TAM_TESELA + 1]) {
	bool isFriend = ((threadIdx.y + (pos * movH)) != -1) && ((threadIdx.x + (pos * movV)) != -1) &&
				    ((threadIdx.y + (pos * movH)) < gridDim.y) && ((threadIdx.x + (pos * movV)) < gridDim.x);
	if (isFriend) {
		long fichaInMov = Tabs[threadIdx.y + (pos * movH)][threadIdx.y + (pos * movV)];
		isFriend = isFriend && (Tabs[threadIdx.y + ((pos - 1) * movH)][threadIdx.x + ((pos - 1) * movV)] != fichaInMov);
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
	bool  *error_play_c, error_play = true;
	cudaMalloc((void **)&error_play_c, sizeof(bool));
	cudaMemcpy(error_play_c, &error_play, sizeof(bool), cudaMemcpyHostToDevice);
	long *tablero_cuda;
	setCudaMalloc(tablero_cuda, numThread);							// Reservamos espacio de memoria para el tablero en la GPU.
	setCudaMemcpyToDevice(tablero_cuda, tablero, numThread);		// Tranferimos el tablero a la GPU.
	dim3 dimGrid_c(numThread / TAM_TESELA, numThread / TAM_TESELA);
	dim3 dimBlock_c(TAM_TESELA, TAM_TESELA);
	/*
		Aqui empieza la fiesta con CUDA, y mi total y asoluta faltas de hora de sueño SORPRISE.
	*/
	DamasBomPlay <<<dimGrid_c, dimBlock_c >> > (tablero_cuda, numThread, error_play_c, jugada[1] - 1, jugada[0] - 1, jugada[2]);
	setCudaMemcpyToHost(tablero, tablero_cuda, numThread);			// Trasferimos el tablero del GPU al HOST.
	cudaMemcpy(&error_play, error_play_c, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(tablero_cuda);											// Liberamos memoria llamamos al recolector de basura.
	return false/*error_play*/;
}