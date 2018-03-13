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
			int type_bom = Tabs[threadIdx.y][threadIdx.x] % 10;
			int num_move = (type_bom > 2) ? type_bom : 1;
			for (size_t i = 0; i < num_move; i++) {
				int movV = threadIdx.x + (new int[2]{ -1, 1 })[(direcion % 10)];						// Determinamos el movimiento vertical en funcion de la direcion recibida
				int movH = threadIdx.y + (new int[2]{ -1, 1 })[((direcion - (direcion % 10)) / 10) - 1];// Determinamos el movimiento Horizontal en funcion de la direcion recibida
				*error_play = isCamarada(movV, movH, Tabs);												// Determinamos si es error de jugada y se lo comunicamos al host.
				if (!*error_play) {
					Tabs[movH][movV] = Tab[(tx)* width + (ty)];											// Insertamos la ficha en la nueva posicion.
					Tabs[threadIdx.y][threadIdx.x] = POS_TAB_JUEGO_EMPTY;								// Ponemos en blanco la poscion previa de mi ficha.
				}
			}
		}
		__syncthreads();
		/*
			Cargamos el contenido de las matrizes teselada (pre cargada)en nuestra matriz 
			resultante.
		*/
		Tab[(Row* width + Col)] = Tabs[threadIdx.y][threadIdx.x];
	}
}

/*
	Determina si el movimiento cuando encuentra una ficha en su camino la puede comer si
	no es ficha amiga si es ficha amiga movimiento no valido;
*/
__device__ bool isCamarada(int movV, int movH, long Tabs[TAM_TESELA][TAM_TESELA + 1]) {
	bool isFriend = (movV != -1) && (movH != -1);
	if (isFriend) {
		long fichaInMov = Tabs[movH][movV];
		isFriend = isFriend && (Tabs[threadIdx.y][threadIdx.x] == fichaInMov);
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