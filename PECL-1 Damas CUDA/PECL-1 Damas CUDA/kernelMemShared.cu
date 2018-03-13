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
__global__ void DamasBomPlay(long *Tab, int numThread, int row, int col, int direcion) {
<<<<<<< HEAD
	__shared__ long Tabs[TAM_TESELA][TAM_TESELA];		// Matriz teselada en memoria compartida.
=======
	__shared__ long Tabs[TAM_TESELA][TAM_TESELA + 1];   // Matriz teselada en memoria compartida.
>>>>>>> master
	int Row = blockIdx.y * gridDim.y + threadIdx.y;		// Calculamos la fila de la matriz teselada.
	int Col = blockIdx.x * gridDim.x + threadIdx.x;		// Calculamos la columna de la matriz teselada.
	/*
		Para evitar las violaciones de acceso esas quye la version 5.3 de nising daban como
		errores de grid pero en realida son violaciones de memoria dpor accesos de hilos.
	*/
	if (Row < TAM_TESELA && Col < TAM_TESELA) {
		int width = numThread / TAM_TESELA;				// Calculamos el tamaño en funcion del ancho.
		/*
			Caragamos la matriz en la matriz teselada con coalalesesian y sin comflitos en
<<<<<<< HEAD
			bancos de memoria para las GPU 2.x y 3.x (Es un precarga de datos).
		*/
		for (size_t i = 0; i < width; i++) {
			Tabs[threadIdx.y][threadIdx.x] = Tab[Row* numThread + (i*TAM_TESELA + threadIdx.x)];
			Tabs[threadIdx.y][threadIdx.x] = Tab[(i*TAM_TESELA + threadIdx.y)* numThread + Col];
		}
=======
			bancos de memoria para las GPU 2.x y 3.x.
		*/
		//for (size_t i = 0; i < TAM_TESELA; i+=(TAM_TESELA/4)) {
			Tabs[threadIdx.y/* + i*/][threadIdx.x] = Tab[(Row/* + i*/)* width + Col];
		//}
>>>>>>> master
		__syncthreads();
		int tx = blockIdx.x * gridDim.x + row;			// Calculamos el indice x de la jugada en la matriz teselda.
		int ty = blockIdx.y * gridDim.y + col;			// Calculamos el indice y de la jugada en la matriz teselda.
		/*
			Cuando encontramos el hilo que coincide con la jugada ejecutamos la jugada.
		*/
		if ((tx == Row) && (ty == Col)) {
			int movV = threadIdx.x + (new int[2]{ -1, 1 })[(direcion % 10)];						// Determinamos el movimiento vertical en funcion de la direcion recibida
			int movH = threadIdx.y + (new int[2]{ -1, 1 })[((direcion - (direcion % 10)) / 10) - 1];// Determinamos el movimiento Horizontal en funcion de la direcion recibida
			if ((movV != -1 && movH != -1)) {
				Tabs[movH][movV] = Tab[tx * width + ty];											// Insertamos la ficha en la nueva posicion.
				Tabs[threadIdx.y][threadIdx.x] = POS_TAB_JUEGO_EMPTY;								// Ponemos en blanco la poscion previa de mi ficha.
			}
			
		}
		__syncthreads();
		/*
			Cargamos el contenido de las matrizes teselada en nuestra matriz resultante.
		*/
		Tab[(Row)* width + Col] = Tabs[threadIdx.y][threadIdx.x];
	}
}

/*
	Realiza la inbocacion al kernel de memoria compartida con coalecencia y teselada.

		- numThread  = Numreo de hilos de nuestra matriz.
		- tablero	 = Puntero al tablero de juego generado en el host.
		- jugada	 = Array de enteros el cual contiene la jugada realizada.
		- error_play = Bolean pasado por referencia para notificar errores de jugada realizados. 
*/
void launchKernelMemShared(double numThread, long* tablero, int* jugada, bool error_play) {
	//const size_t tam_tesela = TAM_TESELA / ((numThread / TAM_TESELA) >= TAM_TESELA) ? 1 : 2;
	long *tablero_cuda;
	setCudaMalloc(tablero_cuda, numThread);							// Reservamos espacio de memoria para el tablero en la GPU.
	setCudaMemcpyToDevice(tablero_cuda, tablero, numThread);		// Tranferimos el tablero a la GPU.
	dim3 dimGrid_c(numThread / TAM_TESELA, numThread / TAM_TESELA);
	dim3 dimBlock_c(TAM_TESELA, TAM_TESELA);
	/*
		Aqui empieza la fiesta con CUDA, y mi total y asoluta faltas de hora de sueño SORPRISE.
	*/
	DamasBomPlay <<<dimGrid_c, dimBlock_c >> > (tablero_cuda, numThread, jugada[1] - 1, jugada[0] - 1, jugada[2]); 
	setCudaMemcpyToHost(tablero, tablero_cuda, numThread);			// Trasferimos el tablero del GPU al HOST.
	cudaFree(tablero_cuda);											// Liberamos memoria llamamos al recolector de basura.
}