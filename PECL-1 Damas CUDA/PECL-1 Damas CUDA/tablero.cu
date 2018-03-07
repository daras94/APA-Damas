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

void setGpuForPlay(cudaDeviceProp *devProp, info_gpu *myConfGpu) {
	int deviceCount = 0;												// Lo utilzamos para contar el numero de GPUs disponibles.	
	cudaError_t error_code = cudaGetDeviceCount(&deviceCount);
	if (error_code != cudaSuccess) {
		string errorMss = "La GPU devolvio el error " ANSI_COLOR_GREEN + to_string(error_code) + ANSI_COLOR_RESET + ": \n\t -> " + cudaGetErrorString(error_code);
		ERROR_MSS(errorMss); // Retornamos un mensaje de error.	
		system("pause");
	} else {
		int  gpuOpc, dimTamblero[3] = { 1, 2, 4 };
		bool isPlay = false, isGpu = false;
		do {
			system("cls");
			cout << "/***************************************************************************************/" << endl;
			cout << "/*  +--> " << ANSI_COLOR_CYAN "Menu de configuracion de partida:" ANSI_COLOR_RESET << setw(47) << "*/" << endl;
			cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
			cout << "/*" << setw(87) << "*/" << endl;
			if (!isGpu) {
				for (int i = 0; i < deviceCount; i++) {
					cudaGetDeviceProperties(devProp, i);
					string modelGPU = devProp->name;
					cout << "/*  " ANSI_COLOR_MAGENTA "GPU " << (i + 1) << ANSI_COLOR_RESET ") - " << modelGPU << setw(76 - modelGPU.length()) << "*/" << endl;
					cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
					getCarrasteristicForGPU(devProp);
					cout << "/*" << setw(87) << "*/" << endl;
				}
				cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
				cout << "/*  - " << ANSI_COLOR_RED "AVISO: " ANSI_COLOR_RESET "Selecione el id de una GPU para jugar." << setw(34) << "*/" << endl;
			}
			else {
				getCofigPlay(gpuOpc - 1, devProp, myConfGpu);						// obtenemos los parametros de la gpu para realizar la configuracion de la gpu.
				int numThread = myConfGpu -> maxDimThreadBlock[1];
				for  (int i = 0; i < 3; i++ ) {
					dimTamblero[i] = numThread / dimTamblero[i];
					cout << "/*\t" ANSI_COLOR_MAGENTA << (i + 1) << ANSI_COLOR_RESET ") - Disponible tablero de juego de " ANSI_COLOR_GREEN << (dimTamblero[i] / TAM_TESELA) << "x" << (dimTamblero[i] / TAM_TESELA) << ANSI_COLOR_RESET << setw(40) << "*/" << endl;
				}
				cout << "/*" << setw(87) << "*/" << endl;
				isPlay = true;
			}
			cout << "/***************************************************************************************/" << endl;
			cout << " - Selecione una opcion para juegar (" ANSI_COLOR_GREEN "Pulse 0 para salir de la configuracion" ANSI_COLOR_RESET "): ";
			cin >> gpuOpc;		// Entrada de texto por teclado.
			if (gpuOpc != 0 && !(isGpu = gpuOpc > 0 && gpuOpc <= deviceCount + 1)) {
				ERROR_MSS("Error opcion de juego introducida no es valida.");
			}
		} while (gpuOpc != 0 && !isPlay);
		
	}
}
