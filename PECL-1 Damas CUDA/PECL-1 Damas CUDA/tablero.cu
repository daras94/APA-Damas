#include "tablero.cuh"

/*
	Recupera las carrasteristicas nesesarias para realizar la configuracion del tablero.

	- devian	 = recibe un entero con la el id de la GPU que se va a usar para realizar la configuracion.
	- deviceProp = recibe un struct para almacenar las carrasteristicas de la GPU.
*/
info_gpu getCofigPlay(int devian, cudaDeviceProp *deviceProp) {
	info_gpu myConfGpu;
	cudaSetDevice(devian);		// Establecemos con que GPU queremos realizar la configuracion en funcion de disponer.
	cudaGetDeviceProperties(deviceProp, devian);
	myConfGpu->numThreadMaxPerSM = deviceProp->maxThreadsPerMultiProcessor;
	myConfGpu->numThreadMasPerBlock = deviceProp->maxThreadsPerBlock;
	myConfGpu->numRegPerBlock = deviceProp->regsPerBlock;
	myConfGpu->sharedMemPerBlock = deviceProp->sharedMemPerBlock;
	for (int i = 0; i < NUM_DIMENSION; i++) {
		myConfGpu->maxDimGridSize[i] = deviceProp->maxGridSize[i];
		myConfGpu->maxDimThreadBlock[i] = deviceProp->maxThreadsDim[i];
	}
	return myConfGpu;
}

void setGpuForPlay(cudaDeviceProp *devProp) {
	int deviceCount = 0, gpuOpc;		// Lo utilzamos para contar el numero de GPUs disponibles.	
	cudaError_t error_code = cudaGetDeviceCount(&deviceCount);
	if (error_code != cudaSuccess) {
		string errorMss = "La GPU devolvio el error " ANSI_COLOR_GREEN + to_string(error_code) + ANSI_COLOR_RESET + ": \n\t -> " + cudaGetErrorString(error_code);
		ERROR_MSS(errorMss); // Retornamos un mensaje de error.	
		system("pause");
	}
	else {
		bool isPlay = false;
		do {
			system("cls");
			cout << "/***************************************************************************************/" << endl;
			cout << "/*  +--> " << ANSI_COLOR_CYAN "Menu de configuracion de partida:" ANSI_COLOR_RESET << setw(47) << "*/" << endl;
			cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
			cout << "/*" << setw(87) << "*/" << endl;
			if (!isPlay) {
				for (int i = 0; i < deviceCount; i++) {
					cudaGetDeviceProperties(devProp, i);
					string modelGPU = devProp->name;
					cout << "/*  " ANSI_COLOR_MAGENTA "GPU " << (i + 1) << ANSI_COLOR_RESET ") - " << modelGPU << setw(76 - modelGPU.length()) << "*/" << endl;
					cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
					getCarrasteristicForGPU(devProp);
					cout << "/*" << setw(87) << "*/" << endl;
				}
				cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
				cout << "/*  - " << ANSI_COLOR_RED "AVISO: " ANSI_COLOR_RESET "Selecione una GPU para jugar." << setw(47) << "*/" << endl;
			} else {
				info_gpu myConfGpu = getCofigPlay(gpuOpc, devProp); // hay un error casca tengo que mirrarlo.
				// ME He quedado trabajando aqui falta la parte de codigo de las tres opciones de configuracion de codigo pero me estoy sobando y es hora de dormir ya.
			}
			cout << "/***************************************************************************************/" << endl;
			cout << " - Selecione una opcion para juegar (" ANSI_COLOR_GREEN "Pulse 0 para salir de la configuracion" ANSI_COLOR_RESET "): ";
			cin >> gpuOpc;		// Entrada de texto por teclado.
			isPlay = (gpuOpc > 0 && gpuOpc <= deviceCount + 1) ? true : false;
			if (gpuOpc != 0) {
				ERROR_MSS("Error opcion de juego introducida no es valida.");
			}
		} while (gpuOpc != 0 && !isPlay);

	}
}