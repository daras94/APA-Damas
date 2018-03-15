#include "GPUCarasteristic.h"

void getCarrasteristicForGPU(cudaDeviceProp *devProp) {
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);	 // Recuperamos la version del driver de CUDA.
	cudaRuntimeGetVersion(&runtimeVersion);  // Recuperamos la version de trabajo de CUDA.
	string globalMem    = ANSI_COLOR_YELLOW + to_string(devProp -> totalGlobalMem/1048576.0f) + ANSI_COLOR_RESET + " MBytes";
	string memConst     = ANSI_COLOR_YELLOW + to_string(devProp -> totalConstMem) + ANSI_COLOR_RESET + " Bytes";
	string memShared    = ANSI_COLOR_YELLOW + to_string(devProp -> sharedMemPerBlock) + ANSI_COLOR_RESET + " Bytes";
	string numReg       = ANSI_COLOR_YELLOW + to_string(devProp -> regsPerBlock) + ANSI_COLOR_RESET + " Registros";
	string numThreadSM  = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsPerMultiProcessor) + ANSI_COLOR_RESET + " Thread";
	string numThreadBL  = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsPerBlock) + ANSI_COLOR_RESET + " Thread";
	string dimThreadBLX = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsDim[0]) + ANSI_COLOR_RESET;
	string dimThreadBLY = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsDim[1]) + ANSI_COLOR_RESET;
	string dimThreadBLZ = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsDim[2]) + ANSI_COLOR_RESET;
	string dimThreadBL  = "(" + dimThreadBLX + ", " + dimThreadBLY + ", " + dimThreadBLZ + ")";
	string dimGridX     = ANSI_COLOR_YELLOW + to_string(devProp ->maxGridSize[0]) + ANSI_COLOR_RESET;
	string dimGridY     = ANSI_COLOR_YELLOW + to_string(devProp -> maxGridSize[1]) + ANSI_COLOR_RESET;
	string dimGridZ     = ANSI_COLOR_YELLOW + to_string(devProp -> maxGridSize[2]) + ANSI_COLOR_RESET;
	string dimGrid      = "(" + dimGridX + ", " + dimGridY + ", " + dimGridZ + ")";
	cout << "/*  " << " - Total global memory          : " << globalMem   << setw(64 - globalMem.length())   << "*/" << endl;
	cout << "/*  " << " - Total constant memory        : " << memConst    << setw(64 - memConst.length())    << "*/" << endl;
	cout << "/*  " << " - Total shared memory/block    : " << memShared   << setw(64 - memShared.length())   << "*/" << endl;
	cout << "/*  " << " - Num REG/Block                : " << numReg      << setw(64 - numReg.length())      << "*/" << endl;
	cout << "/*  " << " - Max Num threads/SM           : " << numThreadSM << setw(64 - numThreadSM.length()) << "*/" << endl;
	cout << "/*  " << " - Max Num threads/Block        : " << numThreadBL << setw(64 - numThreadBL.length()) << "*/" << endl;
	cout << "/*  " << " - Max dim Thread Block (x,y,z) : " << dimThreadBL << setw(90 - dimThreadBL.length()) << "*/" << endl;
	cout << "/*  " << " - Max dim Thread Grid (x,y,z)  : " << dimGrid     << setw(90 - dimGrid.length())     << "*/" << endl;
}

void echoCarGPUs(int devianCurrent, cudaDeviceProp *devProp) {
	int deviceCont = 0;
	if (!getDevCuda(&deviceCont)) {
		system("cls");
		cout << "/***************************************************************************************/" << endl;
		cout << "/*  +--> " << ANSI_COLOR_CYAN "HW Cuda Disponible:" ANSI_COLOR_RESET   << setw(47) << "*/" << endl;
		cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		for (int i = 0; i < deviceCont; i++) {
			cudaGetDeviceProperties(devProp, i);
			string modelGPU = devProp -> name  + ((string)((i == devianCurrent)? ANSI_COLOR_GREEN " CURRENT!!" ANSI_COLOR_RESET : ""));
			cout << "/*  " ANSI_COLOR_MAGENTA << (i+1) << ANSI_COLOR_RESET ") - " << modelGPU << setw(97 - modelGPU.length()) << "*/" << endl;
			cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
			getCarrasteristicForGPU(devProp);
			cout << "/*" << setw(87) << "*/" << endl;
		}
		cout << "/***************************************************************************************/" << endl;
	}
}

void selectGpuCurrent(cudaDeviceProp *devProp, int *devianCurrent) {
	int opc, deviceCont = 0;
	if (!getDevCuda(&deviceCont)) {
		bool isSelectGPUValide = false;
		do {
			cout << " - Seleccione una GPU (" ANSI_COLOR_GREEN "Pulse 0 para salir sin cambios" ANSI_COLOR_RESET "): ";
			cin >> opc;		// Entrada de texto por teclado.
			isSelectGPUValide = (opc > 0 && opc <= deviceCont + 1);
			if (opc != 0 && !isSelectGPUValide) {
				ERROR_MSS("Error: la opcion de juego introducida no es valida.");
			}
		} while (opc != 0 && !isSelectGPUValide);
		*devianCurrent = opc;
	}
}

void fotterCarGPU(cudaDeviceProp *devProp, int deviceCurren) {
	int deviceCont = 0, driverVersion = 0, runtimeVersion = 0;
	if (!getDevCuda(&deviceCont)) {
		if (&deviceCont != 0) {
			cudaDriverGetVersion(&driverVersion);	 // Recuperamos la version del driver de CUDA.
			cudaRuntimeGetVersion(&runtimeVersion);  // Recuperamos la version de trabajo de CUDA.
			string modelGPU = ANSI_COLOR_YELLOW + ((string)devProp -> name) + ANSI_COLOR_RESET;
			string copCapVer = ANSI_COLOR_YELLOW + to_string(devProp -> major) + "." + to_string(devProp -> minor) + ANSI_COLOR_RESET;
			string driverVer = ANSI_COLOR_YELLOW + to_string((driverVersion / 1000)) + "." + to_string(((driverVersion % 100) / 10)) + ANSI_COLOR_RESET;
			string runtimVer = ANSI_COLOR_YELLOW + to_string((runtimeVersion / 1000)) + "." + to_string(((runtimeVersion % 100) / 10)) + ANSI_COLOR_RESET;
			cout << "/* +-> " << ANSI_COLOR_CYAN << "HW GPU CUDA Current: " << ANSI_COLOR_RESET << deviceCont << setw(60) << "*/" << endl;
			cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
			cout << "/*  " << " - Modelo GPU: " << modelGPU << setw(70 - modelGPU.length()) << " - CUDA Driver Ver : " << driverVer << setw(26 - driverVer.length()) << "*/" << endl;
			cout << "/*  " << " - Capability: " << copCapVer << setw(70 - copCapVer.length()) << " - CUDA Runtime Ver: " << runtimVer << setw(26 - runtimVer.length()) << "*/" << endl;
		}
		else {
			string error = "Su host no dipone de un dispositivo preparado para soportar CUDA.";
			cout << "/*" << setw(5) << ANSI_COLOR_RED " - ERROR: " << ANSI_COLOR_RESET << error << setw(65 - error.length()) << "*/" << endl;
		}
		cout << "/***************************************************************************************/" << endl;
	}
}

bool getDevCuda(int *deviceCont) {
	cudaError_t error_code = cudaGetDeviceCount(deviceCont);
	bool  isCudaError = (error_code != cudaSuccess);
	if (isCudaError) {
		string errorMss = "La GPU devolvio el error " ANSI_COLOR_GREEN + to_string(error_code) + ANSI_COLOR_RESET + ": \n\t -> " + cudaGetErrorString(error_code);
		ERROR_MSS(errorMss); // Retornamos un mensaje de error.	
		system("pause");
	}
	return isCudaError;
}

