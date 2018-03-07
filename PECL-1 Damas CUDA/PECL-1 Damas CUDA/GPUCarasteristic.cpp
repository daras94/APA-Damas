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
