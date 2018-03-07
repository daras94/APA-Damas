#include "GPUCarasteristic.h"

void getCarrasteristicForGPU(cudaDeviceProp *devProp) {
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);	 // Recuperamos la version del driver de CUDA.
	cudaRuntimeGetVersion(&runtimeVersion);  // Recuperamos la version de trabajo de CUDA.
	string copCapVer = ANSI_COLOR_YELLOW + to_string(devProp -> major) + "." + to_string(devProp -> minor) + ANSI_COLOR_RESET;
	string driverVer = ANSI_COLOR_YELLOW + to_string((driverVersion / 1000)) + "." + to_string(((driverVersion % 100) / 10)) + ANSI_COLOR_RESET;
	string runtimVer = ANSI_COLOR_YELLOW + to_string((runtimeVersion / 1000)) + "." + to_string(((runtimeVersion % 100) / 10)) + ANSI_COLOR_RESET;
	string numSM     = ANSI_COLOR_YELLOW + to_string(devProp -> multiProcessorCount) + ANSI_COLOR_RESET;
	string globalMem = ANSI_COLOR_YELLOW + to_string((double)devProp -> totalGlobalMem / 1048576.0f) + ANSI_COLOR_RESET + " MB";
	string cpuClock  = ANSI_COLOR_YELLOW + to_string((double)devProp -> clockRate * 1e-3f) + ANSI_COLOR_RESET + " GHz";
	string memClock  = ANSI_COLOR_YELLOW + to_string((double)devProp -> memoryClockRate * 1e-3f) + ANSI_COLOR_RESET + " GHz";
	string memBus    = ANSI_COLOR_YELLOW + to_string(devProp -> memoryBusWidth) + ANSI_COLOR_RESET + " bit";
	string memConst  = ANSI_COLOR_YELLOW + to_string(devProp -> totalConstMem) + ANSI_COLOR_RESET + " B";
	string memShared = ANSI_COLOR_YELLOW + to_string(devProp -> sharedMemPerBlock) + ANSI_COLOR_RESET + " B";
	string numReg    = ANSI_COLOR_YELLOW + to_string(devProp -> regsPerBlock) + ANSI_COLOR_RESET;
	string warpSize  = ANSI_COLOR_YELLOW + to_string(devProp -> warpSize) + ANSI_COLOR_RESET;
	string numThreadSM = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsPerMultiProcessor) + ANSI_COLOR_RESET;
	string numThreadBL = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsPerBlock) + ANSI_COLOR_RESET;
	string dimThreadBLX = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsDim[0]) + ANSI_COLOR_RESET;
	string dimThreadBLY = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsDim[1]) + ANSI_COLOR_RESET;
	string dimThreadBLZ = ANSI_COLOR_YELLOW + to_string(devProp -> maxThreadsDim[2]) + ANSI_COLOR_RESET;
	string dimThreadBL = " (" + dimThreadBLX + ", " + dimThreadBLY + ", " + dimThreadBLZ + ")";
	string dimGridX = ANSI_COLOR_YELLOW + to_string(devProp ->maxGridSize[0]) + ANSI_COLOR_RESET;
	string dimGridY = ANSI_COLOR_YELLOW + to_string(devProp -> maxGridSize[1]) + ANSI_COLOR_RESET;
	string dimGridZ = ANSI_COLOR_YELLOW + to_string(devProp -> maxGridSize[2]) + ANSI_COLOR_RESET;
	string dimGrid = " (" + dimGridX + ", " + dimGridY + ", " + dimGridZ + ")";
	cout << "/*  " << " - Capability: " << copCapVer << setw(60 - copCapVer.length()) << " - CUDA Driver Ver    : " << driverVer << setw(36 - driverVer.length()) << "*/" << endl;
	cout << "/*  " << " - Num Warps : " << warpSize  << setw(60 - warpSize.length())  << " - CUDA Runtime Ver   : " << runtimVer << setw(36 - runtimVer.length()) << "*/" << endl;
	cout << "/*  " << " - Num SM    : " << numSM     << setw(60 - numSM.length())     << " - GPU Max Clock rate : " << cpuClock << setw(36 - cpuClock.length()) << "*/" << endl;
	cout << "/*  " << " - Total global memory: " << globalMem << setw(20 - globalMem.length())       << " - Memory Clock rate  : " << memClock << setw(36 - memClock.length()) << "*/" << endl;
	cout << "/*  " << " - Total constant memory     : " <<  memConst << setw(20 - memConst.length()) << " - Memory Bus Width   : " << memBus << setw(36 - memBus.length()) << "*/" << endl;
	cout << "/*  " << " - Total shared memory/block : " << memShared << setw(20 - memShared.length()) << " - Num REG/Block      : " << numReg << setw(36 - numReg.length()) << "*/" << endl;
	cout << "/*  " << " - Max Num threads/SM        : " << numThreadSM << setw(46 - numThreadSM.length()) << " - Max Num threads/Block: " << numThreadBL << setw(34 - numThreadBL.length()) << "*/" << endl;
	cout << "/*  " << " - Max dim Thread Block (x,y,z): " << dimThreadBL << setw(91 - dimThreadBL.length()) << "*/" << endl;
	cout << "/*  " << " - Max dim Thread Grid (x,y,z) : " << dimGrid << setw(91 - dimGrid.length()) << "*/" << endl;
}
