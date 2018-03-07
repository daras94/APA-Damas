#include "main.cuh"

/*
	El main chico punto de arraque de todo.
*/
int main() {
	SetConsoleTitle("Damas BOM for CUDA");
	int opc, dev;						// Es el valor de la opcion a usar 
	cudaDeviceProp devProp;				// struct de carrasteristicas de la GPU.
	do {
		system("cls");					// Limpiamos el pront
		cout << "/***************************************************************************************/" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		cout << "/*" << setw(35) << " ---> { " << ANSI_COLOR_CYAN " MENU: Damas CUDA " ANSI_COLOR_RESET << " } <--- " << setw(26) << "*/" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		cout << "/***************************************************************************************/" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "1" ANSI_COLOR_RESET ") - Extraer y Configurar la partida en funcion de la GPU Nvidia." << setw(20) << "*/" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		cout << "/***************************************************************************************/" << endl;
		fotterCarGPU();
		cout << " - Selecione una opcion de juego (" ANSI_COLOR_GREEN "Pulse 0 para salir" ANSI_COLOR_RESET "): ";
		cin >> opc;						// Entrada de texto por teclado.
		system("cls");					// Limpiamos el pront
		switch (opc) {
			case 1:
				setGpuForPlay(&devProp);
				break;
			default:
				if (opc != 0) {
					ERROR_MSS("Error opcion de juego introducida no es valida.");
				}
				break;
		}
	} while (opc != 0);
	cout << ANSI_COLOR_GREEN  "\n - Has Salido del juego Amigo espero que te guste CUDA ya que este juego lo patrocinan CUDA y Nvidia !!" ANSI_COLOR_RESET << endl;
	Sleep(1000);
	return EXIT_SUCCESS;
}

/*
	Metodo que imprime las carasteristicas principales de la GPU o que lanza 
	un error si no se dispone del harwared apropiado.
*/
void fotterCarGPU() {
	int numDevice = 0, driverVersion = 0, runtimeVersion = 0;
	cudaError_t error_code; 
	if (( error_code = cudaGetDeviceCount(&numDevice)) != cudaSuccess) {
		string errorMss = "La GPU devolvio el error " ANSI_COLOR_GREEN + to_string(error_code) + ANSI_COLOR_RESET + ": \n\t -> " + cudaGetErrorString(error_code);
		ERROR_MSS(errorMss); // Retornamos un mensaje de error.	
		system("pause");
		exit(EXIT_FAILURE);  // Cerramos el programa devido a error.
	}
	else {
		if (numDevice != 0) {
			cudaDeviceProp devProp;
			for (int i = 0; i < numDevice; i++) {
				cudaGetDeviceProperties(&devProp, i);	 // Recuperamos las carrasteristicas de la GPU.
				cudaDriverGetVersion(&driverVersion);	 // Recuperamos la version del driver de CUDA.
				cudaRuntimeGetVersion(&runtimeVersion);  // Recuperamos la version de trabajo de CUDA.
				string modelGPU  = ANSI_COLOR_YELLOW + ((string)devProp.name) + ANSI_COLOR_RESET;
				string copCapVer = ANSI_COLOR_YELLOW + to_string(devProp.major) + "." + to_string(devProp.minor) + ANSI_COLOR_RESET;
				string driverVer = ANSI_COLOR_YELLOW + to_string((driverVersion / 1000)) + "." + to_string(((driverVersion % 100) / 10)) + ANSI_COLOR_RESET;
				string runtimVer = ANSI_COLOR_YELLOW + to_string((runtimeVersion / 1000)) + "." + to_string(((runtimeVersion % 100) / 10)) + ANSI_COLOR_RESET;
				cout << "/* +-> " << ANSI_COLOR_CYAN << "Carasteristicas GPU: " << ANSI_COLOR_RESET << numDevice << setw(60) << "*/" << endl;
				cout << "/*  ---------------------------------------------------------------------------------  */" << endl;
				cout << "/*  " << " - Modelo GPU: " << modelGPU  << setw(70 - modelGPU.length())  << " - CUDA Driver Ver : " << driverVer << setw(26 - driverVer.length()) << "*/" << endl;
				cout << "/*  " << " - Capability: " << copCapVer << setw(70 - copCapVer.length()) << " - CUDA Runtime Ver: " << runtimVer << setw(26 - runtimVer.length()) << "*/" << endl;
				cout << "/*" << setw(87) << "*/" << endl;
			}
		}
		else {
			string error = "Su Host no dipone de un dispositivo preparado para soportar CUDA.";
			cout << "/*" << setw(5) << ANSI_COLOR_RED " - ERROR: " << ANSI_COLOR_RESET << error << setw(65 - error.length()) << "*/" << endl;
		}
	}
	cout << "/***************************************************************************************/" << endl;
}

