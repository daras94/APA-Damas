#include "main.cuh"

/*
	El main chico punto de arraque de todo.
*/
int main() {
	SetConsoleTitle("Damas BOM for CUDA");
	getCofigPlay(selectGPU, &devProp, &infoMyGPU);						// obtenemos los parametros de la gpu para realizar la configuracion de la gpu.
	int opc,  dificultad;										        // Es el valor de la opcion a usar y la dificultad de juego. 
	double numThread;													// El numero de hilos totales de la matriz escogida.
	long *gui_tablero;
	do {
		system("cls");					// Limpiamos el prompt
		cout << "/***************************************************************************************/" << endl;
		cout << "/*" << setw(35) << " ---> { " << ANSI_COLOR_CYAN " MENU: Damas CUDA " ANSI_COLOR_RESET << " } <--- " << setw(26) << "*/" << endl;
		cout << "/***************************************************************************************/" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "1" ANSI_COLOR_RESET ") - Iniciar partida y configurar tablero en funcion del HW GPU (" ANSI_COLOR_GREEN "MEM SHARED"  ANSI_COLOR_RESET ")." << setw(8) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "2" ANSI_COLOR_RESET ") - Iniciar partida y configurar tablero en funcion del HW GPU (" ANSI_COLOR_GREEN "MULTY BLOCK" ANSI_COLOR_RESET ")." << setw(9) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "3" ANSI_COLOR_RESET ") - Iniciar partida y configurar tablero en funcion del HW GPU (" ANSI_COLOR_GREEN "MEM GLOBAL"  ANSI_COLOR_RESET ")." << setw(8) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "4" ANSI_COLOR_RESET ") - Iniciar paratida establecer configuracion de tablero de forma manual."     << setw(11) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "5" ANSI_COLOR_RESET ") - Iniciar partida damas interfaces grafica."							       << setw(39) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "6" ANSI_COLOR_RESET ") - Ver Carateristicas del Hardware de que dispones."					       << setw(30) << "*/" << endl;
		cout << "/*  " ANSI_COLOR_MAGENTA "7" ANSI_COLOR_RESET ") - Selecionar configuracion de otra GPU disponible."                          << setw(45) << "*/" << endl;
		cout << "/*" << setw(87) << "*/" << endl;
		cout << "/***************************************************************************************/" << endl;
		fotterCarGPU(&devProp, selectGPU);
		cout << " - Selecione una opcion de juego (" ANSI_COLOR_GREEN "Pulse 0 para salir" ANSI_COLOR_RESET "): ";
		cin >> opc;						// Entrada de texto por teclado.
		switch (opc) {
			case 1:
			case 2:
			case 3:
			case 4:
				// Llamamos a establecer la configuracion de la partida con la GPU.
				if (opc >= 1 && opc <= 3) {	
					numThread = setGpuForPlayAuto(&devProp, &infoMyGPU, selectGPU);	// Modo Automatico.
				} else {
					numThread = setGpuForPlayManual(&devProp, &infoMyGPU, selectGPU);	// Modo MAnual.
					cout << "/***************************************************************************************/" << endl;
					cout << " - Selecione un kernel ( Mem-Shared = 1, Block = 2 y Mem & Block = 3): ";
					cin >> opc;
				}
				dificultad = setDificultad();
				if (dificultad != 0) {
					playDamas(opc, numThread, &infoMyGPU, dificultad);
				}
				break;
			case 5:
				numThread = setGpuForPlayAuto(&devProp, &infoMyGPU, selectGPU);	// Modo Automatico.
				dificultad = setDificultad();
				gui_tablero = generarTablero(numThread, dificultad);
				iniciarInterfaz(numThread, gui_tablero);
				break;
			case 6: 
			case 7:
				echoCarGPUs(selectGPU, &devProp);
				if (opc == 7) {
					cudaDeviceProp devShow;								// struct de carasteristicas de la GPU.
					selectGpuCurrent(&devShow, &selectGPU);
					getCofigPlay(selectGPU, &devProp, &infoMyGPU);		// Obtenemos los parametros de la gpu para realizar la configuracion de la gpu.
				} else {
					system("pause");
				}
				break;
			default:
				if (opc != 0) {
					ERROR_MSS("Error opcion de juego introducida no es valida.");
				}
				break;
		}
	} while (opc != 0);
	cout << ANSI_COLOR_GREEN  "\n - Has salido del juego amigo espero que te guste CUDA ya que este juego lo patrocinan CUDA y Nvidia !!" ANSI_COLOR_RESET << endl;
	//Sleep(1000);
	return EXIT_SUCCESS;
}