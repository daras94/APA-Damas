
#include "main.cuh"


int main() {
	int opc; // Es el valor de la opcion a usar 
	do {
		cout << "/**********************************************************************************/" << endl;
		cout << "/*							---> {   MENU: Damas CUDA   } <---					   */" << endl;
		cout << "/**********************************************************************************/" << endl;
		cout << "/*		1 - Extraer y Configurar la partida en funcion de la GPU Nvidia			   */" << endl;
		cout << "/**********************************************************************************/" << endl;
		cout << " - Selecione una opcion de juego (Pulse 0 para salir): " << endl;
		cin >> opc;	// Entrada de texto por teclado.
		switch (opc) {
			case 1:

				break;
			default:
				ERROR_MSS("Error opcion de juego introducida no es valida.");
				break;
		}
		system("cls");
	} while (opc != 0);
	cout << " Has Salido del juego Amigo espero que te gute CUDA ya que este juego lo patrocinan CUDA y Nvidia !!" << endl;
	return EXIT_SUCCESS;
}
