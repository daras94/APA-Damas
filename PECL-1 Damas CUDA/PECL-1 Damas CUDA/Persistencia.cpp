#include "Persistencia.h"


void guardarPartida(long *tablero, int filas, int columnas) {
	ofstream doc;
	doc.open("partida.txt");
	doc << filas << "\n";
	doc << columnas << "\n";
	for (int i = 0; i < filas * columnas; i++) {
		doc << tablero[i] << " ";
	}
	doc.close();
	system("cls");
	cout << "Guardado correctamente.\n\n";
}

long *cargarPartida() {
	const string fichero = "partida.txt";
	ifstream leer;
	leer.open(fichero.c_str());
	long  d, *tablero, *tAux;
	int i = 0;
	int n = 48;
	int f = 0;
	int c = 0;
	char fila[80];
	if (!leer.fail()) {
		leer.getline(fila, 80, '\n');
		while (n > 47 && n < 58) {
			n = (int)fila[i];
			i++;
			if (n > 47 && n < 58) {
				f = f * 10 + (n - 48);
			}
		}

	}
	n = 48;
	i = 0;
	if (!leer.fail()) {
		leer.getline(fila, 80, '\n');
		while (n > 47 && n < 58) {
			n = (int)fila[i];
			i++;
			if (n > 47 && n < 58) {
				c = c * 10 + (n - 48);
			}
		}

	}
	if (!leer.fail()) {
		leer.getline(fila, 80, '\n');
		d = (int)fila[0] - 48;
	}


	tablero = new long[f*c];
	tAux = new long[f*c];
	for (int i = 0; i < f * c; i++) {
		leer.getline(fila, 80, ' ');
		tablero[i] = (int)fila[0] - 48;
	}
	leer.close();
	return tablero;
	//modoManual(tablero, f, c, d);
}