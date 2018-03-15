#include "tableroGUI.cuh"

__global__ void kernelGUI(unsigned char *imagen,double numThread){

	int GPU_COLOR_RED[3] = { 255,0,0 };
	int GPU_COLOR_GREEN[3] = { 0,255,0 };
	int GPU_COLOR_YELLOW[3] = { 255,255,0 };
	int GPU_COLOR_BLUE[3] = { 0,0,255 };
	int GPU_COLOR_MAGENTA[3] = { 255,0,255 };
	int GPU_COLOR_CYAN[3] = { 0,255,255 };
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// coordenada global de cada pixel
	int pixel = x + y * blockDim.x * gridDim.x;
	__syncthreads();
	if ((blockIdx.x + blockIdx.y) % 2 == 0) {
		switch (threadIdx.x) {
		case 1:
			imagen[pixel * 4 + 0] = GPU_COLOR_RED[0] /* x / (blockDim.x * gridDim.x) */; // canal R
			imagen[pixel * 4 + 1] = GPU_COLOR_RED[1] /* y / (blockDim.y * gridDim.y) */; // canal G
			imagen[pixel * 4 + 2] = GPU_COLOR_RED[2] /* blockIdx.x + 2 * blockIdx.y */; // canal B
			break;
		case 2:
			imagen[pixel * 4 + 3] = GPU_COLOR_GREEN[0] /* x / (blockDim.x * gridDim.x) */; // canal R
			imagen[pixel * 4 + 4] = GPU_COLOR_GREEN[1] /* y / (blockDim.y * gridDim.y) */; // canal G
			imagen[pixel * 4 + 5] = GPU_COLOR_GREEN[2] /* blockIdx.x + 2 * blockIdx.y */; // canal B
			break;
		case 3:
			imagen[pixel * 4 + 6] = GPU_COLOR_YELLOW[0] /* x / (blockDim.x * gridDim.x) */; // canal R
			imagen[pixel * 4 + 7] = GPU_COLOR_YELLOW[1] /* y / (blockDim.y * gridDim.y) */; // canal G
			imagen[pixel * 4 + 8] = GPU_COLOR_YELLOW[2] /* blockIdx.x + 2 * blockIdx.y */; // canal B
			break;
		case 4:
			imagen[pixel * 4 + 9] = GPU_COLOR_BLUE[0] /* x / (blockDim.x * gridDim.x) */; // canal R
			imagen[pixel * 4 + 10] = GPU_COLOR_BLUE[1] /* y / (blockDim.y * gridDim.y) */; // canal G
			imagen[pixel * 4 + 11] = GPU_COLOR_BLUE[2] /* blockIdx.x + 2 * blockIdx.y */; // canal B
			break;
		case 5:
			imagen[pixel * 4 + 12] = GPU_COLOR_MAGENTA[0] /* x / (blockDim.x * gridDim.x) */; // canal R
			imagen[pixel * 4 + 13] = GPU_COLOR_MAGENTA[1] /* y / (blockDim.y * gridDim.y) */; // canal G
			imagen[pixel * 4 + 14] = GPU_COLOR_MAGENTA[2] /* blockIdx.x + 2 * blockIdx.y */; // canal B
			break;
		case 6:
			imagen[pixel * 4 + 15] = GPU_COLOR_CYAN[0] /* x / (blockDim.x * gridDim.x) */; // canal R
			imagen[pixel * 4 + 16] = GPU_COLOR_CYAN[1] /* y / (blockDim.y * gridDim.y) */; // canal G
			imagen[pixel * 4 + 17] = GPU_COLOR_CYAN[2] /* blockIdx.x + 2 * blockIdx.y */; // canal B
			break;
		default:
			imagen[pixel * 4 + 0] = 255 /* x / (blockDim.x * gridDim.x) */; // canal R
			imagen[pixel * 4 + 1] = 255 /* y / (blockDim.y * gridDim.y) */; // canal G
			imagen[pixel * 4 + 2] = 255 /* blockIdx.x + 2 * blockIdx.y */; // canal B
			break;
		}
	}else{
		imagen[pixel * 4 + 0] = 255 /* x / (blockDim.x * gridDim.x) */; // canal R
		imagen[pixel * 4 + 1] = 255 /* y / (blockDim.y * gridDim.y) */; // canal G
		imagen[pixel * 4 + 2] = 255 /* blockIdx.x + 2 * blockIdx.y */; // canal B
	}
	imagen[pixel * 4 + 3] = 255; // canal alfa
	__syncthreads();
}

void iniciarInterfaz(double numThreads, long *tablero){
	// declaracion del bitmap
	CPUBitmap bitmap(numThreads,numThreads);
	// tamaño en bytes
	size_t size = bitmap.image_size();
	// reserva en el host
	unsigned char *host_bitmap = bitmap.get_ptr();
	// reserva en el device
	unsigned char *dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, size);
	// generamos el bitmap
	dim3 dimGrid(numThreads/TAM_TESELA, numThreads/TAM_TESELA);
	dim3 dimBlock(TAM_TESELA,TAM_TESELA);
	kernelGUI << <dimGrid, dimBlock >> >(dev_bitmap,numThreads);
	// recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy(host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost);
	// liberacion de recursos
	cudaFree(dev_bitmap);
	// visualizacion y salida
	printf("\n...pulsa ESC para finalizar...");
	bitmap.display_and_exit();
}