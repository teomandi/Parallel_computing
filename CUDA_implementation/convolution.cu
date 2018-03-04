#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <timer.h>

#include "convolution.h"

float convolution(float ***global, int dimension_x, int dimension_y, float filter[FILTER_SIZE][FILTER_SIZE], int *rep){

	int i, j, size, stop=1;
	float **grid_gpu, **new_grid_gpu, **new_grid;

	size = dimension_y * dimension_x * sizeof(float);

	Create_Grid(dimension_x, dimension_y,&new_grid);
	cudaMalloc((void **) &grid_gpu, size);
	cudaMalloc((void **) &new_grid_gpu, size);
	cudaMemcpy(&grid_gpu, global, size, cudaMemcpyHostToDevice);

	/* Kernel invocation */
	dim3 dimBlock(16, 16);
	dim3 dimGrid;
	dimGrid.x = (dimension_x + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (dimension_y + dimBlock.y - 1) / dimBlock.y;
	double start, end; 
	j=0;
	GET_TIME(start);
	while(stop != 0  &&  j < 100){

		Calcutalate_AllCells_Kernel<<<dimGrid, dimBlock>>>(&grid_gpu, &new_grid_gpu, dimension_x, dimension_y, filter);  
		if (cudaGetLastError() != cudaSuccess) {
			printf("Kernel launch failed\n");
		}


		cudaMemcpy(&new_grid, &new_grid_gpu, size, cudaMemcpyDeviceToHost);



		/*check if there is changes in the grids -- Every process check if there was any changes 
		 *in its grid and sends (throught reduce) 0 if there was not or 1 if there was
		 *-- reduce sums up all the values broadcast it to anybody 
		 *-- if everyone sent 0 the broadcasting sum will be 0 so that means that no
		 *changes took place in any process's grid so the convolution has ended */
		if( (j+1)%10 == 0){
			int result = isEqual_Grid(&new_grid, global, dimension_x, dimension_y); 
			if(result)	stop = -1; //terminate.
		}
		//swaping new with old grids				
		float **temp = new_grid;
		new_grid = (*global);
		(*global) = temp;
		j++;

		cudaMemcpy(&grid_gpu, global, size, cudaMemcpyHostToDevice);
	}
	GET_TIME(end);
	(*rep) = j;
	cudaFree(grid_gpu);
	cudaFree(new_grid_gpu);
    Destroy_Grid(&new_grid);
    return end - start ;
}





static inline void Calcutalate_AllCells_(float ***grid, float ***new_grid, int dimension_x, int dimension_y, float filter[FILTER_SIZE][FILTER_SIZE]){
	
	/* The variables below are used to iterate the grid */
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > dimension_x || j > dimension_y) 	return;
	if(i==0 && j==0){ //1) panw aristera gwnia
		((*new_grid)[i])[j]=  
		((*grid)[i+1])[j+1]*filter[0][0]+ ((*grid)[i+1])[j]*filter[0][1]+ ((*grid)[i])[j]*filter[0][2]+
		((*grid)[i])[j+1]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j]*filter[1][2]+
		((*grid)[i])[j]*filter[2][0]+ ((*grid)[i])[j]*filter[2][1]+ ((*grid)[i])[j]*filter[2][2];
	}
	else if(i==dimension_x-1 && j==0){ //2) panw de3ia gwnia
		((*new_grid)[i])[j]=  
		((*grid)[i])[j]*filter[0][0]+ ((*grid)[i])[j]*filter[0][1]+ ((*grid)[i])[j]*filter[0][2]+
		((*grid)[i])[j+1]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j]*filter[1][2]+
		((*grid)[i-1])[j+1]*filter[2][0]+ ((*grid)[i-1])[j]*filter[2][1]+ ((*grid)[i])[j]*filter[2][2];
	}
	else if(i==0 && j==dimension_y-1){ //3) katw aristera gwnia
		((*new_grid)[i])[j]=  
		((*grid)[i])[j]*filter[0][0]+ ((*grid)[i+1])[j]*filter[0][1]+ ((*grid)[i+1])[j-1]*filter[0][2]+
		((*grid)[i])[j]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j-1]*filter[1][2]+
		((*grid)[i])[j]*filter[2][0]+ ((*grid)[i])[j]*filter[2][1]+ ((*grid)[i])[j]*filter[2][2]; 
	}
	else if(i==dimension_x-1 && j=dimension_y-1){ //4) katw de3ia gwnia
		((*new_grid)[i])[j]=  
		((*grid)[i])[j]*filter[0][0]+ ((*grid)[i])[j]*filter[0][1]+ ((*grid)[i])[j]*filter[0][2]+
		((*grid)[i])[j]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j-1]*filter[1][2]+
		((*grid)[i])[j]*filter[2][0]+ ((*grid)[i-1])[j]*filter[2][1]+ ((*grid)[i-1])[j-1]*filter[2][2]; 
	}
	else if(i==0){ // 5)aristeri pleura
		((*new_grid)[i])[j]=  
		((*grid)[i+1])[j+1]*filter[0][0]+ ((*grid)[i+1])[j]*filter[0][1]+ ((*grid)[i+1])[j-1]*filter[0][2]+
		((*grid)[i])[j+1]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j-1]*filter[1][2]+
		((*grid)[i])[j]*filter[2][0]+ ((*grid)[i])[j]*filter[2][1]+ ((*grid)[i])[j]*filter[2][2];
	}
	else if(j==0){ // 6)panw pleura
		((*new_grid)[i])[j]=  
		((*grid)[i+1])[j+1]*filter[0][0]+ ((*grid)[i+1])[j]*filter[0][1]+ ((*grid)[i])[j]*filter[0][2]+
		((*grid)[i])[j+1]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j]*filter[1][2]+
		((*grid)[i-1])[j+1]*filter[2][0]+ ((*grid)[i-1])[j]*filter[2][1]+ ((*grid)[i])[j]*filter[2][2]; 
	}
	else if(i==dimension_x-1){ //7) katw pleura
		((*new_grid)[i])[j]=  
		((*grid)[i])[j]*filter[0][0]+ ((*grid)[i+1])[j]*filter[0][1]+ ((*grid)[i+1])[j-1]*filter[0][2]+
		((*grid)[i])[j]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j-1]*filter[1][2]+
		((*grid)[i])[j]*filter[2][0]+ ((*grid)[i-1])[j]*filter[2][1]+ ((*grid)[i-1])[j-1]*filter[2][2]; 
	}
	else if(j==dimension_y-1){
		((*new_grid)[i])[j]=  
		((*grid)[i])[j]*filter[0][0]+ ((*grid)[i])[j]*filter[0][1]+ ((*grid)[i])[j]*filter[0][2]+
		((*grid)[i])[j+1]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j-1]*filter[1][2]+
		((*grid)[i-1])[j+1]*filter[2][0]+ ((*grid)[i-1])[j]*filter[2][1]+ ((*grid)[i-1])[j-1]*filter[2][2]; 
	}
	else{ // ola ta alla
		((*new_grid)[i])[j]=  
		((*grid)[i+1])[j+1]*filter[0][0]+ ((*grid)[i+1])[j]*filter[0][1]+ ((*grid)[i+1])[j-1]*filter[0][2]+
		((*grid)[i])[j+1]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j-1]*filter[1][2]+
		((*grid)[i-1])[j+1]*filter[2][0]+ ((*grid)[i-1])[j]*filter[2][1]+ ((*grid)[i-1])[j-1]*filter[2][2]; 
	}
}