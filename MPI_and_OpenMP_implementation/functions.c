
#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <time.h>


//init grid
void Create_Grid(int dimension_x, int dimension_y, float ***grid){
	srand(time(NULL));

	int i, j;
	float *p = (float*)malloc(dimension_x * dimension_y * sizeof(float));
	*(grid) = malloc(dimension_x *sizeof(int* ));
	
	for(i=0; i < dimension_x; i++)
		(*grid)[i] = &(p[i * dimension_y]);
	
}



//init grid with random values
void Init_RandomGrid(int dimension_x, int dimension_y, float ***grid){
	int i,j;
	srand(time(NULL));
	for(i=0; i<dimension_x; i++){
		for(j=0; j<dimension_y; j++){
			((*grid)[i])[j] = (float)(rand()%256);
		}
	}
}

/*copies image to grid 
 *start and adder are used in case that the image is colorfull -- so we store the 3d value and we start
 from the 'start'

 *if part > 0 the -p flag was given -- if part < 1 we take values according to the step
 
 * for example if we have the array 0  1  2  3  with part = 0.5 the produced array will be 	0  2 
 									4  5  6  7										  		8  10
 									8  9  10 11
 									12 13 14 15

 *if  part > 1 we store multiple times the values of each element according with part' value
 *for example if we have the array 	0 1 and part = 2 the produced array will be	0 0 1 1
									2 3											0 0 1 1
																				2 2 3 3
																				2 2 3 3 
 
 *if we have a both colorfull immage and a requested partition we fill the produced arraies with the
  appropriate values

 */
void Init_ImageGrid(int dimension_x, int dimension_y, float ***grid, char **image, int start, int adder, float part){
	int  i, j;
	if(part > 0){
		int step;
		if(part < 1){
			float step_f = 1/part;
			step = (int)step_f;
			int i_2 = 0, j_2 = 0;
			for(i = 0; i < dimension_x ; i = i + step){
				j_2 = 0;
				for(j = start; j < adder * dimension_y; j = j + (step*adder)){
					((*grid)[i_2])[j_2] = (float)(image)[i][j];
					j_2++;
				}
				i_2++;
			}
		}
		else{
			step = (int)part;
			int i_2 = 0, j_2 = 0;
			for(i = 0; i < dimension_x ; i++){
				for(j = start; j < adder * dimension_y; j = j + adder){
					for(i_2 = 0; i_2 < step; i_2++){
						for(j_2 = 0; j_2 < step; j_2++){
							((*grid)[i_2 + (step * i)])[j_2 + (step * j)] = (float)(image)[i][j];
						}
					}
				}
			}

		}

	}
	else{
		for(i = 0; i < dimension_x; i++){
			int k = 0;
			for(j = start; j < adder * dimension_y; j = j + adder){
				((*grid)[i])[k] = (float)(image)[i][j];
				k++;
			}
		}
	}	

}



//calculate dimensions of process's local grid 
int  Calculate_SubGrid_dimensions(int dimension_x, int dimension_y, int *local_dimension_x, int* local_dimension_y, int num_of_proc){
	float root= sqrt((float)num_of_proc);
	int i_root= (int)root;
	if(i_root!=root){
		return -1;
	}
	*local_dimension_x = dimension_x/i_root;
	*local_dimension_y = dimension_y/i_root;

	if(dimension_x%i_root != 0 || dimension_y%i_root != 0)
		return -2;

	return 0;
}



//returns 0 if two grids are identical
int isEqual_Grid(float ***new_grid, float ***grid, int dimension_x, int dimension_y){
	int i,j;
	for(i = 0; i < dimension_x; i++){
		for(j = 0; j < dimension_y; j++){
			if((int)((*grid)[i])[j] != (int)((*new_grid)[i])[j]){
				//printf("%.d %.d\n",(int)((*grid)[i])[j], (int)((*new_grid)[i])[j] );
				return 1;
			}
		}
	}
	
	return 0;
}


//destroy grid
void Destroy_Grid(float ***grid){
	free(&((*grid)[0][0]));
	free(*grid);
}



void print_Grid(float ***grid, int dimension_x, int dimension_y){
	printf("PRINT %d X %d:\n\n",dimension_x, dimension_y );
	int i,j;
	for(i=0; i<dimension_x; i++){
		 putchar('|');
		for(j=0; j<dimension_y; j++){
			printf("%.2f ",((*grid)[i])[j]);
		}
		printf("|\n");
	}
	printf("\nEND of PRINT\n");
}
