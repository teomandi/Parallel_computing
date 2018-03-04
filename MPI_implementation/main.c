#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<fcntl.h>


#include "convolution.h"

int main(int argc, char** argv) {
	
	int num_of_proc, rank, i, j;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);

	//grids data and data from command line
	float part = 0;
	char *filename = NULL;
	char *new_filename = NULL;
	int dimension_x = DIMENSION_X, dimension_y= DIMENSION_Y, local_dimension_x, local_dimension_y,  bytes_per_pixel = 1;
	bool random = false, print = false;

	//reading data from command line
	for(i = 0; i < argc; i++){
		if(strcmp(argv[i],"-i")==0){
			filename = malloc(strlen(argv[i+1])+1);
			strncpy(filename, argv[i+1], strlen(argv[i+1])+1);
			i++;
		}
		else if(strcmp(argv[i],"-x")==0){
			dimension_x = atof(argv[i+1]);
			i++;
		}
		else if(strcmp(argv[i],"-y")==0){
			dimension_y = atof(argv[i+1]);
			i++;
		}
		else if(strcmp(argv[i],"-t")==0){
			new_filename = malloc(strlen(argv[i+1]) + strlen(".raw") + 1);
			sprintf(new_filename, "%s%s", argv[i+1], ".raw");			
			i++;
		}
		else if(strcmp(argv[i],"-c")==0){
			//if the picture is colorfull will contain 3 bytes per pixel
 			bytes_per_pixel = 3;
 		}
 		else if(strcmp(argv[i],"-p")==0){
			part = atof(argv[i+1]);
			i++;
		}
		else if(strcmp(argv[i],"-print")==0){
			print =true;
		}

 		else if(strcmp(argv[i],"-r")==0)
			random = true;
	}

	
	/*
	 *initialize grids -- if the picture is colorfull, we create 
	 *3 grids that each one contains the value of a byte of each colour
	 *for instance :	grid[0] contains the first bytes of the pixels
	 *				    grid[1] contains the second bytes of the pixels
	 *			    	grid[2] contains the third bytes of the pixels 
	*/
	float **global[bytes_per_pixel];
	float filter[FILTER_SIZE][FILTER_SIZE] ={
											(float)1/(float)16, (float)2/(float)16, (float)1/(float)16,
   											(float)2/(float)16, (float)4/(float)16, (float)2/(float)16,
   											(float)1/(float)16, (float)2/(float)16, (float)1/(float)16
   									};

  	/*
  	 *in case that not a specific image was given, 
  	 *the programm will use  'waterfall_grey_1920_2520.raw'	
  	 */
   	if(!filename && !random){
		filename = malloc(strlen("../images/waterfall_grey_1920_2520.raw")+1);
		strncpy(filename, "../images/waterfall_grey_1920_2520.raw", strlen("../images/waterfall_grey_1920_2520.raw")+1);
		dimension_x = DIMENSION_X;
		dimension_y= DIMENSION_Y;
		bytes_per_pixel = 1;
	}




	/*if -p flag was given we calculate the new dimensions*/
	int step = 1;
	int picture_dimension_x = dimension_x, picture_dimension_y = dimension_y;
   	if(part > 0){
   		float x = (float)dimension_x * part;
   		float y = (float)dimension_y * part;

   		dimension_x = (int)x;
   		dimension_y = (int)y;
   	}


	/*
	 *calculates the dimensions of the local grids and returns 
	 *if the given values are acceptable*/
	int result = Calculate_SubGrid_dimensions(dimension_x, dimension_y, &local_dimension_x, &local_dimension_y,  num_of_proc);

	//processes per dimensions
	int proc_per_x = dimension_x/ local_dimension_x;
	int proc_per_y = dimension_y/ local_dimension_y;
	
	/*initialize the global grid according to the given data
	 
	 *if result < 0 that means that the given data was not acceptable
	 
	 *if the user gave -r flag then it will initialize a random map with the given dimensions
	 *else the grid will be init from an image
	 Firstly we store the image data in an char** array and then we init the grid via Init_ImageGrid

	 *if the image is colorfull (the -c flag was given), bytes_per_pixel will be 3 so we init 3
	 grids each one contains the values of each color byte

	 *if -p flag was given the grid will have appropirate dimensions and values 

	*/
	if(result < 0){
		if(rank==0){
			if(result == -1)
				printf("Error: Number of processes is not a perfect square!\n");
			if(result == -2)
				printf("Error: The grid can't be devided equally with %d processes!\n", num_of_proc);
		}
		free(filename);
		MPI_Finalize();
		exit(1);
	}
	else {
		if(rank == 0){
			if(!random){
				FILE  *image_raw;
				image_raw = fopen(filename,"rb");
				if(!image_raw){
			    	printf("Error: Image named '%s' doesn't exist\n", filename);
			    	if(filename)free(filename);
			    	MPI_Finalize();
			    	exit(1);
				}	
				char **image_array = malloc(picture_dimension_x * sizeof(char*));
				i = 0;
				int read = 1;
				while(read > 0){   
					
					image_array[i] = malloc(bytes_per_pixel * picture_dimension_y * sizeof(char));
			   		read = fread(image_array[i], sizeof(char) , bytes_per_pixel * picture_dimension_y, image_raw);
			    	i++;
				}
				for(i = 0; i < bytes_per_pixel; i++){
					Create_Grid(dimension_x, dimension_y, &(global[i]));
					Init_ImageGrid(picture_dimension_x, picture_dimension_y, &(global[i]), image_array, i, bytes_per_pixel, part);
				}
				for(i = 0; i < picture_dimension_x; i++)
					free(image_array[i]);
				free(image_array);
				fclose(image_raw);
			}
			else{//random array
				for(i = 0; i < bytes_per_pixel; i++){
					Create_Grid(dimension_x, dimension_y, &(global[i]));
					Init_RandomGrid(dimension_x, dimension_y, &(global[i]));
				}
			}
		}
	}
	
	

	//Creating Cartesian virtual topologies
	int periods[2], dim_size[2];
	periods[0] = 0;
	periods[1] = 0;
	dim_size[0] = proc_per_x;
	dim_size[1] = proc_per_y;
	MPI_Comm comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size, periods, 1, &comm);

	
	/*3 in case the image is colorfull we need to calculculate
	 *the convolution time for 3 grids*/
	float time[3] = { 0, 0, 0 };
	int reps[bytes_per_pixel];
	for(i = 0; i < bytes_per_pixel; i++) //convolution for each grid containing the rgb bytes -- if the image is colorfull
		time[i] = convolution(&(global[i]), dimension_x, dimension_y, local_dimension_x, local_dimension_y,  
			proc_per_x, proc_per_y, filter, rank, num_of_proc, comm, &reps[i]);

	
	if(rank==0){
		//creates the file of the new image -- if no filename was given in command line creates "produced_images/produced.raw"
		FILE* new_file;
		if(new_filename != NULL)
			new_file =  fopen(new_filename, "w+");
		else
			new_file =  fopen("../produced_images/MPI_produced.raw", "w+");

		//writes the data into the new file 
		int k = 0;
		for(i = 0; i < dimension_x; i++){
			for(j = 0; j < dimension_y; j++){
				//in case the image was colorfull
				for(k = 0; k < bytes_per_pixel; k++){
					fprintf(new_file, "%c", (char)((int)((global[k])[i])[j]));
					if(print)printf("%.1f ", ((global[k])[i])[j]);
				}
			}
			if(print)printf("\n");
		}

		/*time[x] contains the duration of convulation of the grid 
		 *if the image was colorfull then 3 convulations took place 
		 *for the 3 grids with the bytes of the pictures, so the duration
		 *of convolution is the sum of those 3 times -- if the picture was
		 *not colorfull then time[1] and time[2] == 0*/
		
		if(random)
			printf("\n\n\nRandom array with Dimensions: %dx%d\n", dimension_x, dimension_y);
		else printf("\n\n\nImage: %s with Dimensions: %dx%d\n", filename, dimension_x, dimension_y);
		printf("Repetitions: ");
		for(i=0; i < bytes_per_pixel; i++)printf("%d ",reps[i]);
		printf("\n");
		if(bytes_per_pixel == 3)
			printf("Colorfull : YES\n");
		else printf("Colorfull : NO\n");
		printf("Duration of Convolution: %.5f sec\nProcesses: %d\n", time[0]+time[1]+time[2], num_of_proc);

		fclose(new_file);
		if(filename) free(filename);
		if(new_filename) free(new_filename);	
		for(k = 0; k < bytes_per_pixel; k++)
			Destroy_Grid(&(global[k]));
	}

	MPI_Finalize();
}


