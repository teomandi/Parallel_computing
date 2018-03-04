#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolution.h"

int rankk;

float convolution(float ***global, int dimension_x, int dimension_y, int local_dimension_x, int local_dimension_y, 
	int proc_per_x, int proc_per_y, float filter[FILTER_SIZE][FILTER_SIZE], int rank, int num_of_proc, MPI_Comm comm, int *rep ){

	rankk = rank;
	int i, j;
	float **local, **new_local;
	Create_Grid(local_dimension_x, local_dimension_y, &local);
	Create_Grid(local_dimension_x, local_dimension_y, &new_local);

	//scatering the array
	int sizes[2]    = {dimension_x, dimension_y};        							 /* global size */
    int subsizes[2] = {local_dimension_x, local_dimension_y};		   	     		 /* local size */
    int starts[2]   = {0,0};                       									 /* where this one starts */

	// the form of the data that we will scatter
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &type);
    MPI_Type_create_resized(type, 0, local_dimension_y*sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);

    float *global_ptr=NULL;
    if (rank == 0) global_ptr = &((*global)[0][0]);


    // calculate the displacement and the send counts
    int sendcounts[proc_per_x * proc_per_y];
    int displs[proc_per_x * proc_per_y];

    if (rank == 0) {
        for ( i=0; i<proc_per_x * proc_per_y; i++) sendcounts[i] = 1;
        int disp = 0;
        for ( i=0; i<proc_per_x; i++) {
            for (j=0; j<proc_per_y; j++) {
                displs[i*proc_per_x + j] = disp;
                disp += 1;
            }	
            disp += (local_dimension_x -1)*proc_per_x;
        }
    }

     MPI_Scatterv(global_ptr, sendcounts, displs, subarrtype, &(local[0][0]), 
     	local_dimension_x * local_dimension_y, MPI_FLOAT, 0, MPI_COMM_WORLD);


     //calculate coordinates of process neighbours 
    int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
    
    int top[2], bot[2], left[2], right[2], top_left[2], top_right[2],bot_left[2], bot_right[2];
   	int top_rank, bot_rank, left_rank, right_rank, top_left_rank, top_right_rank, bot_left_rank, bot_right_rank;

  	top[0] = coords[0] - 1;
	top[1] = coords[1];
	bot[0] = coords[0] + 1;
	bot[1] = coords[1];
	left[0] = coords[0];
	left[1] = coords[1] - 1;
	right[0] = coords[0];
	right[1] = coords[1] + 1;
	top_left[0] = coords[0] - 1;
	top_left[1] = coords[1] - 1;
	top_right[0] = coords[0] - 1;
	top_right[1] = coords[1] + 1;
	bot_left[0] = coords[0] + 1;
	bot_left[1] = coords[1] - 1;
	bot_right[0] = coords[0] + 1;
	bot_right[1] = coords[1] + 1;


	 //buffers for the receiving data
    float columns[2][local_dimension_x], rows[2][local_dimension_y], corners_buff[4] , buffer[local_dimension_y+local_dimension_x];
    bool no_top = false, no_bot = false, no_left = false, no_right = false, no_top_left = false,
    	 no_top_right = false, no_bot_left = false, no_bot_right = false;

	//if the neighbours exist we got his rank, else we indicate it with MPI_PROC_NULL
	//if there isnt a specific neighbor we fill the coresponding bool var to true
	if(top[0]< 0){
		top_rank = MPI_PROC_NULL;
		no_top = true;
	}
	else
		MPI_Cart_rank(comm, top, &top_rank);
	
	if(bot[0] >= proc_per_x){
		bot_rank = MPI_PROC_NULL;
		no_bot = true;
	}
	else
		MPI_Cart_rank(comm, bot, &bot_rank);
	
	if(left[1] < 0){
		left_rank = MPI_PROC_NULL;
		no_left = true;
	}
	else
		MPI_Cart_rank(comm, left, &left_rank);

	if(right[1] >= proc_per_y){
		right_rank = MPI_PROC_NULL;
		no_right = true;

	}
	else
		MPI_Cart_rank(comm, right, &right_rank);
	
	if(top_left[0]<  0 || top_left[1]< 0){
		top_left_rank = MPI_PROC_NULL;
		no_top_left = true;
	}
	else
		MPI_Cart_rank(comm, top_left, &top_left_rank);
	
	if(top_right[0]< 0 || top_right[1] >= proc_per_y ){
		top_right_rank = MPI_PROC_NULL;
		no_top_right = true;
	}
	else
		MPI_Cart_rank(comm, top_right, &top_right_rank);
	
	if(bot_left[0]>= proc_per_x || bot_left[1]< 0){
		bot_left_rank = MPI_PROC_NULL;
		no_bot_left = true;
	}
	else
		MPI_Cart_rank(comm, bot_left, &bot_left_rank);
	
	if(bot_right[0] >= proc_per_x || bot_right[1] >= proc_per_y){
		bot_right_rank = MPI_PROC_NULL;
		no_bot_right = true;
	}
	else
		MPI_Cart_rank(comm, bot_right, &bot_right_rank);



	MPI_Request request[16] , d_request;
	MPI_Status status;

	//Datatype for sending columns
	MPI_Datatype column;
	MPI_Type_vector(local_dimension_x, 1, local_dimension_y, MPI_FLOAT, &column);
	MPI_Type_commit(&column);
	
	int  stop = 1, number_amount;
	double start, end; 
	j=0;
	
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	
	while(stop != 0  &&  j < 300){
	

	    //sending 
	    //sending rows --  top, bot
		MPI_Isend(&local[0][0], local_dimension_y, MPI_FLOAT, top_rank, j,	MPI_COMM_WORLD, &request[0]);
		MPI_Isend(&local[local_dimension_x - 1][0], local_dimension_y, MPI_FLOAT, bot_rank, j, MPI_COMM_WORLD, &request[1]);

		//sending columns -- left, right (use of datatype 'column')
		MPI_Isend(&local[0][0], 1, column, left_rank, j, MPI_COMM_WORLD, &request[2]);
		MPI_Isend(&local[0][local_dimension_y - 1], 1, column, right_rank, j, MPI_COMM_WORLD, &request[3]);
		
		//sending spots -- top_left, top_right, bot_left, bot_right
		MPI_Isend(&local[0][0], 1, MPI_FLOAT, top_left_rank, j, MPI_COMM_WORLD, &request[4]);
		MPI_Isend(&local[0][local_dimension_y - 1], 1, MPI_FLOAT, top_right_rank, j, MPI_COMM_WORLD, &request[5]);
		MPI_Isend(&local[local_dimension_x - 1][0], 1, MPI_FLOAT, bot_left_rank, j, MPI_COMM_WORLD, &request[6]);
		MPI_Isend(&local[local_dimension_x - 1][local_dimension_y - 1], 1, MPI_FLOAT, bot_right_rank, j, MPI_COMM_WORLD, &request[7]);

	 	
	 	
/*	 	//regular receiving! -- not dynamically

		//columns and rows
		MPI_Irecv(&rows[0], local_dimension_y, MPI_FLOAT, top_rank, 0, MPI_COMM_WORLD, &request[8]);
		MPI_Irecv(&rows[1], local_dimension_y, MPI_FLOAT, bot_rank, 0, MPI_COMM_WORLD, &request[9]);				
		MPI_Irecv(&columns[0], local_dimension_x, MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, &request[10]);
		MPI_Irecv(&columns[1], local_dimension_x, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, &request[11]);		
		
		//corners
		MPI_Irecv(&corners_buff[0], 1, MPI_FLOAT, top_left_rank, 0, MPI_COMM_WORLD, &request[12]);
		MPI_Irecv(&corners_buff[1], 1, MPI_FLOAT, top_right_rank, 0,MPI_COMM_WORLD, &request[13]);
		MPI_Irecv(&corners_buff[2], 1, MPI_FLOAT, bot_left_rank, 0, MPI_COMM_WORLD, &request[14]);
		MPI_Irecv(&corners_buff[3], 1, MPI_FLOAT, bot_right_rank, 0, MPI_COMM_WORLD, &request[15]);
		
*/
		

		Calcutalate_InnerCells(&local, &new_local, local_dimension_x, local_dimension_y, filter);  
		


/*		
		//waits to receive all values and then calculates the external cells
		//wait to receive --make sure we have received all of our data
		for(i= 0; i< 8; i++)
			MPI_Wait(&request[i+8], &status);
		
		Calcutalate_ExternalCells(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[0], 
			rows[1], columns[0], columns[1], corners_buff[0], corners_buff[1], corners_buff[2], corners_buff[3],
			no_top, no_bot, no_left, no_right, no_top_left, no_top_right, no_bot_left, no_bot_right);


*/
	
		//dyanamic Receiving -- 
		int k = no_top + no_bot + no_left + no_right + no_top_left + no_top_right + no_bot_left + no_bot_right;		
		bool top = false, bot = false, left = false, right = false, top_left = false, top_right = false, bot_left = false, bot_right = false;
		int flag = -1;
		while(k < 8){
			
			if(flag != 0){
				d_request = MPI_REQUEST_NULL;
				//receives messages with tag j -- in case other processes are in next reps
		        MPI_Irecv(&buffer, local_dimension_x+local_dimension_y, MPI_FLOAT, MPI_ANY_SOURCE, j, MPI_COMM_WORLD, &d_request);
		        flag = 0;
		    }
		
			MPI_Test(&d_request, &flag, &status);
	
			if(flag != 0){			

				k++;	
				if(status.MPI_SOURCE == top_rank && !top){
					memmove(&rows[0], &buffer, sizeof(float)*local_dimension_y );
					Calcutalate_TopRow(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[0], no_top);
					top = true;
				}
					
			 	else if(status.MPI_SOURCE == bot_rank && !bot){
					memmove(&rows[1], &buffer, sizeof(float)*local_dimension_y );
					Calcutalate_BotRow(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[1], no_bot);
					bot = true;
			 	}
						
			 	else if(status.MPI_SOURCE == left_rank && !left){
					memmove(&columns[0], &buffer, sizeof(float)*local_dimension_x );
					Calcutalate_LeftColumn(&local, &new_local, local_dimension_x, local_dimension_y, filter, columns[0], no_left);
					left = true;
			 	}
						
			 	else if(status.MPI_SOURCE == right_rank && !right){
					memmove(&columns[1], &buffer, sizeof(float)*local_dimension_x );
					Calcutalate_RightColumn(&local, &new_local, local_dimension_x, local_dimension_y, filter, columns[1], no_right);
					right = true;
			 	}
						
			 	else if(status.MPI_SOURCE == top_left_rank && !top_left){
					corners_buff[0] = buffer[0];
					if(top && left && !top_left){
						Calcutalate_TopLeftCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[0], columns[0], corners_buff[0],
							no_top, no_left, no_top_left);
						top_left = true;
					}
			 	}
			 	else if(status.MPI_SOURCE == top_right_rank && !top_right){
					corners_buff[1] = buffer[0];
					if(top && right && !top_right){
						Calcutalate_TopRightCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[0], columns[1], corners_buff[1],
							no_top, no_right, no_top_right);
						top_right = true;
					}
			 	}
						
			 	else if(status.MPI_SOURCE == bot_left_rank && !bot_left){
					corners_buff[2] = buffer[0];
					if(bot && left && !bot_left){
						Calcutalate_BotLeftCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[1], columns[0], corners_buff[2],
							no_bot, no_left, no_bot_left);
						bot_left = true;
					}
			 	}
						
			 	else if(status.MPI_SOURCE == bot_right_rank && !bot_right){
					corners_buff[3] = buffer[0];
					if(bot && right && !bot_right){
						Calcutalate_BotRightCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[1], columns[1], corners_buff[3],
							no_bot, no_right, no_bot_right);
						bot_right = true;
					}
				}
		 	}
		}

		//in case that they were not calculated -- caused of an non-existed neighbour 
		if(!top)
			Calcutalate_TopRow(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[0], no_top);
		if(!bot)
			Calcutalate_BotRow(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[1], no_bot);
		if(!left)
			Calcutalate_LeftColumn(&local, &new_local, local_dimension_x, local_dimension_y, filter, columns[0], no_left);
		if(!right)
			Calcutalate_RightColumn(&local, &new_local, local_dimension_x, local_dimension_y, filter, columns[1], no_right);
		if(!top_left)
			Calcutalate_TopLeftCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[0], columns[0], corners_buff[0],
							no_top, no_left, no_top_left);
		if(!top_right)
			Calcutalate_TopRightCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[0], columns[1], corners_buff[1],
							no_top, no_right, no_top_right);
		if(!bot_left)
			Calcutalate_BotLeftCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[1], columns[0], corners_buff[2],
							no_bot, no_left, no_bot_left);
		if(!bot_right)
			Calcutalate_BotRightCell(&local, &new_local, local_dimension_x, local_dimension_y, filter, rows[1], columns[1], corners_buff[3],
							no_bot, no_right, no_bot_right);

	


		//wait to send --make sure we have send the requested data before we change them
		for(i=1; i< 8; i++)
			MPI_Wait(&request[i], &status);


		/*check if there is changes in the grids -- Every process check if there was any changes 
		 *in its grid and sends (throught reduce) 0 if there was not or 1 if there was
		 *-- reduce sums up all the values broadcast it to anybody 
		 *-- if everyone sent 0 the broadcasting sum will be 0 so that means that no
		 *changes took place in any process's grid so the convolution has ended */
		if( (j + 1) % 10 == 0){
			
			int result = isEqual_Grid(&new_local, &local, local_dimension_x, local_dimension_y);
			MPI_Status t_status;

			MPI_Reduce(&result, &stop, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Bcast( &stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
			
		}

		//swaping new with old grids
		float **temp = new_local;
		new_local = local;
		local = temp;
		j++;
	}


	end = MPI_Wtime();	
	MPI_Barrier(MPI_COMM_WORLD);
 	//printf("BARRIR rank : %d stop: %d - j: %d e :%.3f - s :%.3f  time: %.3f\n",rank, stop, j, end, start, end - start );
  	MPI_Gatherv(&(local[0][0]), local_dimension_x * local_dimension_y, MPI_FLOAT,
            global_ptr, sendcounts, displs, subarrtype,
      	    0, MPI_COMM_WORLD);


    MPI_Type_free(&subarrtype);
    Destroy_Grid(&local);
    (*rep) = j;
    return end - start ;
}





static inline void Calcutalate_InnerCells(float ***grid, float ***new_grid, int dimension_x, int dimension_y, 
	float filter[FILTER_SIZE][FILTER_SIZE]){

	int i,j;
	for(i= 1; i< dimension_x-1; i++){
		for(j= 1; j< dimension_y-1; j++){
			((*new_grid)[i])[j]=  
				((*grid)[i+1])[j+1]*filter[0][0]+ ((*grid)[i+1])[j]*filter[0][1]+ ((*grid)[i+1])[j-1]*filter[0][2]+
				((*grid)[i])[j+1]*filter[1][0]+ ((*grid)[i])[j]*filter[1][1]+ ((*grid)[i])[j-1]*filter[1][2]+
				((*grid)[i-1])[j+1]*filter[2][0]+ ((*grid)[i-1])[j]*filter[2][1]+ ((*grid)[i-1])[j-1]*filter[2][2]; 
		}
	}
}






//Calculate all externals cells
static inline void Calcutalate_ExternalCells(float ***grid, float ***new_grid, int dimension_x, int dimension_y, 
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float bot_buff[], float left_buff[],
	float right_buff[], float top_left_value, float top_right_value, float bot_left_value, float bot_right_value,
	bool no_top, bool no_bot, bool no_left, bool no_right, bool no_top_left, bool no_top_right, bool no_bot_left, bool no_bot_right){

	/*in case a buffer is initialised with -1 means that there was no neighbor to provide the needed value
	so we have o use the value of the cell that we are calculatin!*/

	int j;
	//calculate rows without corners
	for(j=1; j< dimension_y-1; j++){
		
		//first
		if(!no_top)
			((*new_grid)[0])[j]=  
				((*grid)[1])[j+1]*filter[0][0]+ ((*grid)[1])[j]*filter[0][1]+ ((*grid)[1])[j-1]*filter[0][2]+
				((*grid)[0])[j+1]*filter[1][0]+ ((*grid)[0])[j]*filter[1][1]+ ((*grid)[0])[j-1]*filter[1][2]+
				top_buff[j+1]*filter[2][0]+ top_buff[j]*filter[2][1]+ top_buff[j-1]*filter[2][2]; 
		else
			((*new_grid)[0])[j]=  
				((*grid)[1])[j+1]*filter[0][0]+ ((*grid)[1])[j]*filter[0][1]+ ((*grid)[1])[j-1]*filter[0][2]+
				((*grid)[0])[j+1]*filter[1][0]+ ((*grid)[0])[j]*filter[1][1]+ ((*grid)[0])[j-1]*filter[1][2]+
				((*grid)[0])[j] *filter[2][0]+ ((*grid)[0])[j]*filter[2][1]+ ((*grid)[0])[j]*filter[2][2]; 

		//last row 
		if(!no_bot)
			((*new_grid)[dimension_x-1])[j]=  
				bot_buff[j+1]*filter[0][0]+ bot_buff[j]*filter[0][1]+ bot_buff[j-1]*filter[0][2]+
				((*grid)[dimension_x-1])[j+1]*filter[1][0]+ ((*grid)[dimension_x-1])[j]*filter[1][1]+ ((*grid)[dimension_x-1])[j-1]*filter[1][2]+
				((*grid)[dimension_x-2])[j+1]*filter[2][0]+ ((*grid)[dimension_x-2])[j]*filter[2][1]+ ((*grid)[dimension_x-2])[j-1]*filter[2][2]; 
		else
			((*new_grid)[dimension_x-1])[j]=  
				((*grid)[dimension_x-1])[j]*filter[0][0]+ ((*grid)[dimension_x-1])[j]*filter[0][1]+ ((*grid)[dimension_x-1])[j]*filter[0][2]+
				((*grid)[dimension_x-1])[j+1]*filter[1][0]+ ((*grid)[dimension_x-1])[j]*filter[1][1]+ ((*grid)[dimension_x-1])[j-1]*filter[1][2]+
				((*grid)[dimension_x-2])[j+1]*filter[2][0]+ ((*grid)[dimension_x-2])[j]*filter[2][1]+ ((*grid)[dimension_x-2])[j-1]*filter[2][2]; 
	}


	//calculate columns without corners
	for(j=1; j< dimension_x-1; j++){

		//left column 
		if(!no_left)
			((*new_grid)[j])[0]=  
					((*grid)[j+1])[1]*filter[0][0]+ ((*grid)[j+1])[0]*filter[0][1]+ left_buff[j+1]*filter[0][2]+
					((*grid)[j])[1]*filter[1][0]+ ((*grid)[j])[0]*filter[1][1]+ left_buff[j]*filter[1][2]+
					((*grid)[j-1])[1]*filter[2][0]+ ((*grid)[j-1])[0]*filter[2][1]+ left_buff[j-1]*filter[2][2]; 
		else
			((*new_grid)[j])[0]=  
					((*grid)[j+1])[1]*filter[0][0]+ ((*grid)[j+1])[0]*filter[0][1]+ ((*grid)[j])[0]*filter[0][2]+
					((*grid)[j])[1]*filter[1][0]+ ((*grid)[j])[0]*filter[1][1]+ ((*grid)[j])[0]*filter[1][2]+
					((*grid)[j-1])[1]*filter[2][0]+ ((*grid)[j-1])[0]*filter[2][1]+ ((*grid)[j])[0]*filter[2][2]; 
	

		//right column
		if(!no_right)
			((*new_grid)[j])[dimension_y-1]=  
					right_buff[j+1]*filter[0][0]+ ((*grid)[j+1])[dimension_y-1]*filter[0][1]+ ((*grid)[j+1])[dimension_y-2]*filter[0][2]+
					right_buff[j]*filter[1][0]+ ((*grid)[j])[dimension_y-1]*filter[1][1]+ ((*grid)[j])[dimension_y-2]*filter[1][2]+
					right_buff[j-1]*filter[2][0]+ ((*grid)[j-1])[dimension_y-1]*filter[2][1]+ ((*grid)[j-1])[dimension_y-2]*filter[2][2]; 
		else
			((*new_grid)[j])[dimension_y-1]=  
					((*grid)[j])[dimension_y-1]*filter[0][0]+ ((*grid)[j+1])[dimension_y-1]*filter[0][1]+ ((*grid)[j+1])[dimension_y-2]*filter[0][2]+
					((*grid)[j])[dimension_y-1]*filter[1][0]+ ((*grid)[j])[dimension_y-1]*filter[1][1]+ ((*grid)[j])[dimension_y-2]*filter[1][2]+
					((*grid)[j])[dimension_y-1]*filter[2][0]+ ((*grid)[j-1])[dimension_y-1]*filter[2][1]+ ((*grid)[j-1])[dimension_y-2]*filter[2][2]; 
	}

	float top[4], bot[4], left[4], right[4], top_left, top_right, bot_left, bot_right;
	
	/*fill the identifiers with the necessery values -- whether the buffers are initialised or not--
	in order to calculate the values of the corners

	if a buffer wasnt given , we use the value of the cell tha we are calculating (corners) */
	//rows
	if(no_top){
		top[0] = ((*grid)[0])[0];
		top[1] = ((*grid)[0])[0];
		top[2] = ((*grid)[0])[dimension_y-1];
		top[3] = ((*grid)[0])[dimension_y-1];
	}
	else{
		top[0] = top_buff[0];
		top[1] = top_buff[1];
		top[2] = top_buff[dimension_y -2];
		top[3] = top_buff[dimension_y -1 ];
	}

	if(no_bot){
		bot[0] = ((*grid)[dimension_x-1])[0];
		bot[1] = ((*grid)[dimension_x-1])[0];
		bot[2] = ((*grid)[dimension_x-1])[dimension_y-1];
		bot[3] = ((*grid)[dimension_x-1])[dimension_y-1];
	}
	else{
		bot[0] = bot_buff[0];
		bot[1] = bot_buff[1];
		bot[2] = bot_buff[dimension_y-1];
		bot[3] = bot_buff[dimension_y-2];
	}


	//columns
	if(no_left){
		left[1] = ((*grid)[0])[0];
		left[0] = ((*grid)[0])[0];	
		left[2] = ((*grid)[dimension_x-1])[0];
		left[3] = ((*grid)[dimension_x-1])[0];	
	}
	else{
		left[0] = left_buff[0];
		left[1] = left_buff[1];
		left[2] = left_buff[dimension_x-2];
		left[3] = left_buff[dimension_x-1];
	}

	if(no_right){
		right[0] = ((*grid)[0])[dimension_y-1];
		right[1] = ((*grid)[0])[dimension_y-1];
		right[2] = ((*grid)[dimension_x-1])[dimension_y-1];
		right[3] = ((*grid)[dimension_x-1])[dimension_y-1];
	}
	else{
		right[0] = right_buff[0];
		right[1] = right_buff[1];
		right[2] = right_buff[dimension_x-1];
		right[3] = right_buff[dimension_x-2];
	}


	//corners
	if(no_top_left)
		top_left =((*grid)[0])[0];
	else
		top_left = top_left_value;

	if(no_top_right)
		top_right = ((*grid)[0])[dimension_y-1];
	else
		top_right = top_right_value;

	if(no_bot_left)
		bot_left = ((*grid)[dimension_x-1])[0];
	else
		bot_left = bot_left_value;

	if(no_bot_right)
		bot_right = ((*grid)[dimension_x-1])[dimension_y-1];
	else
		bot_right = bot_right_value;



	//top-left corner
	((*new_grid)[0])[0]=  
		((*grid)[1])[1]*filter[0][0]+ ((*grid)[1])[0]*filter[0][1]+ left[1]*filter[0][2]+
		((*grid)[0])[1]*filter[1][0]+ ((*grid)[0])[0]*filter[1][1]+ left[0]*filter[1][2]+
		top[1]*filter[2][0]+ top[0]*filter[2][1]+ top_left*filter[2][2]; 

	//top-right corner
	((*new_grid)[0])[dimension_y-1]=  
		right[1]*filter[0][0]+ ((*grid)[1])[dimension_y-1]*filter[0][1]+ ((*grid)[1])[dimension_y-2]*filter[0][2]+
		right[0]*filter[1][0]+ ((*grid)[0])[dimension_y-1]*filter[1][1]+ ((*grid)[0])[dimension_y-2]*filter[1][2]+
		top_right*filter[2][0]+ top[3]*filter[2][1]+ top[2]*filter[2][2]; 


	//bot-left corner
	((*new_grid)[dimension_x-1])[0]=  
		bot[1]*filter[0][0]+ bot[0]*filter[0][1]+ bot_left*filter[0][2]+
		((*grid)[dimension_x-1])[1]*filter[1][0]+ ((*grid)[dimension_x-1])[0]*filter[1][1]+ left[2]*filter[1][2]+
		((*grid)[dimension_x-2])[1]*filter[2][0]+ ((*grid)[dimension_x-2])[0]*filter[2][1]+ left[3]*filter[2][2]; 


	//bot-right corner
	((*new_grid)[dimension_x-1])[dimension_y-1]=  
		bot_right*filter[0][0]+ bot[3]*filter[0][1]+ bot[2]*filter[0][2]+
		right[2]*filter[1][0]+ ((*grid)[dimension_x-1])[dimension_y-1]*filter[1][1]+ ((*grid)[dimension_x-1])[dimension_y-2]*filter[1][2]+
		right[3]*filter[2][0]+ ((*grid)[dimension_x-2])[dimension_y-1]*filter[2][1]+ ((*grid)[dimension_x-2])[dimension_y-2]*filter[2][2]; 

}





static inline void Calcutalate_TopRow(float ***grid, float ***new_grid,int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], bool no_top){
	int j;
	for(j=1; j< dimension_y-1; j++){
		
		//in case they're not init. we use the value of the inspected item
		if(!no_top)
			((*new_grid)[0])[j]=  
				((*grid)[1])[j+1]*filter[0][0]+ ((*grid)[1])[j]*filter[0][1]+ ((*grid)[1])[j-1]*filter[0][2]+
				((*grid)[0])[j+1]*filter[1][0]+ ((*grid)[0])[j]*filter[1][1]+ ((*grid)[0])[j-1]*filter[1][2]+
				top_buff[j+1]*filter[2][0]+ top_buff[j]*filter[2][1]+ top_buff[j-1]*filter[2][2]; 
		else
			((*new_grid)[0])[j]=  
				((*grid)[1])[j+1]*filter[0][0]+ ((*grid)[1])[j]*filter[0][1]+ ((*grid)[1])[j-1]*filter[0][2]+
				((*grid)[0])[j+1]*filter[1][0]+ ((*grid)[0])[j]*filter[1][1]+ ((*grid)[0])[j-1]*filter[1][2]+
				((*grid)[0])[j] *filter[2][0]+ ((*grid)[0])[j]*filter[2][1]+ ((*grid)[0])[j]*filter[2][2]; 

				
	}
}



static inline void Calcutalate_BotRow(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], bool no_bot){

	int j;
	for(j=1; j< dimension_y-1; j++){

		//in case they're not init. we use the value of the inspected item
		if(!no_bot)
			((*new_grid)[dimension_x-1])[j]=  
				bot_buff[j+1]*filter[0][0]+ bot_buff[j]*filter[0][1]+ bot_buff[j-1]*filter[0][2]+
				((*grid)[dimension_x-1])[j+1]*filter[1][0]+ ((*grid)[dimension_x-1])[j]*filter[1][1]+ ((*grid)[dimension_x-1])[j-1]*filter[1][2]+
				((*grid)[dimension_x-2])[j+1]*filter[2][0]+ ((*grid)[dimension_x-2])[j]*filter[2][1]+ ((*grid)[dimension_x-2])[j-1]*filter[2][2]; 
		else
			((*new_grid)[dimension_x-1])[j]=  
				((*grid)[dimension_x-1])[j]*filter[0][0]+ ((*grid)[dimension_x-1])[j]*filter[0][1]+ ((*grid)[dimension_x-1])[j]*filter[0][2]+
				((*grid)[dimension_x-1])[j+1]*filter[1][0]+ ((*grid)[dimension_x-1])[j]*filter[1][1]+ ((*grid)[dimension_x-1])[j-1]*filter[1][2]+
				((*grid)[dimension_x-2])[j+1]*filter[2][0]+ ((*grid)[dimension_x-2])[j]*filter[2][1]+ ((*grid)[dimension_x-2])[j-1]*filter[2][2]; 

		
	}
}	



static inline void Calcutalate_LeftColumn(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float left_buff[], bool no_left){ 	
	int j;
	for(j=1; j< dimension_x-1; j++){

		//in case they're not init. we use the value of the inspected item
		if(!no_left)
			((*new_grid)[j])[0]=  
					((*grid)[j+1])[1]*filter[0][0]+ ((*grid)[j+1])[0]*filter[0][1]+ left_buff[j+1]*filter[0][2]+
					((*grid)[j])[1]*filter[1][0]+ ((*grid)[j])[0]*filter[1][1]+ left_buff[j]*filter[1][2]+
					((*grid)[j-1])[1]*filter[2][0]+ ((*grid)[j-1])[0]*filter[2][1]+ left_buff[j-1]*filter[2][2]; 
		else
			((*new_grid)[j])[0]=  
					((*grid)[j+1])[1]*filter[0][0]+ ((*grid)[j+1])[0]*filter[0][1]+ ((*grid)[j])[0]*filter[0][2]+
					((*grid)[j])[1]*filter[1][0]+ ((*grid)[j])[0]*filter[1][1]+ ((*grid)[j])[0]*filter[1][2]+
					((*grid)[j-1])[1]*filter[2][0]+ ((*grid)[j-1])[0]*filter[2][1]+ ((*grid)[j])[0]*filter[2][2]; 

					
	}
}



static inline void Calcutalate_RightColumn(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float right_buff[], bool no_right){

	int j;
	for(j=1; j< dimension_x-1; j++){
		//in case they're not init. we use the value of the inspected item
		if(!no_right)
			((*new_grid)[j])[dimension_y-1]=  
					right_buff[j+1]*filter[0][0]+ ((*grid)[j+1])[dimension_y-1]*filter[0][1]+ ((*grid)[j+1])[dimension_y-2]*filter[0][2]+
					right_buff[j]*filter[1][0]+ ((*grid)[j])[dimension_y-1]*filter[1][1]+ ((*grid)[j])[dimension_y-2]*filter[1][2]+
					right_buff[j-1]*filter[2][0]+ ((*grid)[j-1])[dimension_y-1]*filter[2][1]+ ((*grid)[j-1])[dimension_y-2]*filter[2][2]; 
		else
			((*new_grid)[j])[dimension_y-1]=  
					((*grid)[j])[dimension_y-1]*filter[0][0]+ ((*grid)[j+1])[dimension_y-1]*filter[0][1]+ ((*grid)[j+1])[dimension_y-2]*filter[0][2]+
					((*grid)[j])[dimension_y-1]*filter[1][0]+ ((*grid)[j])[dimension_y-1]*filter[1][1]+ ((*grid)[j])[dimension_y-2]*filter[1][2]+
					((*grid)[j])[dimension_y-1]*filter[2][0]+ ((*grid)[j-1])[dimension_y-1]*filter[2][1]+ ((*grid)[j-1])[dimension_y-2]*filter[2][2]; 

					
	}
}



static inline void Calcutalate_TopLeftCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float left_buff[], float top_left_value, bool no_top, bool no_left, bool no_top_left){

	float top[2], left[2], top_left;
	
	if(no_top){
		top[0] = ((*grid)[0])[0];
		top[1] = ((*grid)[0])[0];
	}
	else{
		top[0] = top_buff[0];
		top[1] = top_buff[1];
	}

	if(no_left){
		left[1] = ((*grid)[0])[0];
		left[0] = ((*grid)[0])[0];		
	}
	else{
		left[0] = left_buff[0];
		left[1] = left_buff[1];
		
	}

	if(no_top_left)
		top_left =((*grid)[0])[0];
	else
		top_left = top_left_value;
 

	//top-left corner
	((*new_grid)[0])[0]=  
		((*grid)[1])[1]*filter[0][0]+ ((*grid)[1])[0]*filter[0][1]+ left[1]*filter[0][2]+
		((*grid)[0])[1]*filter[1][0]+ ((*grid)[0])[0]*filter[1][1]+ left[0]*filter[1][2]+
		top[1]*filter[2][0]+ top[0]*filter[2][1]+ top_left*filter[2][2]; 

		

}




static inline void Calcutalate_TopRightCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float right_buff[], float top_right_value, bool no_top, bool no_right, bool no_top_right){
	float top[2], right[2], top_right;
	
	if(no_top){
		top[0] = ((*grid)[0])[dimension_y-1];
		top[1] = ((*grid)[0])[dimension_y-1];
	}
	else{
		top[0] = top_buff[dimension_y -2];
		top[1] = top_buff[dimension_y -1];
	}

	if(no_right){
		right[0] = ((*grid)[0])[dimension_y-1];
		right[1] = ((*grid)[0])[dimension_y-1];
	}
	else{
		right[0] = right_buff[0];
		right[1] = right_buff[1];
	}

	if(no_top_right)
		top_right = ((*grid)[0])[dimension_y-1];
	else
		top_right = top_right_value;

	//top-right corner
	((*new_grid)[0])[dimension_y-1]=  
		right[1]*filter[0][0]+ ((*grid)[1])[dimension_y-1]*filter[0][1]+ ((*grid)[1])[dimension_y-2]*filter[0][2]+
		right[0]*filter[1][0]+ ((*grid)[0])[dimension_y-1]*filter[1][1]+ ((*grid)[0])[dimension_y-2]*filter[1][2]+
		top_right*filter[2][0]+ top[1]*filter[2][1]+ top[0]*filter[2][2]; 

	

	
}



static inline void Calcutalate_BotLeftCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], float left_buff[], float bot_left_value, bool no_bot, bool no_left, bool no_bot_left){
	float bot[2], left[2], bot_left;

	if(no_bot){
		bot[0] = ((*grid)[dimension_x-1])[0];
		bot[1] = ((*grid)[dimension_x-1])[0];
	}
	else{
		bot[0] = bot_buff[0];
		bot[1] = bot_buff[1];
	}

	if(no_left){
		left[0] = ((*grid)[dimension_x-1])[0];
		left[1] = ((*grid)[dimension_x-1])[0];
	}
	else{
		left[0] = left_buff[dimension_x-2];
		left[1] = left_buff[dimension_x-1];
	}

	if(no_bot_left)
		bot_left = ((*grid)[dimension_x-1])[0];
	else
		bot_left = bot_left_value;

	//bot-left corner
	((*new_grid)[dimension_x-1])[0]=  
		bot[1]*filter[0][0]+ bot[0]*filter[0][1]+ bot_left*filter[0][2]+
		((*grid)[dimension_x-1])[1]*filter[1][0]+ ((*grid)[dimension_x-1])[0]*filter[1][1]+ left[1]*filter[1][2]+
		((*grid)[dimension_x-2])[1]*filter[2][0]+ ((*grid)[dimension_x-2])[0]*filter[2][1]+ left[0]*filter[2][2]; 

			

}



static inline void Calcutalate_BotRightCell(float ***grid, float ***new_grid,int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], float right_buff[], float bot_right_value, bool no_bot, bool no_right, bool no_bot_right){	

	float bot[2], right[2], bot_right;

	if(no_bot){
		bot[0] = ((*grid)[dimension_x-1])[dimension_y-1];
		bot[1] = ((*grid)[dimension_x-1])[dimension_y-1];
	}
	else{
		bot[1] = bot_buff[dimension_y-1];
		bot[0] = bot_buff[dimension_y-2];
	}

	if(no_right){
		right[0] = ((*grid)[dimension_x-1])[dimension_y-1];
		right[1] = ((*grid)[dimension_x-1])[dimension_y-1];
	}
	else{
		right[1] = right_buff[dimension_x-1];
		right[0] = right_buff[dimension_x-2];
	}

	if(no_bot_right)
		bot_right = ((*grid)[dimension_x-1])[dimension_y-1];
	else
		bot_right = bot_right_value;

	//bot-right corner
	((*new_grid)[dimension_x-1])[dimension_y-1]=  
		bot_right*filter[0][0]+ bot[1]*filter[0][1]+ bot[0]*filter[0][2]+
		right[1]*filter[1][0]+ ((*grid)[dimension_x-1])[dimension_y-1]*filter[1][1]+ ((*grid)[dimension_x-1])[dimension_y-2]*filter[1][2]+
		right[0]*filter[2][0]+ ((*grid)[dimension_x-2])[dimension_y-1]*filter[2][1]+ ((*grid)[dimension_x-2])[dimension_y-2]*filter[2][2];

					
 

}


