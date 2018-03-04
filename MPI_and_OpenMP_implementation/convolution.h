
#include "functions.h"

#define DIMENSION_X 1920
#define DIMENSION_Y 2520

#define FILTER_SIZE 3
#define NUMBER_OF_THREADS 5

float convolution(float ***global, int dimension_x, int dimension_y, int local_dimension_x, int local_dimension_y,  int proc_per_x,
	int proc_per_y, float filter[FILTER_SIZE][FILTER_SIZE], int rank, int num_of_proc, MPI_Comm comm, int num_of_threads, int *reps);



static inline void Calcutalate_InnerCells(float ***grid, float ***new_grid, int dimension_x, int dimension_y, 
	float filter[FILTER_SIZE][FILTER_SIZE], int num_of_threads);

static inline void Calcutalate_ExternalCells(float ***grid, float ***new_grid, int dimension_x, int dimension_y, 
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float bot_buff[], float left_buff[],
	 float right_buff[], float top_left_value, float top_right_value, float bot_left_value, float bot_right_value,
	 bool no_top, bool no_bot, bool no_left, bool no_right, bool no_top_left, bool no_top_right, bool no_bot_left, bool no_bot_right, 
	 int num_of_threads);


static inline void Calcutalate_TopRow(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], bool no_top, 
	int num_of_threads);

static inline void Calcutalate_BotRow(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], bool no_bot, 
	int num_of_threads);

static inline void Calcutalate_LeftColumn(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float left_buff[], bool no_left, 
	int num_of_threads);

static inline void Calcutalate_RightColumn(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float right_buff[], bool no_right, 
	int num_of_threads);

static inline void Calcutalate_TopLeftCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float left_buff[], float top_left_value, bool no_top, bool no_left, bool no_top_left, 
	int num_of_threads);

static inline void Calcutalate_TopRightCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float right_buff[], float top_right_value, bool no_top, bool no_right, bool no_top_right, 
	int num_of_threads);

static inline void Calcutalate_BotLeftCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], float left_buff[], float bot_left_value, bool no_bot, bool no_left, bool no_bot_left, 
	int num_of_threads);

static inline void Calcutalate_BotRightCell(float ***grid, float ***new_grid,int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], float right_buff[], float bot_right_value, bool no_bot, bool no_right, bool no_bot_right, 
	int num_of_threads);

