
#include "functions.h"

#define DIMENSION_X 1920
#define DIMENSION_Y 2520

#define FILTER_SIZE 3

float convolution(float ***global, int dimension_x, int dimension_y, float filter[FILTER_SIZE][FILTER_SIZE], int *rep);



static inline void Calcutalate_InnerCells(float ***grid, float ***new_grid, int dimension_x, int dimension_y, 
	float filter[FILTER_SIZE][FILTER_SIZE]);

static inline void Calcutalate_ExternalCells(float ***grid, float ***new_grid, int dimension_x, int dimension_y, 
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float bot_buff[], float left_buff[],
	 float right_buff[], float top_left_value, float top_right_value, float bot_left_value, float bot_right_value,
	 bool no_top, bool no_bot, bool no_left, bool no_right, bool no_top_left, bool no_top_right, bool no_bot_left, bool no_bot_right);


static inline void Calcutalate_TopRow(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], bool no_top);

static inline void Calcutalate_BotRow(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], bool no_bot);

static inline void Calcutalate_LeftColumn(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float left_buff[], bool no_left);

static inline void Calcutalate_RightColumn(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float right_buff[], bool no_right);

static inline void Calcutalate_TopLeftCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float left_buff[], float top_left_value, bool no_top, bool no_left, bool no_top_left);

static inline void Calcutalate_TopRightCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float top_buff[], float right_buff[], float top_right_value, bool no_top, bool no_right, bool no_top_right);

static inline void Calcutalate_BotLeftCell(float ***grid, float ***new_grid, int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], float left_buff[], float bot_left_value, bool no_bot, bool no_left, bool no_bot_left);

static inline void Calcutalate_BotRightCell(float ***grid, float ***new_grid,int dimension_x, int dimension_y,
	float filter[FILTER_SIZE][FILTER_SIZE], float bot_buff[], float right_buff[], float bot_right_value, bool no_bot, bool no_right, bool no_bot_right);

