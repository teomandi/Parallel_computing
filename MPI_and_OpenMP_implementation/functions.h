
typedef int bool;
#define true 1
#define false 0

void Create_Grid(int dimension_x, int dimension_y, float ***grid);

void Init_RandomGrid(int dimension_x, int dimension_y, float ***grid);

void Init_ImageGrid(int dimension_x, int dimension_y, float ***grid, char **image, int start, int adder, float part);

int Calculate_SubGrid_dimensions(int dimension_x, int dimension_y, int *sub_x, int* sub_y, int num_of_proc);

void Destroy_Grid(float ***grid);

int isEqual_Grid(float ***new_grid, float ***grid, int dimension_x, int dimension_y);

void print_Grid(float ***grid, int dimension_x, int dimension_y);

