
OBJS=  main.o convolution.o functions.o
SRC = main.c convolution.c functions.c
HEADERS =  convolution.h functions.h
OUT = convolution
CC	= mpicc -fopenmp
FLAGS   = -g -lm
MPIP  = -L/usr/local/mpip3/lib -lmpiP -lbfd -liberty
#mpiexec -f <machines> -n <num_of_processes> ./conolution -x <dimension x> -y <dimension_y> -i <image file> -c <image will be colorfull> -n <number_of_threads>

all: $(OBJS)
	$(CC) $(OBJS) $(FLAGS) -o $(OUT)


mpip:	$(OBJS)
	$(CC) $(OBJS) $(MPIP) $(FLAGS)  -o $(OUT)

# clean house
clean:
	rm -f $(OBJS) $(OUT)