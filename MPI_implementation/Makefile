OBJS=  main.o convolution.o functions.o
SRC = main.c convolution.c functions.c
HEADERS =  convolution.h functions.h
OUT = convolution
CC = mpicc
FLAGS = -lm -g 
MPIP  = -L/usr/local/mpip3/lib -lmpiP -lbfd -liberty

#mpiexec -f <machines> -n <num_of_processes> ./conolution -x <dimension x> -y <dimension_y> -i <image file> -c <image will be colorfull>

all: 	$(OBJS)
	$(CC) $(OBJS) -o $(OUT) $(FLAGS)


mpip:	$(OBJS)
	$(CC) $(OBJS) -o $(OUT) $(FLAGS) $(MPIP)


clean:
	rm -f $(OBJS) produced_images/*.raw $(OUT)
