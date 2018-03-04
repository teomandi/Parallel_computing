OBJS=  main.o convolution.o functions.o
SRC = main.cu convolution.cu functions.c
HEADERS =  convolution.h functions.h
OUT = convolution
CC = mpicc
FLAGS = -lm

# ./conolution -x <dimension x> -y <dimension_y> -i <image file> -c <image will be colorfull>

all: $(OBJS)
	$(CC) $(OBJS) -o $(OUT) $(FLAGS)

clean:
	rm -f $(OBJS) ../produced_images/*.raw $(OUT)
