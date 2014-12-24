CC_FLAGS = -Wall -g

all     : analyse ecosy test 

main1.o	: main1.c
	gcc $(GCC_FLAGS) -c main1.c

main2.o	: main2.c
	gcc $(GCC_FLAGS) -c main2.c

ecosys.o: ecosys.c
	gcc $(GCC_FLAGS) -c ecosys.c

analyse : main3.o ecosys.o
	gcc $(GCC_FLAGS) -o analyse main3.o ecosys.o  -lncurses

ecosy   : main2.o ecosys.o
	gcc $(GCC_FLAGS) -o ecosy main2.o ecosys.o  -lncurses

test    : main1.o ecosys.o
	gcc $(GCC_FLAGS) -o test main1.o ecosys.o  -lncurses

clean	:
	rm -f *.o *~ a.out analyse ecosy test 

