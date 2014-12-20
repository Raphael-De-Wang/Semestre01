GCC_FLAGS = -Wall -g

all     : ecosy

main1.o	: main1.c
	gcc $(GCC_FLAGS) -c main1.c

ecosys.o: ecosys.c
	gcc $(GCC_FLAGS) -c ecosys.c

ecosy   : main1.o ecosys.o
	gcc $(GCC_FLAGS) -o ecosy main1.o ecosys.o

clean	:
	rm -f *.o *~ ecosy

