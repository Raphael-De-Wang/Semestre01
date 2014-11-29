CC = gcc
GCC_FLAGS = -Wall -g

all: ex6

randomises_et_approches.o: randomises_et_approches.c
	$(CC) $(GCC_FLAGS) -c randomises_et_approches.c -lgmp

test_cases_ex6.o: test_cases_ex6.c
	$(CC) $(GCC_FLAGS) -c test_cases_ex6.c -lgmp

ex6: randomises_et_approches.o test_cases_ex6.o
	$(CC) $(GCC_FLAGS) -o ex6 randomises_et_approches.o test_cases_ex6.o -lgmp

clean:
	rm -f *.o *\~ ex6 
