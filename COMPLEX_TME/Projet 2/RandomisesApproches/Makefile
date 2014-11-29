CC = gcc
GCC_FLAGS = -Wall -g

all: ex6 ex7

randomises_et_approches.o: randomises_et_approches.c
	$(CC) $(GCC_FLAGS) -c randomises_et_approches.c -lgmp

test_cases_ex6.o: test_cases_ex6.c
	$(CC) $(GCC_FLAGS) -c test_cases_ex6.c -lgmp

test_cases_ex7.o: test_cases_ex7.c
	$(CC) $(GCC_FLAGS) -c test_cases_ex7.c -lgmp

ex6: randomises_et_approches.o test_cases_ex6.o
	$(CC) $(GCC_FLAGS) -o ex6 randomises_et_approches.o test_cases_ex6.o -lgmp

ex7: randomises_et_approches.o test_cases_ex7.o
	$(CC) $(GCC_FLAGS) -o ex7 randomises_et_approches.o test_cases_ex7.o -lgmp

clean:
	rm -f *.o *\~ *gch ex6 
