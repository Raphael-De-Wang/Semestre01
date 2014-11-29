#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"

void test_first_test (void) {
  mpz_t N;
  char *str_N = (char *)malloc(INPUT_MAX_LENGTH);
  
  puts("---- Q 7.1 first test ----");
  puts("Grande Integer: ");
  scanf("%s",str_N);
  mpz_init_set_str(N, str_N, BASE);
  
  if ( first_test(N) == TRUE ) {
    printf("Oui, c'est un primier !\n");
  } else {
    printf("Non, ce n'est pas un primier.\n");
  }
  
  mpz_clear(N);
  FREE(str_N);
}

void test_premier_compteur (void) {
  mpz_t numbre, interval;
  char *str_interv = (char *)malloc(INPUT_MAX_LENGTH);
  char *str_num    = NULL;
  
  puts("---- Q 7.2 premier compteur ----");
  puts("Interval: ");
  scanf("%s",str_interv);
  mpz_init_set_str(interval, str_interv, BASE);
  mpz_init_set_str(numbre, "0", BASE);
  
  naif_premier_compteur(numbre, interval);
  str_num = mpz_get_str(str_num, BASE, numbre);
  printf("Il y a %s numbre premier inferieur que %s", str_num, str_interv);
  
  mpz_clear(numbre);
  mpz_clear(interval);
  FREE(str_interv);
  FREE(str_num);
}

int main( int argc, char *argv[]) {
  puts("==== Exercise 7 Test Naif ====");

  if ( argc > 1 ) {
    switch ( argv[1][0] ) {
    case '1':
      test_first_test();
      break;
    case '2':
      test_premier_compteur ();
      break;
    case '3':
      break;
    default:
      break;
    }
  } else {
    puts("choix de Question: 1, 2, 3, else quit");
  }
  return 0;
}

