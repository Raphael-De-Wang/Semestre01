#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"

void test_my_pgcd (void) {
  mpz_t integ_a, integ_b, pgcd;
  char *str_a = (char *)malloc(INPUT_MAX_LENGTH);
  char *str_b = (char *)malloc(INPUT_MAX_LENGTH);
  char *str   = NULL;
  
  puts("---- Q 6.1 my_pgcd ----");
  puts("Grande Integer 1: ");
  scanf("%s",str_a);
  puts("Grande Integer 2: ");
  scanf("%s",str_b);
  mpz_init_set_str(integ_a, str_a, BASE);
  mpz_init_set_str(integ_b, str_b, BASE);
  mpz_init(pgcd);
  
  my_pgcd(pgcd, integ_a, integ_b);
  str = mpz_get_str(str, BASE, pgcd);
  printf("PGCD: %s\n", str);
  
  mpz_clear(integ_a);
  mpz_clear(integ_b);
  mpz_clear(pgcd);
  FREE(str_a);
  FREE(str_b);
  FREE(str);
}


void test_my_inverse (void) {
  
  mpz_t integ_a, integ_N, inverse;
  char *str_a = (char *)malloc(INPUT_MAX_LENGTH);
  char *str_N = (char *)malloc(INPUT_MAX_LENGTH);
  char *str_b = NULL;
  
  puts("---- Q 6.2 my_inverse ----");
  puts("Grande Integer a: ");
  scanf("%s",str_a);
  puts("Grande Integer N: ");
  scanf("%s",str_N);
  mpz_init_set_str(integ_a, str_a, BASE);
  mpz_init_set_str(integ_N, str_N, BASE);
  mpz_init(inverse);
  
  if ( my_inverse(inverse, integ_a, integ_N) ) {
    str_b = mpz_get_str(str_b, BASE, inverse);
    printf("Inverse: %s\n", str_b);
  } else {
    printf("%s n'est pas inversible modulo %s\n",str_a, str_N);
  }
  
  mpz_clear(integ_a);
  mpz_clear(integ_N);
  mpz_clear(inverse);
  FREE(str_a);
  FREE(str_N);
  FREE(str_b);
}

void test_expo_mod (void) {
  
  mpz_t m, N, e, rop;
  char *str_m   = (char *)malloc(INPUT_MAX_LENGTH);
  char *str_N   = (char *)malloc(INPUT_MAX_LENGTH);
  char *str_e   = (char *)malloc(INPUT_MAX_LENGTH);
  char *str_rop = NULL;
  
  puts("---- Q 6.3 expo_mod ----");
  puts("Base m: ");
  scanf("%s",str_m);
  puts("Exponentiation e: ");
  scanf("%s",str_e);
  puts("Modulo N: ");
  scanf("%s",str_N);
  mpz_init_set_str(m, str_m, BASE);
  mpz_init_set_str(e, str_e, BASE);
  mpz_init_set_str(N, str_N, BASE);
  mpz_init(rop);
  
  expo_mod(rop, m, e, N);
  str_rop = mpz_get_str(str_rop, BASE, rop);
  printf("Exponentiation Modulo: %s\n",str_rop);
  
  mpz_clear(m);
  mpz_clear(N);
  mpz_clear(e);
  mpz_clear(rop);
  FREE(str_m);
  FREE(str_N);
  FREE(str_e);
  FREE(str_rop);
}

int main( int argc, char *argv[]) {
  puts("==== Exercise 6 Arithmetique dans Zn ====");
  if ( argc > 1 ) {
    switch ( argv[1][0] ) {
    case '1':
      test_my_pgcd ();
      break;
    case '2':
      test_my_inverse();
      break;
    case '3':
      test_expo_mod();
      break;
    default:
      puts("choix de Question: 1, 2, 3, else quit");
      break;
    }
  }
  
  return 0;
}
