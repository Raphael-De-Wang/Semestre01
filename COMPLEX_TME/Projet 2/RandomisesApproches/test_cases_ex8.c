#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"
void test_is_carmichael(void){
  
  mpz_t N;
  char *str_N = (char *)malloc(INPUT_MAX_LENGTH);

  puts("---- Q 8.1 verifier nombre carmichael ----");
  puts("Grand Entier : ");
  scanf("%s",str_N);
  mpz_init_set_str(N, str_N, BASE);
  
  if ( Is_Carmichael(N) == TRUE ){
    printf("%s est un nombre carmichael! \n", str_N);
  } else {
    printf("%s n'est pas un nombre carmichael. \n", str_N);
  }

  mpz_clear(N);
  FREE(str_N);

}
int main( int argc, char *argv[]) {
  puts("==== Exercise 8 Test Naif ====");

  if ( argc > 1 ) {
    switch ( argv[1][0] ) {
    case '1':
      test_is_carmichael();
      break;
    case '2':
      break;
    case '3':
      break;
    case '4':
      break;
    case '5':
      break;
    default:
      break;
    }
  } else {
    puts("choix de Question: 1, 2, 3, 4, 5, else quit");
  }
  return 0;
}

