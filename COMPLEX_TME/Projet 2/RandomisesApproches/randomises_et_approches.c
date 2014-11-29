#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"

/** Exercice 6 Arithmetique dans Zn **/
void my_pgcd(mpz_t rop, mpz_t op1, mpz_t op2) {
  mpz_gcd ( rop, op1, op2);
}

int my_inverse(mpz_t b, mpz_t a, mpz_t N) {
  return mpz_invert ( b, a, N);
}

void expo_mod(mpz_t rop, mpz_t m, mpz_t e, mpz_t N) {
  mpz_powm (rop, m, e, N);
}

/** Exercice 7 Test Naif **/
int first_test(mpz_t N) {
  int is_premier = TRUE;
  mpz_t k, div, r, zero;
  mpz_init(k);
  mpz_init(r);
  mpz_init_set_str(div,  "2", BASE);
  mpz_init_set_str(zero, "0", BASE);
  mpz_sqrt (k, N);

  while ( mpz_cmp(div, k) <= 0 ) {
    mpz_mod (r, N, div);
    if (mpz_cmp(r, zero) == 0 ) {
      is_premier = FALSE;
      break;
    }
    mpz_nextprime (div, div);
  }

  mpz_clear(k);
  mpz_clear(div);
  mpz_clear(r);
  mpz_clear(zero);
  
  return is_premier;
}

void naif_premier_compteur(mpz_t numbre, mpz_t interval) {
  
  mpz_t N;
  mpz_init_set_str(N, "2", BASE);
  
  while ( mpz_cmp(N, interval) <= 0 ) {
    mpz_add_ui(numbre, numbre, 1);
    mpz_nextprime (N, N);
  }
  
  mpz_clear(N);
}
