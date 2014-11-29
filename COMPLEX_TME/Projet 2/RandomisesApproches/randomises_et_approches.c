#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"

void my_pgcd(mpz_t rop, mpz_t op1, mpz_t op2) {
  mpz_gcd ( rop, op1, op2);
}

int my_inverse(mpz_t b, mpz_t a, mpz_t N) {
  return mpz_invert ( b, a, N);
}

void expo_mod(mpz_t exp, mpz_t m, mpz_t e, mpz_t N) {
  
}
