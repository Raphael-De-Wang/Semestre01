#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"

void test_TestRabinMiller(void) {
  puts("---- Q 10.1 test TestRabinMiller ----");
  test_callback(TestRabinMiller, "nombre premier");
}

void test_TestRabinMiller_complete(void) {
  puts("---- Q 10.2 test Rabin Miller ----");
  test_callback_complete(TestRabinMiller);
}

void test_TestRobinMiller_probs_erreur(void) {
  puts("---- Q 10.3 Probabilite d'erreur de RabinMiller ----");
  test_probs_erreur_CallBack(TestRobinMiller_probs_erreur);
}

void test_GenPKRSA(void) {
  puts("---- Q 10.4 test GenPKRSA ----");
  int t;
  mpz_t N;
  char * N_str = NULL;
  
  puts("Entrer un Entier >= 0 : ");
  scanf("%d",&t);
  mpz_init(N);
  GenPKRSA(N, t);
  printf("GenPKRSA: %s", mpz_get_str(N_str, BASE, N));
  mpz_clear(N);
  FREE(N_str);
}

void test_plus_grand_entier_GenPKRSA(void) {
  puts("---- Q 10.5  plus grand RSA qu'on peut tester ----");
  test_plus_grand_entier_CallBack(plus_grand_entier_GenPKRSA);
}

int main( int argc, char *argv[]) {
  puts("==== Exercise 10 Test de Rabin et Miller ====");

  if ( argc > 1 ) {
    switch ( argv[1][0] ) {
    case '1':
      test_TestRabinMiller();
      break;
    case '2':
      test_TestRabinMiller_complete();
      break;
    case '3':
      test_TestRobinMiller_probs_erreur();
      break;
    case '4':
      test_GenPKRSA();
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

