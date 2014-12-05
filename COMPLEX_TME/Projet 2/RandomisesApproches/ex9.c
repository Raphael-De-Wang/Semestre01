#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"
void test_TestFermat(void) {
  puts("---- Q 9.1 test fermat ----");
  test_callback(TestFermat, "nombre premier");
}

void test_TestFermat_2(void) {
  puts("---- Q 9.2 test fermat ----");
  test_callback_complete(TestFermat);
}

void test_TestFermat_probs_erreur(void) {
  puts("---- Q 9.3 Probabilite d'erreur de TestFermat ----");
  test_probs_erreur_CallBack(TestFermat_probs_erreur);
}

void test_plus_grand_entier_TestFermat_trouve(void) {
  puts("---- Q 9.4 plus grand entier trouve par TestFermat ----");
  test_plus_grand_entier_trouve_CallBack(plus_grand_entier_TestFermat);
}

int main( int argc, char *argv[]) {
  puts("==== Exercise 9 Test de Fermat ====");
  //test_plus_grand_entier_TestFermat_trouve();
  if ( argc > 1 ) {
    switch ( argv[1][0] ) {
    case '1':
      test_TestFermat();
      break;
    case '2':
      test_TestFermat_2();
      break;
    case '3':
      test_TestFermat_probs_erreur();
      break;
    case '4':
      test_plus_grand_entier_TestFermat_trouve();
      break;
    default:
      puts("choix de Question: 1, 2, 3, 4, else quit");
      break;
    }
  }
  
  return 0;
}

