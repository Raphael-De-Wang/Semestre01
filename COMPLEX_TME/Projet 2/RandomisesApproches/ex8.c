#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

#include "randomises_et_approches.h"

void test_is_carmichael(void){
  
  puts("---- Q 8.1 verifier nombre carmichael ----");
  test_callback(Is_Carmichael, "nombre carmichael");
}

void test_plus_grand_entier_carmichael(void) {
  puts("---- Q 8.2  plus grand entier qu'on peut tester ----");
  test_plus_grand_entier_CallBack(plus_grand_entier_carmichael);
}

void test_Gen_Carmichael(void) {
  mpz_t N;
  mpz_t borne;
  char *borne_str = NULL;
  char *N_str     = NULL;
  
  puts("---- Q 8.3 Generer un nombre de Carmichael ----");  
  puts("Indiquer le borne des facteurs premiers : ");
  scanf("%s", borne_str);
  mpz_init(N);
  mpz_init_set_str(borne, borne_str, BASE);
  Gen_Carmichael(N, borne);
  printf("Nombre de Carmichael Aleatoire: %s \n", mpz_get_str(N_str, BASE, N));
  
  mpz_clears(borne, N, NULL);
  FREE(borne_str);
  FREE(N_str);
}

void test_lister_nombres_carmichael(void) {
  int i, len = 0;
  mpz_t borne;
  mpz_t** list = NULL;
  char *str = (char *)malloc(sizeof(char)*INPUT_MAX_LENGTH);

  puts("---- Q 8.4 lister les nombres de Carmichael ----");  
  puts("Indiquer le borne : ");
  scanf("%s", str);
  puts("Indiquer le longueur du list : ");
  scanf("%d", &len);
  mpz_init_set_str(borne, str, BASE);
  list = lister_nombres_carmichael(borne, len);
  for ( i = 0; i < len ; i++ ) {
    if (list[i]) {
      printf("Nombre de Carmichael [%d]: %s \n", i, mpz_get_str(str, BASE, *list[i]));
    }
  }
  free_nombres_carmichael_list(list, len);
  FREE(str);
}

void test_plus_grand_entier_carmichael_trouve(void){
  puts("---- Q 8.5  plus grand entier qu'on peut tester en temps limite----");  
  test_plus_grand_entier_trouve_CallBack(plus_grand_entier_carmichael_trouve);
}

int main( int argc, char *argv[]) {
  puts("==== Exercise 8 Nombre de Carmichael ====");

  if ( argc > 1 ) {
    switch ( argv[1][0] ) {
    case '1':
      test_is_carmichael();
      break;
    case '2':
      test_plus_grand_entier_carmichael();
      break;
    case '3':
      test_Gen_Carmichael();
      break;
    case '4':
      test_lister_nombres_carmichael();
      break;
    case '5':
      test_plus_grand_entier_carmichael_trouve();
      break;
    default:
      puts("choix de Question: 1, 2, 3, 4, 5, else quit");
      break;
    }
  }
  
  return 0;
}

