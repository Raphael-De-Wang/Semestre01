#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>
#include <time.h>

#include "randomises_et_approches.h"

void plus_grand_entier(mpz_t borne, int sec, int (*CallBack)(mpz_t)) {
  int start = time(NULL);
  while( time(NULL) - start < sec ){
    CallBack(borne);
    mpz_add_ui(borne, borne, 1);
  }
}
 
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

void  plus_grand_entier_TestNaif(mpz_t borne, int sec){
  int (*CallBack)(mpz_t);
  CallBack = &first_test;
  plus_grand_entier( borne, sec, CallBack);
}
 
/** Exercice 8 Nombres de Carmichael **/

int Is_Carmichael(mpz_t N){
  int flag    = TRUE;
  int i       = 0;
  int nom_fac = 3;
  int len_div = nom_fac + 1;
  mpz_t zero;
  mpz_t k;
  mpz_t r;
  mpz_t N_1;
  mpz_t div[len_div];
  
  mpz_init(zero);
  mpz_init(k);
  mpz_init(r);
  mpz_init(N_1);
  for ( i = 0; i < len_div ; i++) {
    mpz_init_set_str(div[i], "2", BASE);
  }
  
  /** facteuriser **/
  i = 0;
  mpz_sqrt (k, N);
  while ( i < len_div  && mpz_cmp(div[i],k) <= 0 ) {
    mpz_mod (r, N, div[i]);
    if (mpz_cmp(r, zero) == 0 ) {
      mpz_set(div[i+1], div[i]);
      i += 1;
    }
    mpz_nextprime (div[i], div[i]);
  }

  /** verification **/
  if ( i == nom_fac ) {
    mpz_sub_ui (N_1, N, 1);
    for (i = 0; i < nom_fac; i++) {
      mpz_sub_ui (div[i], div[i], 1);
      mpz_mod (r, N_1, div[i]);
      if (mpz_cmp(r, zero) != 0 ) {
	flag = FALSE;
      }
    }
  } else {
    flag = FALSE;
  }

  mpz_clear(k);
  mpz_clear(r);
  mpz_clear(N_1);
  for ( i = 0; i < len_div; i++) {
    mpz_clear(div[i]);
  }

  return flag;
}

void plus_grand_entier_carmichael(mpz_t borne, int sec) {
  int (*CallBack)(mpz_t);
  CallBack = &Is_Carmichael;
  plus_grand_entier(borne, sec, CallBack);
}

void Gen_Carmichael(mpz_t rop) {

}
void lister_nombres_carmichael(mpz_t borne) {

}

/** Exercice 9 Test de Fermat **/
int TestFermat(mpz_t N) {
  return 0;
}
float TestFermat_probs_erreur(mpz_t borne) {
  return 0.0;
}

void plus_grand_entier_TestFermat(mpz_t borne, int sec) {
  int (*CallBack)(mpz_t);
  CallBack = &TestFermat;
  plus_grand_entier(borne, sec, CallBack);
}

/** Exercise 10 Test de Rabin et Miller **/
int TestRabinMiller(mpz_t N) {
  return 0;
}

void plus_grand_entier_TestRabinMiller(mpz_t borne, int sec) {
  int (*CallBack)(mpz_t);
  CallBack = &TestRabinMiller;
  plus_grand_entier(borne, sec, CallBack);
}
