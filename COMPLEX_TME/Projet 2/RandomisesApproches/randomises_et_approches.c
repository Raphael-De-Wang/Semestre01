#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>
#include <time.h>

#include "randomises_et_approches.h"

void  test_callback(int (*CallBack)(const mpz_t), char * msg) {
  mpz_t N;
  char *str_N = (char *)malloc(INPUT_MAX_LENGTH);

  puts("Grand Entier : ");
  scanf("%s",str_N);
  mpz_init_set_str(N, str_N, BASE);
  
  if ( CallBack(N) == TRUE ){
    printf("%s est un %s! \n", str_N, msg);
  } else {
    printf("%s n'est pas un %s. \n", str_N, msg);
  }

  mpz_clear(N);
  FREE(str_N);
}

void  test_callback_complete(int (*CallBack)(const mpz_t)) {
  mpz_t N, borne;
  char *str = (char *)malloc(INPUT_MAX_LENGTH);
  mpz_init(N);
  
  puts("Indiquer le borne des facteurs premiers : ");
  scanf("%s",str);
  mpz_init_set_str(borne, str, BASE);

  puts("1) des nombres retournes par Gen_Garmichael: ");
  Gen_Carmichael(N, borne);
  if ( CallBack(N) == TRUE ){
    printf("%s est un nombre premier! \n", mpz_get_str(str, BASE, N));
  } else {
    printf("%s n'est pas un nombre premier. \n", mpz_get_str(str, BASE, N));
  }

  puts("2) des nombres compose: ");
  puts("Grand Entier Compose: ");
  scanf("%s",str);
  mpz_set_str(N, str, BASE);
  if ( CallBack(N) == TRUE ){
    printf("%s est un nombre premier! \n", mpz_get_str(str, BASE, N));
  } else {
    printf("%s n'est pas un nombre premier. \n", mpz_get_str(str, BASE, N));
  }

  puts("3) des nombres aleatoire: ");
  random_nombre_m( N, borne);
  if ( CallBack(N) == TRUE ){
    printf("%s est un nombre premier! \n", mpz_get_str(str, BASE, N));
  } else {
    printf("%s n'est pas un nombre premier. \n", mpz_get_str(str, BASE, N));
  }

  mpz_clears(N,borne,NULL);
  FREE(str);
}

void test_probs_erreur_CallBack(void(*CallBack)(mpf_t probs, const mpz_t borne)) {
  mpz_t borne;
  mpf_t probs;
  char *str = (char *)malloc(INPUT_MAX_LENGTH);
  //mp_exp_t expptr = 0;
    
  puts("Indiquer l'interval du Test : ");
  scanf("%s",str);
  mpf_init(probs);
  mpz_init_set_str(borne, str, BASE);
  CallBack(probs, borne);
  //mpf_get_str( str, &expptr, BASE, FLOAT_DIGITS, probs);
  gmp_printf("Probabilite d'erreur de TestFermat est %.*Ff \n", 6, probs);

  mpz_clear(borne);
  mpf_clear(probs);
  FREE(str);
}

void  test_plus_grand_entier_CallBack(void(*CallBack)(mpz_t, int)) {
  int   sec       = 0;
  char *borne_str = NULL;
  mpz_t borne;
  
  puts("Tester en secondes : ");
  scanf("%d", &sec);
  mpz_init_set_str(borne, "1000000", BASE);
  CallBack(borne, sec);
  printf("Le premier grand nombre que la fonction ne peut pas tester en %d secondes:%s \n", sec, mpz_get_str(borne_str, BASE, borne));
  
  mpz_clear(borne);
  FREE(borne_str);
}

void test_plus_grand_entier_trouve_CallBack(void(*CallBack)(mpz_t, int)) {
  mpz_t borne;
  int sec;
  char *str = NULL;
    
  puts("Tester en secondes : ");
  scanf("%d",&sec);
  mpz_init_set_str(borne, "2", BASE);
  CallBack(borne, sec);
  printf("Le plus grand nombre que la fonction a trouve en %d secondes: %s \n", sec, mpz_get_str(str, BASE, borne));

  mpz_clear(borne);
  FREE(str);
}

void  probs_erreur(mpf_t probs, const mpz_t borne, int (*CallBack)(const mpz_t)) {
  mpz_t N;
  mpf_t borne_f, echec;
  char *str = NULL;
  
  mpf_inits(borne_f,echec,NULL);
  mpz_init_set_str(N,"2",BASE);
  
  while ( mpz_cmp ( N, borne ) < 0 ) {
    if (CallBack(N)) {
      if (!first_test(N)) {
	mpf_add_ui(echec,echec,1);	
      }
    }
    mpz_add_ui(N,N,1);
  }
  str = mpz_get_str(str,BASE,borne);
  mpf_init_set_str(borne_f, str, BASE);
  mpf_div(probs, echec, borne_f);
  mpz_clear(N);
  mpf_clears(borne_f,echec,NULL);
  FREE(str);
}

void plus_grand_entier_sup_tlimit(mpz_t borne, int sec, int (*CallBack)(const mpz_t)) {
  int t, max = 0;
  while( TRUE ) {
    int start = time(NULL);
    CallBack(borne);
    t = time(NULL) - start;
    if (t > max){
      max = t;
      printf("t: %d\t", max);
      gmp_printf("borne: %Zd\n", borne);
    }
    if ( t > sec ) {
      break;
    }
    mpz_add_ui( borne, borne, 1);
  }
}

void plus_grand_entier_inf_tlimit(mpz_t borne, int sec, int (*CallBack)(const mpz_t)) {
  int start = time(NULL);
  while( time(NULL) - start < sec ){
    CallBack(borne);
    mpz_add_ui(borne, borne, 1);
  }
}

void random_nombre_b(mpz_t rand, mp_bitcnt_t n) {
  gmp_randstate_t state;
  gmp_randinit_default (state);
  mpz_urandomb (rand, state, n);
}

void random_nombre_m(mpz_t rand, const mpz_t borne) {
  gmp_randstate_t state;
  gmp_randinit_default (state);
  mpz_urandomm (rand, state, borne);
}

void random_nombre_premier (mpz_t rand, const mpz_t borne) {
  random_nombre_m(rand, borne);
  mpz_nextprime (rand, rand);
}

/** Exercice 6 Arithmetique dans Zn **/
void my_pgcd(mpz_t rop, const mpz_t op1, const mpz_t op2) {
  mpz_gcd ( rop, op1, op2);
}

int my_inverse(mpz_t b, const mpz_t a, const mpz_t N) {
  return mpz_invert ( b, a, N);
}

void expo_mod(mpz_t rop, const mpz_t m, const mpz_t e, const mpz_t N) {
  mpz_powm (rop, m, e, N);
}

/** Exercice 7 Test Naif **/
int first_test(const mpz_t N) {
  int is_premier = TRUE;
  mpz_t k, div, r;
  mpz_inits(k, r, NULL);
  mpz_init_set_str(div,  "2", BASE);
  mpz_sqrt (k, N);
  
  while ( mpz_cmp(div, k) <= 0 ) {
    mpz_mod (r, N, div);
    if (mpz_cmp_si(r, 0) == 0 ) {
      is_premier = FALSE;
      break;
    }
    mpz_nextprime (div, div);
  }
  
  mpz_clears(k,div,r,NULL);
  
  return is_premier;
}

void naif_premier_compteur(mpz_t numbre, const mpz_t interval) {
  
  mpz_t N;
  mpz_init_set_str(N, "2", BASE);
  
  while ( mpz_cmp(N, interval) <= 0 ) {
    mpz_add_ui(numbre, numbre, 1);
    mpz_nextprime (N, N);
  }
  
  mpz_clear(N);
}

void  plus_grand_entier_TestNaif(mpz_t borne, int sec){
  int (*CallBack)(const mpz_t);
  CallBack = &first_test;
  plus_grand_entier_sup_tlimit( borne, sec, CallBack);
}
 
/** Exercice 8 Nombres de Carmichael **/

int Is_Carmichael_Facteuriser(mpz_t div[], int len_div, const mpz_t N) {
  /** facteuriser **/
  int i = 0;
  mpz_t k, r;
  
  mpz_inits(k,r,NULL);
  mpz_sqrt(k, N);
  
  while ( mpz_cmp(div[i],k) <= 0 ) {
    mpz_mod (r, N, div[i]);
    if (mpz_cmp_ui(r, 0) == 0 ) {
      mpz_set(div[i+1], div[i]);
      i++;
    }
    
    if ( i > 3 ) break;
    
    mpz_nextprime (div[i], div[i]);
  }

  mpz_clears(k,r,NULL);
  
  return i;
}

int Is_Carmichael_verification( mpz_t div[], const mpz_t N) {
  int i;
  int flag = TRUE;
  mpz_t r, N_1, div_1;

  mpz_inits(r, N_1, div_1, NULL);
  mpz_sub_ui (N_1, N, 1);
  
  for (i = 0; i < 3; i++) {
    mpz_sub_ui (div_1, div[i], 1);
    mpz_mod (r, N_1, div_1);
    if (mpz_cmp_ui(r, 0) != 0 ) {
      flag = FALSE;
      break;
    }
  }
  
  mpz_clears(N_1,div_1,r,NULL);
  
  return flag;
}

int Is_Carmichael(const mpz_t N){
  
  int flag    = TRUE;
  int i       = 0;
  int nom_fac = 3;
  int len_div = nom_fac + 2;
  mpz_t div[len_div];
  
  for ( i = 0; i < len_div ; i++) {
    mpz_init_set_str(div[i], "2", BASE);
  }

  i = Is_Carmichael_Facteuriser(div, len_div, N);
  
  if ( i == nom_fac ) {
    flag = Is_Carmichael_verification(div, N);
  } else {
    flag = FALSE;
  }

  for ( i = 0; i < len_div; i++) {
    mpz_clear(div[i]);
  }

  return flag;
}

void plus_grand_entier_carmichael(mpz_t borne, int sec) {
  int (*CallBack)(const mpz_t);
  CallBack = &Is_Carmichael;
  plus_grand_entier_sup_tlimit(borne, sec, CallBack);
}

void  plus_grand_entier_carmichael_trouve(mpz_t borne, int sec) {
  int (*CallBack)(const mpz_t);
  CallBack = &Is_Carmichael;
  plus_grand_entier_inf_tlimit(borne, sec, CallBack);
}

void Gen_Carmichael(mpz_t N, const mpz_t borne) {
  int i       = 0;
  int nom_fac = 3;
  mpz_t div[nom_fac];
  
  for ( i = 0; i < nom_fac ; i++) {
    mpz_init(div[i]);
  }
  
  do {
    for ( i = 0; i < nom_fac ; i++) {
      random_nombre_premier(div[i], borne);
    }
    mpz_mul (N, div[0], div[1]);
    mpz_mul (N, N, div[2]);
  } while( !Is_Carmichael(N) );
  
  mpz_clears(div[0], div[1], div[2], NULL);
}

mpz_t** lister_nombres_carmichael(const mpz_t borne, int len) {
  int i = 0;
  mpz_t N;
  mpz_t **carmichaels = NULL;

  mpz_init_set_str(N,"2",BASE);
  carmichaels = (mpz_t **)malloc(sizeof(mpz_t*)*len);
  for ( i = 0; i < len; i++ ) {
    carmichaels[i] = NULL;
  }

  i = 0;
  while ( mpz_cmp(N,borne) <= 0 && i < len) {
    if (Is_Carmichael(N)) {
      carmichaels[i] = (mpz_t *)malloc(sizeof(mpz_t));
      mpz_init(*carmichaels[i]);
      mpz_set(*carmichaels[i], N);
      i++;
    }
    mpz_add_ui(N, N, 1);
  }
  
  mpz_clear(N);
  return carmichaels;
}

mpz_t** free_nombres_carmichael_list(mpz_t** list, int len) {
  int i = 0;
  for ( i = 0; i < len; i++ ) {
    if ( list[i] != NULL ){
      mpz_clear(*list[i]);
      list[i] = NULL;
    }
  }
  FREE(list);
  return list;
}

/** Exercice 9 Test de Fermat **/
int TestFermat(const mpz_t N) {
  int flag = FALSE;
  mpz_t r;
  mpz_t base;
  mpz_t exp;
  
  mpz_inits(r,base,exp,NULL);
  random_nombre_m(base, N);
  mpz_sub_ui(exp, N, 1);
  
  mpz_powm(r, base, exp, N);
  if ( mpz_cmp_si(r,1) == 0 ) {
    flag = TRUE;
  }
  
  mpz_clears(r, base, exp, NULL);
  
  return flag;
}

void TestFermat_probs_erreur(mpf_t probs, const mpz_t borne) {
  probs_erreur( probs, borne, TestFermat);
}

void plus_grand_entier_TestFermat(mpz_t borne, int sec) {
  int (*CallBack)(const mpz_t);
  CallBack = &TestFermat;
  plus_grand_entier_sup_tlimit(borne, sec, CallBack);
}

/** Exercise 10 Test de Rabin et Miller **/
int TestRabinMiller(const mpz_t N) {
  int flag = TRUE;
  mp_bitcnt_t i, s = 0;
  mpz_t N_1, reste, a, r;
  
  mpz_inits( N_1, reste, a, r, NULL);
  mpz_sub_ui(N_1, N, 1);
  
  while ( TRUE ) {
    mpz_fdiv_r_2exp (r, N_1, s);
    if ( mpz_cmp_si(r,0) == 0) {
      s++;
    } else {
      s--;
      break;
    }
  }
  
  if ( s > 0 ) {
    mpz_fdiv_q_2exp (r, N_1, s);
    
    do {
      random_nombre_m(a, N_1);
    } while ( mpz_cmp_si(a,2) < 0 );
    
    mpz_powm_sec(reste, a, r, N);
    if ( mpz_cmp_si(reste,1) != 0 && mpz_cmp(reste,N_1) != 0 ) {
      
      for ( i = 1; i <= s - 1; i++ ) {
	mpz_powm_ui (reste, reste, 2, N);
	if (mpz_cmp_ui(reste,1) == 0) {
	  flag = FALSE;
	  break;
	}
	if (mpz_cmp(reste,N_1) == 0) {
	  break;
	}
      }
      if (mpz_cmp(reste,N_1) != 0) {
	flag = FALSE;
      }
    }
  } else {
    flag = FALSE;
  }

  mpz_clears(N_1, reste, a, r, NULL);
  
  return flag;
}

void  TestRobinMiller_probs_erreur(mpf_t probs, const mpz_t borne) {
  probs_erreur( probs, borne, TestRabinMiller);
}

void plus_grand_entier_TestRabinMiller(mpz_t borne, int sec) {
  int (*CallBack)(const mpz_t);
  CallBack = &TestRabinMiller;
  plus_grand_entier_inf_tlimit(borne, sec, CallBack);
}

void GenPKRSA(mpz_t N, int t) {
  mpz_t p,q;
  mpz_inits(p,q,NULL);
  
  random_nombre_b(p, t);
  random_nombre_b(q, t);
  
  while ( !TestFermat(p) && !TestRabinMiller(p)) {
    mpz_add_ui(p,p,1);
  }
  
  while ( !TestFermat(q) && !TestRabinMiller(q)) {
    mpz_add_ui(q,q,1);
  }

  mpz_mul (N, p, q);

  mpz_clears(p,q,NULL);
  
}

void plus_grand_entier_GenPKRSA(mpz_t borne, int sec) {
  int start, t = 10;
  
  do {
    start = time(NULL);
    GenPKRSA( borne, t++);
  } while ( time(NULL) - start < sec );
  
}
