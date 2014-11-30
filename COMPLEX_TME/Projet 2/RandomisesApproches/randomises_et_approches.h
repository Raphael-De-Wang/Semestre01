#define INPUT_MAX_LENGTH 1000
#define BASE 10
#define TRUE  1
#define FALSE 0
#define FREE(ptr) if ( (ptr) != NULL ) { free(ptr); ptr = NULL; }

void  plus_grand_entier(mpz_t borne, int sec, int (*CallBack)(mpz_t N));

/** Exercice 6 Arithmetique dans Zn **/
void  my_pgcd(mpz_t rop, mpz_t op1, mpz_t op2);
int   my_inverse(mpz_t b, mpz_t a, mpz_t N);
void  expo_mod(mpz_t rop, mpz_t m, mpz_t e, mpz_t N);

/** Exercice 7 Test Naif **/
int   first_test(mpz_t N);
void  naif_premier_compteur(mpz_t numbre, mpz_t interval);
void  plus_grand_entier_TestNaif(mpz_t borne, int sec);

/** Exercice 8 Nombres de Carmichael **/
int   Is_Carmichael(mpz_t N);
void  plus_grand_entier_carmichael(mpz_t borne, int sec);
void  Gen_Carmichael(mpz_t rop);
void  lister_nombres_carmichael(mpz_t borne);

/** Exercice 9 Test de Fermat **/
int   TestFermat(mpz_t N);
float TestFermat_probs_erreur(mpz_t borne);
void  plus_grand_entier_TestFermat(mpz_t borne, int sec);

/** Exercise 10 Test de Rabin et Miller **/
int   TestRabinMiller(mpz_t N);
void  plus_grand_entier_TestRabinMiller(mpz_t borne, int sec);

