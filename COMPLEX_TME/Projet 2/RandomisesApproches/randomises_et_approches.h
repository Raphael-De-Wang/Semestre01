#define INPUT_MAX_LENGTH 1000
#define FLOAT_DIGITS     5
#define BASE 10
#define TRUE  1
#define FALSE 0
#define FREE(ptr) if ( (ptr) != NULL ) { free(ptr); ptr = NULL; }

/** Utils **/
void  test_callback(int (*CallBack)(const mpz_t), char *);
void  test_callback_complete(int (*CallBack)(const mpz_t));
void  test_probs_erreur_CallBack(void(*CallBack)(mpf_t, const mpz_t));
void  test_plus_grand_entier_CallBack(void(*CallBack)(mpz_t, int));
void  test_plus_grand_entier_trouve_CallBack(void(*CallBack)(mpz_t, int));
  
void  plus_grand_entier_sup_tlimit(mpz_t borne, int sec, int (*CallBack)(const mpz_t));
void  plus_grand_entier_inf_tlimit(mpz_t borne, int sec, int (*CallBack)(const mpz_t));

void  random_nombre_b(mpz_t rand, const mp_bitcnt_t n);
void  random_nombre_m(mpz_t rand, const mpz_t borne);
void  random_nombre_premier (mpz_t rand, const mpz_t borne);
void  probs_erreur(mpf_t probs, const mpz_t borne, int (*CallBack)(const mpz_t));

/** Exercice 6 Arithmetique dans Zn **/
void  my_pgcd(mpz_t rop, const mpz_t op1, const mpz_t op2);
int   my_inverse(mpz_t b, const mpz_t a, const mpz_t N);
void  expo_mod(mpz_t rop, const mpz_t m, const mpz_t e, const mpz_t N);

/** Exercice 7 Test Naif **/
int   first_test(const mpz_t N);
void  naif_premier_compteur(mpz_t numbre, const mpz_t interval);
void  plus_grand_entier_TestNaif(mpz_t borne, const int sec);

/** Exercice 8 Nombres de Carmichael **/
int   Is_Carmichael_Facteuriser(mpz_t div[], int len_div, const mpz_t N);
int   Is_Carmichael_verification(mpz_t div[], const mpz_t N);
int   Is_Carmichael(const mpz_t N);
void  plus_grand_entier_carmichael(mpz_t borne, int sec);
void  Gen_Carmichael(mpz_t N, const mpz_t borne);
mpz_t** lister_nombres_carmichael(const mpz_t borne, int len);
mpz_t** free_nombres_carmichael_list(mpz_t** list, int len);
void  plus_grand_entier_carmichael_trouve(mpz_t borne, int sec);

/** Exercice 9 Test de Fermat **/
int   TestFermat(const mpz_t N);
void  TestFermat_probs_erreur(mpf_t probs, const mpz_t borne);
void  plus_grand_entier_TestFermat(mpz_t borne, int sec);

/** Exercise 10 Test de Rabin et Miller **/
int   TestRabinMiller(const mpz_t N);
void  plus_grand_entier_TestRabinMiller(mpz_t borne, int sec);
void  TestRobinMiller_probs_erreur(mpf_t probs, const mpz_t borne);
void  GenPKRSA(mpz_t N, int t);
void  plus_grand_entier_GenPKRSA(mpz_t borne, int sec);
