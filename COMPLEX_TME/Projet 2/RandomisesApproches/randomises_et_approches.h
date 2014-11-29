#define INPUT_MAX_LENGTH 1000
#define BASE 10
#define TRUE  1
#define FALSE 0
#define FREE(ptr) if ( (ptr) != NULL ) { free(ptr); ptr = NULL; }


/** Exercice 6 Arithmetique dans Zn **/
void my_pgcd(mpz_t rop, mpz_t op1, mpz_t op2);
int my_inverse(mpz_t b, mpz_t a, mpz_t N);
void expo_mod(mpz_t exp, mpz_t m, mpz_t e, mpz_t N);

/** Exercice 7 Test Naif **/

/** Exercice 8 Nombres de Carmichael **/

/** Exercice 9 Test de Fermat **/

/** Exercise 10 Test de Rabin et Miller **/

