#include <iostream>
#include <cstdlib>
#include <gmpxx.h>
#include <ctime>
#include <vector>

/*export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib*/
using namespace std;

void my_random(mpz_class& destination, mpz_class borne){
  gmp_randstate_t gmpRandState; /* Random generator state object */
  gmp_randinit_default(gmpRandState);
  gmp_randseed_ui(gmpRandState, time(0));
  mpz_urandomm(destination.get_mpz_t(), gmpRandState, borne.get_mpz_t());
}

void my_pgcd(mpz_class a, mpz_class b, mpz_class& pgcd){
	if(b > a){
		mpz_class c;
		c = b;
		b = a;
		a = c;
	}
	mpz_class r(a%b);
	while (r != 0){
		a = b;
		b = r;
		r = a%b;
	}
	pgcd = b;
}

void testPGCD(){
	mpz_class a, b, pgcd;
	cout << "a = ";
	cin >> a;
	cout << "b = ";
	cin >> b;    
	my_pgcd(b, a, pgcd);
	cout << "pgcd("<<a<<", "<<b<<") = "<<pgcd<<endl;
}

bool first_test(mpz_class n){
	for(int i = 3; i <= sqrt(n); i+=2){
		if (n%i==0){
			return false;
		}
	}
	return true;
}

size_t testNaif(size_t max){
	size_t nb = 0;
	for(unsigned int i = 3; i < max ; i+=2){
		if (first_test(i)){
			nb ++;
		}
	}
	return nb;
}

vector<mpz_class> generer_premiers (mpz_class max){
  vector<mpz_class> premiers;
  for(size_t i = 3; i < max; i = i+2){
    if (first_test(i)){
      premiers.push_back(i);
    }
  }
  return premiers;
}
/*
size_t tempsTestNaif(mpz_class a){
	time_t tbegin;
	bool premier;
	do{
	tbegin=time(NULL);
	premier = first_test(a);
	cout << "temps : " << difftime(time(NULL), tbegin) << "sec"<< "premier ? "<<premier;
	}*/

void plus_grand_premier_naif(){
  time_t tbegin;
  mpz_class a(10000000), prime; 
  do{
    tbegin = time(NULL);
    if(first_test(a)){
      prime = a;;
    }
    a = a + 1;
  }while( difftime(time(NULL), tbegin) < 60 );
  cout << prime << endl;
}
/*
bool Is_Carmichael(mpz_class n){
  mpz_class p = 2, gcd, em, n2;
  do{
    p++;
    my_pgcd(n, p, gcd);    
  }while(gcd != 1);
  n2 = n-1;
  mpz_powm(em.get_mpz_t(), p.get_mpz_t(), n2.get_mpz_t(), n.get_mpz_t());
  return em == 1;
}
*/


bool Is_Carmichael(mpz_class N){
  bool flag(true);
  int i(0), nb_fac(3);
  int len_div = nb_fac + 1;
  mpz_class k(0), r(0), N_1(0), un(1);
  vector<mpz_class> div;
  for ( i = 0; i < len_div ; i++) {
    div.push_back(2);
  }
  /** factoriser **/
  i = 0;
  mpz_sqrt (k.get_mpz_t(), N.get_mpz_t());
  while ( i < len_div  && mpz_cmp(div[i].get_mpz_t(),k.get_mpz_t()) <= 0 ) {
    mpz_mod (r.get_mpz_t(), N.get_mpz_t(), div[i].get_mpz_t());
    if (r  == 0 ) {
      div[i+1] = div[i];
      i += 1;
    }
    mpz_nextprime (div[i].get_mpz_t(), div[i].get_mpz_t());
  }
  /** verification **/
  if ( i == nb_fac ) {
    N_1 = N - un;
    for (i = 0; i < nb_fac; i++) {
      div[i] =  div[i]- un;
      mpz_mod (r.get_mpz_t(), N_1.get_mpz_t(), div[i].get_mpz_t());
      if (r != 0 ) {
	flag = false;
      }
    }
  } else {
    flag = false;
  }
  return flag;
}

void plus_grand_premier_carmichael(){
  time_t tbegin;
  mpz_class a(2), prime; 
  tbegin = time(NULL);
  do{
    if(Is_Carmichael(a)){
      prime = a;;
    }
    a = a + 1;
  }while( difftime(time(NULL), tbegin) < 60 );
  cout << prime << endl;
}

void Gen_Carmichael(mpz_class& n, mpz_class N){
  vector<mpz_class> premiers; 
  mpz_class a, b, c;
  int alea;
  bool carmichael(false);
  premiers = generer_premiers(N);
  while( ! carmichael){
    alea = rand() % premiers.size();
    a = premiers[alea];
    do{
      alea = rand() % premiers.size();
      b = premiers[alea];
    }while(b == a);
    do{
      alea = rand() % premiers.size();
      c = premiers[alea];
    }while(b == a || c == a);
    
    n = a * b * c;
    if (((n-1) % (a - 1) == 0) && ((n-1) % (b - 1) == 0) && ((n-1) % (c - 1) == 0))
      carmichael = true;
  }
}

void TestCarmichael(){
  int nb = 0;
  for(int i = 3; i < 100000; i+=2){
    if (Is_Carmichael(i)){
      nb ++;
      cout << i << endl;
    }
  }
  cout << nb << endl;
}

bool TestFermat(mpz_class N){
  mpz_class a(0);
  N = N - 1;
  my_random(a, N);
  N = N + 1;
  mpz_class reste(0), nm;
  nm = N-1;
  mpz_powm(reste.get_mpz_t(), a.get_mpz_t(), nm.get_mpz_t(), N.get_mpz_t()); 
  return reste == 1;
} 

bool TestRabinMiller(mpz_class N){
  mpz_class n = N, s, r, a, reste;
  n -= 1;
  s = 0;
  while (n%2==0) {
      n /= 2;
      s += 1;
  }
  r = n;
  my_random(a, N-1);
  mpz_powm(reste.get_mpz_t(), a.get_mpz_t(), r.get_mpz_t(), N.get_mpz_t());
  if (reste != 1 && reste != N-1){
    mpz_class j(1), deux(2);
    while(j < s-1){
      mpz_powm(reste.get_mpz_t(), reste.get_mpz_t(), deux.get_mpz_t(), N.get_mpz_t());
      if (reste == 1){
	return false;
      } else if(reste == N-1){
	return true;
      }
    }
    return false;
  }
  return true;
}



int main (void){
  //testPGCD();
  //cout << "nombre de premiers inferieur a 100 000 : " << testNaif(100000) << endl;
  //tempsTestNaif(a);
  //mpz_class a, n;
  //plus_grand_premier_naif();
  //cout << "tirage dans : ";
  //cin >> n;
  //cout << "Carmichael ? " << Is_Carmichael(a);
  //cout << "TestFermat ? " << TestFermat(a);
  //cout << "RabinMiller ? " << TestRabinMiller(a);
  //plus_grand_premier_carmichael();
  //Gen_Carmichael(a, n);
  //Gen_Carmichael(a, n);
  //cout << "Carmichael : " << a << endl;
  TestCarmichael();
  return 0;
}
