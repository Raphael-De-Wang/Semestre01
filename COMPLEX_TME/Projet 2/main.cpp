#include <iostream>
#include <gmpxx.h>
#include <ctime>

/*export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib*/
using namespace std;

void my_random(mpz_class& destination, mpz_class borne){
  gmp_randstate_t gmpRandState; /* Random generator state object */
  gmp_randinit_default(gmpRandState);
  gmp_randseed_ui(gmpRandState, time(0));
  mpz_urandomm(destination, gmpRandState, borne.get_mpz_t());
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
	for(int i = 2; i <= sqrt(n); i++){
		if (n%i==0){
			return false;
		}
	}
	return true;
}

size_t testNaif(size_t max){
	size_t nb = 0; 
	for(int i = 2; i < max; i++){
		if (first_test(i)){
			nb ++;
		}
	}
	return nb;
}

size_t tempsTestNaif(mpz_class a){
	time_t tbegin;
	bool premier;
	tbegin=time(NULL);
	premier = first_test(a);
	cout << "temps : " << difftime(time(NULL), tbegin) << "sec"<< "premier ? "<<premier;
}

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
//return true if prime
bool TestFermat(mpz_class N){
  mpz_t a;
  mpz_init(a);
  N = N - 1;
  my_random(a, N);
  N = N + 1;
  mpz_class reste(0), nm;
  nm = N-1;
  mpz_powm(reste.get_mpz_t(), a, nm.get_mpz_t(), N.get_mpz_t()); 
  return reste == 1;
} 

bool TestRabinMiller(mpz_class N){
  mpz_class n = N, s, r;
  n -= 1;
  s = 0;
  while (n%2==0) {
      n /= 2;
      s += 1;
  }
  r = n;
  // my_random(a, N-1)  
  
  return false;
}



int main (void){
  //testPGCD();
  //cout << "nombre de premiers inferieur a 100 000 : " << testNaif(100000) << endl;
  //tempsTestNaif(a);
  mpz_class a;

  cout << "nombre a tester : ";
  cin >> a;
  //cout << "Carmichael ? " << Is_Carmichael(a);
  cout << "TestFermat ? " << TestFermat(a);
  return 0;
}
