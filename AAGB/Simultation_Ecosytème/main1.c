#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ecosys.h"


#define NB_PROIES 20
#define NB_PREDATEURS 20



int main(void) {
  int i;
  Animal *liste_proie = NULL;
  Animal *liste_predateur = NULL;
  int nb_proies = 0, nb_predateurs = 0;
/*float p_ch_dir=0.01;
float d_proie=1;
float d_predateur=1;
float p_manger=0.2;
float p_reproduce=0.01;*/
float energie=50;



    srand(time(NULL));

  for (i = 0; i < NB_PROIES; ++i) {
    ajouter_animal(rand() % SIZE_X, rand() % SIZE_Y, energie, &liste_proie);
  }
  for (i=0; i < NB_PREDATEURS; ++i) {
    ajouter_animal(rand() % SIZE_X, rand() % SIZE_Y, energie, &liste_predateur);
  }

  nb_proies = compte_animal_rec(liste_proie);
  nb_predateurs = compte_animal_rec(liste_predateur);
  clear_screen();
  printf("Nb proies :     %5d\n", nb_proies);
  printf("Nb predateurs : %5d\n\n", nb_predateurs);

  afficher_ecosys(liste_proie,liste_predateur);  

  assert(nb_proies == NB_PROIES);
  assert(nb_predateurs == NB_PREDATEURS);

  /* Note : libération de la mémoire non demandée dans l'énoncé (et non fournie) */
  liberer_liste_animaux(liste_proie);
  liberer_liste_animaux(liste_predateur);

  return 0;
}
