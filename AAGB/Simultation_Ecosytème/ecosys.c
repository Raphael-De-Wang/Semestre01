#include <stdio.h>
#include <stdlib.h>
#include "ecosys.h"

Animal *creer_animal(int x, int y, float energie) {
  Animal *newBorn = (Animal*)malloc(sizeof(Animal));
  newBorn->x = x;
  newBorn->y = y;
  newBorn->energie   = energie;
  newBorn->precedent = NULL;
  newBorn->suivant   = NULL;
  return newBorn;
}

void enlever_animal(Animal **liste, Animal *animal) {
  if (animal) {
    
    if (animal->precedent){
      animal->precedent->suivant = animal->suivant;
      animal->precedent = NULL;
    } else {
      // animal est la tete du liste
      *liste = animal->suivant;
    }
    
    if (animal->suivant) {
      animal->suivant->precedent = animal->precedent;
      animal->suivant   = NULL;
    }
    
    FREE(animal);
  }
}

void ajouter_animal(int x, int y, float energie, Animal **liste_animal) {
  Animal *newBorn = creer_animal(x, y, energie);
  *liste_animal = ajouter_en_tete_animal(*liste_animal, newBorn);
}

Animal *ajouter_en_tete_animal(Animal *liste,  Animal *animal) {
  if (liste) {
    liste->precedent = animal;
    animal->suivant  = liste;
  }
  // if liste est NULL, animal est le nouveau liste
  return animal;
}

void liberer_liste_animaux(Animal *liste) {
  while(liste) {
    enlever_animal(&liste, liste);
  }
}

unsigned int compte_animal(Animal *la) {
  unsigned int count = 0;
  Animal *ptr = NULL;
  for ( ptr = la; ptr != NULL; ptr = ptr->suivant ) {
    count++;
  }
  return count;
}

unsigned int compte_animal_rec(Animal *la) {
  return compte_animal(la);
}

Animal *animal_en_XY(Animal *liste, int x, int y) {
  Animal *ptr = NULL;
  for ( ptr = liste; ptr != NULL; ptr = ptr->suivant ) {
    if (ptr->x == x && ptr->y == y) {
      return ptr;
    }
  }
  return NULL;
}

void afficher_ecosys(Animal *liste_predateur, Animal *liste_proie) {
  
}



/****
unsigned int compte_animal_it(Animal *la);

void rafraichir_predateurs(Animal **liste_predateur, Animal **liste_proie, float d_predateur, float p_ch_dir,  float p_reproduce, float energie,  float p_manger);
void rafraichir_proies(Animal **liste_proie, float d_proie, float p_ch_dir,  float p_reproduce, float energie);

void clear_screen();
*****/
