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
    animal->precedent->suivant = animal->suivant;
    animal->suivant->precedent = animal->precedent;
    animal->precedent = NULL;
    animal->suivant   = NULL;
    FREE(animal);
  }
}

void ajouter_animal(int x, int y, float energie, Animal **liste_animal) {
  Animal *newBorn = creer_animal(x, y, energie);
  *liste_animal = ajouter_en_tete_animal(*liste_animal, newBorn);
}

Animal *ajouter_en_tete_animal(Animal *liste,  Animal *animal) {
  liste->precedent = animal;
  animal->suivant  = liste;
  return animal;
}

void liberer_liste_animaux(Animal *liste) {
  Animal *ptr = liste;
  while(ptr) {
    ptr = liste->suivant;
    liste->suivant = NULL;
    FREE(liste);
  }
}
unsigned int compte_animal(Animal *la) {
  unsigned int count = 0;
  Animal *ptr = la;
  while(ptr) {
    ptr = la->suivant;
    count++;
  }
  return count;
}

/****
unsigned int compte_animal_rec(Animal *la);
unsigned int compte_animal_it(Animal *la);
Animal *animal_en_XY(Animal *l, int x, int y);

void afficher_ecosys(Animal *liste_predateur, Animal *liste_proie);

void rafraichir_predateurs(Animal **liste_predateur, Animal **liste_proie, float d_predateur, float p_ch_dir,  float p_reproduce, float energie,  float p_manger);
void rafraichir_proies(Animal **liste_proie, float d_proie, float p_ch_dir,  float p_reproduce, float energie);

void clear_screen();
*****/
