#ifndef _ECOSSYS_H_
#define _ECOSYS_H_

#define SIZE_X 50
#define SIZE_Y 50
#define NB_PROIES 20
#define NB_PREDATEURS 20

/* Parametres globaux de l'ecosysteme (externes dans le ecosys.h)*/

typedef struct _animal {
  int x;
  int y;
  int dir[2]; /* direction courante sous la forme (dx, dy) */
  float energie;
  struct _animal *precedent;
  struct _animal *suivant;
} Animal;

Animal *creer_animal(int x, int y, float energie);
void ajouter_animal(int x, int y, float energie, Animal **liste_animal);
Animal *ajouter_en_tete_animal(Animal *liste,  Animal *animal);
void enlever_animal(Animal **liste, Animal *animal);
void liberer_liste_animaux(Animal *liste);

unsigned int compte_animal_rec(Animal *la);
unsigned int compte_animal_it(Animal *la);
Animal *animal_en_XY(Animal *l, int x, int y);

void afficher_ecosys(Animal *liste_predateur, Animal *liste_proie);

void rafraichir_predateurs(Animal **liste_predateur, Animal **liste_proie, float d_predateur, float p_ch_dir,  float p_reproduce, float energie,  float p_manger);
void rafraichir_proies(Animal **liste_proie, float d_proie, float p_ch_dir,  float p_reproduce, float energie);

void clear_screen();

#endif
