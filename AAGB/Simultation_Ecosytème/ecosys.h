#ifndef _ECOSSYS_H_
#define _ECOSYS_H_

#define SIZE_X 50
#define SIZE_Y 50
#define NB_PROIES 20
#define NB_PREDATEURS 20

#define DIR_UP    -1
#define DIR_DOWN   1
#define DIR_STAY   0

#define VIDE      ' '
#define PROIE     '*'
#define PREDATEUR 'O'
#define COEXIST   '@'

#define TRUE       1
#define FALSE      0

#define FREE(ptr) if ( (ptr) != NULL ) { free(ptr); ptr = NULL; }

/* Parametres globaux de l'ecosysteme (externes dans le ecosys.h)*/

typedef struct _animal {
  int x;
  int y;
  int dir[2]; /* direction courante sous la forme (dx, dy) */
  float energie;
  struct _animal *precedent;
  struct _animal *suivant;
} Animal;

/** outil **/
int random_true_false (float prob);

Animal *creer_animal(int x, int y, float energie);
void ajouter_animal(int x, int y, float energie, Animal **liste_animal);
Animal *ajouter_en_tete_animal(Animal *liste,  Animal *animal);
void enlever_animal(Animal **liste, Animal *animal);
void liberer_liste_animaux(Animal *liste);

unsigned int compte_animal(Animal *la);
unsigned int compte_animal_rec(Animal *la);
unsigned int compte_animal_it(Animal *la);
Animal *animal_en_XY(Animal *liste_animal, int x, int y);

void localiser_animal(char ecosys[SIZE_Y][SIZE_X], Animal *liste, char animal);
void init_ecosys(char ecosys[SIZE_Y][SIZE_X]);
void afficher_ecosys(Animal *liste_predateur, Animal *liste_proie);
void draw_ecosys(char ecosys[SIZE_Y][SIZE_X]);

void open_screen(void);
void clear_screen(void);
int write_screen(const char *fmt, ...);
int pose_screen(void);
void close_screen(void);

int random_true_false (float prob);
int random_dir (void);
void change_dir (int *dir);

void bouger_en_dir (Animal *animal);
void bouger_animaux(Animal *la, float p_ch_dir);
void reproduce(Animal **liste_animal, float p_reproduce, float energie);
void user_energie (Animal **liste_animal, Animal *animal, float d_animal);
void random_manger (Animal **liste_proie, Animal *proie, float p_manger);

void rafraichir_predateurs(Animal **liste_predateur, Animal **liste_proie, float d_predateur, float p_ch_dir,  float p_reproduce, float energie,  float p_manger);
void rafraichir_proies(Animal **liste_proie, float d_proie, float p_ch_dir,  float p_reproduce, float energie);

#endif
