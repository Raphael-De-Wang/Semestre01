#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ncurses.h>
#include <stdarg.h>
#include "ecosys.h"

Animal *creer_animal(int x, int y, float energie) {
  Animal *newBorn = (Animal*)malloc(sizeof(Animal));
  newBorn->x = x;
  newBorn->y = y;
  newBorn->energie   = energie;
  newBorn->dir[0] = DIR_STAY;
  newBorn->dir[1] = DIR_STAY;
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

unsigned int compte_animal_it(Animal *la) {
  return compte_animal(la);
}

Animal *animal_en_XY(Animal *liste, int x, int y) {
  Animal *ptr = NULL;
  for ( ptr = liste; ptr != NULL; ptr = ptr->suivant ) {
    if (ptr->x == x && ptr->y == y) {
      break;
    }
  }
  return ptr;
}

void localiser_animal(char ecosys[SIZE_Y][SIZE_X], Animal *liste, char animal) {
  Animal *ptr = NULL;
  int x,y;
  for (ptr = liste; ptr != NULL; ptr = ptr->suivant) {
    x = ptr->x;
    y = ptr->y;
    if (ecosys[y][x] == VIDE) {
      ecosys[x][y] = animal;
    } else if (ecosys[y][x] != animal) {
      ecosys[y][x] = COEXIST;
    }
  }
}

void init_ecosys(char ecosys[SIZE_Y][SIZE_X]) {
  int x, y;
  for (y = 0; y < SIZE_Y; y++) {
    for (x = 0; x < SIZE_X; x++) {
      ecosys[y][x] = VIDE;
    }
  }
}

char *draw_ecosys(char **img, char ecosys[SIZE_Y][SIZE_X]){
  int x, y;
  char *ptr = NULL;
  if (*img == NULL) {
    *img = (char *)malloc(sizeof(char)*SIZE_Y*SIZE_X*2);
  }
  ptr = *img;
  for (y = 0; y < SIZE_Y; y++) {
    for (x = 0; x < SIZE_X; x++ ) {
      memset(ptr,ecosys[y][x],1);
      ptr++;
      memset(ptr++,' ',1);
    }
    ptr--;
    memset(ptr++,'\n',1);
  }
  return *img;
}

void afficher_ecosys(Animal *liste_predateur, Animal *liste_proie) {
  char *img = NULL;
  char ecosys[SIZE_Y][SIZE_X] = {};
  init_ecosys(ecosys);
  localiser_animal(ecosys, liste_predateur, PREDATEUR);
  localiser_animal(ecosys, liste_proie, PROIE);
  draw_ecosys(&img, ecosys);
  write_screan(img);
  FREE(img);
}

void open_screen(void) {
  initscr(); // Start curses mode
}

void clear_screen(void) {
  erase(); // ncurses lib
}

int write_screan(const char *fmt, ...) {
  char printf_buf[10240];
  va_list args;
  int printed;
  va_start(args, fmt);
  printed = vsprintf(printf_buf, fmt, args);
  va_end(args);
  printed = printw(printf_buf);
  refresh();
  return printed;
}

void pose_screen(void) {
  getch();  /* Wait for user input */
}
 
void close_screen(void) {
  endwin();  // End curses mode
}

int random_true_false (float prob) {
  if ( rand() / RAND_MAX > prob) {
    return TRUE;
  }
  return FALSE;
}

int random_dir (void) {
  if ( rand() / RAND_MAX > 0.5 ) {
    return DIR_DOWN;
  }
  return DIR_UP;
}

void change_dir (int *dir) {
  if (*dir == DIR_STAY) {
    *dir = random_dir();
  } else {
    *dir *= -1;
  }
}

void bouger_en_dir (Animal *animal) {
  animal->x += animal->dir[0];
  if (animal->x < 0) {
    animal->x *= -1;
    animal->dir[0] *= -1;
  } else if (animal->x > SIZE_X) {
    animal->x  = SIZE_X - ( animal->x - SIZE_X );
    animal->dir[0] *= -1;
  }
  animal->y += animal->dir[1];
  if (animal->y < 0) {
    animal->y *= -1;
    animal->dir[1] *= -1;
  } else if (animal->y > SIZE_Y) {
    animal->y  = SIZE_Y - ( animal->y - SIZE_Y );
    animal->dir[1] *= -1;
  }
}

void bouger_animaux(Animal *la, float p_ch_dir) {
  if (random_true_false(p_ch_dir)) {
    change_dir (&(la->dir[0]));
  }
  if (random_true_false(p_ch_dir)) {
    change_dir (&(la->dir[1]));
  }
  bouger_en_dir (la);
}

void reproduce(Animal **liste_animal, float p_reproduce, float energie) {
  Animal *ptr = NULL;
  for (ptr = *liste_animal; ptr != NULL; ptr = ptr->suivant) {
    if (random_true_false(p_reproduce)) {
      ajouter_animal(ptr->x, ptr->y, energie, liste_animal);
    }
  }
}

void user_energie (Animal **liste_animal, Animal *animal, float d_animal) {
  animal->energie -= d_animal;
  if ( animal->energie <= 0 ) {
    enlever_animal(liste_animal, animal);
  }
}

void random_manger (Animal **liste_proie, Animal *predateur, float p_manger) {
  Animal *ptr = NULL;
  if (random_true_false(p_manger)) {
    ptr = animal_en_XY(*liste_proie, predateur->x, predateur->y);
    if (ptr) {
      enlever_animal(liste_proie, ptr);
    }
  }
}

void rafraichir_predateurs(Animal **liste_predateur, Animal **liste_proie, float d_predateur, float p_ch_dir,  float p_reproduce, float energie,  float p_manger) {
  Animal *ptr = NULL;
  for (ptr = *liste_predateur; ptr != NULL; ptr = ptr->suivant) {
    bouger_animaux(ptr, p_ch_dir);
    reproduce(liste_predateur, p_reproduce, energie);
    user_energie(liste_predateur, ptr, d_predateur);
    random_manger (liste_proie, ptr, p_manger);
  }
}

void rafraichir_proies(Animal **liste_proie, float d_proie, float p_ch_dir,  float p_reproduce, float energie) {
  Animal *ptr = NULL;
  for (ptr = *liste_proie; ptr != NULL; ptr = ptr->suivant) {
    bouger_animaux(ptr, p_ch_dir);
    reproduce(liste_proie, p_reproduce, energie);
    user_energie(liste_proie, ptr, d_proie);
  }
}
