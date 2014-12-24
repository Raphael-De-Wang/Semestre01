#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ecosys.h"

#define NB_PROIES 20
#define NB_PREDATEURS 20

int main(void) {
  int i, count = 0;
  Animal *liste_proie = NULL;
  Animal *liste_predateur = NULL;
  int nb_proies = 0, nb_predateurs = 0;
  
  char *sb = (char*)malloc(sizeof(char)*1024);
  int  sb_use   = 0;
  
  float p_ch_dir;
  float p_manger;
  float p_reproduce;
  float d_proie;
  float d_predateur;
  float energie;

  // read input parametres

  puts("Nom du fiche de resultat: ");
  scanf("%s",sb);
  FILE *fp = fopen(sb, "w");
  
  puts("Entrer la probablité de changer la direction[float]: ");
  scanf("%f",&p_ch_dir);
  puts("Entrer la probablité de predateur manger proie[float]: ");
  scanf("%f",&p_manger);
  puts("Entrer la probablité de reproduce[float]: ");
  scanf("%f",&p_reproduce);
  puts("Entrer la consomation d'energy de proie[float]: ");
  scanf("%f",&d_proie);
  puts("Entrer la consomation d'energy de predateur[float]: ");
  scanf("%f",&d_predateur);
  puts("Entrer l'energy d'un new born[float]: ");
  scanf("%f",&energie);

  srand(time(NULL));

  for (i = 0; i < NB_PROIES; ++i) {
    ajouter_animal(rand() % SIZE_X, rand() % SIZE_Y, energie, &liste_proie);
  }
  for (i=0; i < NB_PREDATEURS; ++i) {
    ajouter_animal(rand() % SIZE_X, rand() % SIZE_Y, energie, &liste_predateur);
  }

  open_screen();

  while ( TRUE ) {
    clear_screen();
    
    nb_proies = compte_animal_rec(liste_proie);
    nb_predateurs = compte_animal_rec(liste_predateur);
    
    write_screen("Nb proies :     %d\n", nb_proies);
    write_screen("Nb predateurs : %d\n\n", nb_predateurs);
    
    sb_use = sprintf(sb, "%d %d %d\n", count, nb_proies, nb_predateurs);
    fwrite(sb, sizeof(char), sb_use, fp);

    afficher_ecosys(liste_proie,liste_predateur);
    
    write_screen("Iteration Count: %d\n", count++);
    
    if ( !(nb_proies > 0 && nb_predateurs > 0 && count < 2000 ) ) {
      break;
    }
    
    sleep(1);
    
    rafraichir_proies(&liste_proie, d_proie, p_ch_dir,  p_reproduce, energie);
    rafraichir_predateurs(&liste_predateur, &liste_proie, d_predateur, p_ch_dir,  p_reproduce, energie,  p_manger);
  } 
  
  pose_screen();
  
  /* Note : libération de la mémoire non demandée dans l'énoncé (et non fournie) */
  liberer_liste_animaux(liste_proie);
  liberer_liste_animaux(liste_predateur);
  close_screen();
  FREE(sb);
  fclose(fp);
  
  return 0;
}
