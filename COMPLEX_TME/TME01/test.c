#include <stdio.h>
#include <stdlib.h>

int main (void) {
  int  i     = 0;
  char table[10] = "abcde";

  printf("table[%d]: %c\n", i, table[i]);
  printf("address: %c", *table++);
    
}
