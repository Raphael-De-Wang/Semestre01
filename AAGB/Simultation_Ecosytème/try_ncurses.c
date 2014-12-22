#include <stdio.h>
#include <stdlib.h>
#include <ncurses.h>


int main()
{	
	initscr();			/* Start curses mode 		  */
	printw("Hello World !!!");	/* Print Hello World		  */
	refresh();			/* Print it on to the real screen */
	erase();
	printw("Hello World 2 !!!");	/* Print Hello World		  */
	refresh();			/* Print it on to the real screen */
	write_screan("Hello World 3 !!!!!!");
	getch();			/* Wait for user input */
	endwin();			/* End curses mode		  */

	return 0;
}
